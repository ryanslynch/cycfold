import warnings
import random
import string
import subprocess
import pickle
import numpy as np
import matplotlib.pyplot as plt
from presto.polycos import polycos as presto_polycos
from presto.psr_utils import delay_from_DM
from argparse import ArgumentParser
from blimpy.guppi import GuppiRaw
from presto.parfile import psr_par
try:
    import cupy as cp
    from cupy import fft,roll
    cuda = True
except ModuleNotFoundError:
    from numpy import fft,roll
    cuda = False

# Optimal FFT size is chosen to optimize computing many small FFTs vs
# a few large FFTs.  The values below were determined empirically
# using numpy.fft.fft and cupy.fft.fft on a 2**26-point complex-value
# input using an Intel Xeon Gold 5217 CPU and an nVidia RTX 3080 GPU.
# The limiting factor when using CUDA seems to be GPU memory, rather
# than computational speed.
OPTIMAL_FFT_SIZE = 2**22 if cuda else 2**12
# Optimal size of a correlation block is chosen to optimize computing
# many small correlations vs a few large FFTs.  The value below was
# determined empirically using numpy.fft.fft 2**22-point complex-value
# input using an Intel Xeon Gold 5217 CPU.  There was no significant
# performance benefit to using a GPU when testing with an nVidia RTX
# 3080.
OPTIMAL_CORRELATION_SIZE = 2**22 if cuda else 2**12 
verbose = True

observatory_codes = {
    "GBT": 1,
    "ARECIBO": 3,
}

def get_obs_info(header):
    info = {}
    info["obscode"] = observatory_codes[header["TELESCOP"].upper()]
    info["npol"] = int(header["NRCVR"])
    info["nbits"] = int(header["NBITS"])
    info["fctr"] = float(header["OBSFREQ"])
    info["bw"] = float(header["OBSBW"])
    info["band_sign"] = int(np.sign(info["bw"]))
    info["nchan_pfb"] = int(header["OBSNCHAN"])
    info["chanbw"] = info["bw"]/info["nchan_pfb"]
    info["freqs_pfb"] = info["fctr"] - 0.5*info["bw"] \
                        + (np.arange(info["nchan_pfb"])+0.5)*info["chanbw"]
    info["f_low"] = info["fctr"] - 0.5*(np.abs(info["bw"] + info["chanbw"]))
    info["f_high"] = np.max(info["freqs_pfb"])
    info["dt"] = float(header["TBIN"])
    info["start_mjd"] = int(header["STT_IMJD"]) \
                        + (int(header["STT_SMJD"]) \
                        + float(header["STT_OFFS"]))/86400.0
    info["nsamples"] = int(header["BLOCSIZE"])//info["nchan_pfb"]\
                           //info["npol"]//(info["nbits"]//4)
    info["noverlap"] = int(header["OVERLAP"])

    return info


def prepare(header,data,max_lag=0):
    """
    Prepare data for processing and collect important observing
    information.
    """
    info = get_obs_info(header)
    info["max_lag"] = max_lag
    data = np.moveaxis(data,-1,0)

    return info,data

def correlate(in1,in2,lags,axis=-1):
    """
    Compute the point-wise cross-correlation between in1 and in2 for the
    given lags and along the given axis.

    Parameters
    ----------
    in1, in2 : array_like
        Input arrays.  One input must be larger than the other along
        the specified axis by at least the largest lag value to ensure
        complete overlap.
    lags : array_like
        The lags at which to compute the cross-correlation.
    axis : int, optional
        The axis over which to compute the cross-correlation.

    Returns
    -------
    out : ndarray
        The point-wise cross correlation at the given lags.  A new
        axis of size len(lags) is appended to store the correlation
        product for each lag.

    Notes
    -----
    The definition of cross-correlation is not unique.  This function
    uses

    r_{xy}[n,k] = x[n+k] * conj(y[n])

    where n is sample index and k is the lag.  This assumes that x
    extends beyond y to the right, so x is chosen to be the larger of
    in1 or in2, and vice versa for y.
    """
    assert np.abs(in1.shape[axis]-in2.shape[axis]) >= np.max(lags), (
        "One input array must be larger than the other by at least "
        "max(lags) along the specified axis.")
    lags = np.atleast_1d(lags)
    out_type=complex if np.iscomplexobj(in1) or np.iscomplexobj(in2) else float
    if in1.shape[axis] > in2.shape[axis]:
        lagged = in1
        unlagged = in2
    else:
        lagged = in2
        unlagged = in1
    if cuda:
        lagged = cp.asarray(lagged)
        unlagged = cp.asarray(unlagged)
    ncorr = np.min((unlagged.shape[axis],OPTIMAL_CORRELATION_SIZE))
    while unlagged.shape[axis]%ncorr: ncorr += 2
    nseg = unlagged.shape[axis]//ncorr
    out = np.empty((*unlagged.shape,len(lags)),dtype=out_type)
    for iseg in range(nseg):
        if verbose: print(f"Working on correlation segment {iseg+1} of {nseg}")
        for ilag,lag in enumerate(lags):
            to_take_unlagged = range(iseg*ncorr,(iseg+1)*ncorr)
            to_take_lagged = range(iseg*ncorr+lag,(iseg+1)*ncorr+lag)
            tmp_lagged = np.take(lagged,indices=to_take_lagged,axis=axis)
            tmp_unlagged = np.take(unlagged,indices=to_take_unlagged,axis=axis)
            corr = tmp_lagged*tmp_unlagged.conjugate()
            if cuda: corr = corr.get()
            idx = list(map(range,unlagged.shape))
            idx[axis] = range(iseg*ncorr,(iseg+1)*ncorr)
            out[...,ilag][np.ix_(*idx)] = corr
            #del tmp_lagged
            #del tmp_unlagged
            #del corr
            #tmp_out = unlagged.conjugate()*roll(
            #    lagged,-lag,axis=axis)[...,:unlagged.shape[axis]]
            #if cuda:
            #    tmp_out = tmp_out.get()
            #out[...,ilag] = tmp_out

    return out

def get_dedispersion_params(DM,obs_info):
    """
    Calculate optimal dedispersion parameters.
    """
    # Determine the length of the dedispersion sweep in samples.
    kdm = 4.148808e9 # MHz * cm**3 * pc**-1
    f1 = obs_info["f_low"]
    f2 = f1 + np.abs(obs_info["chanbw"])
    tdm = kdm/1e6*DM*(1/f1**2 - 1/f2**2)
    ndm = int(np.ceil(tdm/obs_info["dt"]/2.0))*2 # Rounds to next even
    # integer nout is the number of valid data points in a single raw
    # block after deconvolution. It does not include ndm//2 points of
    # padding that will be added at the start of a block, or ndm//2
    # points at the end of the block that will be discarded to avoid
    # circular deconvultion effects.  It does include nlag-1 points
    # that will be retained in coherent dedispersion but discarded
    # after correlation and cyclic folding.
    nout = obs_info["nsamples"] - obs_info["noverlap"] + obs_info["max_lag"] - 1
    # nprocess is the total number of data points that will be
    # processed during coherent dedispersion. This is equal to nout
    # plus ndm//2 points of padding added at the beginning and end
    # that are then discarded to avoid circular deconvolution effects.
    nprocess = nout + ndm
    # FFT size is calculated in three steps: 
    # 1) Take the larger of 2*ndm and OPTIMAL_FFT_SIZE
    # 2) Take the smaller of step 1) and OPTIMAL_FFT_SIZE
    # 3) Increase nfft until nfft-ndm (the number of valid points
    #    after each FFT) is an integer multiple of nout (total number 
    #    of valid points in the raw block)
    nfft = np.max((2*ndm,OPTIMAL_FFT_SIZE))
    nfft = np.min((nfft,nprocess))
    while nout%(nfft-ndm): nfft += 2
    # nvalid is the total number of valid data points in one FFT step
    nvalid = nfft - ndm
    # nseg is the number of segments that the block will be split into
    nseg = nout//nvalid

    return ndm,nfft,nvalid,nseg,nout

def coherently_dedisperse(x,DM,obs_info,left_pad=None,right_pad=None,axis=-1):
    """
    Apply a coherent dedispersion filter along the given axis.
    """
    kdm = 4.148808e9 # MHz * cm**3 * pc**-1
    ndm,nfft,nvalid,nseg,nout = get_dedispersion_params(DM,obs_info)
    f0 = obs_info["freqs_pfb"][:,None]
    f = np.fft.fftfreq(nfft,d=obs_info["dt"]*1e6)[None,:]
    H = np.exp(obs_info["band_sign"]*2j*np.pi*kdm*DM*f**2/(f0**2*(f+f0)))
    x_padded = x.copy()
    if left_pad is not None:
        assert left_pad.shape[axis] == ndm//2, \
            f"left_pad must have length of {ndm//2} (ndm/2) along axis {axis}"
        x_padded = np.concatenate((left_pad,x),axis=axis)
    if right_pad is not None:
        assert right_pad.shape[axis] == ndm//2, \
            f"right_pad must have length of {ndm//2} (ndm/2) along axis {axis}"
        x_padded = np.concatenate((right_pad,x),axis=axis)
    out_shape = list(x.shape)
    out_shape[axis] = nout
    out_data = np.empty(out_shape,dtype=complex)
    for iseg in range(nseg):
        if verbose: print(f"Working on FFT segment {iseg+1} of {nseg}")
        to_take = range(iseg*nvalid,iseg*nvalid+nfft)
        X = np.fft.fft(np.take(x,indices=to_take,axis=axis),axis=axis)
        X /= H
        dedispersed = np.fft.ifft(X,axis=axis)
        out_data[...,iseg*nvalid:(iseg+1)*nvalid] = np.take(
            dedispersed,indices=range(ndm//2,nfft-ndm//2),axis=axis)
        del X
        del dedispersed
        
    return out_data

def cyclic_spectrum(data,obs_info,nlag):
    """
    Calculate and fold the cyclic spectrum.
    """
    out_type = complex if np.iscomplexobj(data) else float
    x = data[0]
    y = data[1]
    freq_axis = 0
    time_axis = 1
    nchan = obs_info["nchan_pfb"]*2*(nlag-1)
    lags = np.arange(nlag)
    R = np.empty((2,obs_info["nchan_pfb"],data.shape[-1]-(nlag-1),nlag),
                 dtype=out_type)
    R[0] = correlate(x,x[...,:-(nlag-1)],lags,axis=time_axis)
    R[1] = correlate(y,y[...,:-(nlag-1)],lags,axis=time_axis)
    #R[2] = correlate(x,y[...,:-(nlag-1)],lags,axis=time_axis)
    #R[3] = correlate(y,x[...,:-(nlag-1)],lags,axis=time_axis)
    Z = np.fft.fftshift(np.fft.hfft(R,axis=-1),axes=-1)
    Z = np.moveaxis(Z,-1,2).reshape((2,nchan,-1))
    del R

    return Z

def make_polycos(parfile,obs_info,out_file=None):
    par = psr_par(parfile)
    if not out_file:
        tmp = ''.join(random.choices(string.ascii_uppercase+string.digits,k=8))
        out_file = f"/tmp/{par.PSR}_{tmp}.polycos"
    cmd = (f"tempo -f {parfile} -z -ZPSR={par.PSR} -ZOBS={info['obscode']} "
           f"-ZSTART={info['start_mjd']} -ZTOBS=24H -ZFREQ={info['f_high']} "
           f"-ZOUT={out_file}")
    subprocess.call(
        cmd.split(),stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    pcs = presto_polycos(par.PSR,filenm=out_file)

    return pcs

def fold(data,mjds,nbin,polycos,axis=-1):
    profile = np.zeros((data.shape[0],data.shape[1],nbin))
    #profile = np.empty((data.shape[0],nbin))
    samples = np.zeros_like(profile)
    for isamp,mjd in enumerate(mjds):
        if verbose: print(f"Working on sample {isamp+1} of {len(mjds)}")
        mjdi = int(mjd)
        mjdf = mjd - mjdi
        phase = polycos.get_phase(mjdi,mjdf)
        phase_bin = int(phase*nbin)
        profile[...,phase_bin] += data[...,isamp]
        samples[...,phase_bin] += 1
    profile /= samples

    return profile

def incoherently_dedisperse(data,obs_info,DM,f_rot,axis=-1):
    freqs = np.repeat(
        obs_info["freqs_pfb"],data.shape[1]//obs_info["nchan_pfb"])
    DM_time_delays=delay_from_DM(DM,freqs)-delay_from_DM(DM,obs_info["f_high"])
    DM_phase_delays = DM_time_delays*f_rot
    DM_bin_delays = DM_phase_delays*data.shape[axis]
    fft_freqs = np.arange(data.shape[axis]//2+1)
    phasor = np.exp(2j*np.pi*fft_freqs[None,:]*DM_bin_delays[:,None]/data.shape[axis])
    X = np.fft.rfft(data,axis=axis)
    dedispersed = np.fft.irfft(phasor*X,data.shape[axis],axis=axis)
    
    return dedispersed

    

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-c","--ncyc",type=int,default=128,
                        help="Number of cyclic channels per PFB channel")
    parser.add_argument("-b","--nbin",type=int,default=512,
                        help="Number of pulse profile bins")
    parser.add_argument("-E","--parfile",type=str,
                        help="TEMPO1-style parfile")
    parser.add_argument("-P","--polycos",type=str,
                        help="Pre-generated polycos")
    parser.add_argument("-L","--subint-length",type=float,default=10.0,
                        help="Sub-integration length (s)")
    parser.add_argument("-n","--nblocks",type=int,
                        help="Number of raw blocks to process")
    parser.add_argument("infiles",nargs="+")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    nlag = args.ncyc//2 + 1
    par = psr_par(args.parfile)
    total_samples_folded = 0
    for ifile,filenm in enumerate(args.infiles):
        gr = GuppiRaw(filenm)
        blocks_to_process = args.nblocks if args.nblocks else gr.n_blocks
        for iblock in range(blocks_to_process):
            header,data = gr.read_next_data_block()
            info,data = prepare(header,data,max_lag=nlag)
            
            if ifile==0 and iblock==0:
                ndm = get_dedispersion_params(par.DM,info)[0]
                left_pad = np.zeros((info["npol"],info["nchan_pfb"],ndm//2),
                                    dtype=data.dtype)
                if args.polycos is None:
                    pcs = make_polycos(args.parfile,info)
                else:
                    pcs = presto_polycos(par.PSR,filenm=args.polycos)
                profile = np.zeros((2,info["nchan_pfb"]*args.ncyc,args.nbin))
                profiles = []
                last_dump_time = info["start_mjd"]
                
            data_dd = coherently_dedisperse(data,par.DM,info,left_pad=left_pad)
            pad_idx0 = -info["noverlap"]+nlag-1
            pad_idx1 = pad_idx0 + ndm//2
            left_pad = data[...,pad_idx0:pad_idx1] # Used on next pass
            
            Z = cyclic_spectrum(data_dd,info,nlag)
            mjds = info["start_mjd"] + \
                  info["dt"]*(total_samples_folded+np.arange(Z.shape[-1]))/86400
            imjd_mid = int(np.mean(mjds))
            fmjd_mid = np.mean(mjds) - imjd_mid
            tmp_profile = fold(Z,mjds,args.nbin,pcs)
            f_rot = pcs.get_freq(imjd_mid,fmjd_mid)
            profile_dd = incoherently_dedisperse(tmp_profile,info,par.DM,f_rot)
            profile += profile_dd
            total_samples_folded += Z.shape[-1]
            if (mjds[-1] - last_dump_time)*86400 > args.subint_length:
                profiles.append(profile)
                last_dump_time = mjds[-1]
    # Always append last profile
    profiles.append(profile)

    with open(f"{args.infiles[0]+'.pkl'}","wb") as f: pickle.dump(profiles,f)
