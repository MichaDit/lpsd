import math
from pathlib import Path
from sys import platform
from typing import Callable
from warnings import warn

import numpy as np
from scipy.signal import detrend

try:
    import ctypes as ct

    _ctypes_available = True
except ImportError:
    _ctypes_available = False


def c_core_available() -> bool:
    try:
        if not _ctypes_available:
            raise OSError()
        _dft()

        return True
    except OSError as e:
        warn(f"C core not available: {e}", RuntimeWarning)
        return False


def _dft() -> Callable:
    platforms: dict = {
        "win32": dict(libname="ltpda_dft.dll", call="WinDLL"),
        "linux": dict(libname="ltpda_dft.so", call="CDLL"),
        "cygwin": dict(libname="ltpda_dft.so", call="CDLL"),
        "darwin": dict(libname="ltpda_dft.so", call="CDLL"),
    }
    p_dict = platforms[platform]
    dll_abs_path = Path(__file__).parent / p_dict["libname"]
    lib = getattr(ct, p_dict["call"])(str(dll_abs_path), use_last_error=True)

    # function gives pointer to numpy array
    ndpointer = np.ctypeslib.ndpointer
    # make dft function accessible in python
    dft = lib.dft

    # c function to use in python: void dft(double *Pr, double *Vr, long int *Navs, double *xdata, long int nData, long int segLen, double *Cr, double *Ci, double olap, int order)
    # specify return type and argument type of dft function

    dft.restype = None
    dft.argtypes = [
        ct.POINTER(ct.c_double),
        ct.POINTER(ct.c_double),
        ct.POINTER(ct.c_long),
        ndpointer(ct.c_double, flags="C_CONTIGUOUS"),
        ndpointer(ct.c_double, flags="C_CONTIGUOUS"),
        ct.c_long,
        ct.c_long,
        ndpointer(ct.c_double, flags="C_CONTIGUOUS"),
        ndpointer(ct.c_double, flags="C_CONTIGUOUS"),
        ct.c_double,
        ct.c_int,
        ct.c_bool,
    ]
    return dft


def _calc_lcsd_py(
    x1, x2, f, r, m, L, fs, win, psll, order, olap, Lmin, csd
):  # pylint: disable=too-many-arguments,unused-argument,too-many-branches
    """
    Computes the LCSD algorithm like in the Matlab LTPDA implementation,
    generalized to CSD

    Parameters:
        x1 (list of float): first time-series to be processed
        x2 (list of float): second time-series to be processed
        f (list of float): The frequency
        r (list of float): Frequency resolution (not implemented)
        m (int): Bin number
        L (list of float): Segment lengths
        fs (int): Sampling frequency
        psll (float): Peak side-lobe level
        win (function): Window function to be used
        order (int): Order
        olap (float): Overlap (<=1)
        Lmin (int): The minimum segment length
        csd (bool): whether calculating CSD (True) or PSD (False)

    Returns:
        S (list of float): Power spectrum
        Sxx (list of float): Power spectral density
        dev (list of float): Standard deviation PS
        devxx (list of float): Standard deviation PSD
        ENBW (list of float): Equivalent noise bandwidth
    """
    # TODO frequency resolution r not implemented

    # check window function
    win_kaiser = False
    beta = 0
    alpha = 0

    if win is np.kaiser:
        win_kaiser = True
        # calculate kaiser parameter
        alpha = _kaiser_alpha(psll)
        beta = alpha * np.pi

    # define constants
    twopi = 2 * np.pi

    # nx = len(x)
    nf = len(f)
    # initialize outputs
    Sxx = np.zeros(nf, dtype=np.float64)
    S = np.zeros(nf, dtype=np.float64)
    ENBW = np.zeros(nf, dtype=np.float64)
    devxx = np.zeros(nf, dtype=np.float64)
    dev = np.zeros(nf, dtype=np.float64)
    asd = np.zeros(nf, dtype=np.float64)

    # disp_each = _myround(nf / 100) * 10

    x1data = np.array(x1)
    x2data = np.array(x2)

    minReached = False

    for i in range(nf):

        # compute DFT exponent and window
        l = int(L[i])  # segment length

        if not minReached:
            if not win_kaiser:
                window = win(l)
            else:
                # adjust to make window asymmetric (consistent with LTPDA implementation)
                window = win(l + 1, beta)[0:-1]
        if l == Lmin:
            minReached = True

        p = 1j * twopi * m[i] / l * np.arange(0, l)
        C = window * np.exp(p)
        if i == 0:
            C_r = np.array(C.real, dtype=np.float64)
            C_i = np.array(C.imag, dtype=np.float64)
        else:
            C_r = np.append(C_r, C.real)
            C_i = np.append(C_i, C.imag)

        # do segments
        Xr = 0.0
        Qr = 0.0
        Mr = 0.0
        M2 = 0.0

        # Compute the number of averages we want here
        segLen = l  # Segment length
        nData = len(x1data)
        ovfact = 1 / (1 - olap)

        davg = (((nData - segLen)) * ovfact) / segLen + 1
        navg = _myround(davg)

        # Compute steps between segments
        if navg == 1:
            shift = 1.0
        else:
            shift = (float)(nData - segLen) / (float)(navg - 1)
        shift = max(shift, 1.0)

        # changed to 0.0 for python
        start = 0.0

        for j in range(navg):
            # compute start index
            istart = int(_myround(start))
            start = start + shift

            # get segment
            x1s = x1data[istart : istart + l]
            x2s = x2data[istart : istart + l]

            # detrend segment
            _detrend(x1s, order)
            if csd:
                _detrend(x2s, order)
            else:  # x1s is the same data as x2s in this case - simply copy detrending result
                x2s = x1s.copy()

            # make DFT
            a1 = np.dot(C, x1s)
            a2 = np.dot(C, x2s)

            # Welford's algorithm to update mean and variance (see C code)

            if j == 0:
                Mr = a1 * np.conj(a2)
            else:
                Xr = a1 * np.conj(a2)
                Qr = Xr - Mr
                Mr += Qr / j
                M2 += Qr * (Xr - Mr)

        A2ns = 2.0 * Mr.real
        S1 = np.sum(window)
        S12 = S1 * S1
        S2 = np.sum(window**2)

        ENBW[i] = fs * S2 / S12
        Sxx[i] = A2ns / fs / S2
        S[i] = A2ns / S12
        asd[i] = np.sqrt(Sxx[i])

    asdrms, _ = _asdrms(asd, f)

    return [S, Sxx, dev, devxx, ENBW, asd, asdrms]


def _kaiser_alpha(psll):
    """
    KAISER_ALPHA returns the alpha parameter that gives the required input PSLL.

    Taken from C code of Gerhard Heinzel:

    Compute the parameter alpha of Kaiser windows
    from the required PSLL [dB]. Best-fit polynomial
    was obtained from 180 data points between alpha=1
    and alpha=9.95. Maximum error is 0.05
    Maximum error for PSLL > 30 dB is 0.02
    """

    a0 = -0.0821377
    a1 = 4.71469
    a2 = -0.493285
    a3 = 0.0889732

    x = psll / 100
    return ((((a3 * x) + a2) * x) + a1) * x + a0


def _myround(val):
    # implement _myround to be consistent with other implementations
    if (float(val) % 1) >= 0.5:
        x = math.ceil(val)
    else:
        x = round(val)
    return x


def _kaiser_rov(alpha):
    """
    KAISER_ROV returns the recommended overlap for a Kaiser window with parameter alpha.

    Taken from C code of Gerhard Heinzel:

    Compute the 'recommended overlap' (ROV) of Kaiser windows
    from the parameter alpha. Best-fit polynomial
    was obtained from 180 data points between alpha=1
    and alpha=9.95. Maximum error is 1.5%, mainly due
    to insufficient precision in the data points
    """
    a0 = 0.0061076
    a1 = 0.00912223
    a2 = -0.000925946
    a3 = 4.42204e-05
    x = alpha
    return (100 - 1 / (((((a3 * x) + a2) * x) + a1) * x + a0)) / 100


def _ltf_plan(Ndata, fs, olap, bmin, Lmin, Jdes, Kdes):
    """
    Computes the input values needed for the LPSD algorithm

    Parameters:
        Ndata (int): The length of the time-series to be processed
        fs (float): The sample rate of the time-series to be processed
        olap (float): Overlap percentage, usually taken from the window function
        bmin (int): the minimum bin number to be used, usually taken from the window function
        Lmin (float): The minimum segment length
        Jdes (int): The desired number of frequencies
        Kdes (int): The desired number of averages

    Returns:
        f (list of float): The frequency
        r (list of float): Frequency resolution (Hz)
        b (list of float): Bin number
        L (list of float): Segment lengths
        K (list of float): Number of averages
    """

    # set up some variables
    xov = 1 - olap
    fmin = fs / Ndata * bmin
    fmax = fs / 2
    fresmin = fs / Ndata
    freslim = fresmin * (1 + xov * (Kdes - 1))
    logfact = (Ndata / 2) ** (1 / Jdes) - 1

    # prepare outputs
    f = []
    r = []
    m = []
    L = []
    K = []

    # loop over frequencies
    fi = fmin
    while fi < fmax:
        fres = fi * logfact
        # TODO should be less not less equal
        if fres <= freslim:
            fres = np.sqrt(fres * freslim)
        fres = max(fres, fresmin)

        fbin = fi / fres
        if fbin < bmin:
            fbin = bmin
            fres = fi / fbin

        dftlen = _myround(fs / fres)
        dftlen = min(dftlen, Ndata)
        dftlen = max(dftlen, Lmin)

        nseg = _myround((Ndata - dftlen) / (xov * dftlen) + 1)
        if nseg == 1:
            dftlen = Ndata

        fres = fs / dftlen
        fbin = fi / fres

        # store outputs
        f.append(fi)
        r.append(fres)
        m.append(fbin)
        L.append(dftlen)
        K.append(nseg)

        fi = fi + fres

    return [f, r, m, L, K]


def _calc_lcsd(
    x1, x2, f, r, m, L, fs, win, psll, order, olap, Lmin, csd
):  # pylint: disable=too-many-arguments,unused-argument
    """
    Computes the LPSD algorithm like in the Matlab LTPDA implementation

    Parameters:
        x (list of float): The length of the time-series to be processed
        f (list of float): The frequency
        r (list of float): Frequency resolution
        m (int): Bin number
        L (list of float): Segment lengths
        fs (int): Sampling frequency
        psll (float): Peak side-lobe level
        win (function): Window function to be used
        order (int): Order
        olap (float): Overlap percentage, usually taken from the window function
        Lmin (int): The minimum segment length

    Returns:
        S (list of float): Power spectrum
        Sxx (list of float): Power spectral density
        dev (list of float): Standard deviation PS
        devxx (list of float): Standard deviation PSD
        ENBW (list of float): Equivalent noise bandwidth
    """
    olap *= 100

    # check window function
    win_kaiser = False
    beta = 0
    alpha = 0
    if win is np.kaiser:
        win_kaiser = True
        # calculate kaiser parameters
        alpha = _kaiser_alpha(psll)
        beta = alpha * np.pi

    # number of frequency bins
    nf = len(f)

    # get C core
    dft = _dft()

    # initialize outputs
    Sxx = np.zeros(nf, dtype=np.float64)
    S = np.zeros(nf, dtype=np.float64)
    ENBW = np.zeros(nf, dtype=np.float64)
    devxx = np.zeros(nf, dtype=np.float64)
    dev = np.zeros(nf, dtype=np.float64)
    asd = np.zeros(nf, dtype=np.float64)

    # disp_each = _myround(nf / 100) * 10
    min_reached = False

    # initialize dft outputs
    Pr = ct.c_double(0)
    Vr = ct.c_double(0)
    nsegs = ct.c_long(0)

    # pointer to data
    x1data = np.array(x1)
    # length should be the same for both inputs
    nData = ct.c_long(len(x1))
    x2data = np.array(x2)

    for i in range(nf):

        # compute DFT exponent and window
        l = int(L[i])  # segment length

        if not min_reached:
            if not win_kaiser:
                window = win(l)
            else:
                # adjust to make window asymmetric (consistent with LTPDA implementation)
                window = win(l + 1, beta)[0:-1]
        if l == Lmin:
            min_reached = True

        p = 1j * 2 * np.pi * m[i] / l * np.arange(0, l)
        C = window * np.exp(p)

        Cr = np.array(C.real, dtype=np.float64)  # Real part of DFT coefficients
        Ci = np.array(C.imag, dtype=np.float64)  # Imag part of DFT coefficients

        # Core DFT part implemented in C file
        dft(
            ct.byref(Pr),
            ct.byref(Vr),
            ct.byref(nsegs),
            x1data,
            x2data,
            nData,
            ct.c_long(l),
            Cr,
            Ci,
            ct.c_double(olap),
            ct.c_int(order),
            ct.c_bool(csd),
        )

        A2ns = 2.0 * Pr.value
        B2ns = 4.0 * Vr.value / nsegs.value
        S1 = sum(window)
        S12 = S1 * S1
        S2 = sum(window**2)
        ENBW[i] = fs * S2 / S12
        # Scale PS/PSD
        Sxx[i] = A2ns / fs / S2
        S[i] = A2ns / S12
        # Scale sqrt(variance)
        devxx[i] = np.sqrt(B2ns / fs**2 / S2**2)
        dev[i] = np.sqrt(B2ns / S12**2)
        asd[i] = np.sqrt(Sxx[i])

    asdrms, _ = _asdrms(asd, f)

    return [S, Sxx, dev, devxx, ENBW, asd, asdrms]


def _asdrms(asd_in, freq_in, f_start=None):  # TODO: allow user to specify f_start here
    """
    Calculates the high-to-low RMS of an ASD.

    Conor Mow-Lowry June 20 2016, updated November 13 2017 (original MATLAB code)
    Artem Basalaev 20 April 2022 (python version)

    Parameters
    ----------
     asd_in: array_like
        input amplitude spectra density, from low-to-high frequency
     freq_in: array_like
        linearly-spaced frequency vector for ASD [Hz]
     f_start: float
        frequency to begin accumulating RMS (optional)

    Returns
    -------
    rms_out: numpy::Array
        high-to-low frequency cumulative RMS
    freq_out: numpy::Array
        (cropped) output frequency vector (optional)
    """

    freq_out = freq_in
    if f_start is not None:  # Cutting away data above f_start
        n_cut = np.where(freq_in > f_start)[0][0]  # Index for start frequency
        freq_out = freq_in[0:n_cut]
        asd_in = asd_in[0:n_cut]

    bin_widths = np.diff(freq_out)
    bin_widths = np.append(
        bin_widths[0], bin_widths
    )  # add an extra "bin" of the same size as first one
    rms_out = np.sqrt(np.flip(np.cumsum(np.flip(asd_in * asd_in * bin_widths))))
    return rms_out, freq_out


def _detrend(data, order):
    """
    Detrend time series segment. Overwrites input `data` with detrended version

    Parameters
    ----------
     data: array_like
        input time series segment
     order: float
        detrending order
    """

    if order == -1:
        return  # do nothing
    elif order == 0:
        data = data - np.mean(data)
    elif order == 1:
        detrend(data, overwrite_data=True)
    else:
        # TODO implement scipy detrending
        # data = polydetrend(data, order)
        warn(
            "Polynomial detrending is not implemented in Python, yet. Try the C version.",
            UserWarning,
        )
        return
