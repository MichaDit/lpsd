"""Unchanged upstream planning and normalization without the reference wrapper.

These five function bodies are copied verbatim from the pinned lpsd 1.0.6
``lpsd/_helpers.py``. A source-equivalence regression prevents drift. Keeping
this small NumPy-only subset here avoids importing SciPy's detrending backend
for the native API. The original reference package remains byte-for-byte intact.

GPL-3.0-or-later; see docs/provenance.md.
"""
import math

import numpy as np

__all__ = ["_asdrms", "_kaiser_alpha", "_kaiser_rov", "_ltf_plan"]


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
