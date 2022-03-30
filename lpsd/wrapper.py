"""
This module implements the LPSD algorithm by G. Heinzel and M. Troebs.
For the core part of the algorithm it uses an C implementation by M. Hewitson/G.Heinzel.
Additionally a pure python implementation is provided.
It depends on the packages numpy and ctypes.
Ref: Improved spectrum estimation from digitized time series
on a logarithmic frequency axis
https://doi.org/10.1016/j.measurement.2005.10.010
"""
from typing import Callable, Optional, Union
from warnings import warn

import numpy as np
from pandas import DataFrame, Series

from ._helpers import (
    _calc_lpsd,
    _calc_lpsd_py,
    _kaiser_alpha,
    _kaiser_rov,
    _ltf_plan,
    c_core_available,
)


def lpsd(  # pylint: disable=too-many-arguments
    data: Union[Series, DataFrame],
    sample_rate: Optional[
        float
    ] = None,  # None: Use time series data from first column to calculate sampling rate
    window_function: Callable = np.kaiser,
    overlap: Optional[float] = None,  # None: use default overlap
    detrending_order: Optional[int] = 0,
    n_frequencies: int = 1000,  # also called Jdes
    n_averages: int = 100,  # known as Kdes
    n_min_bins: int = 1,  # bmin
    min_segment_length: float = 0,  # known as Lmin
    psll: float = 200,
    use_c_core: bool = True,
) -> Union[DataFrame, dict]:
    """
    Wrapper for LPSD assuming data in a :obj:`pandas.DataFrame`.
    The sample frequency will be calculated from the index column.

    Example
    -------
    .. code-block:: python

        from lpsd import lpsd
        result = lpsd(data["column"])

    Parameters
    ----------
    data: :obj:`pandas.Series` or :obj:`pandas.DataFrame`
        Assuming time series data and takes one column of a :obj:`pandas.DataFrame`
        or a whole `DataFrame`. If multiple columns are provided, it will calculate
        the spectrum for each column and return a dict of `DataFrame`s with all
        results.
    sample_rate: :obj:`float` (optional)
        Sampling rate of the data. Defaults to calculating the mean difference of the first columns elements.
    window_function: :obj:`Callable` (optional)
        Define a window function, defaults to :obj:`np.kaiser`.
    overlap: :obj:`float` (optional)
        Overlap percentage, usually taken from the window function.
        Defaults to recommended overlap.
    detrending_order: :obj:`int` (optional)
        Order for detrending, 0 = offset, 1 = linear, :obj:`None` to disable.
        Also just called *order*.
    n_frequencies: :obj:`int` (optional)
        The desired number of frequencies.
        Also called *Jdes*.
    n_averages: :obj:`int` (optional)
        The desired number of averages
        Also called *Kdes*.
    n_min_bins: :obj:`int` (optional)
        The minimum bin number to be used, usually taken from the window function.
        Also called *bmin*.
    min_segment_length: :obj:`float` (optional)
        The minimum segment length.
        Also called *Lmin*.
    psll: :obj:`float` (optional)
        Peak side-lobe level.
    use_c_core: :obj:`bool`
        Use the C core or the pure Python implementation.

    Returns
    -------
    result: :obj:`openqlab.io.DataFrame` or :obj:`dict(openqlab.io.DataFrame)`
        calculated frequency data with the columns:
        ps, psd, ps_std, psd_std, enbw, asd

    """

    def run_lpsd(data):
        lpsd_runner = _calc_lpsd if use_c_core else _calc_lpsd_py
        result = lpsd_runner(
            data.to_numpy(),
            f,
            r,
            m,
            L,
            sample_rate,
            window_function,
            psll,
            detrending_order,
            overlap,
            min_segment_length,
        )

        dc = DataFrame(
            dict(
                ps=result[0],
                psd=result[1],
                ps_std=result[2],
                psd_std=result[3],
                enbw=result[4],
                asd=result[5],
            ),
            index=f,
        )
        dc.index.name = "frequency"
        return dc

    if sample_rate is None:
        index_diff = np.diff(data.index)
        period_time = np.median(index_diff)
        sample_rate = 1 / period_time
        std = index_diff.std()

        if std / period_time > 1e-6:
            warn(
                "Length of some time steps deviates a lot from the median. Some data maybe corrupt!\n"
                f"Period time: {period_time}, standard deviation: {std}",
                UserWarning,
            )

    if use_c_core and not c_core_available():
        warn(
            "C core should be used, but is not available. Using Python …",
            RuntimeWarning,
        )
        warn(
            f"use_c_core: {use_c_core}, c_core_available: {c_core_available()}",
            RuntimeWarning,
        )
        use_c_core = False

    if isinstance(data, DataFrame) and len(data.columns) == 1:
        data = data[data.columns[0]]

    if detrending_order is None:
        detrending_order = -1

    n_data = len(data)

    if window_function is np.kaiser:
        # calculate kaiser parameters
        alpha = _kaiser_alpha(psll)
        if overlap is None:
            overlap = _kaiser_rov(alpha)

    f, r, m, L, K = _ltf_plan(  # pylint: disable=unused-variable
        n_data,
        sample_rate,
        overlap,
        n_min_bins,
        min_segment_length,
        n_frequencies,
        n_averages,
    )

    if isinstance(data, DataFrame):
        dc = {}
        for col in data.columns:
            dc[col] = run_lpsd(data[col])
    else:
        dc = run_lpsd(data)

    return dc


def lpsd_trad(
    x, fs, olap, bmin, Lmin, Jdes, Kdes, order, win, psll, use_c_core: bool = True
):  # pylint: disable=too-many-arguments
    """
    Traditional wrapper for the LPSD algorithm.

    Parameters:
        x (list of float): The length of the time-series to be processed
        f (list of float): The frequency
        r (list of float): Frequency resolution
        m (int): Bin number
        L (list of float): Segment lengths
        fs (int): Sampling frequency
        win (function): Window function to be used
        psll (float): Peak side-lobe level
        order (int): Order
        olap (float): Overlap percentage, usually taken from the window function
            - choose "default" for recommended overlap
        Lmin (int): The minimum segment length

    Returns:
        S (list of float): Power spectrum
        Sxx (list of float): Power spectral density
        dev (list of float): Standard deviation PS
        devxx (list of float): Standard deviation PSD
        ENBW (list of float): Equivalent noise bandwidth
    """
    if use_c_core and not c_core_available():
        warn(
            "C core should be used, but is not available. Using Python …",
            RuntimeWarning,
        )
        warn(
            f"use_c_core: {use_c_core}, c_core_available: {c_core_available()}",
            RuntimeWarning,
        )
        use_c_core = False

    Ndata = len(x)

    if win is np.kaiser:
        # calculate kaiser parameters
        alpha = _kaiser_alpha(psll)
        if olap == "default":
            olap = _kaiser_rov(alpha)

    f, r, m, L, K = _ltf_plan(  # pylint: disable=unused-variable
        Ndata, fs, olap, bmin, Lmin, Jdes, Kdes
    )

    if use_c_core:
        return [f] + _calc_lpsd(x, f, r, m, L, fs, win, psll, order, olap, Lmin)
    else:
        return [f] + _calc_lpsd_py(x, f, r, m, L, fs, win, psll, order, olap, Lmin)
