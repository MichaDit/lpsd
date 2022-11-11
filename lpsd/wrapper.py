"""
This module implements the LPSD algorithm by G. Heinzel and M. Troebs.
For the core part of the algorithm it uses an C implementation by M. Hewitson/G.Heinzel.
Additionally a pure python implementation is provided.
It depends on the packages numpy and ctypes.
Ref: Improved spectrum estimation from digitized time series
on a logarithmic frequency axis
https://doi.org/10.1016/j.measurement.2005.10.010
"""
from typing import Union
from pandas import DataFrame

from ._lcsd import LCSD


def lcsd(data: DataFrame, *args, **kwargs) -> DataFrame:
    """
    Wrapper for LCSD assuming data in a :obj:`pandas.DataFrame`.
    Only 2-column DataFrames are accepted on input, to calculate CSD(col1,col2)
    The sample frequency will be calculated from the index column.

    Example
    -------
    .. code-block:: python

        from lpsd import lcsd
        result = lcsd(data) #  data is pandas.DataFrame with two columns

    Parameters
    ----------
    data: :obj:`pandas.DataFrame`
        Assuming time series data and takes one column of a :obj:`pandas.DataFrame`
        as signal1 and second column as signal2, returns `DataFrame`s with CSD(signal1,signal2)
        and all additional results
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
    result: :obj:`openqlab.io.DataFrame`
        calculated frequency data with the columns:
        ps, psd, ps_std, psd_std, enbw, asd

    """
    kwargs["csd"] = True
    lcsd = LCSD(data, *args, **kwargs)
    lcsd.check_inputs()
    return lcsd.run()


def lpsd(data: DataFrame, *args, **kwargs) -> Union[DataFrame, dict]:
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

    lcsd = LCSD(data, *args, **kwargs)
    lcsd.check_inputs()
    return lcsd.run()


def lpsd_trad(*args, **kwargs):  # pylint: disable=too-many-arguments
    raise NotImplementedError(
        "Since version 1.0 `lpsd_trad()` is no longer available. Consider using `lpsd()` or "
        "downgrading to version 0.2.2"
    )
