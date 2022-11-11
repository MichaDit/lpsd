"""
This module contains LCSD class which holds all necessary parameters for logarithmic spectral density (LCSD/LPSD)
calculation
"""
from typing import Callable, Optional, Union
from warnings import warn

import numpy as np
from pandas import DataFrame, Series
from pandas.core.indexes.datetimes import DatetimeIndex

from ._helpers import (
    _calc_lcsd,
    _calc_lcsd_py,
    _kaiser_alpha,
    _kaiser_rov,
    _ltf_plan,
    c_core_available,
)


class LCSD:  # pylint: disable=too-many-instance-attributes
    """
    Class to hold paeameters for LCSD/LPSD calculation
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
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
        csd: bool = False,
    ):
        """
        Parameters
        ----------
        data: :obj:`pandas.Series` or :obj:`pandas.DataFrame`
            Assuming time series data and takes one column of a :obj:`pandas.DataFrame`
            or a whole `DataFrame`. If multiple columns are provided, it will calculate
            the spectrum for each column and return a dict of `DataFrame`s with all
            results. Unless csd=True, in which case input DataFrame is assumed to have
            exactly two columns with two input series.
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
        csd: :obj:`bool`
            Calculate CSD instead of PSD: in this case, DataFrame is assumed to have two columns with two input series
        """
        self.data = data
        self.sample_rate = sample_rate
        self.window_function = window_function
        self.overlap = overlap
        self.detrending_order = detrending_order
        self.n_frequencies = n_frequencies
        self.n_averages = n_averages
        self.n_min_bins = n_min_bins
        self.min_segment_length = min_segment_length
        self.psll = psll
        self.use_c_core = use_c_core
        self.csd = csd
        # parameters for LFT planning:
        self._f: list = []
        self._r: list = []
        self._m: list = []
        self._L: list = []
        self._K: list = []

    def _run_lcsd(self, data1, data2):
        lcsd_runner = _calc_lcsd if self.use_c_core else _calc_lcsd_py
        result = lcsd_runner(
            data1.to_numpy(),
            data2.to_numpy(),
            self._f,
            self._r,
            self._m,
            self._L,
            self.sample_rate,
            self.window_function,
            self.psll,
            self.detrending_order,
            self.overlap,
            self.min_segment_length,
            self.csd,
        )

        dc = DataFrame(
            dict(
                ps=result[0],
                psd=result[1],
                ps_std=result[2],
                psd_std=result[3],
                enbw=result[4],
                asd=result[5],
                asdrms=result[6],
            ),
            index=self._f,
        )
        dc.index.name = "frequency"
        return dc

    def check_inputs(self):
        if self.sample_rate is None:
            if isinstance(self.data.index, DatetimeIndex):
                index_diff = self.data.index.to_series().diff()
                period_time = index_diff.median().total_seconds()
                std = index_diff.std().total_seconds()
            else:
                index_diff = np.diff(self.data.index)
                period_time = np.median(index_diff)
                std = index_diff.std()
            self.sample_rate = 1 / period_time

            if std / period_time > 1e-6:
                warn(
                    "Length of some time steps deviates a lot from the median. Some data maybe corrupt!\n"
                    f"Period time: {period_time}, standard deviation: {std}",
                    UserWarning,
                )

        if self.data.isna().any().any():
            warn("Removing NaN values ...", UserWarning)
            self.data.dropna(inplace=True)
        assert (
            not self.data.isna().any().any()
        ), "Bug: There should not be any NaN data!"

        if self.use_c_core and not c_core_available():
            warn(
                "C core should be used, but is not available. Using Python …",
                RuntimeWarning,
            )
            warn(
                f"use_c_core: {self.use_c_core}, c_core_available: {c_core_available()}",
                RuntimeWarning,
            )
            self.use_c_core = False

        if isinstance(self.data, DataFrame) and len(self.data.columns) == 1:
            self.data = self.data[self.data.columns[0]]

        if self.detrending_order is None:
            self.detrending_order = -1

        if self.csd:
            if not isinstance(self.data, DataFrame):
                raise ValueError(
                    "Only 2-column DataFrames are accepted as input for CSD"
                )
            if len(self.data.columns) != 2:
                raise ValueError(
                    "Only 2-column DataFrames are accepted as input for CSD"
                )
            if len(self.data[self.data.columns[0]].to_numpy()) != len(
                self.data[self.data.columns[1]].to_numpy()
            ):
                raise ValueError(
                    "Input time series have different lengths (number of samples)! "
                    "Only time series with equal lengths can be used to calculate CSD"
                )

    def run(self):
        if self.window_function is np.kaiser:
            # calculate kaiser parameters
            alpha = _kaiser_alpha(self.psll)
            if self.overlap is None:
                self.overlap = _kaiser_rov(alpha)

        (
            self._f,
            self._r,
            self._m,
            self._L,
            self._K,
        ) = _ltf_plan(  # pylint: disable=unused-variable
            len(self.data),
            self.sample_rate,
            self.overlap,
            self.n_min_bins,
            self.min_segment_length,
            self.n_frequencies,
            self.n_averages,
        )

        if self.csd:
            dc = self._run_lcsd(
                self.data[self.data.columns[0]], self.data[self.data.columns[1]]
            )
        elif isinstance(self.data, DataFrame):
            dc = {}
            for col in self.data.columns:
                dc[col] = self._run_lcsd(self.data[col], self.data[col])
        else:
            dc = self._run_lcsd(self.data, self.data)

        return dc
