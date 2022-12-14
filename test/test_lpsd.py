# type: ignore
from pathlib import Path
from unittest import TestCase

import numpy as np
import pandas as pd
from pandas import DataFrame

import lpsd
from lpsd.flattop import HFT248D, olap_dict

filedir = Path(__file__).parent
datadir = filedir / "data"


class TestLPSD(TestCase):
    def setUp(self):
        self.fs = 3
        N = 1e4
        self.tt = np.arange(N) / self.fs
        noise_power = 1e-8 * self.fs
        self.y = np.random.normal(scale=np.sqrt(noise_power), size=self.tt.shape)
        self.avg_pow = np.sum((self.y) ** 2 / len(self.y))

        self.amp = 0.001
        freq = 0.001
        self.x = self.amp * np.sin(2 * np.pi * freq * self.tt)
        self.x += self.y

        self.data_x = DataFrame(self.x, index=self.tt)
        self.data_y = DataFrame(self.y, index=self.tt)

    def _max_value_in_window(
        self, values, frequencies, input_frequency, window_size=0.001
    ):
        freq_window = np.where(abs(frequencies - input_frequency) < window_size)
        max_value_index = np.argmax(
            abs(values[freq_window[0][0] : freq_window[0][-1] + 1])
        )
        max_value_index = max_value_index + freq_window[0][0]
        return values[max_value_index]

    def test_c_core_available(self):
        self.assertTrue(
            lpsd._helpers.c_core_available(),
            msg="The LPSD C core is not available and has to be compiled.",
        )

    def test_lpsd_wrapper_default(self):
        for c in (True, False):
            print("Use C core:", c)
            result = lpsd.lpsd(self.data_y[0], use_c_core=c)
            self.assertAlmostEqual(
                np.mean(result["psd"]) * self.fs / 2,
                self.avg_pow,
                delta=self.avg_pow / 10,
            )

    def test_lcsd_wrapper_default(self):
        t = np.arange(10000)
        # define two waves, each with two frequencies, one of which is the same (f1)
        f1 = 0.01
        f2 = 0.1
        f3 = 0.02

        x = 20 * np.sin(2 * np.pi * f1 * t) + 10 * np.sin(2 * np.pi * f2 * t)
        y = 20 * np.sin(2 * np.pi * f1 * t) + 10 * np.sin(2 * np.pi * f3 * t)

        df = DataFrame()
        df["x"] = x
        df["y"] = y

        for c in (True, False):
            print("Use C core:", c)
            result = lpsd.lcsd(df, use_c_core=c)

            # check f1 - the common frequency
            max_csd = self._max_value_in_window(
                result["psd"].to_numpy(), result.index.to_numpy(), f1
            )
            np.testing.assert_allclose(max_csd, 152721.014525, rtol=0.01)

            # with detrending
            result = lpsd.lcsd(df, detrending_order=1, use_c_core=c)
            max_csd = self._max_value_in_window(
                result["psd"].to_numpy(), result.index.to_numpy(), f1
            )
            if not c:  # for python detrending, the result is slightly different,
                check_val = 144984.70236044083  # see https://gitlab.com/uhh-gwd/lpsd/-/issues/26
            else:
                check_val = 152721.014525
            np.testing.assert_allclose(max_csd, check_val, rtol=0.01)

    def test_coherence(self):
        t = np.arange(10000)
        # define two waves, each with two frequencies, one of which is the same (f1)
        f1 = 0.01
        f2 = 0.1
        f3 = 0.02

        x = 20 * np.sin(2 * np.pi * f1 * t) + 10 * np.sin(2 * np.pi * f2 * t)
        y = 20 * np.sin(2 * np.pi * f1 * t) + 10 * np.sin(2 * np.pi * f3 * t)

        df = DataFrame()
        df["x"] = x
        df["y"] = y

        coherence = []
        for c in (True, False):
            print("Use C core:", c)
            psd = lpsd.lpsd(
                df, use_c_core=c, detrending_order=None
            )  # no detrending such that results
            csd = lpsd.lcsd(
                df, use_c_core=c, detrending_order=None
            )  # for c and py are very close
            coh = np.abs(csd["psd"]) ** 2 / psd["x"]["psd"] / psd["y"]["psd"]
            np.testing.assert_array_less(coh, 1.001)
            coherence.append(coh)

            max_coh = self._max_value_in_window(
                coh.to_numpy(), coh.index.to_numpy(), f1
            )
            # common frequency = good coherence
            np.testing.assert_allclose(max_coh, 1.0, rtol=0.01)

            max_coh = self._max_value_in_window(
                coh.to_numpy(), coh.index.to_numpy(), f2
            )
            # frequency only in one signal = poor coherence
            np.testing.assert_allclose(max_coh, 0.0, atol=0.01)

        np.testing.assert_allclose(coherence[0], coherence[1], rtol=0.01)

    def test_lcsd_wrapper_invalid_inputs(self):
        x = [1.0, 2.0, 3.0]
        df = DataFrame()
        df["x"] = x
        self.assertRaises(ValueError, lpsd.lcsd, df)  # wrong input: one time series
        df["y"] = x
        df["z"] = x
        self.assertRaises(ValueError, lpsd.lcsd, df)  # wrong input: 3 time series

    def test_lpsd_wrapper_other_window(self):
        for c in (True, False):
            print("Use C core:", c)
            result = lpsd.lpsd(
                self.data_x[0],
                window_function=HFT248D,
                overlap=olap_dict["HFT248D"],
                n_frequencies=500,
                use_c_core=c,
            )
            self.assertAlmostEqual(
                np.sqrt(2) * np.max(np.sqrt(result["ps"])), self.amp, delta=self.amp / 5
            )

    def test_detrending(self):
        # add some offset and linear curve
        self.data_y[0] += 0.001 * self.data_y.index - 10

        for c in (True, False):
            print("Use C core:", c)
            result = lpsd.lpsd(self.data_y[0], detrending_order=1, use_c_core=c)
            self.assertAlmostEqual(
                result["psd"].mean() * self.fs / 2,
                self.avg_pow,
                delta=self.avg_pow / 10,
            )

    def test_poly_detrending(self):
        self.data_y[0] += 0.001 * self.data_y.index - 10

        _ = lpsd.lpsd(self.data_y[0], detrending_order=2, use_c_core=True)
        # TODO add a value test for C.

        for i in range(2, 9):
            with self.assertWarnsRegex(UserWarning, "Polynomial detrending"):
                _ = lpsd.lpsd(self.data_y[0], detrending_order=i, use_c_core=False)

    def test_no_detrending(self):
        # add some offset and linear curve
        self.data_y[0] += 0.001 * self.data_y.index - 10

        for c in (False, True):
            print("Use C core:", c)
            result = lpsd.lpsd(self.data_y[0], detrending_order=None, use_c_core=c)
            self.assertGreater(result["psd"].mean() * self.fs / 2, self.avg_pow * 10)

    def test_input_dataframe(self):
        out = lpsd.lpsd(self.data_y)
        self.assertIsInstance(out, DataFrame)

        data_y = DataFrame(self.y, index=self.tt)
        out = lpsd.lpsd(data_y)
        self.assertIsInstance(out, DataFrame)

    def test_input_multi_column_datacontainer(self):
        self.data_y["col2"] = self.data_y.copy()
        out = lpsd.lpsd(self.data_y)

        self.assertIsInstance(out, dict)
        self.assertIsInstance(out["col2"], DataFrame)
        self.assertEqual(len(out), 2)
        self.assertIn("asd", out["col2"].columns)
        self.assertIn("asd", out[0].columns)

    def test_with_manual_sample_rate(self):
        for c in (True, False):
            print("Use C core:", c)
            result_auto = lpsd.lpsd(self.data_y[0], use_c_core=c)
            result_manual = lpsd.lpsd(self.data_y[0], sample_rate=self.fs, use_c_core=c)

            np.testing.assert_array_almost_equal(
                result_auto["psd"], result_manual["psd"]
            )

    def test_with_wrong_manual_sample_rate(self):
        for c in (True, False):
            print("Use C core:", c)
            result_auto = lpsd.lpsd(self.data_y[0], use_c_core=c)
            result_manual = lpsd.lpsd(
                self.data_y[0], sample_rate=self.fs * 1.1, use_c_core=c
            )

            np.testing.assert_array_less(result_manual["psd"], result_auto["psd"])

    def test_warning_on_unequal_sample_rate(self):
        self.tt[-1] *= 1.000001
        data = DataFrame(self.y, index=self.tt)
        with self.assertWarns(UserWarning):
            _ = lpsd.lpsd(data)

    def test_asdrms(self):
        spec = lpsd.lpsd(self.data_x)
        asdrms = spec["asdrms"].to_numpy()
        asd = spec["asd"].to_numpy()
        freq = spec.index
        asdrms2, freq_out = lpsd._helpers._asdrms(asd, freq)
        np.testing.assert_array_almost_equal(freq, freq_out)
        np.testing.assert_array_almost_equal(asdrms, asdrms2)

    def test_datetimeindex(self):
        data = pd.read_csv(datadir / "datetimeindex.csv", index_col=0, parse_dates=True)
        results = lpsd.lpsd(data)
        self.assertAlmostEqual(results["temperature"].index[-1], 9.83 / 2, places=2)
        self.assertAlmostEqual(results["temperature"].index[0], 0.00413, places=5)

    def test_removing_nan(self):
        data = pd.read_csv(datadir / "nan.csv", index_col=0, parse_dates=True)
        assert data.isna().any().any()
        results = lpsd.lpsd(data)
        self.assertFalse(results["temperature"]["psd"].isna().any().any())
