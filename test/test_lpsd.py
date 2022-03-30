# type: ignore
from pathlib import Path
from unittest import TestCase

import numpy as np
from pandas import DataFrame

import lpsd
from lpsd.flattop import HFT248D, olap_dict

filedir = Path(__file__).parent


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

    def test_c_core_available(self):
        self.assertTrue(
            lpsd._helpers.c_core_available(),
            msg="The LPSD C core is not available and has to be compiled.",
        )

    def test_lpsd_wrapper_default(self):
        for c in (True, False):
            result = lpsd.lpsd(self.data_y[0], use_c_core=c)
            self.assertAlmostEqual(
                np.mean(result["psd"]) * self.fs / 2,
                self.avg_pow,
                delta=self.avg_pow / 10,
            )

    def test_lpsd_wrapper_other_window(self):
        for c in (True, False):
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

        for c in (True, False):
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
            result_auto = lpsd.lpsd(self.data_y[0], use_c_core=c)
            result_manual = lpsd.lpsd(self.data_y[0], sample_rate=self.fs, use_c_core=c)

            np.testing.assert_array_almost_equal(
                result_auto["psd"], result_manual["psd"]
            )

    def test_with_wrong_manual_sample_rate(self):
        for c in (True, False):
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

    def test_old_wrapper(self):
        # Test data parameters

        fs = self.fs
        y = self.y
        x = self.x
        avg_pow = self.avg_pow
        amp = self.amp

        # LPSD parameters

        # For Kaiser window use "default" as overlap to calculate it automatically. For windows in flattop the olap_dict can be used, to use optimized overlap.
        # (Kaiser window uses psll parameter to determine alpha/olap value)

        olap = "default"
        bmin = 1
        Lmin = 0
        Jdes = 500
        Kdes = 100
        order = 0
        win = np.kaiser
        psll = 200

        raw_f = [[], []]
        raw_S = [[], []]
        raw_Sxx = [[], []]
        raw_dev = [[], []]
        raw_devxx = [[], []]
        raw_ENBW = [[], []]

        for c in (True, False):
            # lpsd method
            (
                raw_f[0],
                raw_S[0],
                raw_Sxx[0],
                raw_dev[0],
                raw_devxx[0],
                raw_ENBW[0],
                _,
            ) = lpsd.lpsd_trad(
                y, fs, olap, bmin, Lmin, Jdes, Kdes, order, win, psll, use_c_core=c
            )
            (
                raw_f[1],
                raw_S[1],
                raw_Sxx[1],
                raw_dev[1],
                raw_devxx[1],
                raw_ENBW[1],
                _,
            ) = lpsd.lpsd_trad(
                x,
                fs,
                olap_dict["HFT248D"],
                bmin,
                Lmin,
                Jdes,
                Kdes,
                order,
                HFT248D,
                psll,
                use_c_core=c,
            )

            self.assertAlmostEqual(
                np.mean(raw_Sxx[0]) * fs / 2, avg_pow, delta=avg_pow / 10
            )
            self.assertAlmostEqual(
                np.sqrt(2) * max(np.sqrt(raw_S[1])), amp, delta=self.amp / 5
            )
