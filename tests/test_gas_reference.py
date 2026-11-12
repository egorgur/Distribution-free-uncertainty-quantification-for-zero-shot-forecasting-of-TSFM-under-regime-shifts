"""Границы отбора, зависимый bootstrap и запрет считать p>0.05 устойчивостью."""

import unittest
from io import StringIO
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.gas_audit import SENSORS
from src.gas_reference import (adjacent_candidates, collapse_candidates, moving_indices,
                               passes, screen_tail, select_tail)


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.segment = SimpleNamespace(boundary_s=200., start_s=50.,
                                       left_censored=False, right_censored=False,
                                       gas_ppm=0., ethylene_ppm=10.)
        self.seconds = pd.DataFrame({"second": np.arange(80, 200), "mixed_setpoint": False,
                                     "gas_ppm_min": 0., "gas_ppm_max": 0.,
                                     "ethylene_ppm_min": 10., "ethylene_ppm_max": 10.})
        for sensor in SENSORS:
            self.seconds[sensor] = np.arange(120, dtype=float)

    def test_exact_guard_accepted(self):
        values, reason, start, end = select_tail(self.seconds, self.segment)
        self.assertEqual((reason, start, end), ("eligible", 80, 200))
        self.assertEqual(values.shape, (120, 16))

    def test_insufficient_guard(self):
        self.segment.start_s = 50.01
        self.assertEqual(select_tail(self.seconds, self.segment)[1], "insufficient_duration")

    def test_censored(self):
        self.segment.right_censored = True
        self.assertEqual(select_tail(self.seconds, self.segment)[1], "censored")

    def test_fractional_end_is_floored(self):
        self.segment.boundary_s = 200.99
        self.assertEqual(select_tail(self.seconds, self.segment)[2:], (80, 200))

    def test_missing_bins(self):
        self.assertEqual(select_tail(self.seconds.iloc[1:], self.segment)[1], "missing_or_duplicate_bins")

    def test_mixed_setpoint(self):
        self.seconds.loc[30, "mixed_setpoint"] = True
        self.assertEqual(select_tail(self.seconds, self.segment)[1], "mixed_or_wrong_setpoint")

    def test_wrong_setpoint(self):
        self.seconds.loc[30, "gas_ppm_max"] = 1.
        self.assertEqual(select_tail(self.seconds, self.segment)[1], "mixed_or_wrong_setpoint")

    def test_nonfinite(self):
        self.seconds.loc[30, "s02"] = np.inf
        self.assertEqual(select_tail(self.seconds, self.segment)[1], "nonfinite_sensor")

    def test_moving_blocks_no_wrap(self):
        idx = moving_indices(40, 10, 20, np.random.default_rng(0))
        self.assertTrue(((idx >= 0) & (idx < 40)).all())
        np.testing.assert_array_equal(np.diff(idx.reshape(20, 4, 10), axis=2), 1)

    def test_invalid_blocks(self):
        with self.assertRaises(ValueError):
            moving_indices(40, 7, 20, np.random.default_rng(0))

    def test_boundary_inclusive_and_nan_rejected(self):
        self.assertTrue(passes(.25, np.log(1.5)))
        self.assertFalse(passes(.2501, 0.))
        self.assertFalse(passes(np.nan, 0.))
        self.assertFalse(passes(0., np.inf))

    def test_constant_not_evidence(self):
        table = screen_tail(np.ones((120, 16)), [0], repeats=40)
        self.assertTrue((table.status == "zero_observed_variance").all())
        self.assertFalse(table.primary.any())

    def test_linear_drift_rejected(self):
        table = screen_tail(np.tile(np.arange(120)[:, None], (1, 16)), [0], repeats=40)
        self.assertFalse(table.primary.any())
        self.assertTrue((table.mean_spread > 2).all())

    def test_variance_change_rejected(self):
        x = np.tile([-1., 1.], 60)
        x[40:80] *= 10
        table = screen_tail(np.tile(x[:, None], (1, 16)), [0], repeats=40)
        self.assertFalse(table.primary.any())
        self.assertTrue((table.log_variance_spread > np.log(1.5)).all())

    def test_reproducible_and_synchronous(self):
        rng = np.random.default_rng(3)
        values = np.tile(rng.normal(size=(120, 1)), (1, 16))
        first = screen_tail(values, [0, 1], repeats=40)
        pd.testing.assert_frame_equal(first, screen_tail(values, [0, 1], repeats=40))
        for _, group in first.groupby("block_s"):
            self.assertEqual(group.mean_upper.nunique(), 1)

    def test_same_channel_adjacent_only(self):
        flags = pd.DataFrame([{"segment": 0, "sensor": "s01", "primary": True, "sensitivity": True},
                              {"segment": 1, "sensor": "s02", "primary": True, "sensitivity": True}])
        self.assertFalse(adjacent_candidates(flags, [0, 1]).primary.any())
        flags.loc[1, "sensor"] = "s01"
        self.assertEqual(adjacent_candidates(flags, [0, 1]).primary.sum(), 1)

    def test_original_csv_precision(self):
        original = pd.DataFrame({"sensor": [1581.5660396039602, -43.523762, .000012345678901234]})
        saved = pd.read_csv(StringIO(original.to_csv(index=False, float_format="%.12g")))
        pd.testing.assert_frame_equal(saved, original, rtol=1e-11, atol=1e-10)

    def test_missing_bootstrap_length_rejected(self):
        frame = pd.DataFrame({"recording": ["co"], "segment": [1], "sensor": ["s01"],
                              "block_s": [5], "primary": [True], "sensitivity": [True]})
        with self.assertRaises(ValueError):
            collapse_candidates(frame)
        other = frame.assign(block_s=10, primary=False)
        self.assertFalse(collapse_candidates(pd.concat([frame, other])).primary.any())
        with self.assertRaises(ValueError):
            collapse_candidates(pd.concat([frame, frame]))


if __name__ == "__main__":
    unittest.main()