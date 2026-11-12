"""Генератор с известным законом, панельные интервалы и решения экрана."""

import unittest

import numpy as np
import pandas as pd

from src.gas_reference import screen_tail
from src.reference_screen_power import (CRITERIA, ar_mean_variance, channel_outcomes,
    mean_only_planning, panel_interval, stationary_ar, summarize, transform)


class PowerTests(unittest.TestCase):
    def test_reproducible_generator(self):
        np.testing.assert_array_equal(stationary_ar(.9, 3, 2), stationary_ar(.9, 3, 2))
        self.assertFalse(np.array_equal(stationary_ar(.9, 3, 2), stationary_ar(.9, 4, 2)))

    def test_stationary_initial_variance(self):
        # Широкий фиксированный допуск, не подбор seed для экспериментальной сетки.
        x = stationary_ar(.98, 123456, 99, length=2, channels=20_000)
        self.assertAlmostEqual(x[0].var(), 1., delta=.05)
        self.assertAlmostEqual(x[1].var(), 1., delta=.05)
        self.assertAlmostEqual(np.corrcoef(x)[0, 1], .98, delta=.005)

    def test_iid_innovations(self):
        x = stationary_ar(0, 99, 99, length=2, channels=20_000)
        self.assertAlmostEqual(np.corrcoef(x)[0, 1], 0., delta=.03)

    def test_invalid_generator(self):
        for phi in (-.1, 1.):
            with self.assertRaises(ValueError):
                stationary_ar(phi, 0, 0)

    def test_transform_no_mutation(self):
        base = np.zeros((120, 16))
        self.assertEqual(transform(base, "linear_0.5")[-1, 0], .5)
        self.assertEqual(transform(base, "linear_2")[-1, 0], 2.)
        self.assertEqual(base.sum(), 0.)

    def test_step_boundary(self):
        x = transform(np.zeros((120, 16)), "mean_step_1")
        self.assertTrue((x[:60] == 0).all())
        self.assertTrue((x[60:] == 1).all())
        y = transform(np.ones((120, 16)), "scale_step_2")
        self.assertTrue((y[:60] == 1).all())
        self.assertTrue((y[60:] == 2).all())

    def test_unknown_condition(self):
        with self.assertRaises(ValueError):
            transform(np.zeros((120, 16)), "unregistered")

    def test_zero_interval_non_degenerate(self):
        rate, lo, hi, method = panel_interval(np.zeros(100))
        self.assertEqual((rate, lo), (0., 0.))
        self.assertAlmostEqual(hi, 1 - .025 ** .01)
        self.assertLess(hi, .05)
        self.assertIn("exact_zero", method)

    def test_one_interval_non_degenerate(self):
        rate, lo, hi, _ = panel_interval(np.ones(100))
        self.assertEqual((rate, hi), (1., 1.))
        self.assertAlmostEqual(lo, .025 ** .01)

    def test_interval_uses_panel_not_sensor_count(self):
        x = np.array([0., .25, .75, 1.])
        rate, lo, hi, _ = panel_interval(x)
        self.assertEqual(rate, .5)
        self.assertGreater(hi - lo, .8)
        with self.assertRaises(ValueError):
            panel_interval([0.])

    def test_missing_block_rejected(self):
        frame = screen_tail(stationary_ar(0, 54321, 99), [0], repeats=20)
        with self.assertRaises(ValueError):
            channel_outcomes(frame.loc[frame.block_s == 5])

    def test_outcomes_match_original(self):
        diagnostics = screen_tail(stationary_ar(0, 54321, 99), [0], repeats=20)
        result = channel_outcomes(diagnostics)
        self.assertEqual(len(result), 32)
        for row in result.itertuples():
            original = diagnostics.loc[diagnostics.sensor == row.sensor, row.criterion]
            self.assertEqual(row.accepted, bool(original.all()))
            self.assertFalse(row.accepted and not row.point_only)

    def test_panel_summary(self):
        rows = []
        for panel, count in enumerate((0, 4, 12, 16)):
            for sensor in range(16):
                rows.append({"phi": 0., "condition": "stationary", "criterion": "primary",
                             "panel": panel, "accepted": sensor < count,
                             "point_only": True, "mean_not_confirmed": sensor >= count,
                             "variance_not_confirmed": False, "undefined": False})
        summary = summarize(pd.DataFrame(rows))
        accepted = summary.loc[summary.measure == "accepted"].iloc[0]
        self.assertEqual(accepted.rate, .5)
        self.assertEqual(accepted.panels, 4)

    def test_ar_mean_variance_exact_sum(self):
        for phi in (0., .5, .9, .98):
            for n in (1, 2, 40):
                direct = (n + 2 * sum((n-k) * phi ** k for k in range(1, n))) / n ** 2
                self.assertAlmostEqual(ar_mean_variance(n, phi), direct, places=11)

    def test_planning_not_120_point_recommendation(self):
        planning = mean_only_planning()
        self.assertEqual(len(planning), 8)
        self.assertTrue(planning.not_a_power_guarantee.all())
        self.assertTrue((planning.illustrative_total_points > 120).all())


if __name__ == "__main__":
    unittest.main()