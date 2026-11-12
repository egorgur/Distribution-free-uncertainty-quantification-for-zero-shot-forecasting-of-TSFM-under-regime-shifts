"""Границы утверждённого H1/H3 v1 без моделей, инференса и новых зависимостей."""

import unittest

import numpy as np
import pandas as pd

from src.evaluation import (
    LOG_FLOOR, SEARCH_GRID, STATUSES, add_ratios, bootstrap_recovery,
    control_thresholds, recovery_v1, reference_audit, safe_log, smoothed_search,
)


def example(cls="change", old=(2.0, 4.0), new=(6.0, 10.0)):
    """H4: old median3, new median8; 648 не должно входить в новый эталон."""
    positions = [-8, -5, -4, -1, 0, 64, 512, 640, 648]
    w = pd.DataFrame({"row": range(len(positions)), "cell": "example", "cls": cls,
                      "kind": "mean", "level": "strong", "series_idx": 0, "pos": positions})
    values = [*old, 8.0, 8.0, 8.0, 8.0, *new, 10000.0]
    f = pd.DataFrame({"row": w.row, "L": 64, "horizon": 4, "mae": values,
                      "crps": np.asarray(values) * 2, "width": np.asarray(values) * 3, "scale": 2.0})
    return f, w


def control_frame():
    """10 series: geometric mean exp(d/1000), not arithmetic mean of windows."""
    d = np.r_[-8, SEARCH_GRID, 512, 640]
    rows = []
    for series in range(10):
        offset = -0.5 if series < 5 else 0.5
        for pos in d:
            rows.append({"series_idx": series, "d": pos, "L": 512, "horizon": 64,
                         "cls": "none", "mae_ratio": np.exp(pos / 1000 + offset)})
    return pd.DataFrame(rows)


class RatioTests(unittest.TestCase):
    def test_mixture_boundaries_minus_h_minus_one_zero(self):
        f, w = example()
        result = add_ratios(f, w, 512).set_index("d")
        self.assertAlmostEqual(result.loc[-4, "mae_reference"], 3 * .75 + 8 * .25)
        self.assertAlmostEqual(result.loc[-4, "mae_ratio"], 8 / 4.25)
        self.assertEqual(result.loc[-1, "mae_reference"], 8)
        self.assertEqual(result.loc[0, "mae_reference"], 8)

    def test_reference_band_inclusive_and_not_moved_to_short_l(self):
        f, w = example()
        result = add_ratios(f, w, 512)
        np.testing.assert_array_equal(result.mae_ref_new, 8)
        np.testing.assert_array_equal(result.mae_ref_new_n, 2)
        self.assertEqual(result.loc[result.d == 64, "mae_reference"].iloc[0], 8)

    def test_length_horizon_series_and_model_isolation(self):
        f, w = example()
        frames, windows = [], []
        for series in range(2):
            ws = w.assign(row=w.row + series * len(w), series_idx=series)
            windows.append(ws)
            for model in ("a", "b"):
                for length in (64, 512):
                    for h in (4, 6):
                        factor = (series + 1) * (1 if model == "a" else 10) * length / 64 * h / 4
                        frames.append(f.assign(row=ws.row, model=model, L=length, horizon=h,
                                               mae=f.mae * factor, expected_factor=factor))
        result = add_ratios(pd.concat(frames), pd.concat(windows), 512)
        # H6: only d=-8 in old band; H4: both -8 and -5.
        np.testing.assert_allclose(result.mae_ref_old,
                                   np.where(result.horizon == 4, 3, 2) * result.expected_factor)
        np.testing.assert_allclose(result.mae_ref_new, 8 * result.expected_factor)

    def test_full_reference_is_computed_before_evaluation_filter(self):
        f, w = example()
        full = add_ratios(f, w, 512)
        evaluation = full[full.d >= -5]
        np.testing.assert_array_equal(evaluation.mae_ref_old, 3)
        # Filtering first would change old median from3 to4: this is deliberately NOT the API usage.
        filtered = add_ratios(f[f.row != 0], w, 512)
        np.testing.assert_array_equal(filtered.mae_ref_old, 4)

    def test_none_and_anomaly_keep_old_reference(self):
        for cls in ("none", "anomaly"):
            with self.subTest(cls=cls):
                f, w = example(cls)
                result = add_ratios(f, w, 512)
                np.testing.assert_array_equal(result.mae_reference, 3)
                np.testing.assert_array_equal(result.mae_ref_new, result.mae_ref_old)

    def test_invalid_old_reference_remains_nan_not_epsilon(self):
        for invalid in (0.0, -2.0, np.nan, np.inf):
            with self.subTest(invalid=invalid):
                f, w = example(old=(invalid, invalid))
                result = add_ratios(f, w, 512)
                self.assertTrue(result.loc[result.d < -1, "mae_ratio"].isna().all())
                self.assertEqual(result.loc[result.d == 0, "mae_ratio"].iloc[0], 1)

    def test_invalid_new_reference_does_not_contaminate_old_only_windows(self):
        f, w = example(new=(0, 0))
        result = add_ratios(f, w, 512)
        self.assertTrue(result.loc[result.d >= -4, "mae_ratio"].isna().all())
        self.assertTrue(result.loc[result.d < -4, "mae_ratio"].notna().all())

    def test_zero_numerator_preserved_log_floor_only(self):
        f, w = example()
        f.loc[w.pos == 0, "mae"] = 0
        result = add_ratios(f, w, 512).set_index("d")
        self.assertEqual(result.loc[0, "mae_ratio"], 0)
        self.assertEqual(result.loc[0, "log_mae_ratio"], np.log(LOG_FLOOR))
        self.assertEqual(result.loc[0, "mae_reference"], 8)

    def test_mase_invalid_scale_not_infinite(self):
        f, w = example()
        f.loc[:3, "scale"] = [0, -1, np.inf, np.nan]
        result = add_ratios(f, w, 512)
        self.assertTrue(result.mase.iloc[:4].isna().all())
        self.assertTrue(result.log_mase.iloc[:4].isna().all())

    def test_drift_marker_and_audit_exclusions(self):
        f, w = example("drift", new=(0, 0))
        result = add_ratios(f, w, 512)
        self.assertTrue(result.reference_approximation.all())
        audit = reference_audit(result)
        self.assertEqual(len(audit), 3)
        self.assertTrue(audit.invalid_new.all())
        np.testing.assert_array_equal(audit.n_excluded, 7)
        np.testing.assert_array_equal(audit.n_valid, 2)

    def test_safe_log_never_repairs_invalid_ratios(self):
        result = safe_log(np.array([0, 1, -1, np.inf, np.nan]))
        np.testing.assert_allclose(result[:2], [np.log(LOG_FLOOR), 0])
        self.assertTrue(np.isnan(result[2:]).all())


class ControlTests(unittest.TestCase):
    def test_geometric_aggregation_log_smoothing_linear_percentiles(self):
        result = control_thresholds(control_frame())
        logs = SEARCH_GRID / 1000
        expected = np.array([logs[max(0, i - 1):min(len(logs), i + 2)].mean()
                             for i in range(len(logs))])
        for row in result.itertuples():
            self.assertAlmostEqual(row.raw_quantile,
                                   np.percentile(np.exp(expected), row.percentile, method="linear"))
            self.assertEqual(row.n_series, 10)
            self.assertEqual(row.n_d, 64)

    def test_smoothing_region_only_no_pre_or_post_boundary_leak(self):
        d = np.r_[-8, SEARCH_GRID, 512]
        logs = np.r_[1000, np.zeros(64), 1000]
        np.testing.assert_array_equal(smoothed_search(logs, d), 0)
        logs[1], logs[-2] = 2, 4
        sm = smoothed_search(logs, d)
        self.assertEqual(sm[0], 1)
        self.assertEqual(sm[-1], 2)

    def test_control_threshold_ignores_reference_zone_and_other_classes(self):
        frame = control_frame()
        before = control_thresholds(frame)
        frame.loc[(frame.d < 0) | (frame.d >= 512), "mae_ratio"] = 1e100
        frame = pd.concat([frame, frame.assign(cls="change", mae_ratio=1e200)])
        pd.testing.assert_frame_equal(before, control_thresholds(frame))

    def test_missing_search_value_not_imputed_by_smoothing(self):
        logs = np.zeros(64)
        logs[20] = np.nan
        result = smoothed_search(logs, SEARCH_GRID)
        self.assertTrue(np.isnan(result[20]))
        self.assertEqual(recovery_v1(result, SEARCH_GRID, 1.1).status, "not_confirmed")

    def test_threshold_floor_one_and_raw_quantile_retained(self):
        result = control_thresholds(control_frame().assign(mae_ratio=.5))
        np.testing.assert_array_equal(result.threshold, 1)
        np.testing.assert_allclose(result.raw_quantile, .5)
        self.assertEqual(result.role.tolist(), ["primary", "sensitivity"])

    def test_thresholds_isolated_per_l_and_h(self):
        f = control_frame()
        changed = f.assign(L=64, horizon=16, mae_ratio=f.mae_ratio * 2)
        result = control_thresholds(pd.concat([f, changed]))
        a = result[result.L == 512].set_index("percentile")
        b = result[result.L == 64].set_index("percentile")
        np.testing.assert_allclose(b.raw_quantile, a.raw_quantile * 2)

    def test_control_rejects_wrong_n_missing_point_invalid_reference(self):
        frame = control_frame()
        cases = [frame[frame.series_idx < 9], frame[frame.d != 8],
                 frame.assign(mae_ratio=lambda f: f.mae_ratio.where(f.d != 0))]
        for bad in cases:
            with self.subTest(size=len(bad)), self.assertRaises(ValueError):
                control_thresholds(bad)


class RecoveryTests(unittest.TestCase):
    def recover(self, values, threshold=1.1):
        return recovery_v1(np.log(values), SEARCH_GRID, threshold)

    def test_no_degradation_has_no_numerical_time(self):
        result = self.recover(np.ones(64))
        self.assertEqual(result.status, "no_degradation")
        self.assertTrue(np.isnan(result.d_star))

    def test_delayed_peak_cannot_use_early_quiet_interval(self):
        values = np.ones(64)
        values[20] = 2
        result = self.recover(values)
        self.assertEqual(result.peak_d, 160)
        self.assertEqual(result.d_star, 168)

    def test_tied_maxima_use_last_peak_strictly_after_it(self):
        values = np.ones(64)
        values[[0, 20]] = 2
        result = self.recover(values)
        self.assertEqual(result.peak_d, 160)
        self.assertEqual(result.d_star, 168)

    def test_nine_not_eight_below_and_threshold_equality_allowed(self):
        values = np.full(64, 1.1)
        values[0], values[9] = 2, 1.2
        result = self.recover(values)
        self.assertEqual(result.d_star, 80)

    def test_confirmation_span64_last_candidate440(self):
        values = np.ones(64)
        values[54] = 2  # peak432: exactly 9 remaining points 440..504
        result = self.recover(values)
        self.assertEqual(result.status, "recovered")
        self.assertEqual(result.d_star, 440)
        self.assertEqual(SEARCH_GRID[-1] - result.d_star, 64)

    def test_insufficient_late_confirmation_not_reported_as_512(self):
        values = np.ones(64)
        values[55] = 2  # only 8 points remain
        result = self.recover(values)
        self.assertEqual(result.status, "not_confirmed")
        self.assertTrue(np.isnan(result.d_star))

    def test_isolated_late_exceedance_tolerated(self):
        values = np.ones(64)
        values[0], values[50] = 2, 1.2
        self.assertEqual(self.recover(values).d_star, 8)

    def test_two_consecutive_late_exceedances_allowed(self):
        values = np.ones(64)
        values[0] = 2
        values[40:42] = 1.2
        self.assertEqual(self.recover(values).d_star, 8)

    def test_three_consecutive_late_exceedances_forbid_early_candidate(self):
        values = np.ones(64)
        values[0] = 2
        values[40:43] = 1.2
        self.assertEqual(self.recover(values).d_star, 344)

    def test_fraction_uses_candidate_suffix_denominator_and_floor(self):
        values = np.ones(64)
        values[50] = 2
        values[62:64] = 1.2  # candidate408 has 13 points: floor(1.3)=1, not2
        self.assertEqual(self.recover(values).status, "not_confirmed")

    def test_one_exceedance_in_eleven_suffix_points_allowed(self):
        values = np.ones(64)
        values[52], values[63] = 2, 1.2
        self.assertEqual(self.recover(values).d_star, 424)

    def test_nonfinite_curve_never_claims_no_degradation(self):
        for invalid in (np.nan, np.inf):
            logs = np.zeros(64)
            logs[20] = invalid
            result = recovery_v1(logs, SEARCH_GRID, 1.1)
            self.assertEqual(result.status, "not_confirmed")
            self.assertEqual(result.reason, "incomplete_curve")

    def test_incomplete_search_grid_and_invalid_threshold_rejected(self):
        with self.assertRaises(ValueError):
            recovery_v1(np.zeros(63), SEARCH_GRID[:-1], 1.1)
        for invalid in (.9, np.inf, np.nan):
            with self.subTest(threshold=invalid), self.assertRaises(ValueError):
                recovery_v1(np.zeros(64), SEARCH_GRID, invalid)


class BootstrapTests(unittest.TestCase):
    def test_all_statuses_counted_and_interval_conditioned_only_on_recovered(self):
        logs = np.zeros((3, 64))
        logs[1, :3], logs[2, :] = np.log(2), np.log(2)
        indices = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        _, summary, outcomes = bootstrap_recovery(logs, SEARCH_GRID, 1.5, n_boot=3, indices=indices)
        self.assertEqual(outcomes.status.tolist(), list(STATUSES))
        for status in STATUSES:
            self.assertEqual(summary[f"boot_{status}_n"], 1)
            self.assertEqual(summary[f"boot_{status}_share"], 1 / 3)
        recovered = outcomes[outcomes.status == "recovered"].d_star.iloc[0]
        self.assertEqual(summary["n_boot_recovered"], 1)
        self.assertEqual(summary["d_star_lo"], recovered)
        self.assertEqual(summary["d_star_hi"], recovered)
        self.assertTrue(outcomes.loc[outcomes.status != "recovered", "d_star"].isna().all())

    def test_bootstrap_without_recovering_replicates_has_no_numeric_interval(self):
        _, summary, _ = bootstrap_recovery(np.zeros((10, 64)), SEARCH_GRID, 1.1, n_boot=5)
        self.assertEqual(summary["n_boot_recovered"], 0)
        self.assertTrue(np.isnan(summary["d_star_lo"]))
        self.assertTrue(np.isnan(summary["d_star_hi"]))
        self.assertEqual(summary["boot_no_degradation_n"], 5)

    def test_bootstrap_whole_series_fixed_seed_is_reproducible(self):
        logs = np.zeros((10, 64))
        logs[:4, :20] = np.log(2)
        result1, summary1, outcomes1 = bootstrap_recovery(logs, SEARCH_GRID, 1.1, n_boot=20, seed=7)
        result2, summary2, outcomes2 = bootstrap_recovery(logs, SEARCH_GRID, 1.1, n_boot=20, seed=7)
        self.assertEqual(result1.status, result2.status)
        self.assertEqual(summary1, summary2)
        pd.testing.assert_frame_equal(outcomes1, outcomes2)


if __name__ == "__main__":
    unittest.main()