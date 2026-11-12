"""Лёгкие проверки H3 v1: unittest, без моделей и настоящего кеша прогнозов."""

from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from scripts import stage4_signals as stage4
from src import metrics
from src.evaluation import add_ratios


def population() -> pd.DataFrame:
    """Три ряда, два H; все исторические сигналы конечны, ошибки намеренно различны."""
    n = 30
    r = np.arange(n)
    one = pd.DataFrame({
        "row": r, "series_idx": r // 10, "d": np.tile(np.arange(-16, 64, 8), 3),
        "cls": np.repeat(["change", "drift", "anomaly"], 10),
        "mae_ratio": 1 + r / 20, "mase": 5 - r / 10,
        "crps_s": 1 + (r % 7) / 10, "mae_fixed": 2 + r / 10,
    })
    for j, key in enumerate(dict.fromkeys([*stage4.MAIN, *stage4.ABLATIONS])):
        one[key] = 0.1 + ((r * (j + 1) + 3 * j) % 31) / 31
    one["s1"] = 1 + (n - r) / n
    one["s4_b"] = 1 + r / n
    return pd.concat([one.assign(horizon=h) for h in (16, 64)], ignore_index=True)


def small_evaluation():
    """Сокращаем только число методов в тестах; настоящие цели/подгруппы сохраняются."""
    stack = ExitStack()
    stack.enter_context(patch.object(stage4, "MAIN", {k: stage4.MAIN[k] for k in ("s1", "s4_b")}))
    stack.enter_context(patch.object(stage4, "COMBOS", {"c_rank_s1_s4": stage4.COMBOS["c_rank_s1_s4"]}))
    stack.enter_context(patch.object(stage4, "ABLATIONS", {}))
    stack.enter_context(patch.object(stage4, "DIFF_PAIRS", [("c_rank_s1_s4", "s1"),
                                                          ("c_rank_s1_s4", "s4_b")]))
    return stack


class Stage4ProtocolTests(unittest.TestCase):
    def test_shared_ratio_implementation(self):
        self.assertIs(stage4.add_ratios, add_ratios)
        self.assertEqual(stage4.MAIN_TARGET, "mae_ratio")

    def test_safe_zero_gain_and_zero_error(self):
        for value in (0.0, 2.0):
            error = np.full(5, value)
            score = np.arange(5, dtype=float)
            order = metrics.rank_order(score)
            got = stage4.signal_metrics(score, error, np.ones(5, dtype=bool), None,
                                        order, metrics.rank_order(error), value, value)
            self.assertTrue(np.isnan(got["gain"]))
            self.assertTrue(np.isnan(got["spearman"]))
            self.assertTrue(np.isnan(got["auroc"]))
            self.assertEqual(float(got["ap"]), 1.0)
            self.assertEqual(float(got["aurc"]), value)
            if value == 0:
                self.assertTrue(np.isnan(got["drop10"]))
        np.testing.assert_allclose(stage4.safe_divide([1, 2], [0, 2]), [np.nan, 1], equal_nan=True)

    def test_invalid_bootstrap_is_not_silently_conditioned(self):
        ci = stage4.interval_info(np.array([1.0, np.nan, 2.0]))
        self.assertEqual(ci["ci_status"], "undefined_replicates")
        self.assertEqual(ci["n_boot_valid"], 2)
        self.assertEqual(ci["n_boot_invalid"], 1)
        self.assertTrue(np.isnan(ci["hi"]))

    def test_constant_error_within_bootstrap_support_has_undefined_gain(self):
        error = np.array([1.0, 1.0, 2.0, 2.0])
        score = np.arange(4, dtype=float)
        weights = np.array([[2, 2, 0, 0], [0, 0, 2, 2], [1, 1, 1, 1]], dtype=float)
        order = metrics.rank_order(error)
        varies = stage4.support_varies(error, weights, order)
        np.testing.assert_array_equal(varies, [False, False, True])
        mean = weights @ error / weights.sum(axis=1)
        got = stage4.signal_metrics(score, error, error >= 2, weights, metrics.rank_order(score),
                                    order, mean, metrics.aurc(error, error, weights), varies)
        self.assertTrue(np.isnan(got["gain"][:2]).all())
        self.assertTrue(np.isnan(got["spearman"][:2]).all())
        self.assertTrue(np.isfinite(got["gain"][2]))

    def test_ranks_fixed_before_invalid_target_and_no_drift_subset(self):
        raw = population()
        raw.loc[raw.row == 4, "mae_ratio"] = np.nan
        expected = stage4.add_combos(raw)
        pop = stage4.prepare_population(raw)
        pd.testing.assert_series_equal(pop.c_rank_s1_s4, expected.c_rank_s1_s4)
        no_drift = pop[stage4.SUBSETS[stage4.NO_DRIFT](pop)]
        self.assertFalse(no_drift.cls.eq("drift").any())
        valid = stage4.valid_target(no_drift, stage4.MAIN_TARGET)
        pd.testing.assert_series_equal(valid.c_rank_s1_s4, expected.loc[valid.index, "c_rank_s1_s4"])
        reranked = stage4.add_combos(valid)
        self.assertFalse(np.allclose(valid.c_rank_s1_s4, reranked.c_rank_s1_s4))
        before = pop.c_rank_s1_s4.copy()
        with small_evaluation():
            met, _ = stage4.evaluate(pop, 7)
        self.assertIn(stage4.NO_DRIFT, set(met.subset))
        pd.testing.assert_series_equal(before, pop.c_rank_s1_s4)

    def test_bad_labels_ap_and_exclusions_are_target_specific(self):
        raw = population()
        raw.loc[raw.row >= 24, "mae_ratio"] = 10.0  # ties => доля плохих больше 0.10
        raw.loc[raw.row == 0, "mae_ratio"] = np.nan
        pop = stage4.prepare_population(raw)
        with small_evaluation():
            met, diffs = stage4.evaluate(pop, 9)
        for h in (16, 64):
            sub = pop[pop.horizon == h]
            for target in stage4.TARGETS:
                valid = stage4.valid_target(sub, target)
                e = valid[target].to_numpy()
                label = e >= np.quantile(e, 0.9)
                rows = met[(met.horizon == h) & (met.subset == stage4.ALL) & (met.target == target)]
                self.assertTrue((rows.actual_bad_share == label.mean()).all())
                self.assertTrue((rows.n_windows == len(valid)).all())
                got = rows[(rows.signal == "s1") & (rows.metric == "ap")].iloc[0]
                self.assertAlmostEqual(got.value, metrics.average_precision(valid.s1.to_numpy(), label))
                if target != stage4.MAIN_TARGET:
                    self.assertTrue((rows.ci_status == "not_requested").all())
            main = met[(met.horizon == h) & (met.subset == stage4.ALL)
                       & (met.target == stage4.MAIN_TARGET)]
            self.assertGreater(main.actual_bad_share.iloc[0], stage4.WORST)
            self.assertTrue((main.n_excluded == 1).all())
        self.assertEqual(set(diffs.metric), {"aurc", "gain", "auroc", "ap", "spearman"})
        audit = stage4.exclusion_audit(pop)
        main_audit = audit[(audit.cls == "ALL") & (audit.subset == stage4.ALL)
                           & (audit.target == stage4.MAIN_TARGET)]
        self.assertTrue((main_audit.n_excluded == 1).all())

    def test_paired_aurc_difference_uses_same_bootstrap_rows(self):
        pop = stage4.prepare_population(population())
        n_boot = 13
        with small_evaluation():
            _, diffs = stage4.evaluate(pop, n_boot)
        valid = stage4.valid_target(pop[pop.horizon == 64], stage4.MAIN_TARGET)
        e = valid.mae_ratio.to_numpy()
        combo, s1 = valid.c_rank_s1_s4.to_numpy(), valid.s1.to_numpy()
        chunks = stage4.bootstrap_weights(valid.series_idx.to_numpy(), n_boot, [stage4.SEED, 64, 0])
        boot_delta = np.concatenate([metrics.aurc(combo, e, w) - metrics.aurc(s1, e, w) for w in chunks])
        row = diffs[(diffs.horizon == 64) & (diffs.subset == stage4.ALL)
                    & (diffs.versus == "s1") & (diffs.metric == "aurc")].iloc[0]
        self.assertAlmostEqual(row.value, metrics.aurc(combo, e) - metrics.aurc(s1, e))
        np.testing.assert_allclose([row.lo, row.hi], np.percentile(boot_delta, [2.5, 97.5]))
        self.assertEqual(row.n_boot_valid, n_boot)

    def test_primary_criterion_and_s4_are_distinct(self):
        rows = [{"horizon": h, "subset": subset, "target": "mae_ratio", "metric": "aurc",
                 "signal": "c_rank_s1_s4", "versus": b, "value": -0.1, "lo": -0.2,
                 "hi": hi} for h, subset in [(64, stage4.ALL), (16, stage4.ALL), (64, stage4.NO_DRIFT)]
                for b, hi in [("s1", -0.01), ("s4_b", 0.01)]]
        diffs = pd.DataFrame(rows)
        criteria = stage4.criteria_table(diffs, "amazon/chronos-bolt-small", 0, 1000)
        primary = criteria[criteria.configuration_status == "primary"]
        self.assertEqual(len(primary), 2)
        self.assertEqual(primary[primary.versus == "s1"].criterion_status.iloc[0], "passed")
        self.assertEqual(primary[primary.versus == "s4_b"].criterion_status.iloc[0], "not_passed")
        self.assertFalse(criteria.beats_both.any())
        for model, seed in [("amazon/chronos-2", 0), ("amazon/chronos-bolt-small", 1)]:
            self.assertTrue((stage4.criteria_table(diffs, model, seed, 1000).configuration_status == "additional").all())
        diagnostic = stage4.criteria_table(diffs, "amazon/chronos-bolt-small", 0, 5)
        self.assertTrue((diagnostic.criterion_status == "diagnostic_n_boot_not_1000").all())

    def test_invalid_scores_fail_before_ranking_not_after_error_mask(self):
        raw = population()
        raw.loc[0, ["s1", "mae_ratio"]] = np.nan
        with self.assertRaisesRegex(ValueError, r"невалидные сигналы.*s1.*1"):
            stage4.prepare_population(raw)

    def test_all_invalid_target_and_constant_target_do_not_crash(self):
        raw = population()
        raw["mae_ratio"] = np.nan
        pop = stage4.prepare_population(raw)
        with small_evaluation():
            met, diffs = stage4.evaluate(pop, 4)
        self.assertTrue((met[met.target == "mae_ratio"].status == "no_valid_windows").all())
        self.assertTrue((diffs.status == "no_valid_windows").all())
        pop["mae_ratio"] = 0.0
        with small_evaluation():
            met, _ = stage4.evaluate(pop, 4)
        gain = met[(met.target == "mae_ratio") & (met.metric == "gain")]
        self.assertTrue(gain.value.isna().all())
        self.assertTrue((gain.status == "undefined_metric").all())

    def test_full_reference_grid_before_s2_filter_with_mock_forecasts(self):
        pos = np.arange(-128, 641, 8)
        n, tau, L = len(pos), 648, 512
        w = pd.DataFrame({"row": np.arange(n), "series_idx": 0, "pos": pos, "t": tau + pos,
                          "cell": "change-mean-strong", "cls": "change", "kind": "mean", "level": "strong"})
        x = (np.sin(np.arange(1400) / 7) + np.arange(1400) / 100)[None, :]
        variants = list(dict.fromkeys(v for vs in stage4.SPREAD_SETS.values() for v in vs))
        ds = {"variants": variants, "horizons": [16, 64], "tau": tau}
        error = np.arange(n, dtype=float) + 1
        error[:2] = 100  # две ранние точки меняют медиану эталона, но исключаются из S2

        def forecasts(root, model_id, horizon, variants=None):
            future = x[0, w.t.to_numpy()[:, None] + 1 + np.arange(horizon)]
            nv = len(ds["variants"]) if variants is None else len(variants)
            q = future[:, None, None, :] - error[:, None, None, None]
            return (q + np.arange(nv)[None, :, None, None] * 0.01
                    + (np.arange(9)[None, None, :, None] - 4) * 0.1)

        archive = MagicMock()
        archive.__enter__.return_value = {"row": np.arange(n),
                                           **{k: np.ones(n) for k in stage4.s5_keys()}}
        with patch.object(stage4, "load_forecasts", side_effect=forecasts), \
                patch.object(stage4.Path, "exists", return_value=True), \
                patch.object(stage4.np, "load", return_value=archive):
            full = stage4.build_table(Path("unused"), "mock", w, x, ds, L)
        pop = stage4.prepare_population(full)
        self.assertEqual(pop.d.min(), -112)
        for h in ds["horizons"]:
            expected = np.median(error[pos < -h])
            wrong = np.median(error[(pos < -h) & (pos >= -112)])
            self.assertNotEqual(expected, wrong)
            np.testing.assert_allclose(pop.loc[pop.horizon == h, "mae_ref_old"], expected)
            np.testing.assert_allclose(pop.loc[pop.horizon == h, "mae_ref_new"], np.median(error[pos >= L]))
        self.assertFalse(pop[["s2_16", "s7_b", "s5_a", "s5_b"]].isna().any().any())

    def test_report_tables_and_risk_figure_use_main_target(self):
        raw = population()
        raw.loc[raw.row < 10, "cls"] = "none"
        raw["kind"] = np.where(raw.cls == "none", "-", "mean")
        raw["level"] = np.where(raw.cls == "none", "-", "strong")
        raw["cell"] = raw.cls + "-" + raw.kind + "-" + raw.level
        raw["zone"] = stage4.assign_zone(raw.d, raw.horizon, 512).where(raw.cls != "none", stage4.CONTROL)
        raw.loc[raw.row == 20, "mae_ratio"] = np.nan
        pop = stage4.prepare_population(raw)
        met, diffs = stage4.evaluate(pop, 3)
        criteria = stage4.criteria_table(diffs, "amazon/chronos-bolt-small", 0, 3)
        zones = [*stage4.zone_names(512), stage4.CONTROL]
        rejected = stage4.rejected_zones(pop, zones)
        groups = stage4.spearman_groups(pop)
        response = stage4.transition_response(pop)
        rc = stage4.risk_coverage_table(pop)
        curves = stage4.signal_curves(pop)
        self.assertEqual(set(rc.target), {"mae_ratio"})
        self.assertEqual(set(rc.subset), set(stage4.SUBSETS))
        self.assertIn("mae_ratio", set(curves.signal))
        self.assertNotIn("mase", set(curves.signal))
        self.assertEqual(set(rejected.target), {"mae_ratio"})
        valid = stage4.valid_target(pop[pop.horizon == 64], "mae_ratio")
        oracle = rc[(rc.horizon == 64) & (rc.subset == stage4.ALL) & (rc.signal == "oracle")]
        np.testing.assert_allclose(oracle.risk, metrics.risk_at_coverage(
            valid.mae_ratio.to_numpy(), valid.mae_ratio.to_numpy(), oracle.coverage.to_numpy()))
        meta = {"model": "amazon/chronos-bolt-small", "date": "test", "cache": "mock", "dataset_hash": "mock",
                "d_eval_min": -112, "n_boot": 3, "criteria": criteria.to_dict("records"), "n_per_cell": 1}
        text = stage4.summarize(met, diffs, rejected, groups, response, meta, zones,
                                list(dict.fromkeys(groups[groups.by == "kind_key"].group)), 512)
        self.assertIn("Основной заранее выбранный критерий", text)
        self.assertIn("без drift", text)
        self.assertNotIn("Спирмен с MASE", text)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "risk.png"
            stage4.risk_coverage_figure(rc, 64, 512, path, stage4.NO_DRIFT)
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()