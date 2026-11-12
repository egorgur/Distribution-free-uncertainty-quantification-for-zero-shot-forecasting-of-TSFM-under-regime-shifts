"""Сверка сохранённых результатов H1/H3 v1, без моделей и нового инференса.

python -m scripts.check_h1_h3_results
Падает при несовпадении; сохраняет validation.json рядом с новыми результатами.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/protocol_h1_h3_v1"


def check() -> dict:
    report = {"protocol_sha256": hashlib.sha256((ROOT / "docs/protocol_h1_h3.md").read_bytes()).hexdigest(),
              "checks": []}
    for model in ("chronos-bolt-small", "chronos-2"):
        p = RESULTS / "stage3" / model
        meta = json.loads((p / "meta.json").read_text())
        assert meta["protocol_sha256"] == report["protocol_sha256"]
        assert meta["n_boot"] == 1000 and meta["control_seed"] == 1
        for name, digest in meta["source_code_sha256"].items():
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
        stars = pd.read_csv(p / "d_star.csv")
        assert set(stars.status) <= {"no_degradation", "recovered", "not_confirmed"}
        assert (stars[["boot_no_degradation_n", "boot_recovered_n", "boot_not_confirmed_n"]].sum(axis=1) == 1000).all()
        assert ((stars.status == "recovered") == stars.d_star.notna()).all()
        recovered = stars[stars.status == "recovered"]
        assert recovered.d_star.between(8, 440).all() and (recovered.d_star % 8 == 0).all()
        thresholds = pd.read_csv(p / "thresholds.csv")
        controls = pd.read_csv(p / "control_curves.csv")
        for row in thresholds.itertuples():
            curve = controls[(controls.L == row.L) & (controls.horizon == row.horizon)]
            np.testing.assert_array_equal(curve.d.to_numpy(), np.arange(0, 505, 8))
            raw = np.percentile(curve.smoothed_value, row.percentile, method="linear")
            np.testing.assert_allclose([raw, max(1, raw)], [row.raw_quantile, row.threshold], rtol=1e-12)
        h1 = pd.read_csv(p / "window_metrics.csv.gz")
        h3 = pd.read_csv(RESULTS / "stage4" / model / "window_signals.csv.gz")
        joined = h3.merge(h1[h1.L == 512], on=["row", "horizon"], suffixes=("_h3", "_h1"), validate="one_to_one")
        assert len(joined) == len(h3) == 64600
        for field in ("mae", "mae_ratio", "mae_ref_old", "mae_ref_new"):
            np.testing.assert_allclose(joined[field + "_h3"], joined[field + "_h1"],
                                       rtol=1e-10, atol=1e-12, equal_nan=True)
        report["checks"].append({"model": model, "h1_h3_matched_rows": len(joined),
                                 "recovery_statuses_and_control_quantiles": "passed"})

    for model in ("chronos-bolt-small", "chronos-2", "synth_v1_seed1/chronos-bolt-small"):
        p = RESULTS / "stage4" / model
        meta = json.loads((p / "meta.json").read_text())
        assert meta["protocol_sha256"] == report["protocol_sha256"] and meta["n_boot"] == 1000
        for name, digest in meta["input_sha256"].items():
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
        metrics = pd.read_csv(p / "metrics.csv")
        criteria = pd.read_csv(p / "criteria.csv")
        assert (criteria.n_boot_valid == 1000).all() and (criteria.n_boot_invalid == 0).all()
        assert ((criteria.criterion_status == "passed") == (criteria.hi < 0)).all()
        assert (metrics[metrics.target == "mae_ratio"].n_excluded == 0).all()
        np.testing.assert_allclose(metrics.actual_bad_share, 0.1, atol=1e-12)
        # Проверка неизменности сигналов и описательных метрик против исторического расчёта.
        # Интервалы и новые подгруппы не сравниваются: для них протокол намеренно изменён.
        old = pd.read_csv(ROOT / "results/stage4" / model / "metrics.csv")
        keys = ["horizon", "subset", "target", "signal", "metric"]
        matched = old.merge(metrics, on=keys, suffixes=("_old", "_v1"), validate="one_to_one")
        unchanged = matched[matched.target.isin(["mase", "crps_s", "mae_fixed"])]
        assert len(unchanged) > 0
        np.testing.assert_allclose(unchanged.value_old, unchanged.value_v1,
                                   rtol=3e-4, atol=2e-5, equal_nan=True)
        report["checks"].append({"model": model, "unchanged_descriptive_metric_rows": len(unchanged),
                                 "primary_error_exclusions": 0, "bad_label_share": 0.1,
                                 "paired_criteria_and_input_hashes": "passed"})
    report["status"] = "passed"
    return report


if __name__ == "__main__":
    result = check()
    (RESULTS / "validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))