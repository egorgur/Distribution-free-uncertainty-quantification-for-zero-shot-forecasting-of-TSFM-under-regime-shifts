"""Расчёт/сверка мощности неизменённого экрана на синтетических контролях."""

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.download_gas_data import sha256_file
from src.gas_reference import BOOTSTRAPS, screen_tail
from src.reference_screen_power import (BOOTSTRAP_SEED, CONDITIONS, DATA_SEED,
    PANELS, PHIS, channel_outcomes, mean_only_planning, stationary_ar,
    suitability, summarize, transform)

ROOT = Path(__file__).resolve().parents[1]
CODE = ["src/gas_reference.py", "src/gas_audit.py", "src/reference_screen_power.py",
        "scripts/check_reference_screen_power.py", "tests/test_reference_screen_power.py",
        "docs/protocol_reference_screen_power.md"]


def dump(path, content):
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def plot_rates(summary, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for axis, criterion in zip(axes, ("primary", "sensitivity")):
        for condition in CONDITIONS:
            group = summary.loc[(summary.criterion == criterion)
                                & (summary.condition == condition)
                                & (summary.measure == "accepted")].sort_values("phi")
            axis.errorbar(np.arange(len(PHIS)), group.rate,
                          yerr=np.vstack([group.rate - group.low, group.high - group.rate]),
                          marker="o", capsize=2, label=condition)
        axis.set_xticks(np.arange(len(PHIS)), [str(p) for p in PHIS])
        axis.set(xlabel="AR(1) phi", title=criterion, ylim=(-.02, 1.02))
        axis.axhline(.8, linestyle="--", color="grey", linewidth=.8)
        axis.axhline(.05, linestyle=":", color="grey", linewidth=.8)
    axes[0].set_ylabel("Acceptance rate, panel MC 95% interval")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(out):
    # Не затирать научный снимок с другими хешами даже тем же именем каталога.
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"nonempty output: {out}; use --check or a new directory")
    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    parts = []
    for pi, phi in enumerate(PHIS):
        for panel in range(PANELS):
            base = stationary_ar(phi, panel, pi)
            for ci, condition in enumerate(CONDITIONS):
                diagnostics = screen_tail(transform(base, condition),
                                          [BOOTSTRAP_SEED, pi, ci, panel], repeats=BOOTSTRAPS)
                flags = channel_outcomes(diagnostics)
                flags.insert(0, "panel", panel)
                flags.insert(0, "condition", condition)
                flags.insert(0, "phi", phi)
                parts.append(flags)
        print(f"phi={phi}: {PANELS} panels x {len(CONDITIONS)} conditions", flush=True)
    outcomes = pd.concat(parts, ignore_index=True)
    summary = summarize(outcomes)
    tables = {"outcomes": outcomes, "summary": summary, "criteria": suitability(summary),
              "mean_only_planning": mean_only_planning()}
    for name, frame in tables.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    plot_rates(summary, out / "acceptance.png")
    # Только автоматически полученные таблицы, без научной интерпретации.
    (out / "summary.md").write_text("# Reference-screen power: generated tables\n\n"
        + tables["criteria"].to_csv(index=False) + "\n")
    files = [f"{name}.csv" for name in tables] + ["acceptance.png", "summary.md"]
    meta = {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__, "elapsed_s": time.perf_counter() - started,
            "data_seed": DATA_SEED, "bootstrap_seed": BOOTSTRAP_SEED,
            "panels": PANELS, "channels_per_panel": 16, "phis": list(PHIS),
            "conditions": list(CONDITIONS), "bootstraps": BOOTSTRAPS,
            "raw_real_data_read": False, "model_inference": False,
            "existing_screen_modified": False, "new_gas_selection_authorized": False,
            "interval_unit": "independent Monte Carlo panel; not sensor",
            "code_sha256": {p: sha256_file(ROOT / p) for p in CODE},
            "artifact_sha256": {p: sha256_file(out / p) for p in files}}
    dump(out / "meta.json", meta)
    return tables


def check(out):
    meta = json.loads((out / "meta.json").read_text())
    for name, expected in meta["code_sha256"].items():
        assert sha256_file(ROOT / name) == expected, name
    for name, expected in meta["artifact_sha256"].items():
        assert sha256_file(out / name) == expected, name
    outcomes = pd.read_csv(out / "outcomes.csv")
    expected_rows = len(PHIS) * len(CONDITIONS) * PANELS * 16 * 2
    assert len(outcomes) == expected_rows
    assert not outcomes.duplicated(["phi", "condition", "panel", "sensor", "criterion"]).any()
    for _, group in outcomes.groupby(["phi", "condition", "criterion"]):
        assert sorted(group.panel.unique().tolist()) == list(range(PANELS))
        assert (group.groupby("panel").size() == 16).all()
    assert not (outcomes.accepted & ~outcomes.point_only).any()
    assert not (outcomes.accepted & (outcomes.mean_not_confirmed | outcomes.variance_not_confirmed
                                    | outcomes.undefined)).any()
    summary = pd.read_csv(out / "summary.csv")
    pd.testing.assert_frame_equal(summary, summarize(outcomes), check_exact=False, atol=1e-12, rtol=1e-12)
    pd.testing.assert_frame_equal(pd.read_csv(out / "criteria.csv"), suitability(summary))
    pd.testing.assert_frame_equal(pd.read_csv(out / "mean_only_planning.csv"), mean_only_planning())
    # Независимая прямая проверка базовых долей по всем решениям.
    for row in summary.itertuples():
        cell = outcomes.loc[(outcomes.phi == row.phi) & (outcomes.condition == row.condition)
                            & (outcomes.criterion == row.criterion)]
        assert abs(float(cell[row.measure].sum()) / len(cell) - row.rate) < 1e-12
        assert 0 <= row.low <= row.rate + 1e-12 and row.rate <= row.high + 1e-12 <= 1 + 1e-12
    # Четыре независимых от сохранённых флагов повторных вызова исходного экрана.
    for pi, ci, panel in ((0, 0, 0), (1, 2, 11), (2, 3, 23), (3, 4, 47)):
        fresh = channel_outcomes(screen_tail(transform(stationary_ar(PHIS[pi], panel, pi), CONDITIONS[ci]),
                                            [BOOTSTRAP_SEED, pi, ci, panel]))
        saved = outcomes.loc[(outcomes.phi == PHIS[pi]) & (outcomes.condition == CONDITIONS[ci])
                             & (outcomes.panel == panel), fresh.columns].reset_index(drop=True)
        pd.testing.assert_frame_equal(saved, fresh)
    result = {"passed": True, "decision_rows": len(outcomes), "cells": len(PHIS) * len(CONDITIONS),
              "independent_panels_per_cell": PANELS, "replayed_panels": 4,
              "real_data_read": False, "model_inference": False,
              "screen_suitability_confirmed": bool(pd.read_csv(out / "criteria.csv").both_ok.all())}
    dump(out / "validation.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results/reference_screen_power_v1")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        print(json.dumps(check(args.out), indent=2))
    else:
        print(run(args.out)["criteria"].to_string(index=False))


if __name__ == "__main__":
    main()