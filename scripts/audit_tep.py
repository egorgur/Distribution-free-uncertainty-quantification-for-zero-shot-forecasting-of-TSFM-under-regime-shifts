"""Аудит TEP MIT: python -m scripts.audit_tep [--check]. Без моделей и сети."""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.tep_audit import CONFIGS, audit, load_dat, rolling_geometry, sha256

ROOT = Path(__file__).resolve().parents[1]
CODE = ("src/tep_audit.py", "scripts/audit_tep.py")


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def run(raw: Path, out: Path) -> None:
    started = time.perf_counter()
    tables = audit(raw)
    out.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
    # Фиксированные fault1/8/13: примеры step/random/drift по описанию Fortran.
    for fault in (0, 1, 8, 13):
        name = f"d{fault:02}_te.dat"
        x, _ = load_dat(raw / "original/TE_process" / name)
        fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
        for ax, j in zip(axes, (6, 8, 22)):
            ax.plot(np.arange(len(x)) * 3 / 60, x[:, j], lw=.7)
            ax.set_ylabel(f"XMEAS{j+1:02}")
            if fault:
                ax.axvline(8, color="C3", ls="--", label="документированное вмешательство 8 ч")
                ax.legend(fontsize=7)
        axes[-1].set_xlabel("Номинальное время, ч; timestamp в DAT отсутствуют")
        fig.suptitle(f"TEP MIT {name}: описание, не доказательство устойчивости")
        fig.tight_layout()
        fig.savefig(out / f"fig_{fault:02}.png", dpi=130)
        plt.close(fig)
    files = tables["files"]
    totals = dict(numeric_files=len(files), rows=int(files.rows.sum()), process_variables=52,
                  nonfinite=int(files.nonfinite.sum()), testing_fault_scenarios=int(((files.split == "testing") & (files.fault != 0)).sum()))
    metadata = dict(dataset="TEP classic MIT (NOT Rieth 2017)", totals=totals,
        versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__),
        model_inference=False, stationarity_proven=False, protocol_selected=False,
        onset="test fault: zero-based row160 after160 baseline rows, documentation-derived; row159 sensitivity unresolved",
        time="180s from temain_mod.f.txt, no timestamps in DAT; not independently verified measurement clock",
        train_onset="not inferred; filename fault is scenario, not pointwise label",
        configurations=[dict(L=l, H=h, stride=8) for l, h in CONFIGS],
        manifest_sha256=sha256(raw / "manifest.json"),
        code_sha256={name: sha256(ROOT / name) for name in CODE},
        artifact_sha256={p.name: sha256(p) for p in sorted(out.iterdir()) if p.suffix in {".csv", ".png"}},
        elapsed_seconds=round(time.perf_counter() - started, 3))
    write_json(out / "audit.json", metadata)
    total_windows = tables["windows"].groupby(["split", "L", "H"])["row_order"].sum()
    (out / "summary.md").write_text("# TEP MIT: таблицы аудита\n\nБез инференса.\n\n" + json.dumps(totals) + "\n\n" + total_windows.to_csv())
    print(json.dumps(totals, indent=2))
    print(total_windows.to_string())


def check(raw: Path, out: Path) -> None:
    meta = json.loads((out / "audit.json").read_text())
    assert sha256(raw / "manifest.json") == meta["manifest_sha256"]
    for name, digest in meta["code_sha256"].items():
        assert sha256(ROOT / name) == digest
    for name, digest in meta["artifact_sha256"].items():
        assert sha256(out / name) == digest
    # Полный повтор разбора и сравнение всех сохранённых таблиц.
    tables = audit(raw)
    for name, table in tables.items():
        pd.testing.assert_frame_equal(pd.read_csv(out / f"{name}.csv"), table, check_dtype=False, check_exact=False)
    # Отдельный прямой перебор геометрии (не тот же векторный код).
    checked = 0
    for record in tables["files"].itertuples():
        for length, horizon in CONFIGS:
            origins = list(range(length - 1, record.rows - horizon, 8))
            onset = 160 if record.split == "testing" and record.fault else None
            result = rolling_geometry(record.rows, length, horizon, onset)
            assert result["row_order"] == len(origins)
            if onset is not None:
                counts = dict(before=0, change_in_horizon=0, mixed_context=0, fully_post=0)
                for t in origins:
                    if t + horizon < onset:
                        counts["before"] += 1
                    elif t < onset:
                        counts["change_in_horizon"] += 1
                    elif t - length + 1 < onset:
                        counts["mixed_context"] += 1
                    else:
                        counts["fully_post"] += 1
                assert all(result[k] == v for k, v in counts.items())
                assert sum(counts.values()) == len(origins)
            checked += 1
    write_json(out / "validation.json", dict(passed=True, original_files_verified=48,
        tables_verified=len(tables), independently_checked_configurations=checked,
        model_inference=False, physical_stability_proven=False))
    print("TEP audit validation: passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw/tep_mit")
    parser.add_argument("--out", type=Path, default=ROOT / "results/tep_audit")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    (check if args.check else run)(args.raw, args.out)


if __name__ == "__main__":
    main()