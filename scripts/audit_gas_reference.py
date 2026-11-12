"""Выполнить/проверить описательный экран кандидатов устойчивого эталона."""

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.check_gas_audit import check as check_originals
from scripts.download_gas_data import sha256_file
from src.gas_audit import audit_file
from src.gas_reference import (BOOTSTRAPS, SEED, adjacent_candidates, collapse_candidates, passes,
                               screen_tail, select_tail)

ROOT = Path(__file__).resolve().parents[1]
CODE = ["src/gas_reference.py", "scripts/audit_gas_reference.py",
        "tests/test_gas_reference.py", "docs/protocol_gas_reference_audit.md"]
RECORDINGS = ("ethylene_CO", "ethylene_methane")


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def run(raw, source, out):
    started = time.perf_counter()
    # Предыдущие хеши оригиналов, CRC и структурные инварианты.
    check_originals(raw, source)
    inputs = {str(source / "audit.json"): sha256_file(source / "audit.json")}
    eligibility, details, flag_parts, event_parts = [], [], [], []
    for recording_id, name in enumerate(RECORDINGS):
        # Повторная агрегация оригинала проверяет и сами средние, не только число строк.
        _, fresh = audit_file(raw / f"{name}.txt")
        folder = source / name
        seconds = pd.read_csv(folder / "seconds.csv.gz")
        segments = pd.read_csv(folder / "segments.csv")
        for key, saved in (("seconds", seconds), ("segments", segments)):
            # Старый аудит сохраняет %.12g: допустима ошибка CSV-округления,
            # но не смысловое изменение средних/границ. Достаточно 1e-11.
            pd.testing.assert_frame_equal(saved, fresh[key], check_dtype=False,
                                          rtol=1e-11, atol=1e-10)
            path = folder / ("seconds.csv.gz" if key == "seconds" else "segments.csv")
            inputs[str(path)] = sha256_file(path)
        for segment in segments.itertuples():
            values, reason, start, end = select_tail(seconds, segment)
            eligibility.append({"recording": name, "segment": segment.segment,
                                "reason": reason, "start_s": start, "end_s": end,
                                "duration_s": segment.duration_s})
            if values is None:
                continue
            table = screen_tail(values, [SEED, recording_id, segment.segment])
            table.insert(0, "segment", segment.segment)
            table.insert(0, "recording", name)
            details.append(table)
        part_list = [part for part in details if part.recording.iloc[0] == name]
        if not part_list:
            raise ValueError(f"{name}: no structurally eligible tails")
        combined = pd.concat(part_list, ignore_index=True)
        flags = collapse_candidates(combined)
        flag_parts.append(flags)
        events = adjacent_candidates(flags, segments.segment)
        events.insert(0, "recording", name)
        event_parts.append(events)
    tables = {"eligibility": pd.DataFrame(eligibility),
              "diagnostics": pd.concat(details, ignore_index=True),
              "candidates": pd.concat(flag_parts, ignore_index=True),
              "events": pd.concat(event_parts, ignore_index=True)}
    tables["counts"] = tables["candidates"].groupby(["recording", "sensor"], as_index=False)[
        ["primary", "sensitivity"]].sum()
    tables["event_counts"] = tables["events"].groupby(["recording", "sensor"], as_index=False)[
        ["primary", "sensitivity"]].sum()
    # Post-hoc проверка мощности/реализации, НЕ подбор допусков или протокола.
    # Непрохождение iid не доказывает нестационарность реальных показаний.
    controls = []
    iid = np.random.default_rng(777).normal(size=(120, 16))
    drift = np.tile(np.arange(120, dtype=float)[:, None], (1, 16))
    for label, values in (("iid_white_noise", iid), ("linear_drift", drift)):
        control = screen_tail(values, [SEED, 999])
        control.insert(0, "control", label)
        controls.append(control)
    tables["implementation_controls"] = pd.concat(controls, ignore_index=True)
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    meta = {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "seed": SEED, "bootstraps": BOOTSTRAPS,
            "elapsed_s": time.perf_counter() - started,
            "model_inference": False, "stationarity_proven": False,
            "reference_approved": False, "seconds_verified_against_raw": True,
            "controls": "post-hoc implementation/power diagnostic, no threshold changes; iid seed777",
            "independence": "sensors, adjacent events and rolling windows are dependent",
            "code_sha256": {p: sha256_file(ROOT / p) for p in CODE},
            "inputs_sha256": inputs,
            "artifacts_sha256": {f"{name}.csv": sha256_file(out / f"{name}.csv") for name in tables}}
    dump(out / "audit.json", meta)
    return tables


def check(out):
    meta = json.loads((out / "audit.json").read_text())
    for name, expected in meta["code_sha256"].items():
        assert sha256_file(ROOT / name) == expected, name
    for name, expected in meta["inputs_sha256"].items():
        assert sha256_file(Path(name)) == expected, name
    for name, expected in meta["artifacts_sha256"].items():
        assert sha256_file(out / name) == expected, name
    diagnostics = pd.read_csv(out / "diagnostics.csv")
    for row in diagnostics.itertuples():
        assert row.primary == passes(row.mean_upper, row.log_variance_upper)
        assert row.sensitivity == passes(row.mean_upper, row.log_variance_upper, .5, 2.)
        if row.status != "ok":
            assert not row.primary and not row.sensitivity
    flags = pd.read_csv(out / "candidates.csv")
    expected = collapse_candidates(diagnostics)
    pd.testing.assert_frame_equal(flags, expected)
    events = pd.read_csv(out / "events.csv")
    lookup = {(r.recording, r.segment, r.sensor): r for r in flags.itertuples()}
    # Независимый простой перебор смежных участков, без вызова adjacent_candidates.
    for row in events.itertuples():
        for criterion in ("primary", "sensitivity"):
            pair = [lookup.get((row.recording, s, row.sensor))
                    for s in (row.new_segment - 1, row.new_segment)]
            expected_value = all(p is not None and getattr(p, criterion) for p in pair)
            assert getattr(row, criterion) == expected_value
    for table, counts in ((flags, "counts"), (events, "event_counts")):
        expected = table.groupby(["recording", "sensor"], as_index=False)[
            ["primary", "sensitivity"]].sum()
        pd.testing.assert_frame_equal(pd.read_csv(out / f"{counts}.csv"), expected)
    controls = pd.read_csv(out / "implementation_controls.csv")
    for row in controls.itertuples():
        assert row.primary == passes(row.mean_upper, row.log_variance_upper)
        assert row.sensitivity == passes(row.mean_upper, row.log_variance_upper, .5, 2.)
    assert not controls.loc[controls.control == "linear_drift", "primary"].any()
    result = {"passed": True, "diagnostics_checked": len(diagnostics),
              "event_channel_pairs_checked": len(events), "model_inference": False,
              "stationarity_proven": False, "reference_approved": False,
              "raw_means_recomputed_during_audit": meta["seconds_verified_against_raw"]}
    dump(out / "validation.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw/uci_gas_dynamic")
    parser.add_argument("--source", type=Path, default=ROOT / "results/real_gas_audit")
    parser.add_argument("--out", type=Path, default=ROOT / "results/gas_reference_audit_v1_recheck")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        print(json.dumps(check(args.out), indent=2))
    else:
        tables = run(args.raw, args.source, args.out)
        print(tables["eligibility"].groupby(["recording", "reason"]).size().to_string())
        print(tables["counts"].to_string(index=False))
        print(tables["event_counts"].to_string(index=False))


if __name__ == "__main__":
    main()