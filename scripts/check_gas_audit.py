"""Сверка исходных хешей и независимых инвариантов таблиц аудита газа."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.download_gas_data import sha256_file, validate_archive


def check(raw, out):
    manifest = json.loads((raw / "manifest.json").read_text())
    meta = json.loads((out / "audit.json").read_text())
    for entry in [manifest["archive"], *manifest["files"]]:
        path = raw / entry["name"]
        assert path.stat().st_size == entry["bytes"], path
        assert sha256_file(path) == entry["sha256"], path
    validate_archive(raw / manifest["archive"]["name"])
    for filename, expected in meta["code_sha256"].items():
        assert sha256_file(Path(filename)) == expected, filename
    checks = []
    for record in meta["recordings"]:
        folder = out / Path(record["file"]).stem
        columns = pd.read_csv(folder / "columns.csv")
        segments = pd.read_csv(folder / "segments.csv")
        events = pd.read_csv(folder / "events.csv")
        seconds = pd.read_csv(folder / "seconds.csv.gz")
        anomalies = pd.read_csv(folder / "time_anomalies.csv.gz")
        assert len(columns) == 19
        assert (columns.finite + columns.nonfinite == record["rows"]).all()
        assert columns.nonfinite.sum() == record["nonfinite_values"]
        assert segments.rows.sum() == record["rows"] == seconds.rows.sum()
        assert len(segments) == record["setpoint_segments"] == len(events) + 1
        assert len(events) == record["setpoint_events"]
        assert len(anomalies) == record["non_nominal_steps"]
        assert (anomalies.delta_ticks == 0).sum() == record["adjacent_duplicate_times"]
        assert (anomalies.delta_ticks < 0).sum() == record["backward_steps"]
        histogram = record["delta_centisecond_histogram"]
        assert sum(histogram.values()) == record["rows"] - 1
        assert sum(int(k) * v for k, v in histogram.items()) == round(
            (record["last_time_s"] - record["first_time_s"]) * 100)
        np.testing.assert_array_equal(segments.end_row_exclusive.iloc[:-1], segments.start_row.iloc[1:])
        np.testing.assert_array_equal(events.row, segments.start_row.iloc[1:])
        np.testing.assert_array_equal(events.pre_rows, segments.rows.iloc[:-1])
        np.testing.assert_array_equal(events.post_rows, segments.rows.iloc[1:])
        assert (segments.rows > 0).all() and (segments.duration_s > 0).all()
        assert segments.right_censored.sum() == segments.left_censored.sum() == 1
        assert int(seconds.mixed_setpoint.sum()) == record["mixed_seconds"]
        for figure in ("overview.png", "transitions.png", "durations_tail.png"):
            assert (folder / figure).stat().st_size > 1000, figure
        checks.append({"file": record["file"], "rows": record["rows"],
                       "events": len(events), "status": "passed"})
    result = {"verified_at_utc": datetime.now(timezone.utc).isoformat(),
              "status": "passed", "recordings": checks,
              "checks": ["raw SHA256 and sizes", "ZIP CRC", "audit source hashes",
                         "column/segment/second row totals", "event boundaries",
                         "timestamp histogram count and span", "censoring", "figures exist"],
              "scope": "Internal consistency, not physical stability or approved H1-H3 protocol"}
    (out / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/uci_gas_dynamic"))
    parser.add_argument("--out", type=Path, default=Path("results/real_gas_audit"))
    args = parser.parse_args()
    print(json.dumps(check(args.raw, args.out), indent=2))


if __name__ == "__main__":
    main()