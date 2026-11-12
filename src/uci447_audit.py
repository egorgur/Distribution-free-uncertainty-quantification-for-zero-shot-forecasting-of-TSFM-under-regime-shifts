"""Bounded structural audit helpers for UCI447 hydraulic cycles."""

import csv
import io
import math
from collections import Counter


def parse_profile(content):
    rows = []
    for line_number, row in enumerate(csv.reader(io.StringIO(content.decode("utf-8-sig")), delimiter="\t"), 1):
        if not row or all(not value.strip() for value in row):
            raise ValueError(f"profile row {line_number}: empty row")
        if len(row) != 5:
            raise ValueError(f"profile row {line_number}: expected 5 fields")
        try:
            values = tuple(int(value) for value in row)
        except ValueError as exc:
            raise ValueError(f"profile row {line_number}: non-integer value") from exc
        domains = ({3, 20, 100}, {73, 80, 90, 100}, {0, 1, 2}, {90, 100, 115, 130}, {0, 1})
        if any(value not in domain for value, domain in zip(values, domains)):
            raise ValueError(f"profile row {line_number}: value outside documented domain")
        rows.append(values)
    if not rows:
        raise ValueError("empty profile")
    runs = []
    start = 0
    for stop in range(1, len(rows) + 1):
        if stop == len(rows) or rows[stop] != rows[start]:
            runs.append({"start_cycle": start, "stop_exclusive": stop,
                         "rows": stop - start, "profile": list(rows[start])})
            start = stop
    return {"cycles": len(rows),
            "columns": ["cooler", "valve", "pump_leakage", "accumulator", "stable_flag"],
            "unique_values": [sorted({row[i] for row in rows}) for i in range(5)],
            "value_counts": [{str(k): v for k, v in sorted(Counter(row[i] for row in rows).items())}
                             for i in range(5)],
            "stable_flag_counts": {str(k): v for k, v in sorted(Counter(row[-1] for row in rows).items())},
            "profile_runs": runs,
            "stable_flag_is_cycle_label": True,
            "stationarity_established": False}


def parse_ps1(content, expected_cycles=2205, expected_samples=6000):
    """Summarise PS1 without joining rows into a continuous time series."""
    if expected_samples < 2 or expected_samples % 2 or expected_cycles < 1:
        raise ValueError("Expected positive cycles and a positive even sample count")
    cycle_stats = []
    reader = csv.reader(io.StringIO(content.decode("utf-8-sig")), delimiter="\t")
    for cycle, row in enumerate(reader):
        if not row or all(not value.strip() for value in row):
            raise ValueError(f"PS1 cycle {cycle}: empty row")
        if len(row) != expected_samples:
            raise ValueError(f"PS1 cycle {cycle}: expected {expected_samples} values")
        try:
            values = [float(value) for value in row]
        except ValueError as exc:
            raise ValueError(f"PS1 cycle {cycle}: nonnumeric value") from exc
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"PS1 cycle {cycle}: nonfinite value")
        mean = math.fsum(values) / len(values)
        first = math.fsum(values[:len(values) // 2]) / (len(values) // 2)
        second = math.fsum(values[len(values) // 2:]) / (len(values) // 2)
        variance = math.fsum((value - mean) ** 2 for value in values) / len(values)
        cycle_stats.append({"cycle": cycle, "mean": mean, "std": math.sqrt(variance),
                            "min": min(values), "max": max(values),
                            "first_half_mean": first, "second_half_mean": second,
                            "half_mean_difference": second - first})
    if len(cycle_stats) != expected_cycles:
        raise ValueError(f"PS1: expected {expected_cycles} cycles, got {len(cycle_stats)}")
    return {"cycles": len(cycle_stats), "samples_per_cycle": expected_samples,
            "sample_rate_hz": 100, "cycle_seconds": 60,
            "finite_values": len(cycle_stats) * expected_samples,
            "cycle_stats": cycle_stats,
            "continuity_between_cycles_established": False,
            "stationarity_established": False}


def cycle_geometry(cycle_seconds=60, sample_rates=(1, 10, 100), length=512, horizon=64):
    result = []
    for rate in sample_rates:
        samples = cycle_seconds * rate
        needed = length + horizon
        result.append({"sample_rate_hz": rate, "samples_per_cycle": samples,
                       "needed_samples": needed,
                       "full_windows_per_cycle_stride1": max(0, samples - needed + 1),
                       "full_window_seconds": needed / rate,
                       "length_seconds": length / rate,
                       "horizon_seconds": horizon / rate})
    return result