"""Descriptive MCC5 diagnostics in original units; no stability acceptance rule."""

import csv
import numpy as np

FS = 12800
SIGNALS = ("key_phase", "torque", "vibration_x", "vibration_y", "vibration_z",
           "current_a", "current_b", "current_c")
PLATEAUS = ((10, 30, 20), (35, 55, 15), (60, 80, 20))


def load_recording(path):
    """Check every logical row, including blanks, before creating a numeric array."""
    blocks, rows, header = [], [], None
    width = None
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        for number, row in enumerate(reader, 1):
            if not row or all(not value.strip() for value in row):
                raise ValueError(f"Empty CSV row {number}")
            if width is None:
                width = len(row)
                if width not in (8, 9):
                    raise ValueError(f"Unexpected CSV width: {width}")
            if len(row) != width:
                raise ValueError(f"CSV row {number}: width differs")
            try:
                values = [float(value) for value in row]
            except ValueError as exc:
                # A header is recognised only at the very first logical row.
                all_text = True
                for value in row:
                    try:
                        float(value)
                    except ValueError:
                        continue
                    all_text = False
                if number == 1 and all_text and all(value.strip() for value in row):
                    header = row
                    continue
                raise ValueError(f"CSV row {number}: nonnumeric value") from exc
            if not np.isfinite(values).all():
                raise ValueError(f"CSV row {number}: nonfinite value")
            rows.append(values)
            if len(rows) == 8192:
                blocks.append(np.asarray(rows, dtype=np.float64))
                rows = []
    if rows:
        blocks.append(np.asarray(rows, dtype=np.float64))
    if not blocks:
        raise ValueError("No numeric rows")
    data = np.concatenate(blocks)
    return data, {"columns": width, "header": header, "data_rows": len(data),
                  "names_from_article_table4": (["time"] if width == 9 else []) + list(SIGNALS),
                  "unit_conversion_applied": False}


def basic_stats(values):
    return {"count": len(values), "mean": float(values.mean()),
            "std": float(values.std()), "min": float(values.min()), "max": float(values.max())}


def aggregate(signals, fs=FS, seconds: float = 1.0, columns=None):
    """Non-overlapping index bins; diagnostic only, never a model resampling."""
    size = round(fs * seconds)
    if size < 1 or not np.isclose(size, fs * seconds):
        raise ValueError("Bin size must be a positive integer number of samples")
    columns = range(signals.shape[1]) if columns is None else columns
    rows = []
    for start in range(0, len(signals), size):
        stop = min(start + size, len(signals))
        for column in columns:
            rows.append({"start_s": start / fs, "stop_exclusive_s": stop / fs,
                         "channel": SIGNALS[column], **basic_stats(signals[start:stop, column])})
    return rows


def plateau_stats(signals, fs=FS):
    """Keep documented intervals fixed even if the measured trace differs."""
    rows = []
    for start, stop, target in PLATEAUS:
        intervals = [(start, stop, "whole")] + [(a, a + 5, "five_second") for a in range(start, stop, 5)]
        for a, b, kind in intervals:
            lo, hi = a * fs, b * fs
            if hi > len(signals):
                continue
            torque = signals[lo:hi, 1]
            index = np.arange(len(torque), dtype=np.float64)
            index -= index.mean()
            slope = float(np.dot(index, torque - torque.mean()) / np.dot(index, index) * fs)
            rows.append({"start_s": a, "stop_exclusive_s": b, "kind": kind,
                         "parent_start_s": start, "command_example_Nm": target,
                         **basic_stats(torque), "slope_original_units_per_s": slope,
                         "stability_established": False})
    return rows


def key_phase_stats(phase, fs=FS):
    low, high = float(phase.min()), float(phase.max())
    threshold = (low + high) / 2
    indices = np.flatnonzero((phase[:-1] <= threshold) & (phase[1:] > threshold)) + 1
    gaps = np.diff(indices) / fs
    rate = [{"start_s": start / fs,
             "stop_exclusive_s": min(start + fs, len(phase)) / fs,
             "rising_edges": int(np.count_nonzero((indices >= start) & (indices < min(start + fs, len(phase)))))}
            for start in range(0, len(phase), fs)]
    return {"threshold_raw": threshold, "threshold_rule": "midpoint of full-run min/max; diagnostic only",
            "rising_edges": len(indices), "rising_edges_per_s": len(indices) / (len(phase) / fs),
            "inter_edge_seconds_quantiles": ([float(v) for v in np.quantile(gaps, [0, .01, .5, .99, 1])]
                                             if len(gaps) else []),
            "rpm_derived": False, "pulses_per_revolution_verified": False,
            "per_second": rate}


def analyse(data, fs=FS):
    if data.ndim != 2 or data.shape[1] not in (8, 9) or len(data) < 2 or not np.isfinite(data).all():
        raise ValueError("Expected at least two finite 8/9-column rows")
    if fs < 1 or int(fs) != fs:
        raise ValueError("Expected positive integer sample rate")
    has_time = data.shape[1] == 9
    signals = data[:, 1:] if has_time else data
    time = data[:, 0] if has_time else np.arange(len(data)) / fs
    grid = time[0] + np.arange(len(data)) / fs
    dt = np.diff(time)
    time_info = {"timestamp_present": has_time, "sample_rate_documented_hz": fs,
                 "first_s": float(time[0]), "last_s": float(time[-1]),
                 "count_divided_by_fs_s": len(data) / fs,
                 "dt_min_s": float(dt.min()), "dt_max_s": float(dt.max()),
                 "dt_zero": int(np.count_nonzero(dt == 0)), "dt_negative": int(np.count_nonzero(dt < 0)),
                 "max_abs_deviation_from_nominal_s": float(np.max(np.abs(time - grid))),
                 "hardware_continuity_verified": False,
                 "bins_basis": "row-index / documented fs; compare separately against raw time"}
    channels = []
    for column, name in enumerate(SIGNALS):
        x = signals[:, column]
        channels.append({"channel": name, **basic_stats(x), "nonfinite": 0,
                         "unique": len(np.unique(x)), "zeros": int(np.count_nonzero(x == 0)),
                         "negative": int(np.count_nonzero(x < 0)),
                         "adjacent_equal": int(np.count_nonzero(x[1:] == x[:-1]))})
    duplicates = [[SIGNALS[a], SIGNALS[b]] for a in range(8) for b in range(a + 1, 8)
                  if np.array_equal(signals[:, a], signals[:, b])]
    return {"time": time_info, "channels": channels, "exact_duplicate_channels": duplicates,
            "seconds": aggregate(signals, fs), "torque_quarters": aggregate(signals, fs, .25, [1]),
            "plateaus": plateau_stats(signals, fs), "key_phase": key_phase_stats(signals[:, 0], fs),
            "geometry": [{"sample_rate_hz": rate, "length_seconds": 512 / rate,
                          "horizon_seconds": 64 / rate, "full_window_seconds": 576 / rate,
                          "nominal_full_windows_in_20s_stride1": max(0, 20 * rate - 576 + 1)}
                         for rate in (FS, 100, 10, 1)],
            "stability_established": False, "model_inference": False}