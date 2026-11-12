"""Потоковый описательный аудит газа UCI; исходные строки не исправляются.

Секундные агрегаты предназначены только для графиков, не для моделей.
Индекс строки в результатах — от нуля, без строки заголовка.
"""

from pathlib import Path

import numpy as np
import pandas as pd

SENSORS = [f"s{i:02d}" for i in range(1, 17)]
TYPES = ["TGS2602", "TGS2602", "TGS2600", "TGS2600",
         "TGS2610", "TGS2610", "TGS2620", "TGS2620"] * 2
COLUMNS = ["time_s", "gas_ppm", "ethylene_ppm", *SENSORS]


def validate_row_width(path):
    """Проверить поля каждой физической строки: pandas дополняет короткие NaN."""
    rows = 0
    with Path(path).open("rb") as source:
        source.readline()
        for rows, line in enumerate(source, 1):
            fields = len(line.split())
            if fields != 19:
                raise ValueError(f"{Path(path).name}: data row {rows - 1} has {fields} fields, expected 19")
    if not rows:
        raise ValueError(f"{Path(path).name}: no numeric rows")
    return rows


def run_summary(x, previous=None, trailing=0):
    """Число равных соседних пар, максимальная и последняя серии; NaN не равны."""
    finite = np.isfinite(x)
    joins = (x[1:] == x[:-1]) & finite[1:] & finite[:-1]
    starts = np.r_[0, np.flatnonzero(~joins) + 1]
    lengths = np.diff(np.r_[starts, len(x)])
    boundary = previous is not None and np.isfinite(previous) and finite[0] and x[0] == previous
    if boundary:
        lengths[0] += trailing
    valid = finite[starts]
    maximum = int(lengths[valid].max()) if valid.any() else 0
    return int(joins.sum()) + int(boundary), maximum, int(lengths[-1]) if finite[-1] else 0


def late_tail_diagnostics(seconds, segments):
    """Сдвиг хвоста задания, без объявления физической устойчивости.

    Только целые неперемешанные секунды, полностью внутри наблюдённого
    отрезка. Сравниваются первые/последние 10 секунд последних 30 секунд.
    Нормировка — std секундных средних в этих 30 секундах, не шум датчика.
    """
    rows = []
    for segment in segments.itertuples():
        end = segment.boundary_s
        if end - segment.start_s < 30:
            continue
        bins = seconds.loc[(seconds.second >= end - 30)
                           & (seconds.second >= segment.start_s)
                           & (seconds.second + 1 <= end)
                           & ~seconds.mixed_setpoint
                           & (seconds.gas_ppm_min == segment.gas_ppm)
                           & (seconds.ethylene_ppm_min == segment.ethylene_ppm)]
        first = bins.loc[bins.second + 1 <= end - 20]
        last = bins.loc[bins.second >= end - 10]
        for sensor in SENSORS:
            delta = last[sensor].mean() - first[sensor].mean()
            scale = bins[sensor].std(ddof=0)
            rows.append({"segment": segment.segment, "sensor": sensor,
                         "right_censored": segment.right_censored,
                         "first_bins": len(first), "last_bins": len(last),
                         "tail_bins": len(bins), "delta": delta,
                         "tail_std_1s_means": scale,
                         "delta_over_tail_std": delta / scale if scale > 0 else np.nan})
    return pd.DataFrame(rows, columns=["segment", "sensor", "right_censored",
                                      "first_bins", "last_bins", "tail_bins", "delta",
                                      "tail_std_1s_means", "delta_over_tail_std"])


def length_geometry(segments):
    """Только длины заданий, не доказательство устойчивости или выбор частоты."""
    durations = segments.duration_s.to_numpy()
    rows = []
    for rate in (100, 10, 1):
        # Первое/последнее задание цензурированы границами записи.
        interior = durations[1:-1]
        old, new = durations[:-1], durations[1:]
        strict = np.ones(len(old), dtype=bool)
        if len(strict):
            strict[0] = strict[-1] = False
        rows.append({"candidate_hz": rate, "L512_seconds": 512 / rate,
                     "H64_seconds": 64 / rate,
                     "interior_plateaus_L_plus_H": int((interior >= 576 / rate).sum()),
                     "events_both_sides_L": int(((old >= 512 / rate)
                                                  & (new >= 512 / rate) & strict).sum()),
                     "events_old640_new705": int(((old >= 640 / rate)
                                                   & (new >= 705 / rate) & strict).sum())})
    return pd.DataFrame(rows)


def audit_file(path: Path, chunksize=200_000):
    """Проверить каждую строку, вернуть summary, tables (DataFrame).

    Память: один числовой чанк, вектор всех времён и секундные суммы.
    Структурно некорректные строки/нечисловые значения вызывают ошибку;
    нечисловое NaN распознаётся и учитывается. Не конечные время/задания
    исключают корректную разметку и вызывают явную ошибку, не пропуск строки.
    """
    if chunksize < 1:
        raise ValueError("chunksize must be positive")
    path = Path(path)
    physical_rows = validate_row_width(path)
    with path.open(encoding="utf-8") as source:
        header = source.readline().strip()
    count = np.zeros(19, dtype=np.int64)
    nonfinite = count.copy()
    total = np.zeros(19)
    square = np.zeros(19)
    minimum, maximum = np.full(19, np.inf), np.full(19, -np.inf)
    at_min, at_max = count.copy(), count.copy()
    zeros, negatives = count.copy(), count.copy()
    equal_pairs, longest, trailing = (np.zeros(16, dtype=np.int64) for _ in range(3))
    times, anomalies, second_parts, boundaries = [], [], [], []
    delta_hist = {}
    unique_gas, unique_ethylene = set(), set()
    n, previous, tick_error = 0, None, 0.0
    reader = None
    try:
        reader = pd.read_csv(path, sep=r"\s+", skiprows=1, header=None,
                             dtype=float, chunksize=chunksize)
        for frame in reader:
            if frame.shape[1] != 19:
                raise ValueError(f"{path.name}: expected 19 columns, got {frame.shape[1]}")
            x = frame.to_numpy()
            finite = np.isfinite(x)
            if not finite[:, :3].all():
                bad = np.argwhere(~finite[:, :3])[0]
                raise ValueError(f"{path.name}: nonfinite time/setpoint at data row {n + bad[0]}, column {bad[1]}")
            t = x[:, 0]
            times.append(t.copy())
            ticks = np.rint(t * 100).astype(np.int64)
            tick_error = max(tick_error, float(np.max(np.abs(t * 100 - ticks))))
            extended = x if previous is None else np.vstack([previous, x])
            dt = np.diff(extended[:, 0])
            dt_ticks = np.diff(np.rint(extended[:, 0] * 100).astype(np.int64))
            values, frequencies = np.unique(dt_ticks, return_counts=True)
            for value, frequency in zip(values, frequencies):
                delta_hist[int(value)] = delta_hist.get(int(value), 0) + int(frequency)
            offset = n + (1 if previous is None else 0)
            bad = np.flatnonzero(np.abs(dt - .01) > 1e-7)
            if len(bad):
                anomalies.append(pd.DataFrame({"row": offset + bad,
                    "previous_s": extended[bad, 0], "time_s": extended[bad + 1, 0],
                    "delta_s": dt[bad], "delta_ticks": dt_ticks[bad]}))
            changes = np.flatnonzero(np.any(extended[1:, 1:3] != extended[:-1, 1:3], axis=1)) + 1
            if previous is None:
                boundaries.append({"start_row": 0, "start_s": float(t[0]),
                                   "gas_ppm": x[0, 1], "ethylene_ppm": x[0, 2],
                                   "previous_s": np.nan})
            for k in changes:
                boundaries.append({"start_row": offset + k - 1, "start_s": extended[k, 0],
                    "gas_ppm": extended[k, 1], "ethylene_ppm": extended[k, 2],
                    "previous_s": extended[k - 1, 0]})
            unique_gas.update(x[:, 1])
            unique_ethylene.update(x[:, 2])
            clean = np.where(finite, x, 0)
            count += finite.sum(axis=0)
            nonfinite += (~finite).sum(axis=0)
            total += clean.sum(axis=0)
            square += (clean * clean).sum(axis=0)
            zeros += ((x == 0) & finite).sum(axis=0)
            negatives += ((x < 0) & finite).sum(axis=0)
            lo, hi = np.where(finite, x, np.inf).min(axis=0), np.where(finite, x, -np.inf).max(axis=0)
            at_min = np.where(lo < minimum, (x == lo).sum(axis=0),
                              at_min + np.where(lo == minimum, (x == lo).sum(axis=0), 0))
            at_max = np.where(hi > maximum, (x == hi).sum(axis=0),
                              at_max + np.where(hi == maximum, (x == hi).sum(axis=0), 0))
            minimum, maximum = np.minimum(minimum, lo), np.maximum(maximum, hi)
            for j in range(16):
                pairs, run, tail = run_summary(x[:, j + 3],
                    None if previous is None else previous[j + 3], int(trailing[j]))
                equal_pairs[j] += pairs
                longest[j] = max(longest[j], run)
                trailing[j] = tail
            frame.columns = COLUMNS
            frame["second"] = np.floor(t).astype(np.int64)
            # pandas не считает inf пропуском; здесь они явно превращены в NaN.
            frame[SENSORS] = frame[SENSORS].where(np.isfinite(frame[SENSORS]))
            grouped = frame.groupby("second", sort=True)
            part = grouped[SENSORS].agg(["sum", "count"])
            part.columns = [f"{a}_{b}" for a, b in part.columns]
            for column in ("gas_ppm", "ethylene_ppm"):
                part[f"{column}_min"] = grouped[column].min()
                part[f"{column}_max"] = grouped[column].max()
            part["rows"] = grouped.size()
            second_parts.append(part)
            previous = x[-1].copy()
            n += len(x)
    except pd.errors.EmptyDataError as error:
        raise ValueError(f"{path.name}: no numeric rows") from error
    finally:
        if reader is not None:
            reader.close()
    if n != physical_rows:
        raise ValueError(f"{path.name}: parsed {n} of {physical_rows} physical data rows")
    all_times = np.concatenate(times)
    per_column = pd.DataFrame({"column": COLUMNS, "finite": count,
        "nonfinite": nonfinite, "min": np.where(count, minimum, np.nan),
        "max": np.where(count, maximum, np.nan), "zeros": zeros, "negative": negatives,
        "at_min": at_min, "at_max": at_max})
    with np.errstate(invalid="ignore", divide="ignore"):
        per_column["mean"] = total / count
        per_column["std"] = np.sqrt(np.maximum(0, square / count - (total / count) ** 2))
    per_column["sensor_type"] = [None] * 3 + TYPES
    per_column["equal_adjacent_pairs"] = [None] * 3 + equal_pairs.tolist()
    per_column["longest_equal_run_rows"] = [None] * 3 + longest.tolist()
    segments = pd.DataFrame(boundaries)
    segments.insert(0, "segment", np.arange(len(segments)))
    segments["end_row_exclusive"] = np.r_[segments.start_row.to_numpy()[1:], n]
    segments["last_s"] = np.r_[segments.previous_s.to_numpy()[1:], all_times[-1]]
    # Последнее время границы приближено номинальным интервалом, отмечено цензурой.
    segments["boundary_s"] = np.r_[segments.start_s.to_numpy()[1:], all_times[-1] + .01]
    segments["rows"] = segments.end_row_exclusive - segments.start_row
    segments["duration_s"] = segments.boundary_s - segments.start_s
    segments["left_censored"] = segments.segment == 0
    segments["right_censored"] = segments.segment == len(segments) - 1
    segments = segments.drop(columns="previous_s")
    event_rows = []
    for i in range(1, len(segments)):
        old, new = segments.iloc[i - 1], segments.iloc[i]
        event_rows.append({"event": i - 1, "row": int(new.start_row), "time_s": new.start_s,
            "old_gas_ppm": old.gas_ppm, "new_gas_ppm": new.gas_ppm,
            "old_ethylene_ppm": old.ethylene_ppm, "new_ethylene_ppm": new.ethylene_ppm,
            "gas_changed": bool(old.gas_ppm != new.gas_ppm),
            "ethylene_changed": bool(old.ethylene_ppm != new.ethylene_ppm),
            "pre_duration_s": old.duration_s, "post_duration_s": new.duration_s,
            "pre_rows": int(old.rows), "post_rows": int(new.rows),
            "edge_censored": bool(old.left_censored or new.right_censored)})
    events = pd.DataFrame(event_rows)
    combined = pd.concat(second_parts)
    rules = {c: "min" if c.endswith("_min") else "max" if c.endswith("_max") else "sum"
             for c in combined.columns}
    seconds = combined.groupby(level=0).agg(rules)
    for sensor in SENSORS:
        seconds[sensor] = seconds[f"{sensor}_sum"] / seconds[f"{sensor}_count"]
    seconds["mixed_setpoint"] = ((seconds.gas_ppm_min != seconds.gas_ppm_max)
                                  | (seconds.ethylene_ppm_min != seconds.ethylene_ppm_max))
    seconds = seconds.reset_index()
    anomaly_columns = ["row", "previous_s", "time_s", "delta_s", "delta_ticks"]
    time_anomalies = (pd.concat(anomalies, ignore_index=True) if anomalies
                      else pd.DataFrame(columns=anomaly_columns))
    summary = {"file": path.name, "header": header, "rows": n, "columns": 19,
        "first_time_s": float(all_times[0]), "last_time_s": float(all_times[-1]),
        "duration_s": float(all_times[-1] - all_times[0]),
        "nonfinite_values": int(nonfinite.sum()),
        "time_tick_rounding_error_max": tick_error,
        "delta_centisecond_histogram": {str(k): v for k, v in sorted(delta_hist.items())},
        "adjacent_duplicate_times": delta_hist.get(0, 0),
        "backward_steps": sum(v for k, v in delta_hist.items() if k < 0),
        "gaps_over_nominal": sum(v for k, v in delta_hist.items() if k > 1),
        "duplicate_time_labels_beyond_first": n - len(np.unique(all_times)),
        "non_nominal_steps": len(time_anomalies),
        "max_delta_s": float(np.max(np.diff(all_times))) if n > 1 else None,
        "setpoint_events": len(events), "setpoint_segments": len(segments),
        "gas_setpoints": sorted(float(v) for v in unique_gas),
        "ethylene_setpoints": sorted(float(v) for v in unique_ethylene),
        "simultaneous_events": int((events.gas_changed & events.ethylene_changed).sum()) if len(events) else 0,
        "mixed_seconds": int(seconds.mixed_setpoint.sum()),
        "seconds_bins": len(seconds)}
    tables = {"columns": per_column, "segments": segments, "events": events,
        "time_anomalies": time_anomalies, "seconds": seconds,
        "length_geometry": length_geometry(segments),
        "late_tail": late_tail_diagnostics(seconds, segments),
        "correlation_1s": seconds[SENSORS].corr().rename_axis("sensor").reset_index()}
    return summary, tables