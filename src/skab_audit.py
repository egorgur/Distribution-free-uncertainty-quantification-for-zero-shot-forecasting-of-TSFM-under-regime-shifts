"""Структурный аудит SKAB: метки и длины не доказывают устойчивость процесса.

Индексы строк с нуля; все отрезки полуоткрытые [start, stop).
Оригиналы не исправляются, файлы не склеиваются, сеть/модели не используются.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

COMMIT = "b2c0d46c2971dcbfe71e26087b6d231998bb91c2"
CONFIGS = ((512, 64), (128, 16))
STRIDE = 8
SENSORS = (
    "Accelerometer1RMS", "Accelerometer2RMS", "Current", "Pressure",
    "Temperature", "Thermocouple", "Voltage", "Volume Flow RateRMS",
)
SECOND_NS = 1_000_000_000


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_originals(root: Path, *, pinned: bool = True) -> pd.DataFrame:
    """Сверить каждый оригинал, включая README/LICENSE, с манифестом.

    Git blob SHA1 = SHA1(b'blob ' + byte_length + b'\\0' + original_bytes).
    Это сверка с зафиксированным манифестом, не новое обращение к GitHub.
    """
    manifest = json.loads((root / "manifest.json").read_text())
    if pinned and manifest["commit"] != COMMIT:
        raise ValueError("Unexpected SKAB commit")
    expected_csv = {p for p in manifest["files"] if p.endswith(".csv")}
    actual_csv = {p.relative_to(root).as_posix() for p in (root / "data").rglob("*.csv")}
    if actual_csv != expected_csv or (pinned and len(actual_csv) != 35):
        raise ValueError("CSV inventory differs from manifest / expected 35 files")
    rows = []
    for name, record in sorted(manifest["files"].items()):
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Unsafe manifest path")
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        ok = len(raw) == record["bytes"] and digest == record["sha256"]
        ok = ok and blob == record["git_blob_sha1"]
        if not ok:
            raise ValueError(f"Original integrity mismatch: {name}")
        rows.append(dict(file=name, bytes=len(raw), sha256=digest, git_blob_sha1=blob,
                         verified=True))
    return pd.DataFrame(rows)


def load_csv(path: Path) -> pd.DataFrame:
    """Строгая схема и число полей каждой физической CSV-строки."""
    with path.open(newline="") as stream:
        reader = csv.reader(stream, delimiter=";")
        header = next(reader, [])
        allowed = [["datetime", *SENSORS], ["datetime", *SENSORS, "anomaly", "changepoint"]]
        if header not in allowed:
            raise ValueError(f"Unexpected schema: {path}: {header}")
        count = 0
        for count, row in enumerate(reader, 1):
            if len(row) != len(header):
                raise ValueError(f"Wrong field count at data row {count - 1}: {path}")
        if not count:
            raise ValueError(f"Empty CSV: {path}")
    frame = pd.read_csv(path, sep=";")
    if len(frame) != count:
        raise ValueError("Parsed / physical row counts differ")
    frame["datetime"] = pd.to_datetime(frame.datetime, format="%Y-%m-%d %H:%M:%S", errors="raise")
    if frame.datetime.isna().any():
        raise ValueError("Missing timestamp")
    for sensor in SENSORS:
        frame[sensor] = pd.to_numeric(frame[sensor], errors="raise")
    for label in ("anomaly", "changepoint"):
        if label in frame:
            if not frame[label].isin([0, 1]).all():
                raise ValueError(f"Nonbinary/missing label: {label}")
            frame[label] = frame[label].astype(int)
    return frame


def time_ns(frame: pd.DataFrame) -> np.ndarray:
    return frame.datetime.to_numpy(dtype="datetime64[ns]").astype(np.int64)


def cp_rows(frame: pd.DataFrame) -> np.ndarray:
    return np.flatnonzero(frame.changepoint.to_numpy() == 1) if "changepoint" in frame else np.array([], dtype=int)


def bounds_from_cuts(n: int, cuts: np.ndarray) -> list[tuple[int, int]]:
    edges = np.unique(np.r_[0, cuts, n]).astype(int)
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]


def continuous(times: np.ndarray, start: int, stop: int) -> bool:
    return bool(np.all(np.diff(times[start:stop]) == SECOND_NS))


def longest_equal_run(values: np.ndarray) -> int:
    longest = current = 0
    previous = np.nan
    for value in values:
        current = current + 1 if np.isfinite(value) and value == previous else int(np.isfinite(value))
        longest = max(longest, current)
        previous = value
    return longest


def rolling_windows(frame: pd.DataFrame, length: int, horizon: int, stride: int = STRIDE) -> pd.DataFrame:
    """Не перезапускать сетку t после CP/разрыва/начала сегмента.

    Контекст [t-L+1,t+1), цель [t+1,t+H+1). CP в первой точке окна
    допустим для same_cp_segment; CP внутри объединения — нет.
    В strict_1s проверяются ВСЕ L+H-1 интервалов, включая стык цели.
    """
    if min(length, horizon, stride) < 1:
        raise ValueError("L, H, stride must be positive")
    n = len(frame)
    ends = np.arange(length - 1, n - horizon, stride, dtype=int)
    starts, stops = ends - length + 1, ends + horizon + 1
    times = time_ns(frame)
    bad_dt = np.r_[0, np.cumsum(np.diff(times) != SECOND_NS)]
    result = pd.DataFrame(dict(L=length, H=horizon, t=ends, start=starts, stop=stops))
    result["strict_1s"] = (bad_dt[stops - 1] - bad_dt[starts]) == 0
    if "changepoint" in frame:
        segment = np.cumsum(frame.changepoint.to_numpy() == 1)
        result["same_cp_segment"] = segment[starts] == segment[stops - 1]
        result["cp_segment"] = segment[starts]
    else:
        result["same_cp_segment"] = pd.array([pd.NA] * len(ends), dtype="boolean")
        result["cp_segment"] = pd.array([pd.NA] * len(ends), dtype="Int64")
    if "anomaly" in frame:
        abnormal = np.r_[0, np.cumsum(frame.anomaly.to_numpy() != 0)]
        result["all_anomaly0"] = (abnormal[stops] - abnormal[starts]) == 0
    else:
        # Отсутствующая разметка normal-файла не превращается в наблюдаемые нули.
        result["all_anomaly0"] = pd.array([pd.NA] * len(ends), dtype="boolean")
    return result


def window_counts(windows: pd.DataFrame) -> dict:
    strict = windows.strict_1s
    same = windows.same_cp_segment.fillna(False).astype(bool)
    zero = windows.all_anomaly0.fillna(False).astype(bool)
    return dict(row_order=len(windows), strict_1s=int(strict.sum()),
                same_cp=int(same.sum()), same_cp_strict=int((same & strict).sum()),
                anomaly0=int(zero.sum()), anomaly0_strict=int((zero & strict).sum()),
                same_cp_anomaly0=int((same & zero).sum()),
                same_cp_anomaly0_strict=int((same & zero & strict).sum()))


def event_geometry(frame: pd.DataFrame, length: int, horizon: int) -> pd.DataFrame:
    """CP — первая точка правого сегмента; соседний CP не включается.

    before = [prev_cp, cp), after = [cp, next_cp). Края файла цензурированы.
    L_before_H_after — геометрия вокруг CP, не наличие устойчивых режимов.
    """
    times, marks = time_ns(frame), cp_rows(frame)
    rows = []
    for i, mark in enumerate(marks):
        mark = int(mark)
        previous = int(marks[i - 1]) if i else 0
        following = int(marks[i + 1]) if i + 1 < len(marks) else len(frame)
        pre, post = mark - previous, following - mark
        row = dict(L=length, H=horizon, cp_index=i, row=mark, previous_boundary=previous,
                   next_boundary=following, before_rows=pre, after_rows=post,
                   left_file_censored=i == 0, right_file_censored=i + 1 == len(marks),
                   L_before=pre >= length, H_after=post >= horizon,
                   LH_after=post >= length + horizon,
                   L_before_H_after=pre >= length and post >= horizon,
                   L_before_LH_after=pre >= length and post >= length + horizon)
        row["L_before_H_after_strict"] = bool(row["L_before_H_after"] and continuous(
            times, mark - length, mark + horizon))
        row["L_before_LH_after_strict"] = bool(row["L_before_LH_after"] and continuous(
            times, mark - length, mark + length + horizon))
        rows.append(row)
    return pd.DataFrame(rows)


def tail_diagnostics(frame: pd.DataFrame, segments: list[tuple[int, int]]) -> pd.DataFrame:
    """Последние 128 строк CP-сегмента: |mean(last64)-mean(first64)|/std128.

    Только описание (ddof=0), без порога/теста стационарности и без отбора.
    При нулевой дисперсии отношение не определено, а не равно нулю.
    """
    rows = []
    times = time_ns(frame)
    for segment, (start, stop) in enumerate(segments):
        if stop - start < 128:
            continue
        for sensor in SENSORS:
            values = frame[sensor].to_numpy(dtype=float)[stop - 128:stop]
            finite = bool(np.isfinite(values).all())
            std = float(np.std(values)) if finite else np.nan
            delta = float(np.mean(values[64:]) - np.mean(values[:64])) if finite else np.nan
            rows.append(dict(segment=segment, start=stop - 128, stop=stop, sensor=sensor,
                strict_1s=continuous(times, stop - 128, stop), std128=std, delta=delta,
                abs_mean_difference_over_std=abs(delta) / std if std > 0 else np.nan,
                constant_tail=finite and std == 0, finite=finite))
    return pd.DataFrame(rows)


def audit_frame(frame: pd.DataFrame, name: str) -> tuple[dict, dict[str, pd.DataFrame]]:
    n, times = len(frame), time_ns(frame)
    dt = np.diff(times)
    marks = cp_rows(frame)
    has_labels = "anomaly" in frame
    anomaly = frame.anomaly.to_numpy() if has_labels else None
    transitions = np.flatnonzero(np.diff(anomaly) != 0) + 1 if has_labels else np.array([], dtype=int)
    onsets = transitions[anomaly[transitions] == 1] if has_labels else np.array([], dtype=int)
    offsets = transitions[anomaly[transitions] == 0] if has_labels else np.array([], dtype=int)
    anomaly_positions = np.flatnonzero(anomaly == 1) if has_labels else np.array([], dtype=int)
    first = int(anomaly_positions[0]) if len(anomaly_positions) else None
    pre_start = 0
    if first is not None:
        bad = np.flatnonzero(dt[:first] != SECOND_NS)
        pre_start = int(bad[-1] + 1) if len(bad) else 0
    columns = []
    for sensor in SENSORS:
        values = frame[sensor].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        columns.append(dict(sensor=sensor, rows=n, finite=len(finite), nan=int(np.isnan(values).sum()),
            posinf=int(np.isposinf(values).sum()), neginf=int(np.isneginf(values).sum()),
            min=float(finite.min()) if len(finite) else np.nan,
            max=float(finite.max()) if len(finite) else np.nan,
            mean=float(finite.mean()) if len(finite) else np.nan,
            std=float(finite.std()) if len(finite) else np.nan,
            unique_finite=len(np.unique(finite)), constant=len(finite) == n and len(np.unique(finite)) == 1,
            longest_equal_run_rows=longest_equal_run(values)))
    gaps = pd.DataFrame([dict(row=int(i + 1), previous_time=str(frame.datetime.iloc[i]),
        time=str(frame.datetime.iloc[i + 1]), dt_s=float(dt[i] / SECOND_NS))
        for i in np.flatnonzero(dt != SECOND_NS)],
        columns=["row", "previous_time", "time", "dt_s"])
    hist_values, hist_counts = np.unique(dt, return_counts=True)
    dt_hist = pd.DataFrame(dict(dt_s=hist_values / SECOND_NS, count=hist_counts))
    uninterrupted = bounds_from_cuts(n, np.flatnonzero(dt != SECOND_NS) + 1)
    continuous_runs = pd.DataFrame([dict(run=i, start=a, stop=b, rows=b - a,
        span_s=float((times[b - 1] - times[a]) / SECOND_NS))
        for i, (a, b) in enumerate(uninterrupted)])
    segment_bounds = bounds_from_cuts(n, marks)
    segments = []
    for i, (start, stop) in enumerate(segment_bounds):
        segments.append(dict(segment=i, start=start, stop=stop, rows=stop - start,
            span_s=float((times[stop - 1] - times[start]) / SECOND_NS),
            strict_1s=continuous(times, start, stop), labels_observed=has_labels,
            anomaly0_rows=int((anomaly[start:stop] == 0).sum()) if has_labels else None,
            anomaly1_rows=int((anomaly[start:stop] == 1).sum()) if has_labels else None,
            left_file_censored=start == 0, right_file_censored=stop == n))
    runs = []
    if has_labels:
        for i, (start, stop) in enumerate(bounds_from_cuts(n, transitions)):
            runs.append(dict(run=i, label=int(anomaly[start]), start=start, stop=stop, rows=stop - start,
                span_s=float((times[stop - 1] - times[start]) / SECOND_NS),
                strict_1s=continuous(times, start, stop), cp_marks=int(((marks >= start) & (marks < stop)).sum()),
                left_file_censored=start == 0, right_file_censored=stop == n))
    transition_rows = []
    for position in transitions:
        nearest = int(marks[np.argmin(abs(marks - position))]) if len(marks) else None
        transition_rows.append(dict(row=int(position), kind="onset" if anomaly[position] else "offset",
            cp_at_transition=bool(position in marks), nearest_cp=nearest,
            nearest_cp_minus_transition_rows=nearest - int(position) if nearest is not None else None,
            nearest_cp_minus_transition_s=float((times[nearest] - times[position]) / SECOND_NS) if nearest is not None else None))
    cp_table = []
    for i, mark in enumerate(marks):
        nearest = int(transitions[np.argmin(abs(transitions - mark))]) if len(transitions) else None
        cp_table.append(dict(cp_index=i, row=int(mark), time=str(frame.datetime.iloc[mark]),
            anomaly_at_cp=int(anomaly[mark]), at_onset=bool(mark in onsets), at_offset=bool(mark in offsets),
            cp_0_to_1=bool(mark == 0 or frame.changepoint.iloc[mark - 1] == 0),
            nearest_anomaly_transition=nearest,
            nearest_transition_minus_cp_rows=nearest - int(mark) if nearest is not None else None,
            nearest_transition_minus_cp_s=float((times[nearest] - times[mark]) / SECOND_NS) if nearest is not None else None))
    windows, counts, geometry = [], [], []
    for length, horizon in CONFIGS:
        win = rolling_windows(frame, length, horizon)
        windows.append(win)
        counts.append(dict(L=length, H=horizon, labels_observed=has_labels, **window_counts(win)))
        geometry.append(event_geometry(frame, length, horizon))
    summary = dict(file=name, group=Path(name).parent.name, rows=n, labels_observed=has_labels,
        start_time=str(frame.datetime.iloc[0]), end_time=str(frame.datetime.iloc[-1]),
        span_s=float((times[-1] - times[0]) / SECOND_NS),
        dt_min_s=float(dt.min() / SECOND_NS) if len(dt) else None,
        dt_median_s=float(np.median(dt) / SECOND_NS) if len(dt) else None,
        dt_max_s=float(dt.max() / SECOND_NS) if len(dt) else None,
        max_continuous_1s_rows=max(b - a for a, b in uninterrupted),
        non1s_steps=int((dt != SECOND_NS).sum()), gaps_over1s=int((dt > SECOND_NS).sum()),
        missing_seconds_if_integer_grid=int(np.maximum(dt // SECOND_NS - 1, 0).sum()),
        duplicate_steps=int((dt == 0).sum()), duplicate_times=n - len(np.unique(times)),
        backward_steps=int((dt < 0).sum()), cp_marks=len(marks),
        cp_0_to_1_runs=sum(bool(r["cp_0_to_1"]) for r in cp_table),
        anomaly_onsets=len(onsets), anomaly_offsets=len(offsets),
        anomaly0_rows=int((anomaly == 0).sum()) if has_labels else None,
        anomaly1_rows=int((anomaly == 1).sum()) if has_labels else None,
        anomaly0_runs=sum(r["label"] == 0 for r in runs), anomaly1_runs=sum(r["label"] == 1 for r in runs),
        first_anomaly_row=first, prehistory_rows=first,
        contiguous_prehistory_rows=first - pre_start if first is not None else None,
        first_anomaly_elapsed_s=float((times[first] - times[0]) / SECOND_NS) if first is not None else None,
        nonfinite_sensor_values=sum(r["nan"] + r["posinf"] + r["neginf"] for r in columns),
        constant_sensors=sum(r["constant"] for r in columns))
    tables = dict(columns=pd.DataFrame(columns), time_gaps=gaps, dt_histogram=dt_hist,
        continuous_runs=continuous_runs,
        cp_marks=pd.DataFrame(cp_table), anomaly_transitions=pd.DataFrame(transition_rows),
        anomaly_runs=pd.DataFrame(runs), cp_segments=pd.DataFrame(segments),
        windows=pd.concat(windows, ignore_index=True), window_counts=pd.DataFrame(counts),
        event_geometry=pd.concat(geometry, ignore_index=True),
        tail128=tail_diagnostics(frame, segment_bounds))
    for table in tables.values():
        table.insert(0, "file", name)
    return summary, tables


def brute_window_counts(frame: pd.DataFrame, length: int, horizon: int) -> dict:
    """Независимая простая сверка без префиксных сумм rolling_windows."""
    counts = dict.fromkeys(("row_order", "strict_1s", "same_cp", "same_cp_strict", "anomaly0",
                           "anomaly0_strict", "same_cp_anomaly0", "same_cp_anomaly0_strict"), 0)
    times = time_ns(frame)
    for t in range(length - 1, len(frame) - horizon, STRIDE):
        a, b = t - length + 1, t + horizon + 1
        strict = all(int(times[j + 1] - times[j]) == SECOND_NS for j in range(a, b - 1))
        same = "changepoint" in frame and not frame.changepoint.iloc[a + 1:b].any()
        zero = "anomaly" in frame and bool((frame.anomaly.iloc[a:b] == 0).all())
        for key, flag in dict(row_order=True, strict_1s=strict, same_cp=same,
            same_cp_strict=same and strict, anomaly0=zero, anomaly0_strict=zero and strict,
            same_cp_anomaly0=same and zero, same_cp_anomaly0_strict=same and zero and strict).items():
            counts[key] += int(flag)
    return counts


def cross_file_overlaps(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Совпавшее время и восемь датчиков между файлами — риск зависимости.

    Соединение только для аудита дубликатов, не склейка временного ряда.
    Совпавшие часы без совпавших датчиков сами по себе не доказывают дубликат.
    """
    rows = []
    names = sorted(frames)
    for i, left_name in enumerate(names):
        left = frames[left_name]
        for right_name in names[i + 1:]:
            right = frames[right_name]
            start = max(left.datetime.min(), right.datetime.min())
            end = min(left.datetime.max(), right.datetime.max())
            if start > end:
                continue
            joined = left.merge(right, on="datetime", suffixes=("_left", "_right"))
            matches = np.ones(len(joined), dtype=bool)
            for sensor in SENSORS:
                matches &= (joined[sensor + "_left"] == joined[sensor + "_right"]).to_numpy()
            comparable = "anomaly" in left and "anomaly" in right
            conflicts = int(((joined.anomaly_left != joined.anomaly_right).to_numpy() & matches).sum()) if comparable else None
            rows.append(dict(left_file=left_name, right_file=right_name,
                overlap_start=str(start), overlap_end=str(end), overlap_span_s=(end - start).total_seconds(),
                shared_timestamp_pairs=len(joined), identical_8sensor_timestamp_pairs=int(matches.sum()),
                anomaly_labels_comparable=comparable, anomaly_conflicts_on_identical_rows=conflicts))
    return pd.DataFrame(rows, columns=["left_file", "right_file", "overlap_start", "overlap_end", "overlap_span_s",
        "shared_timestamp_pairs", "identical_8sensor_timestamp_pairs", "anomaly_labels_comparable",
        "anomaly_conflicts_on_identical_rows"])