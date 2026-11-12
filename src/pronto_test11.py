"""Структурный аудит Test11; nominal grid проверяется, но не подменяет raw time."""

import csv
import io
import math
from collections import Counter
from datetime import datetime
from decimal import Decimal


def parse_test11(content):
    rows = list(csv.reader(io.StringIO(content.decode("utf-8-sig"), newline="")))
    if len(rows) < 4 or len(set(map(len, rows))) != 1:
        raise ValueError("CSV schema or logical row width changed")
    meta, tags, labels = rows[:3]
    if meta[0] != "Start Time" or meta[2] != "End Time" or labels[0] != "TIMESTAMP":
        raise ValueError("Unknown CSV headers")
    header_start = datetime.strptime(meta[1], "%m/%d/%Y %H:%M")
    header_end = datetime.strptime(meta[3], "%m/%d/%Y %H:%M")
    interval_column = meta.index("Interval")
    if meta[interval_column + 1] != "1s":
        raise ValueError("Not documented as nominal 1s")
    values = rows[3:]
    times = [Decimal(row[0]) for row in values]
    if not all(t.is_finite() for t in times):
        raise ValueError("Nonfinite time")
    origin = Decimal(str((header_start - datetime(1899, 12, 30)).days))
    origin += Decimal(header_start.hour * 3600 + header_start.minute * 60) / Decimal(86400)
    # Excel serial экспортирован до 5 знаков (0.864с шаг представления).
    errors = [(t - origin) * Decimal(86400) - i for i, t in enumerate(times)]
    rounding_consistent = all(abs(error) <= Decimal("0.432001") for error in errors)
    diffs = [(b - a) * Decimal(86400) for a, b in zip(times, times[1:])]
    channels = []
    for j in range(1, len(tags)):
        strings = [row[j] for row in values]
        numeric, nonfinite, nonnumeric = [], 0, 0
        for value in strings:
            if not value.strip():
                continue
            try:
                number = float(value)
            except ValueError:
                nonnumeric += 1
                continue
            if math.isfinite(number):
                numeric.append(number)
            else:
                nonfinite += 1
        mean = math.fsum(numeric) / len(numeric) if numeric else None
        std = math.sqrt(math.fsum((x - mean) ** 2 for x in numeric) / len(numeric)) if numeric else None
        channels.append({"column_index": j, "tag": tags[j], "label": labels[j],
                         "finite": len(numeric), "blank": sum(not s.strip() for s in strings),
                         "nonnumeric": nonnumeric, "nonfinite": nonfinite,
                         "unique_raw": len(set(strings)), "mean": mean, "std": std,
                         "min": min(numeric) if numeric else None,
                         "max": max(numeric) if numeric else None})
    duplicates = [{"left_index": i, "right_index": j, "tag": tags[i]}
                  for i in range(1, len(tags)) for j in range(i + 1, len(tags))
                  if tags[i] and tags[j] and all(row[i] == row[j] for row in values)]
    summary = {"logical_data_rows": len(values), "columns_including_time": len(tags),
               "header_start": header_start.isoformat(), "header_end": header_end.isoformat(),
               "raw_timestamp_unique": len(set(times)),
               "duplicate_timestamps": len(times) - len(set(times)),
               "nonpositive_time_steps": sum(d <= 0 for d in diffs),
               "exact_one_second_steps": sum(d == 1 for d in diffs),
               "raw_time_step_seconds": dict(Counter(str(d.normalize()) for d in diffs)),
               "nominal_grid_consistent_with_5_decimal_rounding": rounding_consistent,
               "nominal_endpoint_matches_header": len(values) - 1 == (header_end - header_start).total_seconds(),
               "max_abs_nominal_time_error_seconds": float(max(map(abs, errors))),
               "finite_numeric_columns": sum(c["finite"] == len(values) for c in channels),
               "varying_finite_columns": sum(c["finite"] == len(values) and c["unique_raw"] > 1 for c in channels),
               "duplicate_column_pairs": duplicates, "model_inference": False,
               "stationarity_established": False}
    return summary, channels


def window_geometry(n, start, stop, length=512, horizon=64, stride=8):
    """Индексы на кандидатной nominal сетке; [start, stop), конец исключён."""
    origins = list(range(length - 1, n - horizon, stride))
    accepted = [t for t in origins if t - length + 1 >= start and t + horizon < stop]
    # Независимый полный перебор не перекрывающихся target-блоков — лишь геометрия.
    return {"start_index": start, "stop_exclusive": stop, "samples": max(0, stop - start),
            "length": length, "horizon": horizon, "stride": stride,
            "rolling_windows_global_phase": len(accepted),
            "first_origin": accepted[0] if accepted else None,
            "last_origin": accepted[-1] if accepted else None,
            "max_disjoint_target_blocks": max(0, (stop - start - length) // horizon),
            "independent_replicates": "not_established"}