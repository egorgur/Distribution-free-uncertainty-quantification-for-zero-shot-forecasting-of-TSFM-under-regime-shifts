"""Дневные SCADA PRONTO: raw-время сохраняется, nominal-сетка лишь проверяется."""

import csv
import io
import math
import re
from collections import Counter
from datetime import datetime
from decimal import Decimal, InvalidOperation

EPOCH = datetime(1899, 12, 30)
DAY = Decimal(86400)


def seconds(dt):
    return Decimal(str((dt - EPOCH).total_seconds()))


def header_time(raw):
    """Явно распознаём избыточное PM после 24-часового времени в оригинале."""
    clean = " ".join(raw.split())
    suffix = clean[-3:] if clean.endswith((" AM", " PM")) else ""
    if suffix:
        clean = clean[:-3]
    for fmt in ("%m/%d/%Y %H:%M", "%m/%d/%Y %H:%M:%S"):
        try:
            dt = datetime.strptime(clean, fmt)
            break
        except ValueError:
            continue
    else:
        raise ValueError(f"Unknown header time: {raw}")
    if suffix and ((suffix == " PM") != (dt.hour >= 12)):
        raise ValueError("Ambiguous AM/PM header")
    return dt, "explicit_24h_with_redundant_suffix" if suffix else "24h"


def timestamp(raw):
    """Значение, тип и неопределённость: минуты — нижняя граница, serial — ±0.432с."""
    try:
        dt = datetime.strptime(raw, "%m/%d/%Y %H:%M")
        return seconds(dt), "text_minute"
    except ValueError:
        pass
    try:
        value = Decimal(raw)
        if value.is_finite():
            return value * DAY, "excel_serial"
    except InvalidOperation:
        pass
    return None, "unparsed"


def parse_scada(content):
    rows = list(csv.reader(io.StringIO(content.decode("utf-8-sig"), newline="")))
    if len(rows) < 4 or len(set(map(len, rows))) != 1:
        raise ValueError("Unequal logical row widths or no data")
    meta, tags, labels = rows[:3]
    if meta[0] != "Start Time" or meta[2] != "End Time" or labels[0] != "TIMESTAMP":
        raise ValueError("Unknown CSV headers")
    if meta[meta.index("Interval") + 1] != "1s":
        raise ValueError("Not nominal 1s")
    start, start_format = header_time(meta[1])
    end, end_format = header_time(meta[3])
    values, origin = rows[3:], seconds(start)
    parsed = [timestamp(row[0]) for row in values]
    errors, consistent, kinds = [], [], []
    for i, (value, kind) in enumerate(parsed):
        error = value - origin - i if value is not None else None
        # text_minute: исходник содержит только минуту, не секунду.
        ok = (Decimal(-60) < error <= 0 if kind == "text_minute" else
              abs(error) <= Decimal("0.432001")) if error is not None else False
        errors.append(error)
        consistent.append(ok)
        kinds.append(kind)
    runs = []
    first = 0
    for stop in range(1, len(values) + 1):
        if stop == len(values) or kinds[stop] != kinds[first]:
            runs.append({"start_index": first, "stop_exclusive": stop,
                         "encoding": kinds[first], "rows": stop - first,
                         "nominal_consistent": all(consistent[first:stop]),
                         "first_raw": values[first][0], "last_raw": values[stop - 1][0]})
            first = stop
    differences = [b[0] - a[0] for a, b in zip(parsed, parsed[1:])
                   if a[0] is not None and b[0] is not None]
    channels, columns = [], {}
    for j in range(1, len(tags)):
        strings = tuple(row[j] for row in values)
        columns[j] = strings
        finite, nonnumeric, nonfinite = [], 0, 0
        for s in strings:
            if not s.strip():
                continue
            try:
                number = float(s)
            except ValueError:
                nonnumeric += 1
                continue
            if math.isfinite(number):
                finite.append(number)
            else:
                nonfinite += 1
        mean = math.fsum(finite) / len(finite) if finite else None
        std = math.sqrt(math.fsum((x - mean) ** 2 for x in finite) / len(finite)) if finite else None
        channels.append({"column_index": j, "tag": tags[j], "label": labels[j],
                         "finite": len(finite), "blank": sum(not s.strip() for s in strings),
                         "nonnumeric": nonnumeric, "nonfinite": nonfinite,
                         "unique_raw": len(set(strings)), "mean": mean, "std": std,
                         "min": min(finite) if finite else None,
                         "max": max(finite) if finite else None})
    duplicates = [{"left_index": i, "right_index": j, "left_tag": tags[i], "right_tag": tags[j]}
                  for i in columns for j in columns if i < j and tags[i] and tags[j]
                  and columns[i] == columns[j]]
    summary = {"logical_data_rows": len(values), "columns_including_time": len(tags),
               "header_start_raw": meta[1], "header_end_raw": meta[3],
               "header_start": start.isoformat(), "header_end": end.isoformat(),
               "header_start_format": start_format, "header_end_format": end_format,
               "timestamp_encodings": dict(Counter(kinds)),
               "unparsed_timestamps": kinds.count("unparsed"),
               "duplicate_raw_timestamps": len(values) - len({v[0] for v in values}),
               "nonpositive_parsed_steps": sum(d <= 0 for d in differences),
               "backwards_parsed_steps": sum(d < 0 for d in differences),
               "parsed_time_step_seconds": dict(Counter(str(d.normalize()) for d in differences)),
               "nominal_consistent_rows": sum(consistent),
               "nominal_grid_consistent": all(consistent),
               "nominal_endpoint_matches_header": len(values) - 1 == int((end - start).total_seconds()),
               "first_inconsistent_index": next((i for i, ok in enumerate(consistent) if not ok), None),
               "finite_numeric_columns": sum(c["finite"] == len(values) for c in channels),
               "varying_finite_columns": sum(c["finite"] == len(values) and c["std"] > 0 for c in channels),
               "duplicate_column_pairs": duplicates,
               "model_inference": False, "stationarity_established": False,
               "raw_time_modified": False}
    return summary, channels, runs


def journal_rows(cells):
    """Полные строки с координатами: никогда не заполняем отсутствующее время."""
    grouped = {}
    for c in cells:
        col, row = re.fullmatch(r"([A-Z]+)(\d+)", c["cell"]).groups()
        grouped.setdefault((c["sheet"], int(row)), {})[col] = c["display"]
    rows, timeline = [], []
    for (sheet, row), values in grouped.items():
        rows.append({"sheet": sheet, "excel_row": row,
                     "cells_json": __import__("json").dumps(values, ensure_ascii=False, sort_keys=True)})
        if sheet.endswith("0907") or sheet.endswith("0911"):
            action, clock = values.get("J", ""), values.get("K", "")
            payload = {k: values.get(k, "") for k in ("J", "K", "L", "M")}
            time_cell = f"K{row}"
        elif sheet.endswith("0912 "):
            action, clock = values.get("M", ""), values.get("L", "")
            payload = {k: values.get(k, "") for k in ("L", "M", "N", "O")}
            time_cell = f"L{row}"
        else:
            action, clock = values.get("B", ""), values.get("A", "")
            payload = values
            time_cell = f"A{row}"
        # Включаем действия без времени и противоречивые заметки; заголовки
        # остаются в полном rows, но не превращаются в события.
        if action and (re.fullmatch(r"\d\d:\d\d:\d\d", clock) or
                       any(word in action.lower() for word in ("valve", "flow", "start", "stabled"))):
            timeline.append({"sheet": sheet, "excel_row": row, "kind": "action_or_note",
                             "time_cell": time_cell, "clock": clock, "description": action,
                             "payload_json": __import__("json").dumps(payload, ensure_ascii=False)})
        if "start labview" in values.get("D", "").lower():
            timeline.append({"sheet": sheet, "excel_row": row, "kind": "measurement_start",
                             "time_cell": f"C{row}", "clock": values.get("C", ""),
                             "description": values.get("B", ""),
                             "payload_json": __import__("json").dumps(values, ensure_ascii=False)})
    return rows, timeline