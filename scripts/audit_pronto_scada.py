"""Описательный аудит четырёх дневных PRONTO CSV; никаких TSFM/отбора эталонов."""

import argparse
import csv
import io
import json
from datetime import datetime, timezone

from scripts.audit_pronto_documents import ROOT, save, write_json
from scripts.download_pronto_scada import RAW, digest, verify
from src.pronto_scada import parse_scada, journal_rows
from src.pronto_test11 import window_geometry

# Первый снимок сохранён. После усиления проверки состава входов — новый v2,
# не переписываем хеши v1. Численные таблицы должны остаться теми же.
OUT = ROOT / "results/pronto_scada_v2"
LOG = ROOT / "results/pronto_documents_v2/log_cells.csv"
# Проверенные вручную пары координат, а НЕ автоматически найденные эталоны.
# Точность границ минутная; весь расчёт лишь на совместимой nominal-сетке.
CANDIDATES = (
    ("0907Testday2.csv", "Operation Log 0907", "C26", "K25", "measurement_to_next_action"),
    ("0907Testday2.csv", "Operation Log 0907", "C197", "K196", "measurement_to_next_action"),
    ("0911Testday3.csv", "Operation Log 0911", "C11", "K18", "measurement_to_next_action"),
    ("0912Testday4.csv", "Operation Log 0912 ", "C86", "L82", "measurement_to_next_action"),
    ("0626Testday5.csv", "Operation log 180626", "A4", "A6", "reaches_setpoint_not_explicitly_stabled"),
    ("0626Testday5.csv", "Operation log 180626", "A8", "A10", "explicit_system_stabled"),
)


def csv_bytes(rows):
    out = io.StringIO(newline="")
    if rows:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return out.getvalue().encode()


def products():
    verify()
    summaries, channels, runs = {}, [], []
    for path in sorted(RAW.glob("*.csv")):
        summary, cols, times = parse_scada(path.read_bytes())
        summaries[path.name] = summary
        channels.extend({"file": path.name, **c} for c in cols)
        runs.extend({"file": path.name, **t} for t in times)
    with LOG.open(newline="") as stream:
        cells = list(csv.DictReader(stream))
    rows, timeline = journal_rows(cells)
    lookup = {(c["sheet"], c["cell"]): c["display"] for c in cells}
    geometry = []
    for file, sheet, left, right, basis in CANDIDATES:
        summary = summaries[file]
        if not summary["nominal_grid_consistent"] or not summary["nominal_endpoint_matches_header"]:
            raise ValueError("Cannot calculate nominal geometry for inconsistent day")
        origin = datetime.fromisoformat(summary["header_start"])
        def offset(cell):
            clock = datetime.strptime(lookup[sheet, cell], "%H:%M:%S").time()
            return int((datetime.combine(origin.date(), clock) - origin).total_seconds())
        start, stop = offset(left), offset(right)
        if not 0 <= start < stop <= summary["logical_data_rows"]:
            raise ValueError("Journal interval outside nominal day")
        geometry.append({"file": file, "sheet": sheet, "start_cell": left, "stop_cell": right,
                         "basis": basis, **window_geometry(summary["logical_data_rows"], start, stop)})
    manifest = json.loads((RAW / "manifest.json").read_text())
    return {"days.json": (json.dumps(summaries, ensure_ascii=False, indent=2) + "\n").encode(),
            "columns.csv": csv_bytes(channels), "time_runs.csv": csv_bytes(runs),
            "journal_rows.csv": csv_bytes(rows), "timeline.csv": csv_bytes(timeline),
            "candidate_geometry.csv": csv_bytes(geometry),
            "archive_aliases.csv": csv_bytes(manifest["files"])}


def run():
    if (OUT / "meta.json").exists():
        return check()
    outputs = products()
    for name, content in outputs.items():
        save(OUT / name, content)
    inputs = [RAW / "manifest.json", LOG,
              ROOT / "results/pronto_documents_v2/archive_inventory.csv",
              ROOT / "docs/decisions.md", ROOT / "src/pronto_scada.py",
              ROOT / "src/pronto_test11.py", ROOT / "src/pronto_documents.py",
              ROOT / "scripts/download_pronto_scada.py", ROOT / "scripts/audit_pronto_documents.py",
              ROOT / "tests/test_pronto_scada.py", ROOT / "scripts/audit_pronto_scada.py"]
    # Журнал решений расширяется: сохраняем его снимок, а не запрещаем новые решения.
    snapshot = OUT / "decisions_snapshot.md"
    save(snapshot, (ROOT / "docs/decisions.md").read_bytes())
    inputs.remove(ROOT / "docs/decisions.md")
    inputs.append(snapshot)
    write_json(OUT / "meta.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
               "inputs": {str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in inputs},
               "outputs": {name: digest(data) for name, data in outputs.items()},
               "model_inference": False, "reference_selection": False})
    return check()


def check():
    meta = json.loads((OUT / "meta.json").read_text())
    for path, sha in meta["inputs"].items():
        if digest((ROOT / path).read_bytes()) != sha:
            raise ValueError(f"Input changed: {path}")
    replay = products()
    if set(replay) != set(meta["outputs"]):
        raise ValueError("Output set changed")
    for name, data in replay.items():
        if (OUT / name).read_bytes() != data or digest(data) != meta["outputs"][name]:
            raise ValueError(f"Recalculation differs: {name}")
    # Прямой перебор геометрии независимо от window_geometry.
    with (OUT / "candidate_geometry.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            a, b = int(row["start_index"]), int(row["stop_exclusive"])
            count = sum(t % 8 == 7 and t - 511 >= a and t + 64 < b for t in range(a, b))
            if count != int(row["rolling_windows_global_phase"]):
                raise ValueError("Independent geometry check failed")
    result = {**verify(), "tables_replayed": len(replay), "candidate_intervals": len(CANDIDATES),
              "stationarity_established": False, "reference_selected": False}
    write_json(OUT / "validation.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    print(json.dumps(check() if args.check else run(), ensure_ascii=False, indent=2))