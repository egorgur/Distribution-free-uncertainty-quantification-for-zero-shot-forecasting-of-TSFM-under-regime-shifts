"""Документы PRONTO: inventory и малый журнал, без полной загрузки ZIP/TSFM."""

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.request import urlopen
from zipfile import ZipFile

from src.pronto_documents import RangeReader, workbook_cells

ROOT = Path(__file__).resolve().parents[1]
API = "https://zenodo.org/api/records/1341583"
ARCHIVE = "PRONTO benchmark case study.zip"
# v1 сохраняется как первоначальный снимок: исправлены отображение процентов
# и ранние даты Excel, не относящиеся к датам текущего журнала.
RAW = ROOT / "data/raw/pronto_documents_v2"
OUT = ROOT / "results/pronto_documents_v2"


def save(path, content):
    """Не перезаписываем отличающийся исследовательский снимок."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != content:
        raise ValueError(f"Different existing file: {path}")
    if not path.exists():
        path.write_bytes(content)


def write_json(path, data):
    save(path, (json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode())


def write_csv(path, rows):
    import io
    stream = io.StringIO(newline="")
    if rows:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save(path, stream.getvalue().encode())


def fetch_small(url, size):
    if size > 8 * 1024 * 1024:
        raise ValueError("Document too large")
    with urlopen(url, timeout=60) as response:
        data = response.read(size + 1)
    if len(data) != size:
        raise ValueError("Unexpected document size")
    return data


def run():
    if (RAW / "manifest.json").exists():
        return check()
    with urlopen(API, timeout=60) as response:
        data = response.read(1024 * 1024)
    meta = json.loads(data)
    save(RAW / "zenodo_record.json", data)
    archive = next(f for f in meta["files"] if f["key"] == ARCHIVE)
    remote = RangeReader(archive["links"]["self"], archive["size"])
    inventory, originals = [], []
    with ZipFile(remote) as zipped:
        for info in zipped.infolist():
            inventory.append({"member": info.filename, "size": info.file_size,
                              "compressed_size": info.compress_size, "crc32": f"{info.CRC:08x}",
                              "header_offset": info.header_offset})
        # Только журналы/описания. Сигналы, фото, видео и прочие XLSX не загружаем.
        selected = [i for i in zipped.infolist() if not i.is_dir() and
                    (("log" in i.filename.lower() and i.filename.lower().endswith(".xlsx")) or
                     i.filename.lower().endswith("readme.txt"))]
        if not selected:
            raise ValueError("No log found; inspect inventory, do not download whole archive")
        for info in selected:
            path = PurePosixPath(info.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in info.filename:
                raise ValueError("Unsafe ZIP member")
            if info.file_size > 8 * 1024 * 1024:
                raise ValueError("Selected document too large")
            content = zipped.read(info)  # zipfile проверяет CRC распакованного члена.
            relative = Path("members") / Path(*path.parts)
            save(RAW / relative, content)
            originals.append({"path": relative.as_posix(), "member": info.filename,
                              "size": len(content), "sha256": hashlib.sha256(content).hexdigest(),
                              "crc32": f"{info.CRC:08x}"})
    for f in meta["files"]:
        if f["key"] == ARCHIVE:
            continue
        content = fetch_small(f["links"]["self"], f["size"])
        if "md5:" + hashlib.md5(content).hexdigest() != f["checksum"]:
            raise ValueError("Published document MD5 mismatch")
        save(RAW / f["key"], content)
        originals.append({"path": f["key"], "member": "", "size": len(content),
                          "sha256": hashlib.sha256(content).hexdigest(),
                          "published_checksum": f["checksum"]})
    write_csv(OUT / "archive_inventory.csv", inventory)
    all_cells, all_sheets = [], []
    for f in originals:
        if f["path"].lower().endswith(".xlsx"):
            cells, sheets = workbook_cells((RAW / f["path"]).read_bytes())
            all_cells.extend({"source": f["path"], **c} for c in cells)
            all_sheets.extend({"source": f["path"], **s} for s in sheets)
    write_csv(OUT / "log_cells.csv", all_cells)
    write_json(OUT / "log_sheets.json", all_sheets)
    # Только локальные источники/таблицы, не обещание подлинности всего ZIP.
    artifacts = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in [RAW / "zenodo_record.json", OUT / "archive_inventory.csv",
                           OUT / "log_cells.csv", OUT / "log_sheets.json",
                           ROOT / "src/pronto_documents.py", Path(__file__)]}
    write_json(RAW / "manifest.json", {
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(), "source": API,
        "doi": meta["doi"], "archive": archive, "full_archive_downloaded": False,
        "full_archive_md5_verified": False, "range_bytes": remote.transferred,
        "range_requests": remote.requests, "files": originals, "artifacts": artifacts,
        "model_inference": False, "sensor_data_downloaded": False,
    })
    return check()


def check():
    manifest = json.loads((RAW / "manifest.json").read_text())
    for relative, expected in manifest["artifacts"].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Artifact hash mismatch: {relative}")
    expected_cells, expected_sheets = [], []
    for f in manifest["files"]:
        content = (RAW / f["path"]).read_bytes()
        if len(content) != f["size"] or hashlib.sha256(content).hexdigest() != f["sha256"]:
            raise ValueError(f"Original mismatch: {f['path']}")
        if "crc32" in f:
            import zlib
            if f"{zlib.crc32(content):08x}" != f["crc32"]:
                raise ValueError("CRC mismatch")
        if f["path"].lower().endswith(".xlsx"):
            cells, sheets = workbook_cells(content)
            expected_cells.extend({"source": f["path"], **c} for c in cells)
            expected_sheets.extend({"source": f["path"], **s} for s in sheets)
    with (OUT / "log_cells.csv").open(newline="") as stream:
        if list(csv.DictReader(stream)) != expected_cells:
            raise ValueError("Parsed log cells differ")
    if json.loads((OUT / "log_sheets.json").read_text()) != expected_sheets:
        raise ValueError("Parsed sheets differ")
    result = {"passed": True, "originals": len(manifest["files"]),
              "worksheets": len(expected_sheets), "nonempty_cells": len(expected_cells),
              "full_archive_downloaded": False, "full_archive_md5_verified": False,
              "sensor_data_downloaded": False, "model_inference": False}
    write_json(OUT / "validation.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    print(json.dumps(check() if args.check else run(), ensure_ascii=False, indent=2))