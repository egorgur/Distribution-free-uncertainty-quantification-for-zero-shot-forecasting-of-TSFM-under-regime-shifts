"""Получение одного непрерывного Test11; описательный аудит, не H1–H3."""

import argparse
import hashlib
import json
import csv
import zlib
from pathlib import Path
from zipfile import ZipFile

from scripts.audit_pronto_documents import ROOT, RAW as DOCUMENTS, save, write_csv, write_json
from src.pronto_documents import RangeReader
from src.pronto_test11 import parse_test11, window_geometry

RAW = ROOT / "data/raw/pronto_test11"
OUT = ROOT / "results/pronto_test11"
MEMBER = "PRONTO benchmark case study/C0 Normal and Slugging conditions/Test11/Process Data/0626Testday5.csv"


def download():
    if (RAW / "manifest.json").exists():
        manifest = json.loads((RAW / "manifest.json").read_text())
        data = (RAW / "0626Testday5.csv").read_bytes()
        if hashlib.sha256(data).hexdigest() != manifest["sha256"]:
            raise ValueError("Different local original")
        return manifest
    documents = json.loads((DOCUMENTS / "manifest.json").read_text())
    archive = documents["archive"]
    remote = RangeReader(archive["links"]["self"], archive["size"])
    with ZipFile(remote) as zipped:
        info = zipped.getinfo(MEMBER)
        if info.file_size > 8 * 1024 * 1024:
            raise ValueError("Unexpected Test11 size")
        data = zipped.read(info)
    save(RAW / "0626Testday5.csv", data)
    manifest = {"source_archive": archive, "member": MEMBER, "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(), "crc32": f"{info.CRC:08x}",
                "range_bytes": remote.transferred, "range_requests": remote.requests,
                "full_archive_downloaded": False, "full_archive_md5_verified": False,
                "model_inference": False}
    write_json(RAW / "manifest.json", manifest)
    return manifest


def audit():
    download()
    data = (RAW / "0626Testday5.csv").read_bytes()
    manifest = json.loads((RAW / "manifest.json").read_text())
    if f"{zlib.crc32(data):08x}" != manifest["crc32"]:
        raise ValueError("Original CRC differs")
    summary, channels = parse_test11(data)
    write_csv(OUT / "columns.csv", channels)
    # Эти интервалы прямо прочитаны из журнала, не отобраны по сигналу.
    intervals = [("after_reaches_setpoint_not_explicitly_stabled", 1080, 1980),
                 ("explicit_system_stabled_to_shutdown", 2220, 3120)]
    geometry = [{"interval": name, **window_geometry(summary["logical_data_rows"], a, b)}
                for name, a, b in intervals]
    write_csv(OUT / "window_geometry.csv", geometry)
    write_json(OUT / "audit.json", summary)
    artifacts = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in [OUT / "columns.csv", OUT / "window_geometry.csv", OUT / "audit.json",
                              ROOT / "src/pronto_test11.py", ROOT / "src/pronto_documents.py",
                              ROOT / "tests/test_pronto_test11.py", Path(__file__)]}
    write_json(OUT / "meta.json", {"original_sha256": manifest["sha256"],
                                  "artifacts": artifacts, "model_inference": False,
                                  "nominal_grid_is_diagnostic_not_approved_preprocessing": True})
    return check()


def check():
    data = (RAW / "0626Testday5.csv").read_bytes()
    original = json.loads((RAW / "manifest.json").read_text())
    meta = json.loads((OUT / "meta.json").read_text())
    if len(data) != original["size"] or hashlib.sha256(data).hexdigest() != original["sha256"]:
        raise ValueError("Original mismatch")
    if original["sha256"] != meta["original_sha256"]:
        raise ValueError("Analysis input differs")
    for relative, digest in meta["artifacts"].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Artifact mismatch: {relative}")
    summary, channels = parse_test11(data)
    if summary != json.loads((OUT / "audit.json").read_text()):
        raise ValueError("Summary differs")
    with (OUT / "columns.csv").open(newline="") as stream:
        actual = list(csv.DictReader(stream))
    if actual != [{k: "" if v is None else str(v) for k, v in row.items()} for row in channels]:
        raise ValueError("Columns differ")
    with (OUT / "window_geometry.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            a, b = int(row["start_index"]), int(row["stop_exclusive"])
            valid = [t for t in range(511, summary["logical_data_rows"] - 64, 8)
                     if t - 511 >= a and t + 64 < b and t + 64 < summary["logical_data_rows"]]
            if len(valid) != int(row["rolling_windows_global_phase"]):
                raise ValueError("Independent geometry differs")
    result = {"passed": True, "rows": summary["logical_data_rows"],
              "columns": len(channels), "nominal_grid_rounding_consistent":
              summary["nominal_grid_consistent_with_5_decimal_rounding"],
              "model_inference": False, "stationarity_established": False}
    write_json(OUT / "validation.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    print(json.dumps(check() if args.check else audit(), indent=2))