"""Все Process Data PRONTO: сверка 10 ZIP-членов, хранение уникальных дней."""

import csv
import hashlib
import json
import zlib
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from scripts.audit_pronto_documents import ROOT, RAW as DOCUMENTS, save, write_json
from src.pronto_documents import RangeReader

RAW = ROOT / "data/raw/pronto_scada"


def digest(content):
    return hashlib.sha256(content).hexdigest()


def run():
    if (RAW / "manifest.json").exists():
        return verify()
    documents = json.loads((DOCUMENTS / "manifest.json").read_text())
    archive = documents["archive"]
    remote = RangeReader(archive["links"]["self"], archive["size"])
    members, unique = [], {}
    with ZipFile(remote) as zipped:
        selected = [i for i in zipped.infolist() if "/Process Data/" in i.filename
                    and i.filename.lower().endswith(".csv")]
        if len(selected) != 10:
            raise ValueError("Upstream Process Data inventory changed")
        for info in selected:
            filename = PurePosixPath(info.filename).name
            if info.file_size > 16 * 1024 * 1024 or not filename:
                raise ValueError("Unexpected SCADA member")
            content = zipped.read(info)  # CRC каждого из десяти членов проверяется.
            checksum = digest(content)
            if filename in unique and unique[filename] != checksum:
                raise ValueError("Same basename has different contents; do not merge")
            unique[filename] = checksum
            if filename == "0626Testday5.csv":
                previous = ROOT / "data/raw/pronto_test11" / filename
                if previous.exists() and previous.read_bytes() != content:
                    raise ValueError("Earlier Test11 differs")
            save(RAW / filename, content)
            members.append({"member": info.filename, "local_file": filename,
                            "size": info.file_size, "compressed_size": info.compress_size,
                            "crc32": f"{info.CRC:08x}", "sha256": checksum})
    write_json(RAW / "manifest.json", {
        "source": documents["source"], "doi": documents["doi"], "archive": archive,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(), "files": members,
        "range_bytes": remote.transferred, "range_requests": remote.requests,
        "full_archive_downloaded": False, "full_archive_md5_verified": False,
        "model_inference": False,
    })
    return verify()


def verify():
    manifest = json.loads((RAW / "manifest.json").read_text())
    with (ROOT / "results/pronto_documents_v2/archive_inventory.csv").open(newline="") as stream:
        inventory = {row["member"]: row for row in csv.DictReader(stream)}
    files = manifest["files"]
    expected = {name for name in inventory if "/Process Data/" in name and name.endswith(".csv")}
    if len(files) != len(expected) or {f["member"] for f in files} != expected:
        raise ValueError("Member set differs")
    if {p.name for p in RAW.glob("*.csv")} != {f["local_file"] for f in files}:
        raise ValueError("Unmanifested or missing local CSV")
    for f in files:
        path = Path(f["local_file"])
        if path.name != str(path):
            raise ValueError("Invalid local filename")
        data = (RAW / path).read_bytes()
        inv = inventory[f["member"]]
        if len(data) != f["size"] or digest(data) != f["sha256"]:
            raise ValueError("Original hash mismatch")
        if f"{zlib.crc32(data):08x}" != f["crc32"] or f["crc32"] != inv["crc32"]:
            raise ValueError("CRC mismatch")
        if f["size"] != int(inv["size"]):
            raise ValueError("Inventory size mismatch")
    return {"passed": True, "archive_members": len(files),
            "unique_files": len({f["sha256"] for f in files}),
            "full_archive_md5_verified": False, "model_inference": False}


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))