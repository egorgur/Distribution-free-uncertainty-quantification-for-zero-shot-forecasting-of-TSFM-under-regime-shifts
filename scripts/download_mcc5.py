"""Save the MCC5 torque inventory, then exactly one pre-selected complete CSV."""

import argparse
import json
import zlib
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

from scripts.audit_pronto_documents import save, write_json
from src.mcc5_download import (ARCHIVE_URL, COMMIT, README_URL, TREE_URL,
                               BudgetRangeReader, archive_metadata, safe_member,
                               select_member, sha256, small_document)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/mcc5_one_run"
CODE = ("src/mcc5_download.py", "src/pronto_documents.py", "scripts/download_mcc5.py")


def code_hashes():
    return {name: sha256((ROOT / name).read_bytes()) for name in CODE}


def make_inventory():
    if (RAW / "inventory.json").exists():
        return json.loads((RAW / "inventory.json").read_text())
    tree = small_document(TREE_URL)
    metadata = archive_metadata(tree)
    readme = small_document(README_URL)
    remote = BudgetRangeReader(ARCHIVE_URL, metadata["size"])
    with ZipFile(remote) as zipped:
        infos = zipped.infolist()
        if len(infos) > 4096 or len({info.filename for info in infos}) != len(infos):
            raise ValueError("Too many or duplicate ZIP entries")
        inventory = [{"member": info.filename, "size": info.file_size,
                      "compressed_size": info.compress_size, "crc32": f"{info.CRC:08x}",
                      "header_offset": info.header_offset, "compression": info.compress_type}
                     for info in infos]
    for item in inventory:
        safe_member(item["member"])
    selected = select_member(inventory)
    save(RAW / "hf_tree.json", tree)
    save(RAW / "source_README.md", readme)
    write_json(RAW / "inventory.json", inventory)
    write_json(RAW / "selection.json", selected)
    write_json(RAW / "inventory_provenance.json", {
        "commit": COMMIT, "archive_url": ARCHIVE_URL, "archive_metadata": metadata,
        "tree_url": TREE_URL, "readme_url": README_URL,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "range_bytes": remote.transferred, "range_requests": remote.requests,
        "code_sha256": code_hashes(), "full_archive_downloaded": False,
        "full_archive_lfs_sha256_verified": False,
        "artifacts_sha256": {name: sha256((RAW / name).read_bytes()) for name in
                             ("hf_tree.json", "source_README.md", "inventory.json", "selection.json")}})
    return inventory


def verify_inventory():
    provenance = json.loads((RAW / "inventory_provenance.json").read_text())
    if provenance["commit"] != COMMIT or provenance["code_sha256"] != code_hashes():
        raise ValueError("Inventory source/version/code changed; preserve the old snapshot")
    for name, expected in provenance["artifacts_sha256"].items():
        if sha256((RAW / name).read_bytes()) != expected:
            raise ValueError(f"Inventory artifact differs: {name}")
    inventory = json.loads((RAW / "inventory.json").read_text())
    chosen = select_member(inventory)
    if json.loads((RAW / "selection.json").read_text()) != chosen:
        raise ValueError("Metadata-only selection differs")
    return provenance, chosen


def download():
    if (RAW / "manifest.json").exists():
        return verify()
    provenance, selected = verify_inventory()
    meta = provenance["archive_metadata"]
    remote = BudgetRangeReader(ARCHIVE_URL, meta["size"])
    path = RAW / "selected.csv"
    part = path.with_suffix(".part")
    with ZipFile(remote) as zipped:
        info = zipped.getinfo(selected["member"])
        if (info.file_size != selected["size"] or info.compress_size != selected["compressed_size"]
                or f"{info.CRC:08x}" != selected["crc32"]):
            raise ValueError("Remote entry differs from saved inventory")
        count = 0
        with zipped.open(info) as source, part.open("wb") as stream:
            while chunk := source.read(1024**2):
                count += len(chunk)
                if count > selected["size"]:
                    raise ValueError("Uncompressed member exceeds advertised size")
                stream.write(chunk)
    if count != selected["size"]:
        raise ValueError("Truncated selected member")
    content = part.read_bytes()
    if f"{zlib.crc32(content):08x}" != selected["crc32"]:
        raise ValueError("Selected member CRC mismatch")
    save(path, content)
    part.unlink()
    write_json(RAW / "manifest.json", {
        "commit": COMMIT, "archive_url": ARCHIVE_URL, "selected": selected,
        "local_file": "selected.csv", "sha256": sha256(content),
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "range_bytes": remote.transferred, "range_requests": remote.requests,
        "inventory_provenance_sha256": sha256((RAW / "inventory_provenance.json").read_bytes()),
        "code_sha256": code_hashes(), "full_archive_downloaded": False,
        "full_archive_lfs_sha256_verified": False, "model_inference": False})
    return verify()


def verify():
    verify_inventory()
    manifest = json.loads((RAW / "manifest.json").read_text())
    if manifest["code_sha256"] != code_hashes():
        raise ValueError("Downloader code differs")
    if manifest["inventory_provenance_sha256"] != sha256((RAW / "inventory_provenance.json").read_bytes()):
        raise ValueError("Inventory provenance differs")
    if manifest["selected"] != json.loads((RAW / "selection.json").read_text()):
        raise ValueError("Manifest selection differs")
    if {p.name for p in RAW.glob("*.csv")} != {"selected.csv"}:
        raise ValueError("Missing or unmanifested CSV")
    content = (RAW / "selected.csv").read_bytes()
    if (len(content) != manifest["selected"]["size"] or sha256(content) != manifest["sha256"]
            or f"{zlib.crc32(content):08x}" != manifest["selected"]["crc32"]):
        raise ValueError("Original CSV integrity check failed")
    return {"passed": True, "member": manifest["selected"]["member"],
            "bytes": len(content), "range_bytes": manifest["range_bytes"],
            "full_archive_downloaded": False, "model_inference": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.inventory_only:
        make_inventory()
        _, selected = verify_inventory()
        print(json.dumps(selected, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(verify() if args.check else download(), ensure_ascii=False, indent=2))