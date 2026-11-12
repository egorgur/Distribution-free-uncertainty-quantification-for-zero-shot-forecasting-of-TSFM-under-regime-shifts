"""Download a bounded UCI447 snapshot and audit profile/one raw channel."""

import argparse
import hashlib
import json
import urllib.request
import sys
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

from src.uci447_audit import cycle_geometry, parse_profile, parse_ps1

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/uci447"
OUT = ROOT / "results/uci447_audit_v1"
URL = "https://archive.ics.uci.edu/static/public/447/condition%2Bmonitoring%2Bof%2Bhydraulic%2Bsystems.zip"
MAX_ARCHIVE_BYTES = 100 * 1024 * 1024
SELECTED = ("description.txt", "documentation.txt", "profile.txt", "PS1.txt")
CODE = ("src/uci447_audit.py", "scripts/audit_uci447.py", "tests/test_uci447_audit.py")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != data:
        raise ValueError(f"Different existing file: {path}")
    if not path.exists():
        path.write_bytes(data)


def write_json(path, value):
    save(path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode())


def fetch_archive(local_archive=None):
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / "condition_monitoring_of_hydraulic_systems.zip"
    if path.exists():
        data = path.read_bytes()
    elif local_archive is not None:
        if local_archive.stat().st_size > MAX_ARCHIVE_BYTES:
            raise ValueError("Local archive exceeds bounded audit limit")
        data = local_archive.read_bytes()
        with ZipFile(local_archive) as archive:
            for name in SELECTED:
                archive.getinfo(name)
        save(path, data)
    else:
        part = path.with_suffix(".part")
        hasher = hashlib.sha256()
        size = 0
        with urllib.request.urlopen(URL, timeout=120) as response, part.open("wb") as stream:
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_ARCHIVE_BYTES:
                    raise ValueError("UCI447 archive exceeds bounded audit limit")
                hasher.update(chunk)
                stream.write(chunk)
        part.replace(path)
        data = path.read_bytes()
        if hasher.hexdigest() != digest(data):
            raise ValueError("Downloaded archive hash changed while saving")
    if len(data) > MAX_ARCHIVE_BYTES:
        raise ValueError("UCI447 archive exceeds bounded audit limit")
    with ZipFile(path) as archive:
        names = set(archive.namelist())
        if not set(SELECTED) <= names:
            raise ValueError("UCI447 selected members missing")
        members = [{"name": info.filename, "size": info.file_size,
                    "compressed_size": info.compress_size, "crc32": f"{info.CRC:08x}"}
                   for info in archive.infolist()]
    return path, data, members


def run(local_archive=None):
    path, data, members = fetch_archive(local_archive)
    with ZipFile(path) as archive:
        profile = archive.read("profile.txt")
        description = archive.read("description.txt")
        documentation = archive.read("documentation.txt")
        ps1 = archive.read("PS1.txt")
    save(RAW / "profile.txt", profile)
    save(RAW / "description.txt", description)
    save(RAW / "documentation.txt", documentation)
    profile_summary = parse_profile(profile)
    ps1_summary = parse_ps1(ps1, expected_cycles=profile_summary["cycles"])
    write_json(OUT / "profile_summary.json", profile_summary)
    write_json(OUT / "ps1_summary.json", {k: v for k, v in ps1_summary.items() if k != "cycle_stats"})
    write_json(OUT / "cycle_stats.json", ps1_summary["cycle_stats"])
    write_json(OUT / "cycle_geometry.json", cycle_geometry())
    save(OUT / "description.txt", description)
    save(OUT / "documentation.txt", documentation)
    manifest_path = RAW / "manifest.json"
    retrieved_at = (json.loads(manifest_path.read_text())["retrieved_at_utc"] if manifest_path.exists()
                    else datetime.now(timezone.utc).isoformat())
    write_json(manifest_path, {
        "source": URL, "doi": "10.24432/C5CW21", "article_doi": "10.1109/I2MTC.2015.7151267",
        "archive": {"path": str(path.relative_to(ROOT)), "size": len(data), "sha256": digest(data)},
        "members": members, "selected_members": SELECTED, "model_inference": False,
        "selected_sha256": {name: digest(value) for name, value in
                            zip(SELECTED, (description, documentation, profile, ps1))},
        "cycles_joined": False, "stationarity_established": False,
        "retrieved_at_utc": retrieved_at})
    write_json(OUT / "meta.json", {"python": sys.version,
                                   "code_sha256": {name: digest((ROOT / name).read_bytes()) for name in CODE},
                                   "archive_sha256": digest(data),
                                   "scope": "profile, documentation and PS1 only; other channels not audited"})
    return check()


def check():
    manifest = json.loads((RAW / "manifest.json").read_text())
    path = ROOT / manifest["archive"]["path"]
    data = path.read_bytes()
    if len(data) != manifest["archive"]["size"] or digest(data) != manifest["archive"]["sha256"]:
        raise ValueError("UCI447 archive hash mismatch")
    meta = json.loads((OUT / "meta.json").read_text())
    for name, expected in meta["code_sha256"].items():
        if digest((ROOT / name).read_bytes()) != expected:
            raise ValueError(f"Code hash changed: {name}")
    if meta["archive_sha256"] != digest(data):
        raise ValueError("Metadata archive hash mismatch")
    with ZipFile(path) as archive:
        inventory = [{"name": info.filename, "size": info.file_size,
                      "compressed_size": info.compress_size, "crc32": f"{info.CRC:08x}"}
                     for info in archive.infolist()]
        if inventory != manifest["members"]:
            raise ValueError("Archive inventory differs")
        for name in SELECTED:
            content = archive.read(name)  # zipfile validates CRC on every selected member.
            if digest(content) != manifest["selected_sha256"][name]:
                raise ValueError(f"Selected member hash differs: {name}")
            if name != "PS1.txt" and (RAW / name).read_bytes() != content:
                raise ValueError(f"Extracted metadata differs: {name}")
            if name in ("description.txt", "documentation.txt") and (OUT / name).read_bytes() != content:
                raise ValueError(f"Report source differs: {name}")
        profile = archive.read("profile.txt")
        ps1 = archive.read("PS1.txt")
    profile_summary = parse_profile(profile)
    ps1_summary = parse_ps1(ps1, expected_cycles=profile_summary["cycles"])
    if json.loads((OUT / "profile_summary.json").read_text()) != profile_summary:
        raise ValueError("Profile replay differs")
    if json.loads((OUT / "cycle_stats.json").read_text()) != ps1_summary["cycle_stats"]:
        raise ValueError("PS1 replay differs")
    if json.loads((OUT / "ps1_summary.json").read_text()) != {k: v for k, v in ps1_summary.items() if k != "cycle_stats"}:
        raise ValueError("PS1 summary differs")
    if json.loads((OUT / "cycle_geometry.json").read_text()) != cycle_geometry():
        raise ValueError("Cycle geometry differs")
    result = {"passed": True, "archive_size": len(data), "cycles": profile_summary["cycles"],
              "ps1_values": ps1_summary["finite_values"], "cycle_seconds": 60,
              "stable_flag_counts": profile_summary["stable_flag_counts"],
              "continuity_between_cycles_established": False, "stationarity_established": False,
              "model_inference": False}
    write_json(OUT / "validation.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--archive", type=Path, help="Reuse an already downloaded original ZIP")
    args = parser.parse_args()
    print(json.dumps(check() if args.check else run(args.archive), ensure_ascii=False, indent=2))