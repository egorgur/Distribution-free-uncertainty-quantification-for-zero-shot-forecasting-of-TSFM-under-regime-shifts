"""Download and verify the original UCI dynamic gas-mixture files (stdlib only).

Run with ``python -m scripts.download_gas_data [--out PATH]``.
The two text files need about 1.2 GiB, in addition to the retained ZIP.
SHA256 values are computed locally, NOT trusted upstream checksums.
"""

import argparse
import hashlib
import json
import sys
import zipfile
from datetime import datetime, timezone
from http.client import HTTPException
from pathlib import Path, PurePosixPath
from typing import BinaryIO
from urllib.error import URLError
from urllib.request import Request, urlopen


ARCHIVE_URL = (
    "https://archive.ics.uci.edu/static/public/322/"
    "gas+sensor+array+under+dynamic+gas+mixtures.zip"
)
ARCHIVE_NAME = "gas_sensor_array_under_dynamic_gas_mixtures.zip"
EXPECTED_FILES = ("ethylene_CO.txt", "ethylene_methane.txt")
DEFAULT_OUT = Path("data/raw/uci_gas_dynamic")
DOI = "10.24432/C5WP4C"
CHUNK_BYTES = 1024 * 1024
PROGRESS_BYTES = 32 * 1024 * 1024
SOURCE_HEADERS = ("Content-Type", "Content-Length", "ETag", "Last-Modified")


def _stream_hash(source: BinaryIO, target: BinaryIO | None = None) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    while chunk := source.read(CHUNK_BYTES):
        digest.update(chunk)
        size += len(chunk)
        if target is not None:
            target.write(chunk)
    return size, digest.hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a file without loading it into memory."""
    with path.open("rb") as source:
        return _stream_hash(source)[1]


def _expected_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = {}
    for member in archive.infolist():
        name = member.filename
        path = PurePosixPath(name)
        # Even directory entries must be safe. Never interpret ZIP paths locally.
        if "\\" in name or path.is_absolute() or ".." in path.parts or ":" in name:
            raise ValueError(f"Unsafe ZIP member: {name!r}")
        if member.is_dir():
            continue
        basename = path.name
        if basename not in EXPECTED_FILES:
            raise ValueError(f"Unexpected ZIP file: {name!r}")
        if basename in members:
            raise ValueError(f"Duplicate ZIP file: {basename!r}")
        members[basename] = member
    missing = set(EXPECTED_FILES) - members.keys()
    if missing:
        raise ValueError(f"Missing ZIP files: {sorted(missing)}")
    return [members[name] for name in EXPECTED_FILES]


def validate_archive(path: Path) -> list[zipfile.ZipInfo]:
    """Require exactly the expected basenames and verify every member's CRC."""
    with zipfile.ZipFile(path) as archive:
        members = _expected_members(archive)
        bad = archive.testzip()
        if bad is not None:
            raise zipfile.BadZipFile(f"CRC check failed: {bad}")
        return members


def download_archive(
    path: Path, *, url: str = ARCHIVE_URL, timeout: float = 60, attempts: int = 3,
) -> dict[str, str]:
    """Reuse a valid ZIP, or download fully to .part, validate and rename.

    Retries always restart; no Range requests or appends are used. Returned
    headers are a small provenance allowlist, never cookies or authorization.
    An invalid existing archive raises instead of being silently replaced.
    """
    if timeout <= 0 or attempts < 1:
        raise ValueError("timeout and attempts must be positive")
    if path.is_symlink():
        raise ValueError(f"Refusing archive symlink: {path}")
    if path.exists():
        validate_archive(path)
        return {}
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    for attempt in range(1, attempts + 1):
        part.unlink(missing_ok=True)
        try:
            request = Request(url, headers={"Accept-Encoding": "identity"})
            with urlopen(request, timeout=timeout) as response, part.open("xb") as target:
                if response.getcode() != 200:
                    raise ValueError("Expected HTTP 200 for a complete download")
                headers = {key: response.headers[key] for key in SOURCE_HEADERS
                           if response.headers.get(key) is not None}
                expected = headers.get("Content-Length")
                size, next_progress = 0, PROGRESS_BYTES
                while chunk := response.read(CHUNK_BYTES):
                    target.write(chunk)
                    size += len(chunk)
                    if size >= next_progress:
                        print(f"Downloaded {size / 1024**2:.0f} MiB", file=sys.stderr)
                        next_progress = size + PROGRESS_BYTES
                if expected is not None and size != int(expected):
                    raise ValueError(f"Incomplete download: {size} != {expected} bytes")
            validate_archive(part)
            part.replace(path)
            return headers
        except (OSError, URLError, HTTPException, ValueError, zipfile.BadZipFile):
            if attempt == attempts:
                raise
            print(f"Download attempt {attempt} failed; restarting", file=sys.stderr)
        finally:
            part.unlink(missing_ok=True)
    raise RuntimeError("Download attempts exhausted")


def extract_archive(path: Path, out: Path) -> list[dict]:
    """Stream original bytes to flat filenames, refusing differing local files.

    Existing files are hashed and compared with a fresh stream of the ZIP
    member. Matching files are not rewritten. ZIP paths are never extracted.
    """
    validate_archive(path)
    out.mkdir(parents=True, exist_ok=True)
    files = []
    with zipfile.ZipFile(path) as archive:
        for member in _expected_members(archive):
            name = PurePosixPath(member.filename).name
            target = out / name
            part = out / (name + ".part")
            if target.is_symlink():
                raise ValueError(f"Refusing extracted-file symlink: {target}")
            part.unlink(missing_ok=True)
            try:
                if target.exists():
                    local_hash = sha256_file(target)
                    with archive.open(member) as source:
                        size, digest = _stream_hash(source)
                    if local_hash != digest or target.stat().st_size != size:
                        raise ValueError(f"Existing file differs from archive: {target}")
                else:
                    with archive.open(member) as source, part.open("xb") as destination:
                        size, digest = _stream_hash(source, destination)
                    if size != member.file_size:
                        raise zipfile.BadZipFile(f"Incorrect member size: {name}")
                    part.replace(target)
                files.append({"name": name, "bytes": size, "sha256": digest,
                              "zip_crc32": f"{member.CRC:08x}"})
            finally:
                part.unlink(missing_ok=True)
    return files


def download_data(
    out: Path = DEFAULT_OUT, *, url: str = ARCHIVE_URL,
    timeout: float = 60, attempts: int = 3,
) -> dict:
    """Download, verify, extract and atomically save manifest.json."""
    out = Path(out)
    archive = out / ARCHIVE_NAME
    headers = download_archive(archive, url=url, timeout=timeout, attempts=attempts)
    files = extract_archive(archive, out)
    manifest = {
        "url": url,
        "doi": DOI,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive": {"name": ARCHIVE_NAME, "bytes": archive.stat().st_size,
                    "sha256": sha256_file(archive)},
        "files": files,
        "source_headers": headers,
        "hash_note": "SHA256 values computed locally; not trusted upstream hashes.",
    }
    part = out / "manifest.json.part"
    part.unlink(missing_ok=True)
    try:
        with part.open("x", encoding="utf-8") as destination:
            json.dump(manifest, destination, indent=2)
            destination.write("\n")
        part.replace(out / "manifest.json")
    finally:
        part.unlink(missing_ok=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--url", default=ARCHIVE_URL)
    args = parser.parse_args()
    download_data(args.out, url=args.url)
    print(f"Verified data and manifest: {args.out}")


if __name__ == "__main__":
    main()