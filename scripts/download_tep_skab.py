"""Оригиналы SKAB (фиксированный commit) и классический TEP MIT, без обработки."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen
from zipfile import ZipFile

SKAB_COMMIT = "b2c0d46c2971dcbfe71e26087b6d231998bb91c2"
TEP_URL = "https://web.mit.edu/braatzgroup/TE_process.zip"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_path(name: str) -> Path:
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts or "\\" in name or ":" in name:
        raise ValueError(f"Небезопасный путь: {name}")
    return Path(*p.parts)


def fetch(url: str) -> bytes:
    # Ограниченное число попыток; никаких исполняемых скачанных файлов.
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers={"User-Agent": "TSFM-research-audit"}), timeout=90) as response:
                return response.read()
        except OSError:
            if attempt == 2:
                raise
    raise RuntimeError("download failed")


def store(path: Path, data: bytes) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Отказ от перезаписи изменённого оригинала: {path}")
    else:
        temporary = path.with_name(path.name + ".part")
        temporary.write_bytes(data)
        temporary.replace(path)
    return {"bytes": len(data), "sha256": digest(data)}


def download_skab(root: Path) -> None:
    base = f"https://raw.githubusercontent.com/waico/SKAB/{SKAB_COMMIT}/"
    tree_url = f"https://api.github.com/repos/waico/SKAB/git/trees/{SKAB_COMMIT}?recursive=1"
    tree = json.loads(fetch(tree_url))
    if tree.get("truncated"):
        raise ValueError("Неполное дерево GitHub")
    selected = [x for x in tree["tree"] if x["type"] == "blob" and (
        (x["path"].startswith("data/") and x["path"].endswith(".csv"))
        or x["path"] in {"README.md", "data/README.md", "LICENSE"})]
    if sum(x["path"].endswith(".csv") for x in selected) != 35:
        raise ValueError("Ожидались ровно 35 CSV SKAB")
    files = {}
    for item in selected:
        name = item["path"]
        path = root / safe_path(name)
        transport_url = base + name
        if path.exists():
            data = path.read_bytes()
        else:
            try:
                data = fetch(transport_url)
            except OSError:
                # Тот же Git object, а не зеркало другой версии данных.
                transport_url = item["url"]
                blob = json.loads(fetch(transport_url))
                if blob.get("encoding") != "base64":
                    raise ValueError("Неожиданная кодировка Git blob")
                data = base64.b64decode(blob["content"])
        git_hash = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if git_hash != item["sha"]:
            raise ValueError(f"Git blob SHA не совпал: {name}")
        files[name] = {**store(path, data), "url": base + name,
                   "transport_url": transport_url, "git_blob_sha1": git_hash}
    manifest = {"dataset": "SKAB v0.9", "commit": SKAB_COMMIT, "tree_url": tree_url, "files": files}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"SKAB: {len(files)} файлов, все Git blob проверены", flush=True)


def download_tep(root: Path) -> None:
    archive = root / "TE_process.zip"
    data = archive.read_bytes() if archive.exists() else fetch(TEP_URL)
    archive_meta = store(archive, data)
    files = {}
    with ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError("Ошибка ZIP CRC")
        seen = set()
        for member in z.infolist():
            relative = safe_path(member.filename)
            if member.is_dir():
                continue
            if str(relative) in seen:
                raise ValueError("Повторяющееся имя в ZIP")
            seen.add(str(relative))
            files[str(relative)] = {**store(root / "original" / relative, z.read(member)), "crc32": f"{member.CRC:08x}"}
    manifest = {"dataset": "TEP classic MIT archive (not Rieth 2017)", "url": TEP_URL,
                "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                "archive": archive_meta, "zip_crc_ok": True, "files": files}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"TEP: ZIP {len(data)} байт, {len(files)} файлов, CRC проверен", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["skab", "tep", "both"], default="both")
    parser.add_argument("--root", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    if args.dataset in {"skab", "both"}:
        download_skab(args.root / "skab")
    if args.dataset in {"tep", "both"}:
        download_tep(args.root / "tep_mit")


if __name__ == "__main__":
    main()