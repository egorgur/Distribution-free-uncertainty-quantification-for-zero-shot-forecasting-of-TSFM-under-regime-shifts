"""Загрузка реальных экземпляров Petrobras 3W по закреплённому commit (docs/protocol_3w_audit.md).

    python -m scripts.download_3w            # оглавление + загрузка 1119 WELL-*.parquet (~516 МБ)
    python -m scripts.download_3w --check    # без сети: сверить файлы с манифестом

Каждый файл берётся по raw-ссылке закреплённого commit и проверяется git-blob SHA1 из дерева
репозитория (sha1(b"blob <размер>\\0" + содержимое)); при несовпадении файл удаляется и скрипт падает.
Файлы — data/raw/3w/files/<класс>/ (вне git); манифест и оглавление — data/raw/3w/ (в git).
Загруженные файлы — недоверенные данные: только читаются как Parquet.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = "petrobras/3W"
COMMIT = "c86cc7c70063646910be7292de8a54a6fb2a8399"
ROOT = Path("data/raw/3w")
FILES = ROOT / "files"
DOCS = ("dataset/dataset.ini", "dataset/README.md", "dataset/LICENSE-CC-BY")
UA = {"User-Agent": "research-client"}


def git_blob_sha1(data: bytes) -> str:
    """SHA1 git-объекта blob: так GitHub адресует содержимое файла в дереве."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def fetch(url: str, tries: int = 3) -> bytes:
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
                return r.read()
        except OSError:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))
    raise RuntimeError("unreachable")


def raw_url(path: str) -> str:
    return f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{path}"


def real_entries(tree: dict) -> list[dict]:
    """Только реальные экземпляры: dataset/<класс>/WELL-*.parquet."""
    return sorted((x for x in tree["tree"] if x["type"] == "blob" and x["path"].startswith("dataset/")
                   and x["path"].split("/")[-1].startswith("WELL-") and x["path"].endswith(".parquet")),
                  key=lambda x: x["path"])


def check(manifest: dict) -> None:
    bad = []
    for e in manifest["files"]:
        f = ROOT / e["local"]
        if not f.exists() or git_blob_sha1(f.read_bytes()) != e["git_sha"]:
            bad.append(e["local"])
    if bad:
        raise SystemExit(f"не совпали {len(bad)} файлов, например {bad[:3]}")
    print(f"проверено {len(manifest['files'])} файлов: OK")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="без сети: сверить файлы с манифестом")
    args = parser.parse_args()
    if args.check:
        check(json.loads((ROOT / "manifest.json").read_text()))
        return
    ROOT.mkdir(parents=True, exist_ok=True)
    tree_bytes = fetch(f"https://api.github.com/repos/{REPO}/git/trees/{COMMIT}?recursive=1")
    tree = json.loads(tree_bytes)
    if tree.get("truncated"):
        raise SystemExit("оглавление репозитория усечено API — нужен другой способ получения дерева")
    (ROOT / "tree.json").write_bytes(tree_bytes)
    for d in DOCS:
        (ROOT / Path(d).name).write_bytes(fetch(raw_url(d)))
    entries = real_entries(tree)
    out, t0, done = [], time.perf_counter(), 0
    for i, e in enumerate(entries):
        cls, name = e["path"].split("/")[1], e["path"].split("/")[-1]
        local = Path("files") / cls / name
        f = ROOT / local
        f.parent.mkdir(parents=True, exist_ok=True)
        if not (f.exists() and git_blob_sha1(f.read_bytes()) == e["sha"]):  # продолжение прерванной загрузки
            data = fetch(raw_url(e["path"]))
            if git_blob_sha1(data) != e["sha"] or len(data) != e["size"]:
                raise SystemExit(f"{e['path']}: хеш или размер не совпадает с деревом commit")
            f.write_bytes(data)
            done += 1
        out.append({"path": e["path"], "local": str(local), "class_dir": cls, "size": e["size"], "git_sha": e["sha"],
                    "sha256": hashlib.sha256(f.read_bytes()).hexdigest()})
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(entries)}, {time.perf_counter() - t0:.0f} с", flush=True)
    manifest = {"repo": REPO, "commit": COMMIT, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "protocol": "docs/protocol_3w_audit.md", "n_files": len(out),
                "bytes": sum(e["size"] for e in out), "downloaded_this_run": done, "files": out}
    (ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    check(manifest)
    print(f"готово: {len(out)} файлов, {manifest['bytes'] / 1e6:.1f} МБ, {time.perf_counter() - t0:.0f} с")


if __name__ == "__main__":
    main()
