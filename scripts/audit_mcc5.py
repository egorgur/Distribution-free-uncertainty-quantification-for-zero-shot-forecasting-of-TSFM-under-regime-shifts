"""Reproducible descriptive audit of one pre-recorded MCC5 run, without TSFM."""

import argparse
import csv
import io
import json
import platform
import time
from pathlib import Path
from xml.etree import ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.audit_pronto_documents import save, write_json
from scripts.download_mcc5 import RAW, ROOT, verify
from src.mcc5_audit import SIGNALS, analyse, load_recording
from src.mcc5_download import sha256, small_document

OUT = ROOT / "results/mcc5_one_run_v1"
ARTICLE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12936562/fullTextXML"
CODE = ("src/mcc5_audit.py", "scripts/audit_mcc5.py", "tests/test_mcc5_audit.py",
        "docs/protocol_mcc5_bounded_audit.md")
JSON_PRODUCTS = ("schema", "time", "channels", "exact_duplicate_channels", "key_phase", "geometry")
CSV_PRODUCTS = ("seconds", "torque_quarters", "plateaus")


def csv_bytes(rows):
    stream = io.StringIO(newline="")
    if rows:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return stream.getvalue().encode()


def code_hashes():
    return {name: sha256((ROOT / name).read_bytes()) for name in CODE}


def products():
    data, schema = load_recording(RAW / "selected.csv")
    result = analyse(data)
    result["schema"] = schema
    return data, result


def article_extract(content):
    root = ET.fromstring(content)
    extracted = {}
    for node in root.iter():
        ident = node.attrib.get("id")
        if ident in ("para0012", "para0014", "para0015", "tbl0004"):
            extracted[ident] = " ".join(" ".join(node.itertext()).split())
    if len(extracted) != 4:
        raise ValueError("Expected primary article schema/profile passages missing")
    return extracted


def figures(result):
    fig, axes = plt.subplots(4, 2, figsize=(13, 10), sharex=True)
    for ax, channel in zip(axes.flat, SIGNALS):
        rows = [r for r in result["seconds"] if r["channel"] == channel]
        x = np.array([(r["start_s"] + r["stop_exclusive_s"]) / 2 for r in rows])
        means = np.array([r["mean"] for r in rows])
        stds = np.array([r["std"] for r in rows])
        ax.fill_between(x, [r["min"] for r in rows], [r["max"] for r in rows], color="gray", alpha=.15)
        ax.fill_between(x, means - stds, means + stds, alpha=.25)
        ax.plot(x, means, linewidth=1)
        ax.set_title(channel + " (original units)")
        ax.grid(alpha=.25)
    fig.supxlabel("Row-index time (s), 12800 Hz; 1 s diagnostic bins")
    fig.tight_layout()
    stream = io.BytesIO()
    fig.savefig(stream, format="png", dpi=130)
    plt.close(fig)
    save(OUT / "fig_channels.png", stream.getvalue())
    fig, ax = plt.subplots(figsize=(12, 4))
    rows = result["torque_quarters"]
    x = np.array([(r["start_s"] + r["stop_exclusive_s"]) / 2 for r in rows])
    y = np.array([r["mean"] for r in rows])
    std = np.array([r["std"] for r in rows])
    ax.plot(x, y, label="Measured torque channel, original units")
    ax.fill_between(x, y - std, y + std, alpha=.25, label="within-bin std")
    for a, b, target in ((10, 30, 20), (35, 55, 15), (60, 80, 20)):
        ax.axvspan(a, b, alpha=.1, color="green")
        ax.text((a + b) / 2, .97, f"command example {target} Nm", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8)
    ax.set(xlabel="Row-index time (s), 12800 Hz; 0.25 s diagnostic bins", ylabel="Original units (no conversion)")
    ax.legend(loc="lower right")
    ax.grid(alpha=.25)
    fig.tight_layout()
    stream = io.BytesIO()
    fig.savefig(stream, format="png", dpi=130)
    plt.close(fig)
    save(OUT / "fig_torque.png", stream.getvalue())


def run():
    if OUT.exists() and any(OUT.iterdir()):
        raise ValueError("Use --check for an existing snapshot; preserve historical outputs")
    download = verify()
    start = time.perf_counter()
    article = small_document(ARTICLE_URL)
    save(RAW / "source_article.xml", article)
    write_json(OUT / "article_passages.json", article_extract(article))
    _, result = products()
    for name in JSON_PRODUCTS:
        write_json(OUT / f"{name}.json", result[name])
    for name in CSV_PRODUCTS:
        save(OUT / f"{name}.csv", csv_bytes(result[name]))
    figures(result)
    write_json(OUT / "meta.json", {
        "source_manifest_sha256": sha256((RAW / "manifest.json").read_bytes()),
        "article_url": ARTICLE_URL, "article_sha256": sha256(article),
        "code_sha256": code_hashes(), "python": platform.python_version(),
        "numpy": np.__version__, "matplotlib": matplotlib.__version__,
        "audit_wall_seconds": time.perf_counter() - start,
        "download": download, "diagnostic_only": True, "stability_established": False,
        "model_inference": False,
        "artifacts_sha256": {path.name: sha256(path.read_bytes()) for path in sorted(OUT.iterdir()) if path.is_file()}})
    return check()


def check():
    verify()
    meta = json.loads((OUT / "meta.json").read_text())
    if meta["code_sha256"] != code_hashes():
        raise ValueError("Audit code/protocol/tests changed; preserve snapshot")
    if sha256((RAW / "manifest.json").read_bytes()) != meta["source_manifest_sha256"]:
        raise ValueError("Source manifest changed")
    article = (RAW / "source_article.xml").read_bytes()
    if sha256(article) != meta["article_sha256"]:
        raise ValueError("Primary source changed")
    if article_extract(article) != json.loads((OUT / "article_passages.json").read_text()):
        raise ValueError("Primary-source extraction differs")
    for name, expected in meta["artifacts_sha256"].items():
        if sha256((OUT / name).read_bytes()) != expected:
            raise ValueError(f"Artifact changed: {name}")
    data, result = products()
    for name in JSON_PRODUCTS:
        if json.loads((OUT / f"{name}.json").read_text()) != result[name]:
            raise ValueError(f"Numeric replay differs: {name}")
    for name in CSV_PRODUCTS:
        if (OUT / f"{name}.csv").read_bytes() != csv_bytes(result[name]):
            raise ValueError(f"Numeric replay differs: {name}")
    # Independent direct raw slices for the 15 pre-specified torque intervals.
    column = 2 if data.shape[1] == 9 else 1
    for row in result["plateaus"]:
        raw = data[row["start_s"] * 12800:row["stop_exclusive_s"] * 12800, column]
        if not np.isclose(float(np.sum(raw) / len(raw)), row["mean"], rtol=1e-14, atol=1e-14):
            raise ValueError("Raw interval mean check failed")
    validation = {"passed": True, "data_rows": len(data), "columns": data.shape[1],
                  "timestamp_present": result["time"]["timestamp_present"],
                  "duration_count_over_fs_s": result["time"]["count_divided_by_fs_s"],
                  "finite_signal_values": len(data) * 8,
                  "replayed_torque_intervals": len(result["plateaus"]),
                  "one_run_only": True, "stability_established": False, "model_inference": False}
    write_json(OUT / "validation.json", validation)
    return validation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT, help="New empty snapshot directory, or directory to check")
    args = parser.parse_args()
    OUT = args.out.resolve()
    print(json.dumps(check() if args.check else run(), ensure_ascii=False, indent=2))