"""Полный аудит UCI gas: python -m scripts.audit_gas_data. Без TSFM/ресемплинга модели."""

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.download_gas_data import EXPECTED_FILES, sha256_file
from src.gas_audit import SENSORS, TYPES, audit_file


def plot_audit(summary, tables, out):
    seconds, segments, events = (tables[k] for k in ("seconds", "segments", "events"))
    gas = "CO" if "_CO" in summary["file"] else "Methane"
    fig, axes = plt.subplots(5, 1, figsize=(15, 12), sharex=True)
    hours = seconds.second / 3600
    axes[0].plot(hours, seconds.gas_ppm_min, label=f"{gas} setpoint (ppm)")
    axes[0].plot(hours, seconds.ethylene_ppm_min, label="Ethylene setpoint (ppm)")
    axes[0].legend(loc="upper right")
    # Не сглаживание исходных данных: прорежены только точки обзорного рисунка.
    for ax, kind in zip(axes[1:], dict.fromkeys(TYPES)):
        for sensor, sensor_type in zip(SENSORS, TYPES):
            if sensor_type == kind:
                ax.plot(hours.iloc[::5], seconds[sensor].iloc[::5], lw=.65, label=sensor)
        ax.set_ylabel(f"{kind}\nraw units")
        ax.legend(ncol=4, loc="upper right")
    axes[-1].set_xlabel("Time (hours); 1-second means, plotted every 5 seconds")
    fig.suptitle(summary["file"] + ": full record (setpoints are not sensor regimes)")
    fig.tight_layout()
    fig.savefig(out / "overview.png", dpi=140)
    plt.close(fig)

    if len(events):
        eligible = events.loc[(events.pre_duration_s >= 30) & (events.post_duration_s >= 30)]
        if len(eligible):
            positions = np.unique(np.rint(np.linspace(0, len(eligible) - 1, 5)).astype(int))
            chosen = eligible.iloc[positions]
            chosen.to_csv(out / "plotted_events.csv", index=False)
            fig, axes = plt.subplots(len(chosen), 2, figsize=(15, 3 * len(chosen)), squeeze=False)
            for (left, right), event in zip(axes, chosen.itertuples()):
                lo, hi = max(-30, -event.pre_duration_s), min(60, event.post_duration_s)
                part = seconds.loc[(seconds.second >= event.time_s + lo)
                                   & (seconds.second < event.time_s + hi)]
                rel = part.second - event.time_s
                left.plot(rel, part.gas_ppm_min, label=f"{gas} ppm")
                left.plot(rel, part.ethylene_ppm_min, label="Ethylene ppm")
                left.set_title(f"Event {event.event}, t={event.time_s:.2f}s")
                for sensor in SENSORS:
                    values = part[sensor]
                    scale = values.std(ddof=0)
                    if scale > 0:
                        right.plot(rel, (values - values.mean()) / scale, lw=.7)
                for ax in (left, right):
                    ax.axvline(0, color="black", ls="--", lw=.8)
                    ax.set_xlabel("Seconds relative to command switch")
                right.set_ylabel("16 channels, z-score within panel")
            axes[0, 0].legend()
            fig.suptitle("Fixed time-order event quantiles; ends clipped at adjacent commands")
            fig.tight_layout()
            fig.savefig(out / "transitions.png", dpi=130)
            plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(segments.duration_s, bins=40)
    axes[0].set_xlabel("Constant-setpoint duration (s), including censored edges")
    tail = tables["late_tail"].delta_over_tail_std.dropna()
    if len(tail):
        axes[1].hist(tail, bins=60)
    axes[1].set_xlabel("Late-tail delta / std of 1s means (not a stability test)")
    fig.tight_layout()
    fig.savefig(out / "durations_tail.png", dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/uci_gas_dynamic"))
    parser.add_argument("--out", type=Path, default=Path("results/real_gas_audit"))
    parser.add_argument("--chunksize", type=int, default=200_000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    manifest = json.loads((args.raw / "manifest.json").read_text())
    summaries = []
    for filename in EXPECTED_FILES:
        path = args.raw / filename
        record = next(entry for entry in manifest["files"] if entry["name"] == filename)
        digest = sha256_file(path)
        if path.stat().st_size != record["bytes"] or digest != record["sha256"]:
            raise ValueError(f"Raw file differs from manifest: {path}")
        print(f"Auditing every row: {filename}", flush=True)
        summary, tables = audit_file(path, chunksize=args.chunksize)
        summary.update(sha256=digest, bytes=path.stat().st_size)
        durations = tables["segments"].iloc[1:-1].duration_s
        summary["interior_duration_quantiles_s"] = {
            str(q): float(durations.quantile(q)) if len(durations) else None
            for q in (0, .1, .25, .5, .75, .9, 1)}
        correlations = tables["correlation_1s"].drop(columns="sensor").to_numpy()
        pairs = np.abs(correlations[np.triu_indices(16, 1)])
        summary["abs_correlation_1s_quantiles"] = {
            str(q): float(np.nanquantile(pairs, q)) if np.isfinite(pairs).any() else None
            for q in (0, .5, 1)}
        tail = tables["late_tail"].delta_over_tail_std.abs().dropna()
        summary["late_tail_abs_delta_over_std_quantiles"] = {
            str(q): float(tail.quantile(q)) if len(tail) else None for q in (.1, .5, .9)}
        summaries.append(summary)
        out = args.out / path.stem
        out.mkdir(parents=True, exist_ok=True)
        for name, table in tables.items():
            suffix = ".csv.gz" if name in ("seconds", "time_anomalies") else ".csv"
            table.to_csv(out / (name + suffix), index=False, float_format="%.12g")
        plot_audit(summary, tables, out)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    meta = {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_url": manifest["url"], "dataset_doi": manifest["doi"],
        "archive_sha256": manifest["archive"]["sha256"],
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "matplotlib": matplotlib.__version__,
        "chunksize": args.chunksize, "elapsed_s": time.perf_counter() - start,
        "code_sha256": {p: sha256_file(Path(p)) for p in
            ("src/gas_audit.py", "scripts/audit_gas_data.py")},
        "settings": {"raw_data_modified": False, "model_inference": False,
            "seconds_bins": "floor(time_s), means for plots only; mixed bins flagged",
            "late_tail": "first/last 10 seconds of final 30; no stability threshold",
            "geometry_hz": [100, 10, 1], "frequency_selected": False,
            "sensor_std": "population, sum-of-squares; small roundoff possible",
            "last_boundary": "last printed timestamp + nominal 0.01s, right censored",
            "independence": "channels and adjacent episodes are dependent"},
        "recordings": summaries}
    (args.out / "audit.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False,
                                                   allow_nan=False) + "\n")


if __name__ == "__main__":
    main()