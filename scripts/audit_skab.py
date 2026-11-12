"""Полный воспроизводимый аудит SKAB без сети и инференса.

Запуск: python -m scripts.audit_skab; сверка: python -m scripts.audit_skab --check.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.skab_audit import (COMMIT, CONFIGS, SENSORS, STRIDE, audit_frame,
    brute_window_counts, cp_rows, cross_file_overlaps, load_csv, sha256, verify_originals)

ROOT = Path(__file__).resolve().parents[1]
CODE = ("scripts/audit_skab.py", "src/skab_audit.py", "tests/test_skab_audit.py")
# Фиксированный выбор по имени, а не по качеству прогноза/детектора.
ILLUSTRATIONS = ("data/anomaly-free/anomaly-free.csv", "data/other/1.csv",
                 "data/valve1/0.csv", "data/valve2/0.csv")


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def illustrate(frame: pd.DataFrame, name: str, out: Path) -> None:
    elapsed = (frame.datetime - frame.datetime.iloc[0]).dt.total_seconds().to_numpy()
    figure, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
    gaps = np.flatnonzero(np.diff(elapsed) != 1) + 1
    for axis, sensor in zip(axes[:4], ("Accelerometer1RMS", "Pressure", "Thermocouple", "Volume Flow RateRMS")):
        # Не рисовать непрерывную линию через пропущенное время.
        for positions in np.split(np.arange(len(frame)), gaps):
            axis.plot(elapsed[positions], frame[sensor].iloc[positions], lw=.65, color="C0")
        axis.set_ylabel(sensor, fontsize=8)
        for mark in cp_rows(frame):
            axis.axvline(elapsed[mark], color="C3", lw=.7, alpha=.7)
        for gap in gaps:
            axis.axvspan(elapsed[gap - 1], elapsed[gap], color="grey", alpha=.2)
    if "anomaly" in frame:
        axes[-1].step(elapsed, frame.anomaly, where="post", label="anomaly", color="C1")
        marks = cp_rows(frame)
        axes[-1].scatter(elapsed[marks], np.full(len(marks), .5), marker="|", color="C3", label="CP")
        axes[-1].legend(loc="upper right")
    else:
        axes[-1].text(.03, .5, "Метки отсутствуют; normal по описанию источника", transform=axes[-1].transAxes)
    axes[-1].set_xlabel("Секунды от начала файла (без склейки разрывов)")
    figure.suptitle(f"SKAB: {name}; иллюстрация по фиксированному имени, не проверка устойчивости")
    figure.tight_layout()
    figure.savefig(out / ("fig_" + name.removeprefix("data/").replace("/", "_").replace(".csv", ".png")), dpi=130)
    plt.close(figure)


def aggregate_windows(counts: pd.DataFrame) -> pd.DataFrame:
    data = counts.copy()
    data["scope"] = np.where(data.labels_observed, "34_labeled_episodes", "normal_unlabeled")
    keys = [key for key in counts if key not in ("file", "L", "H", "labels_observed")]
    groups = data.groupby(["scope", "L", "H"], as_index=False)[keys].sum()
    total = data.groupby(["L", "H"], as_index=False)[keys].sum()
    total.insert(0, "scope", "all35")
    return pd.concat([groups, total], ignore_index=True)


def combine_tables(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """Пустой normal CP-table не должен превращать индексы событий в float."""
    nonempty = [part for part in parts if not part.empty]
    return pd.concat(nonempty or parts, ignore_index=True)


def run_audit(raw: Path, out: Path) -> None:
    started = time.perf_counter()
    integrity = verify_originals(raw)
    out.mkdir(parents=True, exist_ok=True)
    summaries, collections, frames = [], {}, {}
    for name in integrity.file[integrity.file.str.endswith(".csv")]:
        frame = load_csv(raw / name)
        frames[name] = frame
        summary, tables = audit_frame(frame, name)
        summaries.append(summary)
        for key, table in tables.items():
            collections.setdefault(key, []).append(table)
        if name in ILLUSTRATIONS:
            illustrate(frame, name, out)
    tables = {key: combine_tables(parts) for key, parts in collections.items()}
    tables["files"] = pd.DataFrame(summaries)
    tables["integrity"] = integrity
    tables["cross_file_overlaps"] = cross_file_overlaps(frames)
    tables["window_totals"] = aggregate_windows(tables["window_counts"])
    geometry_keys = ["L_before", "H_after", "LH_after", "L_before_H_after", "L_before_LH_after",
                     "L_before_H_after_strict", "L_before_LH_after_strict"]
    tables["geometry_totals"] = tables["event_geometry"].groupby(["L", "H"], as_index=False)[geometry_keys].sum()
    for key, table in tables.items():
        table.to_csv(out / f"{key}.csv", index=False)
    files = tables["files"]
    totals = dict(files=len(files), labeled_episodes=int(files.labels_observed.sum()),
        rows=int(files.rows.sum()), cp_marks=int(files.cp_marks.sum()),
        anomaly_onsets=int(files.anomaly_onsets.sum()), anomaly_offsets=int(files.anomaly_offsets.sum()),
        time_non1s_steps=int(files.non1s_steps.sum()), nonfinite_sensor_values=int(files.nonfinite_sensor_values.sum()))
    metadata = dict(dataset="SKAB v0.9", commit=COMMIT, totals=totals,
        versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, matplotlib=matplotlib.__version__),
        configurations=[dict(L=l, H=h, stride=STRIDE) for l, h in CONFIGS],
        definitions=dict(indices="zero-based; [start, stop)", origin="t=L-1+8k per file; target t+1..t+H",
            strict="all L+H-1 adjacent timestamp intervals exactly 1 second",
            same_cp="no CP in (start,stop); a CP at start belongs to the right segment",
            anomaly0="all L+H observed labels are zero; missing labels are NOT filled",
            geometry="pre=[previous_CP,CP), post=[CP,next_CP), file edges censored",
            tail128="absolute difference of means of halves / population std of last128; no stability threshold"),
        illustrations=list(ILLUSTRATIONS), model_inference=False, protocol_selected=False,
        manifest_sha256=sha256(raw / "manifest.json"),
        code_sha256={name: sha256(ROOT / name) for name in CODE},
        elapsed_seconds=round(time.perf_counter() - started, 3),
        artifact_sha256={path.name: sha256(path) for path in sorted(out.iterdir())
                         if path.suffix in (".csv", ".png")})
    write_json(out / "audit.json", metadata)
    # Только таблицы. Научные выводы находятся в docs/results/skab_audit.md.
    text = ["# SKAB: таблицы аудита", "", "Без инференса; метки не доказывают физическую устойчивость.", "",
            json.dumps(totals, ensure_ascii=False), "", "## Окна", "", tables["window_totals"].to_csv(index=False),
            "## Геометрия CP", "", tables["geometry_totals"].to_csv(index=False)]
    (out / "summary.md").write_text("\n".join(text) + "\n")
    print(json.dumps(metadata["totals"], indent=2))
    print(tables["window_totals"].to_string(index=False))
    print(tables["geometry_totals"].to_string(index=False))


def check_audit(raw: Path, out: Path) -> None:
    """Перепроверить оригиналы, хеши, полноту таблиц и окна простым циклом.

    CP-геометрия проверяется отдельно непосредственными срезами исходников.
    Проверка не является тестом физической устойчивости / качества TSFM.
    """
    integrity = verify_originals(raw)
    metadata = json.loads((out / "audit.json").read_text())
    assert metadata["manifest_sha256"] == sha256(raw / "manifest.json"), "manifest hash"
    for name, digest in metadata["code_sha256"].items():
        assert sha256(ROOT / name) == digest, f"code hash: {name}"
    for name, digest in metadata["artifact_sha256"].items():
        assert sha256(out / name) == digest, f"artifact hash: {name}"
    read = lambda name: pd.read_csv(out / f"{name}.csv")
    files, counts, windows = read("files"), read("window_counts"), read("windows")
    segments, runs, geometry = read("cp_segments"), read("anomaly_runs"), read("event_geometry")
    cp, transitions = read("cp_marks"), read("anomaly_transitions")
    columns, hist, gaps, tails = read("columns"), read("dt_histogram"), read("time_gaps"), read("tail128")
    continuous_runs = read("continuous_runs")
    verified_counts, geometry_checks = 0, 0
    assert set(files.file) == set(integrity.file[integrity.file.str.endswith(".csv")])
    assert len(counts) == len(files) * len(CONFIGS)
    for row in files.itertuples():
        frame = load_csv(raw / row.file)
        assert row.rows == len(frame)
        assert len(columns[columns.file == row.file]) == len(SENSORS)
        assert hist.loc[hist.file == row.file, "count"].sum() == row.rows - 1
        assert len(gaps[gaps.file == row.file]) == row.non1s_steps
        uninterrupted = continuous_runs[continuous_runs.file == row.file]
        assert uninterrupted.rows.sum() == row.rows
        assert uninterrupted.rows.max() == row.max_continuous_1s_rows
        for run1s in uninterrupted.itertuples():
            stamps = frame.datetime.iloc[run1s.start:run1s.stop]
            assert len(stamps) == run1s.rows
            assert (stamps.diff().iloc[1:] == pd.Timedelta(seconds=1)).all()
        assert len(cp[cp.file == row.file]) == row.cp_marks
        assert len(transitions[transitions.file == row.file]) == row.anomaly_onsets + row.anomaly_offsets
        seg = segments[segments.file == row.file]
        assert seg.start.iloc[0] == 0 and seg.stop.iloc[-1] == row.rows
        assert (seg.start.to_numpy()[1:] == seg.stop.to_numpy()[:-1]).all()
        assert seg.rows.sum() == row.rows
        if row.labels_observed:
            run = runs[runs.file == row.file]
            assert run.rows.sum() == row.rows
            assert run.start.iloc[0] == 0 and run.stop.iloc[-1] == row.rows
            assert (run.start.to_numpy()[1:] == run.stop.to_numpy()[:-1]).all()
            assert run.loc[run.label == 0, "rows"].sum() == row.anomaly0_rows
            assert run.loc[run.label == 1, "rows"].sum() == row.anomaly1_rows
        for length, horizon in CONFIGS:
            observed = counts[(counts.file == row.file) & (counts.L == length) & (counts.H == horizon)]
            assert len(observed) == 1
            expected = brute_window_counts(frame, length, horizon)
            for key, value in expected.items():
                assert observed.iloc[0][key] == value, (row.file, length, key)
            detail = windows[(windows.file == row.file) & (windows.L == length) & (windows.H == horizon)]
            assert len(detail) == expected["row_order"]
            assert detail.t.tolist() == list(range(length - 1, row.rows - horizon, STRIDE))
            assert (detail.start == detail.t - length + 1).all()
            assert (detail.stop == detail.t + horizon + 1).all()
            # Сверка всех окон и их флагов, не только итоговых сумм.
            for w in detail.itertuples():
                times = frame.datetime.iloc[w.start:w.stop]
                strict = bool((times.diff().iloc[1:] == pd.Timedelta(seconds=1)).all())
                assert w.strict_1s == strict
                if row.labels_observed:
                    assert w.same_cp_segment == (not frame.changepoint.iloc[w.start + 1:w.stop].any())
                    assert w.all_anomaly0 == bool((frame.anomaly.iloc[w.start:w.stop] == 0).all())
                else:
                    assert pd.isna(w.same_cp_segment) and pd.isna(w.all_anomaly0)
            verified_counts += 1
            geo = geometry[(geometry.file == row.file) & (geometry.L == length) & (geometry.H == horizon)]
            marks = cp_rows(frame).tolist()
            assert geo.row.tolist() == marks
            for i, event in enumerate(geo.itertuples()):
                previous = marks[i - 1] if i else 0
                following = marks[i + 1] if i + 1 < len(marks) else len(frame)
                assert event.before_rows == event.row - previous
                assert event.after_rows == following - event.row
                for label, post in (("H", horizon), ("LH", length + horizon)):
                    available = event.row - previous >= length and following - event.row >= post
                    assert getattr(event, f"L_before_{label}_after") == available
                    stamps = frame.datetime.iloc[event.row - length:event.row + post] if available else None
                    strict = available and bool((stamps.diff().iloc[1:] == pd.Timedelta(seconds=1)).all())
                    assert getattr(event, f"L_before_{label}_after_strict") == strict
                    geometry_checks += 1
        for tail in tails[tails.file == row.file].itertuples():
            values = frame[tail.sensor].iloc[tail.start:tail.stop].to_numpy()
            assert len(values) == 128
            assert np.isclose(tail.std128, values.std())
            assert np.isclose(tail.delta, values[64:].mean() - values[:64].mean())
    pd.testing.assert_frame_equal(aggregate_windows(counts), read("window_totals"), check_dtype=False)
    geometry_keys = [key for key in read("geometry_totals") if key not in ("L", "H")]
    pd.testing.assert_frame_equal(geometry.groupby(["L", "H"], as_index=False)[geometry_keys].sum(),
                                 read("geometry_totals"), check_dtype=False)
    assert metadata["totals"]["rows"] == int(files.rows.sum())
    assert metadata["totals"]["cp_marks"] == len(cp)
    pd.testing.assert_frame_equal(cross_file_overlaps({name: load_csv(raw / name) for name in files.file}),
                                 read("cross_file_overlaps"), check_dtype=False)
    validation = dict(passed=True, verified_originals=len(integrity), verified_csv_files=len(files),
        independently_checked_file_configurations=verified_counts,
        independently_checked_rolling_windows=len(windows), geometry_predicates_checked=geometry_checks,
        code_hashes_verified=True, artifacts_verified=len(metadata["artifact_sha256"]),
        physical_stability_proven=False, model_inference=False)
    write_json(out / "validation.json", validation)
    print(json.dumps(validation, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw/skab")
    parser.add_argument("--out", type=Path, default=ROOT / "results/skab_audit")
    parser.add_argument("--check", action="store_true", help="Only validate saved audit; no model/network")
    args = parser.parse_args()
    if args.check:
        check_audit(args.raw, args.out)
    else:
        run_audit(args.raw, args.out)


if __name__ == "__main__":
    main()