"""Аудит классического архива TEP MIT, не расширенного корпуса Rieth.

Сетка 180 с и граница 8 ч взяты из поставленного Fortran, не из timestamp:
в числовых файлах их нет. Это геометрия, не доказательство устойчивости.
"""
from __future__ import annotations

import hashlib
import json
from itertools import combinations
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

CONFIGS = ((512, 64), (128, 16))
CHANNELS = [f"XMEAS{i:02}" for i in range(1, 42)] + [f"XMV{i:02}" for i in range(1, 12)]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_originals(raw: Path) -> list[dict]:
    manifest = json.loads((raw / "manifest.json").read_text())
    archive = raw / "TE_process.zip"
    if sha256(archive) != manifest["archive"]["sha256"] or archive.stat().st_size != manifest["archive"]["bytes"]:
        raise ValueError("Archive checksum mismatch")
    rows = []
    with ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError("ZIP CRC mismatch")
        expected = {m.filename for m in z.infolist() if not m.is_dir()}
        if expected != set(manifest["files"]):
            raise ValueError("Manifest and archive inventory mismatch")
        actual = {p.relative_to(raw / "original").as_posix() for p in (raw / "original").rglob("*") if p.is_file()}
        if actual != expected:
            raise ValueError("Extracted inventory mismatch")
        for name, record in manifest["files"].items():
            path = (raw / "original" / name).resolve()
            if not path.is_relative_to((raw / "original").resolve()):
                raise ValueError("Unsafe manifest path")
            data = path.read_bytes()
            if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"] or data != z.read(name):
                raise ValueError(f"Original differs: {name}")
            rows.append(dict(file=name, bytes=len(data), sha256=record["sha256"], verified=True))
    return rows


def load_dat(path: Path) -> tuple[np.ndarray, bool]:
    # Только d00.dat в поставке имеет другую ориентацию; не угадываем её
    # по качеству ряда и не меняем значения. Каждая строка должна быть полной.
    transposed = path.name == "d00.dat"
    widths = [len(line.split()) for line in path.read_text().splitlines()]
    if not widths or min(widths) == 0 or len(set(widths)) != 1:
        raise ValueError("Empty/ragged DAT")
    values = np.loadtxt(path, ndmin=2)
    if transposed:
        if values.shape[0] != 52:
            raise ValueError("Expected 52 rows in transposed d00.dat")
        values = values.T
    if values.shape[1] != 52:
        raise ValueError("Expected 52 process variables")
    return values, transposed


def rolling_geometry(n: int, length: int, horizon: int, onset: int | None) -> dict:
    if min(n, length, horizon) < 1:
        raise ValueError("Lengths must be positive")
    t = np.arange(length - 1, n - horizon, 8)
    row = dict(L=length, H=horizon, row_order=len(t),
               first_t=int(t[0]) if len(t) else None, last_t=int(t[-1]) if len(t) else None)
    if onset is not None:
        row.update(before=int((t + horizon < onset).sum()),
                   change_in_horizon=int(((t < onset) & (t + horizon >= onset)).sum()),
                   mixed_context=int(((t >= onset) & (t - length + 1 < onset)).sum()),
                   fully_post=int((t - length + 1 >= onset).sum()),
                   first_d=int(t[0] - onset) if len(t) else None,
                   last_d=int(t[-1] - onset) if len(t) else None)
    return row


def compare_runs(frames: dict[str, np.ndarray]) -> pd.DataFrame:
    """Точные совпадения по номеру строки и общий префикс; не тест независимости."""
    rows = []
    for (left, x), (right, y) in combinations(sorted(frames.items()), 2):
        n = min(len(x), len(y))
        equal = np.all(x[:n] == y[:n], axis=1)
        mismatch = np.flatnonzero(~equal)
        rows.append(dict(left=left, right=right, compared_rows=n,
                         identical_rows_same_index=int(equal.sum()),
                         common_prefix_rows=int(mismatch[0]) if len(mismatch) else n))
    return pd.DataFrame(rows)


def audit(raw: Path) -> dict[str, pd.DataFrame]:
    integrity = verify_originals(raw)
    files, columns, windows, tails, frames = [], [], [], [], {}
    paths = sorted((raw / "original" / "TE_process").glob("*.dat"))
    expected = {f"d{i:02}{suffix}.dat" for i in range(22) for suffix in ("", "_te")}
    if {p.name for p in paths} != expected:
        raise ValueError("Expected classic 44 numeric files")
    for path in paths:
        x, transposed = load_dat(path)
        frames[path.name] = x
        testing = path.stem.endswith("_te")
        fault = int(path.stem[1:3])
        # Граница только в test fault; точная семантика train-файлов не выдумывается.
        onset = 160 if testing and fault else None
        files.append(dict(file=path.name, split="testing" if testing else "training", fault=fault,
                          rows=len(x), columns=x.shape[1], transposed_on_read=transposed,
                          nonfinite=int((~np.isfinite(x)).sum()),
                          nominal_step_seconds=180, onset_assumed=onset))
        for j, channel in enumerate(CHANNELS):
            v = x[:, j]
            f = v[np.isfinite(v)]
            columns.append(dict(file=path.name, channel=channel, finite=len(f),
                nonfinite=len(v) - len(f), min=float(f.min()) if len(f) else np.nan,
                max=float(f.max()) if len(f) else np.nan, std=float(f.std()) if len(f) else np.nan,
                unique_finite=len(np.unique(f)), constant=len(f) == len(v) and len(np.unique(f)) == 1,
                equal_adjacent=int((v[1:] == v[:-1]).sum())))
            # Последние128: только диагностика поздней динамики, не отбор стационарных.
            if len(v) >= 128:
                tail = v[-128:]
                std = float(tail.std())
                delta = float(tail[64:].mean() - tail[:64].mean())
                tails.append(dict(file=path.name, channel=channel, std128=std, delta=delta,
                                  abs_mean_difference_over_std=abs(delta) / std if std > 0 else np.nan))
        for length, horizon in CONFIGS:
            windows.append(dict(file=path.name, split="testing" if testing else "training", fault=fault,
                                **rolling_geometry(len(x), length, horizon, onset)))
    return dict(integrity=pd.DataFrame(integrity), files=pd.DataFrame(files),
                columns=pd.DataFrame(columns), windows=pd.DataFrame(windows),
                tail128=pd.DataFrame(tails), run_comparisons=compare_runs(frames))