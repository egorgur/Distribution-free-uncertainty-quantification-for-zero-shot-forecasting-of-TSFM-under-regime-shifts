"""Чтение кеша прогнозов, который строит scripts/cache_forecasts.py.

Структура results/cache/<name>/:
    dataset.json            — хеш конфига, τ, N, имена вариантов, число окон;
    config.yaml             — копия конфига, по которому построены данные;
    series.npz, series.csv  — сами ряды (ячейка × номер) и их параметры;
    windows.csv             — одна строка на окно: ячейка, номер ряда, t, d, …;
    <модель>/<ячейка>.npz   — квантили q{H}: (окна ячейки, варианты, 9 квантилей, H)
                              и row — номера строк windows.csv для этих окон;
    <модель>/meta.json      — версии, время, short_horizon_is_prefix;
    <модель>/signals/      — сигналы, для которых нужен отдельный прогон модели
                              (s5.npz — scripts/cache_s5.py); не путать с файлами ячеек.

Пример:
    from src.cache import load_windows, load_forecasts
    w = load_windows("results/cache/synth_v1")
    q = load_forecasts("results/cache/synth_v1", "amazon/chronos-bolt-small", horizon=16, variants=[0])
    q.shape  # (len(w), 1, 9, 16); строки в том же порядке, что w
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def model_dir(root: Path | str, model_id: str) -> Path:
    return Path(root) / model_id.split("/")[-1]


def signals_path(root: Path | str, model_id: str, name: str) -> Path:
    """Файл сигнала с отдельным прогоном модели, например name = "s5". Лежит в подпапке,
    потому что load_forecasts читает как прогнозы все *.npz самой папки модели."""
    return model_dir(root, model_id) / "signals" / f"{name}.npz"


def load_dataset_meta(root: Path | str) -> dict:
    return json.loads((Path(root) / "dataset.json").read_text())


def load_windows(root: Path | str) -> pd.DataFrame:
    return pd.read_csv(Path(root) / "windows.csv")


def load_series(root: Path | str) -> tuple[np.ndarray, pd.DataFrame]:
    """Ряды (число рядов, N) и таблица их параметров; строка i таблицы = ряд i."""
    return np.load(Path(root) / "series.npz")["x"], pd.read_csv(Path(root) / "series.csv")


def load_forecasts(
    root: Path | str, model_id: str, horizon: int, variants: list[int] | None = None
) -> np.ndarray:
    """Квантили для всех окон windows.csv: (окна, варианты, 9, horizon), float32.

    variants — номера вариантов (порядок из dataset.json["variants"]); None — все.
    Окна ячеек, которые ещё не посчитаны, заполнены NaN.
    """
    root = Path(root)
    ds = load_dataset_meta(root)
    meta = json.loads((model_dir(root, model_id) / "meta.json").read_text())
    key, max_h = f"q{horizon}", max(ds["horizons"])
    if horizon not in ds["horizons"]:
        raise ValueError(f"горизонт {horizon} не входит в {ds['horizons']}")
    v_idx = list(range(len(ds["variants"]))) if variants is None else list(variants)
    out = np.full((ds["n_windows"], len(v_idx), 9, horizon), np.nan, dtype=np.float32)
    for f in sorted(model_dir(root, model_id).glob("*.npz")):
        with np.load(f) as z:
            if key in z:
                q = z[key][:, v_idx]
            elif meta["short_horizon_is_prefix"]:
                q = z[f"q{max_h}"][:, v_idx, :, :horizon]  # у Chronos-Bolt короткий горизонт = начало длинного
            else:
                raise KeyError(f"{f}: нет {key}")
            out[z["row"]] = q
    return out
