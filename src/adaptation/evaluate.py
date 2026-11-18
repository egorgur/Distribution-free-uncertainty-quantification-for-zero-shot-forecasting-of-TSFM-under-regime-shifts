"""Общие показатели новых методов v3 (docs/protocol_v3_multicontext.md, разд. 5–6).

r — MAE метода / MAE Full L512 на том же окне. Сводка всегда двухступенчатая: сначала
среднее log r по окнам каждого ряда, затем среднее по рядам. Так каждый ряд весит
одинаково, сколько бы окон в него ни попало.

Transition Gain (TG) = 1 − exp(среднее): доля, на которую метод снижает ошибку Full в
переходе (TG = 0.1 — ошибка в среднем на 10% меньше).
Stable Harm (SH) = exp(среднее) − 1: на сколько метод ухудшает Full без смены режима.
Бутстреп — ряды с возвращением; для парного сравнения методов одни и те же выборки.
"""

from __future__ import annotations

import numpy as np

from src.evaluation import safe_log


def log_ratio(mae_method: np.ndarray, mae_full: np.ndarray) -> np.ndarray:
    """log(MAE метода / MAE Full) с floor 1e-6, как в протоколе v1."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return safe_log(np.asarray(mae_method, float) / np.asarray(mae_full, float))


def series_means(values: np.ndarray, series: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Среднее values по окнам mask внутри каждого ряда: (номера рядов, средние)."""
    v, s = np.asarray(values, float)[mask], np.asarray(series)[mask]
    if not np.isfinite(v).all():
        raise ValueError("non-finite values inside the evaluated group")
    ids, inv = np.unique(s, return_inverse=True)
    sums = np.bincount(inv, weights=v)
    return ids, sums / np.bincount(inv)


def bootstrap_draws(n_series: int, n_boot: int, seed) -> np.ndarray:
    """Индексы рядов бутстреп-выборок: (выборки, ряды)."""
    return np.random.default_rng(seed).integers(0, n_series, (n_boot, n_series))


def group_mean(values: np.ndarray, series: np.ndarray, mask: np.ndarray,
               draws: np.ndarray) -> tuple[float, np.ndarray]:
    """Среднее по рядам средних по окнам и его бутстреп-повторы (draws — из bootstrap_draws)."""
    _, m = series_means(values, series, mask)
    if draws.shape[1] != len(m):
        raise ValueError("bootstrap draws do not match the number of series in the group")
    return float(m.mean()), m[draws].mean(axis=1)


def ci(boot: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def gain(point: float, boot: np.ndarray) -> tuple[float, float, float]:
    """TG и 95% интервал из среднего log r (TG убывает с ростом log r, поэтому границы меняются местами)."""
    lo, hi = ci(boot)
    return 1 - np.exp(point), 1 - np.exp(hi), 1 - np.exp(lo)


def harm(point: float, boot: np.ndarray) -> tuple[float, float, float]:
    """SH и 95% интервал из среднего log r."""
    lo, hi = ci(boot)
    return np.exp(point) - 1, np.exp(lo) - 1, np.exp(hi) - 1
