"""Regime-Component: какая характеристика ряда изменилась (протокол docs/protocol_v3_component.md).

Простыми словами. Контекст делится, как в Regime-Aligned, на старую часть O (448 точек) и недавнюю R
(64 точки). Для каждой компоненты считается одно число z — насколько R отличается от O:
    level — сдвиг медианы в единицах разброса O;
    scale — во сколько раз изменился разброс (модуль логарифма отношения MAD);
    trend — изменение прироста за 64 шага в единицах разброса O;
    freq  — среднее различие автокорреляций на лагах 1…24 (периодичность).
Порог каждой компоненты — 95-й процентиль z на рядах без смены (другой seed). Если ни один порог
не превышен — No Adaptation; иначе выбирается компонента с наибольшим отношением z / порог.
"""

from __future__ import annotations

import numpy as np

from src.adaptation import alignment as al

COMPONENTS = ("none", "level", "scale", "trend", "freq")  # индекс 0 — No Adaptation
STATS = COMPONENTS[1:]
EPS = 1e-3  # масштабы меньше EPS · s(t) считаются нулевыми
MAX_LAG = 24
QUANTILE = 0.95

# Истинный тип смены → компонента (для Oracle). ar, none, anomaly → No Adaptation.
TRUE_COMPONENT = {
    ("change", "mean"): 1, ("change", "sensor_bias"): 1, ("drift", "mean"): 1,
    ("change", "variance"): 2,
    ("change", "trend"): 3,
    ("change", "seasonality"): 4, ("change", "frequency"): 4, ("drift", "frequency"): 4,
}


def acf(a: np.ndarray, max_lag: int) -> np.ndarray:
    """Выборочная автокорреляция по строкам на лагах 1…max_lag: (окна, max_lag)."""
    c = a - a.mean(axis=1, keepdims=True)
    den = (c * c).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.stack([(c[:, :-k] * c[:, k:]).sum(axis=1) / den for k in range(1, max_lag + 1)], axis=1)


def statistics(ctx: np.ndarray, scale: np.ndarray, w: int = al.W_RECENT) -> dict[str, np.ndarray]:
    """z каждой компоненты для контекстов (окна, L); scale — наивный масштаб s(t) контекста."""
    O, R = al.split(ctx, w)
    L = ctx.shape[1]
    tiny = EPS * scale
    mo, mr = al.mad(O), al.mad(R)
    m = np.where(mo >= tiny, mo, scale)  # масштаб старой части; при залипании — s(t)
    ok = np.isfinite(scale) & (scale > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        z_level = np.abs(np.median(R, axis=1) - np.median(O, axis=1)) / m
        z_scale = np.where((mo >= tiny) & (mr >= tiny), np.abs(np.log(mr / mo)), 0.0)
        _, b_o = al.ols_line(O, np.arange(L - w, dtype=float))
        _, b_r = al.ols_line(R, np.arange(L - w, L, dtype=float))
        z_trend = np.abs(b_r - b_o) * w / m
        flat = (O.std(axis=1) < tiny) | (R.std(axis=1) < tiny)
        z_freq = np.where(flat, 0.0, np.abs(acf(O, MAX_LAG) - acf(R, MAX_LAG)).mean(axis=1))
    out = {"level": z_level, "scale": z_scale, "trend": z_trend, "freq": z_freq}
    return {k: np.where(ok & np.isfinite(v), v, 0.0) for k, v in out.items()}


def thresholds(z: dict[str, np.ndarray], mask: np.ndarray, q: float = QUANTILE) -> dict[str, float]:
    """Порог каждой компоненты — квантиль q её z на окнах mask (ряды none другого seed)."""
    return {k: float(np.quantile(z[k][mask], q)) for k in STATS}


def decide(z: dict[str, np.ndarray], thr: dict[str, float]) -> np.ndarray:
    """Номер компоненты (0 — No Adaptation) по наибольшему z / порог среди превысивших порог."""
    ratio = np.stack([z[k] / thr[k] if thr[k] > 0 else np.zeros_like(z[k]) for k in STATS], axis=1)
    best = np.argmax(ratio, axis=1)
    return np.where(ratio.max(axis=1) > 1, best + 1, 0)


def true_component(cls: np.ndarray, kind: np.ndarray) -> np.ndarray:
    """Компонента по истинному типу смены (только для Oracle и разбора ошибок)."""
    return np.array([TRUE_COMPONENT.get((c, k), 0) for c, k in zip(cls, kind)], dtype=int)
