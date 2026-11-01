"""Каскад (H5): надёжность (этап A) → локализация (этап B) → обрезка контекста.

Определения политик, метрик и критерия — журнал решений docs/decisions.md (2026-11-02). Здесь только
примитивы, общие для scripts/cache_cascade.py (прогнозы по обрезанным контекстам) и
scripts/stage6_cascade.py (оценка политик).

Соглашение об индексах — как в src/windows.py и src/localization.py: t — последняя точка контекста,
основной контекст — x[t − L + 1 … t]; k — первая точка правого сегмента в точках основного контекста,
то есть τ′ = t − L + 1 + k. «Начало» start — индекс в ряду первой точки обрезанного контекста;
обрезанный контекст — x[start … t], его длина t − start + 1. NO_CROP — обрезки нет (полный контекст).
"""

from __future__ import annotations

import numpy as np

from src import localization as loc

NO_CROP = -1
MIN_POINTS = loc.MIN_POINTS  # самый короткий обрезанный контекст, точек (минимальный сегмент этапа B)

# Этап B каскада — уровень, выбранный в H4 по seed 1, детектор rbf. Для каждой модели: горизонт
# прохода → (конфигурация в results/stage5/meta.json, уровень). None — уровни от горизонта не зависят (Bolt).
STAGE_B = {
    "amazon/chronos-bolt-small": {None: ("bolt-small", "block6")},
    "amazon/chronos-2": {64: ("chronos-2", "block8"), 16: ("chronos-2-H16", "block8")},
}
RAW_DETECTOR = "p2"  # детектор без модели (только описание): р2 этапа B, разбиение rbf на статистиках патча

# Исход одной обрезки (crop_outcome).
CORRECT, FALSE, ANOMALY, EXCLUDED, NONE = "верная", "ложная", "аномалия", "вне оценки", ""


def crop_start(t: np.ndarray, k: np.ndarray, L: int) -> np.ndarray:
    """Начало контекста при обрезке по τ′ этапа B: τ′ = t − L + 1 + k."""
    return np.asarray(t) - L + 1 + np.asarray(k)


def oracle_start(cls: np.ndarray, d: np.ndarray, tau: int, L: int) -> np.ndarray:
    """П4: начало контекста по истинной τ (у drift — начало дрейфа) или NO_CROP.

    Обрезка есть у change и drift, если τ строго внутри основного контекста (d < L − 1) и после
    неё не меньше MIN_POINTS точек (d + 1 ≥ 32, то есть 31 ≤ d < L − 1).
    """
    d = np.asarray(d, dtype=np.float64)  # у none d = NaN: сравнения дают False
    ok = np.isin(cls, ["change", "drift"]) & (d >= MIN_POINTS - 1) & (d < L - 1)
    return np.where(ok, tau, NO_CROP).astype(np.int64)


def reset_and_grow_starts(
    series: np.ndarray, t: np.ndarray, detected: np.ndarray, start_detected: np.ndarray, L: int
) -> np.ndarray:
    """П3 (reset-and-grow, приближение): начало контекста каждого окна или NO_CROP.

    Окна каждого ряда обходятся по возрастанию t. Когда детектор срабатывает (detected), точка
    сброса становится max(прежняя точка сброса, start_detected) — отброшенная история не
    возвращается. Контекст окна — [max(точка сброса, t − L + 1), t]: после сброса он растёт с
    каждым окном, пока снова не станет полным (длина L).
    """
    series, t = np.asarray(series), np.asarray(t)
    out = np.full(len(t), NO_CROP, dtype=np.int64)
    reset, prev = NO_CROP, None
    for i in np.lexsort((t, series)):
        if series[i] != prev:
            reset, prev = NO_CROP, series[i]
        if detected[i]:
            reset = max(reset, int(start_detected[i]))
        if reset > t[i] - L + 1:
            out[i] = reset
    return out


def forecast_metrics(q: np.ndarray, fut: np.ndarray, levels: list[float]) -> dict[str, np.ndarray]:
    """Метрики окна в единицах ряда (без деления на масштаб): q (окна, квантили, H), fut (окна, H).

    mae — MAE медианы; crps — 2 · средняя pinball-потеря по уровням (как на этапе 4);
    coverage — доля шагов внутри [q0.1, q0.9]; width — средняя ширина этого интервала.
    """
    q = np.asarray(q, dtype=np.float64)
    i_lo, i_med, i_hi = (levels.index(v) for v in (0.1, 0.5, 0.9))
    lv = np.asarray(levels)[None, :, None]
    diff = fut[:, None, :] - q
    pinball = np.maximum(lv * diff, (lv - 1) * diff).mean(axis=2).mean(axis=1)
    return {
        "mae": np.abs(fut - q[:, i_med]).mean(axis=1),
        "crps": 2 * pinball,
        "coverage": ((fut >= q[:, i_lo]) & (fut <= q[:, i_hi])).mean(axis=1),
        "width": (q[:, i_hi] - q[:, i_lo]).mean(axis=1),
    }


def crop_outcome(
    cropped: np.ndarray, start: np.ndarray, zone: np.ndarray, cls: np.ndarray, tau: int, delta: int, drift_duration: int
) -> np.ndarray:
    """Исход каждой обрезки по разметке этапа B.

    «ложная» — обрезка в окне «без смены» или в окне «со сменой» мимо допуска (|τ′ − τ| > δ;
    у drift τ′ ∉ [τ − δ, τ + duration + δ]); «верная» — в окне «со сменой» в допуске; «аномалия» и
    «вне оценки» — обрезки в окнах этих зон (в долю ложных не входят); "" — обрезки нет.
    τ′ здесь — начало обрезанного контекста start (у П3 оно могло быть найдено в одном из прежних окон).
    """
    off = np.asarray(start) - tau
    upper = np.where(cls == "drift", drift_duration, 0)
    hit = (off >= -delta) & (off <= upper + delta)
    out = np.full(len(start), NONE, dtype=object)
    out[cropped & (zone == loc.POSITIVE) & hit] = CORRECT
    out[cropped & (((zone == loc.POSITIVE) & ~hit) | (zone == loc.NEGATIVE))] = FALSE
    out[cropped & (zone == loc.ANOMALY)] = ANOMALY
    out[cropped & (zone == loc.EXCLUDED)] = EXCLUDED
    return out
