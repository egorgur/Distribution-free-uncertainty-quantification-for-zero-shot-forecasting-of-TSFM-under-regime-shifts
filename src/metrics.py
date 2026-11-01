"""Метрики качества прогноза одного окна (определения — CLAUDE.md, раздел 8).

Все функции работают с одним окном: future — факт на горизонте (H,),
quantiles — квантильный прогноз (число квантилей, H), levels — уровни квантилей.
Масштаб `scale` считается по контексту (см. seasonal_naive_scale) и делает
метрики безразмерными, чтобы их можно было усреднять по рядам разного масштаба.

Метрики этапа A (AURC, AUROC, AUPRC, Спирмен) — в конце файла; метрики этапа B добавим позже.
"""

from __future__ import annotations

import numpy as np


def seasonal_naive_scale(context: np.ndarray, m: int = 1) -> float:
    """Знаменатель MASE: средняя |x_t − x_{t−m}| по контексту.

    Это ошибка сезонного наивного прогноза «как m шагов назад» внутри контекста.
    m = 1 — обычный наивный прогноз, m = период — сезонный.
    """
    if len(context) <= m:
        raise ValueError(f"контекст длины {len(context)} слишком короткий для m={m}")
    return float(np.mean(np.abs(context[m:] - context[:-m])))


def mase(future: np.ndarray, median: np.ndarray, scale: float) -> float:
    """MASE медианного прогноза: MAE / scale. Меньше 1 — лучше наивного эталона."""
    return float(np.mean(np.abs(future - median)) / scale)


def quantile_loss(future: np.ndarray, quantiles: np.ndarray, levels: list[float]) -> np.ndarray:
    """Квантильная (pinball) потеря для каждого уровня, усреднённая по горизонту → (число квантилей,)."""
    levels_col = np.asarray(levels)[:, None]
    diff = future[None, :] - quantiles  # > 0, если факт выше квантиля
    loss = np.maximum(levels_col * diff, (levels_col - 1) * diff)
    return loss.mean(axis=1)


def scaled_crps(future: np.ndarray, quantiles: np.ndarray, levels: list[float], scale: float) -> float:
    """Приближение CRPS по набору квантилей, делённое на scale.

    CRPS = 2·∫₀¹ pinball_τ dτ; интеграл заменяем средним по имеющимся уровням.
    При 9 квантилях 0.1…0.9 хвосты (τ < 0.1 и τ > 0.9) не учитываются. Усреднение
    то же, что в mean weighted quantile loss (GluonTS, GIFT-Eval), но там делят на
    сумму |факта|, а мы — на масштаб наивного прогноза, как в MASE.
    Для точечного прогноза (все квантили равны) получается ровно MASE.
    """
    return float(2 * quantile_loss(future, quantiles, levels).mean() / scale)


def interval_coverage(future: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Доля шагов горизонта, где факт внутри [lower, upper]."""
    return float(np.mean((future >= lower) & (future <= upper)))


def quantile_hits(future: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """Для каждого уровня — доля шагов, где факт ≤ квантиля → (число квантилей,).

    У откалиброванной модели в среднем по окнам доля для уровня τ ≈ τ.
    """
    return np.mean(future[None, :] <= quantiles, axis=1)


def pce(mean_hits: np.ndarray, levels: list[float]) -> float:
    """Ошибка калибровки по квантилям (PCE, как у Adler et al., ICLR 2026).

    mean_hits — quantile_hits, усреднённые по окнам. PCE = среднее |доля − τ| по уровням.
    0 — идеально, 0.05 — «в среднем на 5 п. п. мимо».
    """
    return float(np.mean(np.abs(np.asarray(mean_hits) - np.asarray(levels))))


# ---------- этап A: насколько сигнал надёжности отделяет плохие прогнозы ----------
#
# Соглашение: большой сигнал = прогнозу доверять нельзя; при отказе первыми отбрасываются окна
# с наибольшим сигналом. weights — веса окон формы (n,) или (B, n), то есть B бутстреп-выборок
# сразу: вес окна = сколько раз его ряд попал в выборку. None — все веса 1. При весе больше 1
# все повторы окна принимаются (или отбрасываются) одновременно; от явного повторения окон
# AURC и AUPRC отличаются на доли процента (Спирмен и AUROC совпадают).
# order — порядок окон по возрастанию сигнала (rank_order); его можно посчитать один раз
# и передавать во все функции.


def rank_order(score: np.ndarray, seed: int = 0) -> np.ndarray:
    """Номера окон по возрастанию сигнала; равные значения — в случайном порядке с фиксированным seed.

    У дискретных сигналов (статистика KS) много совпадений. Без случайного порядка результат
    зависел бы от того, в каком порядке окна лежат в таблице.
    """
    tie_break = np.random.default_rng(seed).random(len(score))
    return np.lexsort((tie_break, score))


def _sorted_weights(n: int, weights: np.ndarray | None, order: np.ndarray) -> np.ndarray:
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)
    return w[..., order]


def risk_coverage(score, error, weights=None, order=None) -> tuple[np.ndarray, np.ndarray]:
    """Кривая риск–покрытие. Окна принимаются по возрастанию сигнала; после каждого окна —
    доля принятых (покрытие) и средняя ошибка принятых (риск). Формы (n,) или (B, n)."""
    order = rank_order(score) if order is None else order
    w = _sorted_weights(len(score), weights, order)
    cw = np.cumsum(w, axis=-1)
    ce = np.cumsum(w * error[order], axis=-1)
    risk = np.divide(ce, cw, out=np.zeros_like(ce), where=cw > 0)
    return cw / cw[..., -1:], risk


def aurc(score, error, weights=None, order=None):
    """AURC — площадь под кривой риск–покрытие, то есть средний риск по всем уровням покрытия.

    Меньше — лучше. Случайный сигнал в среднем даёт среднюю ошибку всех окон; сигнал,
    равный самой ошибке (оракул), — наименьшее возможное значение.
    """
    order = rank_order(score) if order is None else order
    w = _sorted_weights(len(score), weights, order)
    _, risk = risk_coverage(score, error, weights, order)
    return np.sum(w * risk, axis=-1) / np.sum(w, axis=-1)


def risk_at_coverage(score, error, coverages, weights=None, order=None) -> np.ndarray:
    """Средняя ошибка принятых окон при покрытии c (принята доля c окон с наименьшим сигналом).

    Одно значение на каждое c из coverages → (..., len(coverages)).
    """
    cov, risk = risk_coverage(score, error, weights, order)
    k = np.stack([np.sum(cov < c - 1e-12, axis=-1) for c in coverages], axis=-1)  # первое окно с покрытием ≥ c
    return np.take_along_axis(risk, k, axis=-1)


def auroc(score, label, weights=None, order=None):
    """AUROC: вероятность, что у случайного плохого окна (label = True) сигнал больше,
    чем у случайного хорошего. 0.5 — случайный сигнал, 1 — идеальное разделение."""
    order = rank_order(score) if order is None else order
    w = _sorted_weights(len(score), weights, order)
    pos = label[order]
    wp, wn = w * pos, w * ~pos
    neg_below = np.cumsum(wn, axis=-1) - wn  # вес хороших окон с меньшим сигналом
    return np.sum(wp * neg_below, axis=-1) / (wp.sum(axis=-1) * wn.sum(axis=-1))


def average_precision(score, label, weights=None, order=None):
    """AUPRC в форме average precision.

    Отбрасываем окна по убыванию сигнала. Каждый раз, когда отброшено очередное плохое окно,
    считаем долю плохих среди отброшенных; AUPRC — среднее этих долей. Случайный сигнал
    даёт долю плохих окон во всей выборке.
    """
    order = (rank_order(score) if order is None else order)[::-1]
    w = _sorted_weights(len(score), weights, order)
    wp = w * label[order]
    cw = np.cumsum(w, axis=-1)
    precision = np.divide(np.cumsum(wp, axis=-1), cw, out=np.zeros_like(cw), where=cw > 0)
    return np.sum(wp * precision, axis=-1) / wp.sum(axis=-1)


def spearman(score, error, weights=None, order=None, error_order=None):
    """Ранговая корреляция Спирмена между сигналом и ошибкой.

    С весами — для выборки, где окно повторено weight раз; повторы одного окна получают
    общий средний ранг.
    """
    w = np.ones(len(score)) if weights is None else np.asarray(weights, dtype=np.float64)
    r1 = _ranks(w, rank_order(score) if order is None else order)
    r2 = _ranks(w, rank_order(error) if error_order is None else error_order)
    total = w.sum(axis=-1, keepdims=True)
    d1 = r1 - (w * r1).sum(axis=-1, keepdims=True) / total
    d2 = r2 - (w * r2).sum(axis=-1, keepdims=True) / total
    return (w * d1 * d2).sum(axis=-1) / np.sqrt((w * d1**2).sum(axis=-1) * (w * d2**2).sum(axis=-1))


def _ranks(w: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Ранги окон в выборке с весами; при весах 1 — просто 1 … n."""
    ws = w[..., order]
    sorted_ranks = np.cumsum(ws, axis=-1) - (ws - 1) / 2
    ranks = np.empty_like(sorted_ranks)
    ranks[..., order] = sorted_ranks
    return ranks
