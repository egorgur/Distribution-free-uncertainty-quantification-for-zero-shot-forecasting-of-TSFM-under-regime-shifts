"""Сигналы надёжности zero-shot прогноза для этапа A (определения — CLAUDE.md, разделы 5 и 8).

Все функции векторные: первая ось — окна. Чем больше сигнал, тем меньше прогнозу можно доверять.
Сигналы в единицах ряда делятся на масштаб наивного прогноза по контексту (знаменатель MASE),
чтобы их можно было сравнивать между рядами разного масштаба.

    S1  interval_width       ширина 80%-интервала основного прогноза          0 доп. прогонов
    S2  backcast_error       ошибка прогноза «назад» на последних точках      1
    S3  cusum_statistic,     классические статистики смены по самому          0, без модели
        ks_recent            контексту
    S4  median_spread,       расхождение прогнозов при длине контекста,       K
        quantile_distance    изменённой в разы
    S7  те же две меры       расхождение при малых возмущениях контекста      M
                             (длина ± несколько точек, слабый шум)
    S5  embedding_shift      сдвиг эмбеддингов последних патчей контекста     0 (тот же проход
                             относительно прежних                         энкодера)

S6 — этап 7.
"""

from __future__ import annotations

import itertools

import numpy as np


def naive_scale(ctx: np.ndarray) -> np.ndarray:
    """Средняя |x_t − x_{t−1}| по контексту (знаменатель MASE); ctx: (окна, L) → (окна,)."""
    return np.abs(np.diff(ctx, axis=1)).mean(axis=1)


def interval_width(lower: np.ndarray, upper: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """S1: средняя по горизонту ширина интервала [lower, upper]; lower, upper: (окна, H)."""
    return (upper - lower).mean(axis=1) / scale


def backcast_error(median_past: np.ndarray, actual: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """S2: MAE прогноза, сделанного h0 шагов назад, на последних h0 уже известных точках.

    median_past — медиана прогноза по контексту, который кончается за h0 шагов до t;
    actual — факт x[t − h0 + 1 … t]. Оба (окна, h0). Будущее за t не используется.
    """
    return np.abs(actual - median_past).mean(axis=1) / scale


def cusum_statistic(ctx: np.ndarray) -> np.ndarray:
    """S3: CUSUM-статистика одного сдвига среднего по всему контексту.

    max_k |Σ_{i ≤ k} (x_i − x̄)| / (σ̂ · √L). Большая, если где-то в контексте сменилось
    среднее, — неважно, давно или только что.
    """
    dev = ctx - ctx.mean(axis=1, keepdims=True)
    return np.abs(np.cumsum(dev, axis=1)).max(axis=1) / (ctx.std(axis=1) * np.sqrt(ctx.shape[1]))


def ks_recent(ctx: np.ndarray, w: int) -> np.ndarray:
    """S3: статистика Колмогорова–Смирнова между последними w точками контекста и остальными.

    Максимум |F_последние(v) − F_прежние(v)| по всем значениям v. Большая, если распределение
    значений в конце контекста не похоже на прежнее. Принимает значения, кратные 1 / НОК(w, L − w),
    поэтому у разных окон часто совпадает.
    """
    n = ctx.shape[1]
    order = np.argsort(ctx, axis=1, kind="stable")
    values = np.take_along_axis(ctx, order, axis=1)
    step = np.where(order >= n - w, 1.0 / w, -1.0 / (n - w))  # точка из конца контекста или из начала
    diff = np.cumsum(step, axis=1)  # F_последние − F_прежние после очередного значения
    # При равных значениях (залипание датчика) разность считаем только после последнего из них.
    last_of_tie = np.concatenate([values[:, 1:] != values[:, :-1], np.ones((len(ctx), 1), bool)], axis=1)
    return np.abs(np.where(last_of_tie, diff, 0.0)).max(axis=1)


def median_spread(medians: list[np.ndarray], scale: np.ndarray) -> np.ndarray:
    """S4а / S7а: std медианных прогнозов разных вариантов контекста, средняя по горизонту.

    medians — по одному массиву (окна, H) на вариант.
    """
    return np.std(np.stack(medians), axis=0, dtype=np.float64).mean(axis=1) / scale


def quantile_distance(quantiles: list[np.ndarray], scale: np.ndarray) -> np.ndarray:
    """S4б / S7б: расстояние между квантильными прогнозами, среднее по парам вариантов.

    Для пары: |q_k(i) − q_k(j)|, среднее по уровням квантилей и горизонту. Среднее по
    равномерной сетке уровней приближает расстояние Вассерштейна-1 между двумя прогнозными
    распределениями (без хвостов ниже q0.1 и выше q0.9). quantiles — по одному массиву
    (окна, число квантилей, H) на вариант.
    """
    pairs = list(itertools.combinations(range(len(quantiles)), 2))
    total = sum(np.abs(quantiles[i] - quantiles[j]).mean(axis=(1, 2), dtype=np.float64) for i, j in pairs)
    return total / len(pairs) / scale


def embedding_shift(tokens: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    """S5: насколько эмбеддинги последних m патчей контекста отличаются от прежних.

    tokens: (окна, патчи, D) — эмбеддинги патчей по порядку времени, последний кончается в t.
    Среднее последних m патчей μ_new сравнивается с остальными (прежними) патчами:
      (а) косинусное расстояние 1 − cos(μ_new, μ_old) между средними;
      (б) диагональное расстояние Махаланобиса: среднеквадратичное по координатам
          (μ_new − μ_old) / σ_old, где σ_old — std прежних патчей по каждой координате.
          Полная ковариация не оценивается: прежних патчей (28 при L = 512, m = 4) меньше,
          чем координат (512 у Chronos-Bolt-small).
    Сигналы безразмерны: модель сама нормирует контекст, делить на масштаб ряда не нужно.
    """
    t = np.asarray(tokens, dtype=np.float64)
    new, old = t[:, -m:].mean(axis=1), t[:, :-m]
    mu, sd = old.mean(axis=1), old.std(axis=1)
    cos = (new * mu).sum(axis=1) / (np.linalg.norm(new, axis=1) * np.linalg.norm(mu, axis=1))
    # Защита от деления на 0 в координатах, которые у прежних патчей почти не меняются.
    floor = np.maximum(1e-3 * sd.mean(axis=1, keepdims=True), 1e-12)
    maha = np.sqrt((((new - mu) / np.maximum(sd, floor)) ** 2).mean(axis=1))
    return 1 - cos, maha
