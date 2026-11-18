"""Regime-Aligned Context: подгонка старой части контекста под недавний режим (протокол docs/protocol_v3_aligned.md).

Простыми словами. Контекст из 512 точек делится на старую часть O (первые 448) и недавнюю R
(последние W = 64). R не меняется. O преобразуется так, чтобы её уровень, разброс или тренд
совпадали с R, а форма колебаний сохранялась. Модель получает контекст той же длины.

Все функции векторные: ctx — (окна, L). Используется только история до t (сам контекст),
поэтому функции можно применять и к контексту окна t − 16 для прогноза «назад».
"""

from __future__ import annotations

import numpy as np

W_RECENT = 64
MAD_EPS = 1e-3  # MAD меньше MAD_EPS · s(t) считается нулевым → вместо scale применяется level
K_MIN, K_MAX = 0.1, 10.0  # ограничение коэффициента масштаба
KINDS = ("level", "scale", "trend")


def split(ctx: np.ndarray, w: int = W_RECENT) -> tuple[np.ndarray, np.ndarray]:
    """Старая часть O (без последних w точек) и недавняя R (последние w точек)."""
    if ctx.ndim != 2 or ctx.shape[1] <= w:
        raise ValueError("ctx must be (windows, L) with L > w")
    return ctx[:, :-w], ctx[:, -w:]


def mad(a: np.ndarray) -> np.ndarray:
    """Медиана абсолютных отклонений от медианы по строкам (без множителя 1.4826)."""
    return np.median(np.abs(a - np.median(a, axis=1, keepdims=True)), axis=1)


def ols_line(y: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """МНК-прямая y ≈ a + b · idx по строкам: (a, b)."""
    xm = idx.mean()
    xc = idx - xm
    b = (y - y.mean(axis=1, keepdims=True)) @ xc / (xc @ xc)
    a = y.mean(axis=1) - b * xm
    return a, b


def align_old(ctx: np.ndarray, kind: str, scale: np.ndarray, w: int = W_RECENT) -> tuple[np.ndarray, np.ndarray]:
    """Преобразованная старая часть O′ (окна, L − w) и флаг «scale заменён на level» (окна,)."""
    O, R = split(ctx, w)
    fallback = np.zeros(len(ctx), bool)
    med_o = np.median(O, axis=1, keepdims=True)
    med_r = np.median(R, axis=1, keepdims=True)
    if kind == "level":
        return O - med_o + med_r, fallback
    if kind == "scale":
        mo, mr = mad(O), mad(R)
        fallback = ~((mo >= MAD_EPS * scale) & (mr >= MAD_EPS * scale))
        with np.errstate(divide="ignore", invalid="ignore"):
            k = np.clip(mr / mo, K_MIN, K_MAX)
        k = np.where(fallback, 1.0, k)  # при нулевом MAD — только сдвиг уровня
        return med_r + k[:, None] * (O - med_o), fallback
    if kind == "trend":
        L = ctx.shape[1]
        i_old, i_new = np.arange(L - w, dtype=float), np.arange(L - w, L, dtype=float)
        a_o, b_o = ols_line(O, i_old)
        a_r, b_r = ols_line(R, i_new)
        # Остатки старой части + прямая недавнего участка, продолженная назад.
        return O - (a_o[:, None] + b_o[:, None] * i_old) + (a_r[:, None] + b_r[:, None] * i_old), fallback
    raise ValueError(f"unknown kind {kind}")


def aligned_context(ctx: np.ndarray, kind: str, alpha: float, scale: np.ndarray,
                    w: int = W_RECENT) -> tuple[np.ndarray, np.ndarray]:
    """Контекст с частичной коррекцией старой части: O_α = (1 − α)·O + α·O′, R без изменений."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must lie in [0, 1]")
    O, R = split(ctx, w)
    O2, fallback = align_old(ctx, kind, scale, w)
    return np.concatenate([(1 - alpha) * O + alpha * O2, R], axis=1), fallback


def choose(errors: np.ndarray, scale: np.ndarray, tie_rel: float = 1e-6) -> np.ndarray:
    """Номер кандидата с наименьшей ошибкой «назад» (окна, кандидаты) → (окна,).

    При разнице < tie_rel · s(t) берётся первый по порядку: кандидаты передаются в порядке
    наименьшего вмешательства. Окна без ошибки или с s(t) = 0 получают кандидата 0 (Original).
    """
    ok = np.isfinite(errors).all(axis=1) & np.isfinite(scale) & (scale > 0)
    e = np.where(ok[:, None], errors, 0.0)
    near = e <= e.min(axis=1, keepdims=True) + tie_rel * np.where(ok, scale, 1.0)[:, None]
    return np.where(ok, np.argmax(near, axis=1), 0)
