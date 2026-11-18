"""Multi-Context: выбор и смешивание прогнозов одной TSFM по контекстам разной длины.

Протокол — docs/protocol_v3_multicontext.md (v1, 2026-11-18). Здесь только чистые функции
на numpy, без модели и без чтения кеша.

Идея простыми словами. Из одного момента t модель делает четыре прогноза: по последним
64, 128, 256 и 512 точкам. Короткий контекст быстрее «забывает» старый режим, длинный
даёт больше истории. Мы оцениваем, какой длине сейчас можно доверять, ТОЛЬКО по уже
известным данным, и либо выбираем одну длину, либо смешиваем прогнозы с весами.

Соглашения о формах массивов:
    q       — квантили кандидатов: (окна, K длин, 9 квантилей, H шагов);
    scores  — признак надёжности: (окна, K), меньше — надёжнее;
    weights — веса длин: (окна, K), неотрицательны, сумма 1.
Порядок длин всегда LENGTHS (по возрастанию), поэтому последний индекс — L512.
"""

from __future__ import annotations

import numpy as np

LENGTHS = (64, 128, 256, 512)
FULL = len(LENGTHS) - 1  # индекс полного контекста L512 — запасной вариант
TIE_TOL = 1e-6  # разница признаков меньше этой — равенство (в единицах масштаба s(t))


# ---------- признаки надёжности длины ----------


def previous_rows(series: np.ndarray, pos: np.ndarray, row: np.ndarray, lag: int) -> np.ndarray:
    """Номер строки окна того же ряда с pos − lag или −1, если такого окна нет.

    Окно pos − lag кончается на lag точек раньше t, поэтому его прогноз на первые lag шагов —
    честный прогноз «назад»: значения x[t−lag+1 … t] в его контекст не входят.
    """
    index = {(int(s), int(p)): int(r) for s, p, r in zip(series, pos, row)}
    return np.array([index.get((int(s), int(p) - lag), -1) for s, p in zip(series, pos)], dtype=np.int64)


def hindcast_scores(median_prev: np.ndarray, actual: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Ошибка «назад» каждой длины: MAE медианы прогноза окна t−16 на x[t−15 … t] / s(t).

    median_prev — (окна, K, h0): первые h0 шагов медианы прогноза окна t − h0 по каждой длине;
    actual — (окна, h0): x[t − h0 + 1 … t]; scale — (окна,) наивный масштаб контекста L512 в t.
    """
    return np.abs(median_prev - actual[:, None, :]).mean(axis=2) / scale[:, None]


def width_scores(q: np.ndarray, scale: np.ndarray, i_lo: int, i_hi: int) -> np.ndarray:
    """Средняя по горизонту ширина интервала [q_lo, q_hi] каждой длины / s(t)."""
    return (q[:, :, i_hi] - q[:, :, i_lo]).mean(axis=2) / scale[:, None]


def consistency_scores(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Несогласованность длины L с остальными: среднее |q_k(L) − q_k(L′)| по L′ ≠ L, квантилям, шагам / s(t).

    Это S4б (docs/implementation.md), но посчитанный для одной длины против остальных.
    """
    n, K = q.shape[:2]
    out = np.zeros((n, K))
    for a in range(K):
        for b in range(K):
            if a != b:
                out[:, a] += np.abs(q[:, a] - q[:, b]).mean(axis=(1, 2))
    return out / (K - 1) / scale[:, None]


# ---------- выбор и веса ----------


def valid_rows(scores: np.ndarray) -> np.ndarray:
    """Окна, где признак определён для всех длин (конечен)."""
    return np.isfinite(scores).all(axis=1)


def select_index(scores: np.ndarray, tol: float = TIE_TOL) -> np.ndarray:
    """Жёсткий выбор: длина с наименьшим признаком; при равенстве (разница < tol) — более длинная.

    Окна с неопределённым признаком получают L512 (FULL).
    """
    ok = valid_rows(scores)
    safe = np.where(ok[:, None], scores, 0.0)
    near_best = safe <= safe.min(axis=1, keepdims=True) + tol
    # Последний True по оси длин = самая длинная из почти лучших.
    idx = scores.shape[1] - 1 - np.argmax(near_best[:, ::-1], axis=1)
    return np.where(ok, idx, FULL)


def one_hot(idx: np.ndarray, k: int = len(LENGTHS)) -> np.ndarray:
    return np.eye(k)[idx]


def softmax_weights(scores: np.ndarray, temperature: float) -> np.ndarray:
    """Веса w_L ∝ exp(−признак_L / T). Меньше T — ближе к жёсткому выбору, больше — к равным весам.

    Вычитание минимума не меняет весов, но защищает exp от переполнения.
    Окна с неопределённым признаком получают весь вес на L512.
    """
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    ok = valid_rows(scores)
    safe = np.where(ok[:, None], scores, 0.0)
    e = np.exp(-(safe - safe.min(axis=1, keepdims=True)) / temperature)
    w = e / e.sum(axis=1, keepdims=True)
    return np.where(ok[:, None], w, one_hot(np.full(len(scores), FULL), scores.shape[1]))


# ---------- объединение прогнозов ----------


def pick(q: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Квантили выбранной длины: (окна, 9, H)."""
    return q[np.arange(len(q)), idx]


def vincent(q: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Взвешенное среднее ОДНОИМЁННЫХ квантилей («среднее квантилей»). Это НЕ смесь распределений."""
    return np.einsum("nk,nkqh->nqh", weights, q.astype(np.float64))


def _cdf(xp: np.ndarray, taus: np.ndarray, count: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Кусочно-линейная CDF в точках u по числу узлов count (слева от u или в u).

    xp — (M, 11) узлы одного кандидата, count — (M, U) в 0…11. При 1 ≤ count ≤ 10 точка лежит
    на отрезке между узлами count−1 и count, длина которого строго положительна.
    """
    c = np.clip(count, 1, xp.shape[1] - 1)
    x0 = np.take_along_axis(xp, c - 1, axis=1)
    x1 = np.take_along_axis(xp, c, axis=1)
    t0, t1 = taus[c - 1], taus[c]
    with np.errstate(divide="ignore", invalid="ignore"):
        f = t0 + (u - x0) / (x1 - x0) * (t1 - t0)
    return np.where(count == 0, 0.0, np.where(count == xp.shape[1], 1.0, f))


def mixture_quantiles(q: np.ndarray, weights: np.ndarray, levels_in: list[float],
                      levels_out: list[float], chunk: int = 50_000) -> tuple[np.ndarray, int]:
    """Квантили СМЕСИ распределений кандидатов: решение Σ w_L F_L(y) = τ (протокол, разд. 4.2).

    1) Квантили каждого кандидата упорядочиваются (в кеше бывают малые нарушения порядка).
    2) F_L — кусочно-линейная CDF через (q_k, τ_k) с хвостами до τ = 0 и 1 по наклону соседнего
       отрезка; совпадающие квантили дают скачок (точечную массу).
    3) G(y) = Σ w_L F_L(y) кусочно-линейна между узлами всех кандидатов; квантиль уровня τ —
       inf{y : G(y) ≥ τ}, считается точно, без численного поиска.

    q — (окна, K, 9, H), weights — (окна, K). Возвращает ((окна, len(levels_out), H), число
    окон, где пришлось упорядочивать квантили).
    """
    n, K, nq, H = q.shape
    taus = np.r_[0.0, levels_in, 1.0]
    out_levels = np.asarray(levels_out, dtype=np.float64)
    if not ((out_levels > 0) & (out_levels < 1)).all():
        raise ValueError("output levels must lie strictly inside (0, 1)")
    # Каждое распределение — пара (окно, шаг горизонта): переставляем в (окна·H, K, 9).
    dist = np.moveaxis(q, 3, 1).reshape(n * H, K, nq).astype(np.float64)
    crossed = (np.diff(dist, axis=2) < 0).any(axis=(1, 2)).reshape(n, H).any(axis=1)
    w_all = np.repeat(weights.astype(np.float64), H, axis=0)
    out = np.empty((n * H, len(out_levels)))
    for a in range(0, n * H, chunk):
        d = np.sort(dist[a:a + chunk], axis=2)
        w = w_all[a:a + chunk]
        lo = d[:, :, :1] - (d[:, :, 1:2] - d[:, :, :1])
        hi = d[:, :, -1:] + (d[:, :, -1:] - d[:, :, -2:-1])
        xp = np.concatenate([lo, d, hi], axis=2)  # (M, K, 11)
        M, U = len(d), K * xp.shape[2]
        flat = xp.reshape(M, U)
        order = np.argsort(flat, axis=1, kind="stable")
        u = np.take_along_axis(flat, order, axis=1)  # (M, U) все узлы всех кандидатов по возрастанию
        label = order // xp.shape[2]  # какому кандидату принадлежит узел
        # Границы серий равных значений: число узлов кандидата «≤ u» берётся в конце серии, «< u» — до её начала.
        cols = np.broadcast_to(np.arange(U), (M, U))
        starts = np.ones((M, U), bool)
        starts[:, 1:] = u[:, 1:] != u[:, :-1]
        ends = np.ones((M, U), bool)
        ends[:, :-1] = starts[:, 1:]
        run_start = np.maximum.accumulate(np.where(starts, cols, 0), axis=1)
        run_end = np.minimum.accumulate(np.where(ends, cols, U - 1)[:, ::-1], axis=1)[:, ::-1]
        g_right = np.zeros_like(u)
        g_left = np.zeros_like(u)
        for k in range(K):
            mine = label == k
            cs = np.cumsum(mine, axis=1)
            cnt_le = np.take_along_axis(cs, run_end, axis=1)  # узлов кандидата ≤ u: CDF справа
            cnt_lt = (np.take_along_axis(cs, run_start, axis=1)
                      - np.take_along_axis(mine, run_start, axis=1))  # узлов < u: предел слева
            g_right += w[:, k:k + 1] * _cdf(xp[:, k], taus, cnt_le, u)
            g_left += w[:, k:k + 1] * _cdf(xp[:, k], taus, cnt_lt, u)
        rows = np.arange(len(d))
        for j, tau in enumerate(out_levels):
            # Первый узел, где G (справа) уже ≥ τ. При совпадающих узлах берётся первый из них.
            reached = g_right >= tau - 1e-12  # допуск — только на округление суммы весов
            js = np.where(reached.any(axis=1), np.argmax(reached, axis=1), u.shape[1] - 1)
            prev = np.maximum(js - 1, 0)
            u1, u0 = u[rows, js], u[rows, prev]
            gl1, gr0 = g_left[rows, js], g_right[rows, prev]
            on_segment = (js > 0) & (gl1 >= tau)  # τ достигается на линейном участке до узла
            with np.errstate(divide="ignore", invalid="ignore"):
                y = u0 + (tau - gr0) / (gl1 - gr0) * (u1 - u0)
            out[a:a + chunk, j] = np.where(on_segment, y, u1)  # иначе — скачок в узле u1
    return np.moveaxis(out.reshape(n, H, len(out_levels)), 1, 2), int(crossed.sum())
