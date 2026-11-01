"""Этап B: локализация точки смены в контексте (H4). Определения — журнал решений docs/decisions.md, 2026-11-01.

Детектор ищет одну точку смены τ′ в основном контексте (L = 512) или отвечает «смены нет».
Меняется только представление контекста, детектор и настройки одни:

    представление                                       точки       детектор
    уровни энкодера TSFM (вход, блоки, final)           32 патча    split_rbf (абляция — split_l2)
    р1  нормированные значения патча                    32 патча    split_rbf
    р2  статистики патча: среднее, std, наклон, ACF(1)  32 патча    split_rbf
    р3  нормированные значения по точкам                512 точек   pelt_path (PELT с rbf, точно)
    р4  нормированные значения по точкам                512 точек   split_cusum (CUSUM среднего)

Соглашение об индексах: k — номер первой точки правого сегмента В ТОЧКАХ контекста (0 … L − 1),
у патчей k = 16 · (номер первого патча справа). Точка контекста k — это момент t − L + 1 + k ряда.
Минимальный сегмент — 32 точки (2 патча), поэтому k ∈ [32, L − 32].

Стоимость rbf — как CostRbf в ruptures 1.1.10: ядро exp(−clip(γ·‖a − b‖², 0.01, 100)) вне диагонали
и 1 на диагонали, γ = 1 / медиана квадратов попарных расстояний (1, если медиана 0). Стоимость
отрезка [a, b): b − a − (сумма ядра по отрезку) / (b − a).

Все функции векторные по окнам (первая ось), кроме pelt_path — она для одного окна
(pelt_path_many — для набора окон в несколько процессов).
"""

from __future__ import annotations

import numpy as np

PATCH = 16
MIN_POINTS = 32  # минимальный сегмент, точек
MIN_PATCHES = MIN_POINTS // PATCH
MAX_CHANGES = 15  # больше при L = 512 и сегменте 32 не помещается


# ---------- представления сырого контекста ----------


def normalize_context(ctx: np.ndarray) -> np.ndarray:
    """Контекст на своё среднее и std: (окна, L) → (окна, L). Постоянный контекст → нули."""
    ctx = np.asarray(ctx, dtype=np.float64)
    sd = ctx.std(axis=1, keepdims=True)
    return (ctx - ctx.mean(axis=1, keepdims=True)) / np.where(sd > 0, sd, 1.0)


def raw_patches(z: np.ndarray) -> np.ndarray:
    """р1: нормированные значения патча, (окна, L) → (окна, L / 16, 16)."""
    return z.reshape(len(z), -1, PATCH)


def patch_stats(z: np.ndarray) -> np.ndarray:
    """р2: статистики патча (окна, L / 16, 4): среднее, std, изменение за патч, ACF(1).

    Изменение за патч — наклон МНК-прямой × 15 (разница значений прямой в начале и в конце патча).
    ACF(1) — автокорреляция с лагом 1 внутри патча; у постоянного патча 0.
    Каждая статистика затем нормируется на своё среднее и std по патчам окна (standardize_columns),
    иначе расстояние определяла бы статистика с наибольшим разбросом.
    """
    p = raw_patches(z)
    mean = p.mean(axis=2)
    dev = p - mean[..., None]
    var = (dev**2).mean(axis=2)
    pos = np.arange(PATCH) - (PATCH - 1) / 2
    slope = (dev * pos).sum(axis=2) / (pos**2).sum()
    lag = (dev[..., 1:] * dev[..., :-1]).mean(axis=2)
    acf = np.divide(lag, var, out=np.zeros_like(lag), where=var > 1e-12)
    stats = np.stack([mean, np.sqrt(var), slope * (PATCH - 1), acf], axis=-1)
    return standardize_columns(stats)


def standardize_columns(tokens: np.ndarray) -> np.ndarray:
    """Каждая координата — на своё среднее и std по точкам окна: (окна, n, D) → (окна, n, D).

    Координаты, которые в окне почти не меняются (std < 1e-6 · медианной std координат или 0),
    обнуляются: деление на почти ноль раздуло бы шум округления.
    """
    t = np.asarray(tokens, dtype=np.float64)
    mu, sd = t.mean(axis=1, keepdims=True), t.std(axis=1, keepdims=True)
    floor = np.maximum(1e-6 * np.median(sd, axis=2, keepdims=True), 1e-12)
    return np.where(sd > floor, (t - mu) / np.where(sd > floor, sd, 1.0), 0.0)


# ---------- ядро rbf ----------


def sq_distances(tokens: np.ndarray) -> np.ndarray:
    """Квадраты попарных расстояний между точками окна: (окна, n, D) → (окна, n, n), float64."""
    t = np.asarray(tokens, dtype=np.float64)
    if t.shape[2] == 1:  # одномерный ряд: точно, без вычитания больших чисел
        v = t[..., 0]
        return (v[:, :, None] - v[:, None, :]) ** 2
    norm = (t**2).sum(axis=2)
    d2 = norm[:, :, None] + norm[:, None, :] - 2 * np.einsum("wid,wjd->wij", t, t)
    d2 = np.maximum(d2, 0.0)
    idx = np.arange(t.shape[1])
    d2[:, idx, idx] = 0.0
    return d2


def rbf_gram(tokens: np.ndarray) -> np.ndarray:
    """Матрица ядра rbf как в CostRbf (ruptures 1.1.10): (окна, n, D) → (окна, n, n)."""
    d2 = sq_distances(tokens)
    n = d2.shape[1]
    iu = np.triu_indices(n, 1)
    med = np.median(d2[:, iu[0], iu[1]], axis=1)
    gamma = np.where(med > 0, 1.0 / np.where(med > 0, med, 1.0), 1.0)
    gram = np.exp(-np.clip(gamma[:, None, None] * d2, 1e-2, 1e2))
    idx = np.arange(n)
    gram[:, idx, idx] = 1.0  # в CostRbf диагональ не обрезается: exp(0) = 1
    return gram


def _prefix2d(gram: np.ndarray) -> np.ndarray:
    """P[a, b] = сумма gram[:a, :b]; (окна, n, n) → (окна, n + 1, n + 1)."""
    w, n, _ = gram.shape
    p = np.zeros((w, n + 1, n + 1))
    p[:, 1:, 1:] = gram.cumsum(axis=1).cumsum(axis=2)
    return p


# ---------- детекторы одного разбиения ----------


def _split_from_inner(inner: np.ndarray, total: np.ndarray, n: int, min_seg: int, unit: int):
    """inner[:, j] — Σ внутри левого / |левого| + Σ внутри правого / |правого| для разбиения
    j = 1 … n − 1. Уменьшение стоимости = inner − total / n. Возвращает (счёт / n, k в точках)."""
    gain = inner - total[:, None] / n
    valid = np.arange(1, n)
    valid = (valid >= min_seg) & (valid <= n - min_seg)
    gain = np.where(valid, gain, -np.inf)
    j = gain.argmax(axis=1)
    return gain[np.arange(len(gain)), j] / n, (j + 1) * unit


def split_rbf(tokens: np.ndarray, unit: int = PATCH) -> tuple[np.ndarray, np.ndarray]:
    """Лучшее разбиение окна на две части по стоимости rbf.

    tokens: (окна, n, D); unit — точек контекста на одну точку tokens (16 у патчей, 1 у точек).
    Возвращает (счёт, k): счёт — уменьшение стоимости rbf при лучшем разбиении, делённое на n;
    k — первая точка правого сегмента в точках контекста. Минимальный сегмент — 32 точки.
    """
    gram = rbf_gram(tokens)
    n = gram.shape[1]
    p = _prefix2d(gram)
    j = np.arange(1, n)
    left = p[:, j, j] / j  # сумма ядра внутри [0, j) / j
    right = (p[:, n, n][:, None] - 2 * p[:, j, n] + p[:, j, j]) / (n - j)  # внутри [j, n) / (n − j); ядро симметрично
    return _split_from_inner(left + right, p[:, n, n], n, MIN_POINTS // unit, unit)


def split_l2(tokens: np.ndarray, unit: int = PATCH) -> tuple[np.ndarray, np.ndarray]:
    """Абляция: лучшее разбиение по стоимости l2 (сдвиг среднего) после нормировки каждой координаты
    на её std по точкам окна. Счёт — уменьшение суммы квадратов отклонений, делённое на n · D."""
    z = standardize_columns(tokens)
    w, n, dim = z.shape
    c = z.cumsum(axis=1)[:, :-1]  # суммы левых частей при разбиении j = 1 … n − 1
    total = z.sum(axis=1)
    j = np.arange(1, n)[None, :, None]
    # Уменьшение SSE при разбиении: |Σ_left|² / j + |Σ_right|² / (n − j) − |Σ|² / n
    inner = ((c**2) / j).sum(axis=2) + (((total[:, None] - c) ** 2) / (n - j)).sum(axis=2)
    score, k = _split_from_inner(inner, (total**2).sum(axis=1) * 1.0, n, MIN_POINTS // unit, unit)
    return score / dim, k


def split_cusum(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """р4: CUSUM одного сдвига среднего по точкам, как S3 (signals.cusum_statistic), но разбиение
    не ближе 32 точек к краям. z: (окна, L). Возвращает (max |S_k| / (σ̂ √L), k)."""
    n = z.shape[1]
    dev = z - z.mean(axis=1, keepdims=True)
    s = np.abs(np.cumsum(dev, axis=1))[:, :-1]  # S после точки j − 1, то есть разбиение j = 1 … n − 1
    j = np.arange(1, n)
    s = np.where((j >= MIN_POINTS) & (j <= n - MIN_POINTS), s, -np.inf)
    best = s.argmax(axis=1)
    sd = z.std(axis=1)
    score = s[np.arange(len(z)), best] / (np.where(sd > 0, sd, 1.0) * np.sqrt(n))
    return score, best + 1


# ---------- PELT (р3), точно через динамическое программирование ----------


def pelt_path(z: np.ndarray, max_changes: int = MAX_CHANGES) -> tuple[np.ndarray, np.ndarray]:
    """Для одного окна z (L,): наименьшая стоимость rbf разбиения с m = 0 … max_changes точками смены
    и последняя точка смены такого разбиения.

    Возвращает (cost (max_changes + 1,), last (max_changes + 1,)); last[0] = 0 (смены нет).
    PELT со штрафом β выдаёт разбиение с наименьшим cost[m] + β·m (ruptures штрафует каждый
    сегмент, что отличается на постоянное β), поэтому его ответ при любом β берётся отсюда:
    pelt_decision. Минимальный сегмент — 32 точки, в том числе первый и последний.
    """
    n, s = len(z), MIN_POINTS
    p = _prefix2d(rbf_gram(z[None, :, None]))[0]
    a = np.arange(n + 1)
    diag = p[a, a]
    # Сумма ядра по квадрату [a, b) × [a, b) = P[b, b] − P[a, b] − P[b, a] + P[a, a].
    block = diag[None, :] - p - p.T + diag[:, None]
    seg = (a[None, :] - a[:, None]).astype(float)  # b − a
    with np.errstate(divide="ignore", invalid="ignore"):
        cost = seg - block / seg  # cost[a, b] — стоимость отрезка [a, b)
    cost[seg < s] = np.inf  # допустимы только отрезки не короче s
    best = np.empty(max_changes + 1)
    last = np.zeros(max_changes + 1, dtype=np.int64)
    f = cost[0].copy()  # f[j] — лучшая стоимость [0, j) с m точками смены
    best[0] = f[n]
    for m in range(1, max_changes + 1):
        total = f[:, None] + cost  # [последняя точка смены a, конец j]
        arg = total.argmin(axis=0)
        f = total[arg, np.arange(n + 1)]
        best[m], last[m] = f[n], arg[n]
    return best, last


def _pelt_worker(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    out = [pelt_path(row) for row in z]
    return np.stack([o[0] for o in out]), np.stack([o[1] for o in out])


def pelt_path_many(z: np.ndarray, processes: int = 1, chunk: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """pelt_path для всех окон z (окна, L): (cost (окна, 16), last (окна, 16))."""
    parts = [z[a : a + chunk] for a in range(0, len(z), chunk)]
    if processes > 1:
        import multiprocessing as mp

        with mp.get_context("fork").Pool(processes) as pool:
            out = pool.map(_pelt_worker, parts)
    else:
        out = [_pelt_worker(p) for p in parts]
    return np.concatenate([o[0] for o in out]), np.concatenate([o[1] for o in out])


def pelt_critical_penalty(cost: np.ndarray) -> np.ndarray:
    """Наибольший штраф β, при котором PELT ещё находит смену: max_m (C₀ − C_m) / m. (окна, 16) → (окна,)."""
    m = np.arange(1, cost.shape[1])
    return ((cost[:, :1] - cost[:, 1:]) / m).max(axis=1)


def pelt_decision(cost: np.ndarray, last: np.ndarray, beta: float) -> tuple[np.ndarray, np.ndarray]:
    """Ответ PELT со штрафом β: (число точек смены m, последняя точка смены k; 0 при m = 0)."""
    m = (cost + beta * np.arange(cost.shape[1])).argmin(axis=1)
    return m, last[np.arange(len(m)), m]


# ---------- разметка окон этапа B ----------

POSITIVE, NEGATIVE, ANOMALY, EXCLUDED = "смена", "без смены", "аномалия", "вне оценки"


def window_zones(w, L: int, drift_duration: int) -> np.ndarray:
    """Зона каждого окна (таблица windows.csv) по журналу решений:
    «смена» — change и drift при 0 ≤ d < L − 1;
    «без смены» — d < 0 любого класса, весь none, change при d ≥ L − 1, drift при d ≥ L − 1 + duration,
                  spike при d ≥ L, freeze при d ≥ L − 1 + k;
    «аномалия» — spike и freeze, пока аномалия в контексте;
    «вне оценки» — drift при L − 1 ≤ d < L − 1 + duration (в контексте только часть подъёма).
    """
    cls, kind, d = w.cls.to_numpy(), w.kind.to_numpy(), w.d.to_numpy()
    value = w.value.to_numpy()
    zone = np.full(len(w), EXCLUDED, dtype=object)
    zone[(cls == "none") | (d < 0)] = NEGATIVE
    in_ctx = (d >= 0) & (d < L - 1)
    zone[np.isin(cls, ["change", "drift"]) & in_ctx] = POSITIVE
    zone[(cls == "change") & (d >= L - 1)] = NEGATIVE
    zone[(cls == "drift") & (d >= L - 1 + drift_duration)] = NEGATIVE
    spike, freeze = (cls == "anomaly") & (kind == "spike"), (cls == "anomaly") & (kind == "freeze")
    zone[spike & (d >= 0) & (d < L)] = ANOMALY
    zone[spike & (d >= L)] = NEGATIVE
    with np.errstate(invalid="ignore"):
        gone = d >= L - 1 + value  # freeze: value = длина залипания
    zone[freeze & (d >= 0) & ~gone] = ANOMALY
    zone[freeze & gone] = NEGATIVE
    return zone


def change_offset(k: np.ndarray, d: np.ndarray, L: int) -> np.ndarray:
    """τ′ − τ для точки k контекста окна с данным d: точка k — момент t − L + 1 + k, t = τ + d."""
    return d - L + 1 + k


def is_hit(k: np.ndarray, d: np.ndarray, cls: np.ndarray, L: int, delta: int, drift_duration: int) -> np.ndarray:
    """Попадание: |τ′ − τ| ≤ δ; у drift — τ′ ∈ [τ − δ, τ + duration + δ]."""
    off = change_offset(k, d, L)
    upper = np.where(cls == "drift", drift_duration, 0)
    return (off >= -delta) & (off <= upper + delta)
