"""Описательный аудит Petrobras 3W (протокол docs/protocol_3w_audit.md).

Только чистые функции на numpy/pandas: периоды меток, структура события, пригодность переменных,
агрегирование по блокам и достаточность длин для окон протокола H1/H3 v1. Без моделей.

Термины. Период — максимальный подряд идущий участок с одной меткой class (NaN — «неизвестно»).
Событие — экземпляр с переходной меткой 101–109; τ — первое наблюдение первого перехода.
Старый режим — период 0 прямо перед τ; новый — устойчивое состояние события (1–9) после перехода.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

NAN_LABEL = -1
FROZEN_RUN = 60
INTERP_TOL = 1e-9
SCALES = (1, 10, 60)
CONFIGS = ((512, 64), (128, 16))
REFERENCE_END = 640  # граница нового эталона протокола v1: L ≤ d ≤ 640


def periods(labels: np.ndarray) -> list[tuple[int, int, int]]:
    """Периоды (метка, начало, длина); NaN → NAN_LABEL."""
    lab = np.where(np.isnan(labels.astype(float)), NAN_LABEL, labels).astype(int)
    if len(lab) == 0:
        return []
    cut = np.flatnonzero(np.diff(lab) != 0) + 1
    starts = np.r_[0, cut]
    ends = np.r_[cut, len(lab)]
    return [(int(lab[s]), int(s), int(e - s)) for s, e in zip(starts, ends)]


def is_transient(label: int) -> bool:
    return 101 <= label <= 109


def is_event_steady(label: int) -> bool:
    return 1 <= label <= 9


@dataclass(frozen=True)
class Event:
    structure: str        # "0>T>S", "0>T", "T>S", "T", "0 only", "S only", "other"
    tau: int              # индекс τ или −1
    old_len: int          # период 0 прямо перед τ (0, если нет)
    trans_len: int        # первый переходный период
    new_len: int          # устойчивое состояние сразу после перехода (0, если нет)
    new_start: int        # индекс начала нового режима или −1
    total_from_tau: int   # наблюдений от τ до конца экземпляра


def event_of(labels: np.ndarray) -> Event:
    """Разбор структуры экземпляра по меткам class."""
    ps = periods(labels)
    n = int(len(labels))
    kinds = [p[0] for p in ps]
    i_t = next((i for i, k in enumerate(kinds) if is_transient(k)), None)
    if i_t is None:
        if kinds and all(k in (0, NAN_LABEL) for k in kinds) and 0 in kinds:
            return Event("0 only", -1, 0, 0, 0, -1, 0)
        if any(is_event_steady(k) for k in kinds):
            return Event("S only", -1, 0, 0, 0, -1, 0)
        return Event("other", -1, 0, 0, 0, -1, 0)
    lab_t, tau, trans = ps[i_t]
    old = ps[i_t - 1][2] if i_t > 0 and ps[i_t - 1][0] == 0 else 0
    nxt = ps[i_t + 1] if i_t + 1 < len(ps) else None
    new_len, new_start = (nxt[2], nxt[1]) if nxt and is_event_steady(nxt[0]) else (0, -1)
    struct = ("0>" if old else "") + "T" + (">S" if new_len else "")
    return Event(struct, tau, old, trans, new_len, new_start, n - tau)


def blocks(n_obs: int, k: int) -> int:
    """Число полных непересекающихся блоков по k наблюдений."""
    return n_obs // k


def sufficient(ev: Event, k: int, L: int, H: int) -> dict[str, bool]:
    """Достаточность длин события при агрегировании блоками k и окне (L, H).

    old: старый режим ≥ L + 2H блоков; span: от τ ≥ 640 + H блоков; new: новый режим ≥ L + H блоков.
    """
    old = blocks(ev.old_len, k) >= L + 2 * H
    span = blocks(ev.total_from_tau, k) >= REFERENCE_END + H
    new = blocks(ev.new_len, k) >= L + H
    return {"old": old, "span": span, "new": new, "all": old and span and new}


def usable(values: np.ndarray) -> bool:
    """Переменная пригодна в периоде: нет NaN и больше одного значения."""
    v = values.astype(float)
    return bool(len(v) > 0 and np.isfinite(v).all() and np.unique(v).size > 1)


def frozen_share(values: np.ndarray, run: int = FROZEN_RUN) -> float:
    """Доля наблюдений в сериях ≥ run подряд одинаковых конечных значений."""
    v = values.astype(float)
    if len(v) == 0:
        return float("nan")
    same = np.r_[False, (v[1:] == v[:-1]) & np.isfinite(v[1:])]
    total, cur = 0, 1
    for s in same[1:]:
        if s:
            cur += 1
        else:
            if cur >= run:
                total += cur
            cur = 1
    if cur >= run:
        total += cur
    return total / len(v)


def interp_share(values: np.ndarray, tol: float = INTERP_TOL) -> float:
    """Доля шагов с |Δ²x| ≤ tol · медиана |Δx| — мера кусочно-линейности (интерполяция PI)."""
    v = values.astype(float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return float("nan")
    d1 = np.abs(np.diff(v))
    scale = np.median(d1)
    if scale <= 0:
        return float("nan")
    return float((np.abs(np.diff(v, 2)) <= tol * scale).mean())


def time_check(index: pd.Index) -> dict:
    """Шаг времени: медиана, доля шагов ровно 1 с, дубли, монотонность."""
    t = pd.DatetimeIndex(index)
    # В pandas 3 единица хранения зависит от данных (с, мс, мкс, нс) — явно переводим в наносекунды.
    d = np.diff(t.values.astype("datetime64[ns]").astype(np.int64)) / 1e9
    return {"n": len(t), "monotonic": bool(t.is_monotonic_increasing), "duplicates": int(t.duplicated().sum()),
            "median_step_s": float(np.median(d)) if len(d) else float("nan"),
            "share_step_1s": float((d == 1).mean()) if len(d) else float("nan"),
            "max_step_s": float(d.max()) if len(d) else float("nan"),
            "start": str(t[0]) if len(t) else "", "end": str(t[-1]) if len(t) else ""}
