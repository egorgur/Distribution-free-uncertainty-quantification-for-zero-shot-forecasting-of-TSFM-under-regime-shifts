"""Окна скользящей оценки (rolling origin) и альтернативные контексты для S4/S7.

Соглашение об индексах (0-based):
    τ — индекс первой точки нового режима;
    t — момент начала прогноза = индекс ПОСЛЕДНЕЙ точки контекста;
    контекст = x[t − L + 1 : t + 1], будущее = x[t + 1 : t + 1 + H];
    d = t − τ.
Тогда:
    d < −H        — и контекст, и горизонт целиком до смены;
    −H ≤ d < 0    — смена внутри горизонта, в контексте её нет;
    0 ≤ d         — в контексте d + 1 точек нового режима;
    d ≥ L − 1     — старого режима в контексте нет (в CLAUDE.md граница округлена до d ≥ L).

Варианты контекста одного окна (конец у всех — момент t, поэтому прогнозы сравнимы):
    L512           — основной;
    L256, L128, …  — S4, длина в разы меньше;
    tail-9 … tail+9 — S7, длина L + k (начало сдвигается на k точек);
    noise1, …      — S7, основной контекст + слабый шум.
"""

from __future__ import annotations

import zlib

import numpy as np


def window_ends(tau: int, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Моменты t и соответствующие d = t − τ для всех окон одного ряда."""
    w = cfg["windows"]
    d = np.arange(w["d_min"], w["d_max"] + 1, w["d_step"])
    return tau + d, d


def variant_names(cfg: dict) -> list[str]:
    v, L = cfg["variants"], cfg["windows"]["context"]
    return (
        [f"L{L}"]
        + [f"L{n}" for n in v["s4_lengths"]]
        + [f"tail{k:+d}" for k in v["s7_tail_k"]]
        + [f"noise{i + 1}" for i in range(v["s7_noise_draws"])]
    )


def noise_rng(seed: int, cell_id: str, series: int, t: int) -> np.random.Generator:
    """Свой генератор шума S7 на каждое окно: окно можно восстановить отдельно от остальных."""
    return np.random.default_rng([seed, zlib.crc32(cell_id.encode()), series, t, 7])


def window_contexts(x: np.ndarray, t: int, cfg: dict, rng: np.random.Generator) -> list[np.ndarray]:
    """Все варианты контекста окна с концом в t, в порядке variant_names."""
    v, L = cfg["variants"], cfg["windows"]["context"]
    end = t + 1
    base = x[end - L : end]
    ctxs = [base]
    ctxs += [x[end - n : end] for n in v["s4_lengths"]]
    for k in v["s7_tail_k"]:
        if end - (L + k) < 0:
            raise ValueError(f"окну t={t} не хватает истории для L{k:+d}")
        ctxs.append(x[end - (L + k) : end])
    sigma = np.std(np.diff(base))
    ctxs += [base + rng.normal(0, v["s7_noise_alpha"] * sigma, L) for _ in range(v["s7_noise_draws"])]
    return ctxs
