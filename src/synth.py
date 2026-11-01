"""Генератор синтетических рядов с известной точкой смены (этап 1).

Модель ряда до масштабирования (σ = 1 — стационарное std шума):

    x_t = u_t + A_t · sin(θ_t) + c_t + bias_t

    u_t = φ_t · u_{t−1} + (1 − φ_t) · m_t + ε_t,   ε_t ~ N(0, s_t² · (1 − φ_t²))
          AR(1)-шум вокруг уровня m_t; в установившемся режиме std(u) = s_t при любом φ_t.
    θ_t = θ_0 + 2π · (f_1 + … + f_t)   фаза сезонности с частотой f_t; фаза без скачков.
    c_t = b_1 + … + b_t                тренд как накопленный наклон b_t; уровень без скачка.
    bias_t                             смещение показаний датчика (мгновенное).

Смена в точке τ (индекс первой точки нового режима) меняет один параметр:
    mean → m, variance → s, trend → b, ar → φ, seasonality → A, frequency → f, sensor_bias → bias.
Класс change меняет параметр скачком, drift — линейно за duration шагов.
Класс anomaly портит сами наблюдения (выброс, «залипание»), процесс не меняется.
Класс none — ряд без изменений.

mean и sensor_bias различаются только переходом: уровень процесса m доходит до нового
значения через AR-динамику (при φ = 0.7 за ~10 шагов на 97%), смещение показаний — сразу.

Итоговый ряд: offset + scale · x. Силы смены заданы в σ до масштабирования.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np

LEVELS = ("weak", "medium", "strong")
CHANGE_KINDS = ("mean", "variance", "trend", "ar", "seasonality", "frequency", "sensor_bias")
ANOMALY_KINDS = ("spike", "freeze")
TREND_SPAN = 64  # сила trend: прирост уровня (σ) за столько шагов


@dataclass(frozen=True)
class Cell:
    """Ячейка сетки факторов: класс изменения × тип × сила."""

    cls: str  # change / drift / anomaly / none
    kind: str  # тип: mean, variance, …, spike, freeze; у none — "none"
    level: str  # weak / medium / strong; у none — "-"
    value: float  # сила в единицах конфига; у none — NaN

    @property
    def cell_id(self) -> str:
        return "none" if self.cls == "none" else f"{self.cls}-{self.kind}-{self.level}"


def grid(cfg: dict) -> list[Cell]:
    """Все ячейки сетки из конфига, в фиксированном порядке."""
    allowed = {"change": CHANGE_KINDS, "drift": CHANGE_KINDS, "anomaly": ANOMALY_KINDS}
    cells = []
    for cls, kinds in allowed.items():
        for kind, values in cfg.get(cls, {}).items():
            if kind == "duration":
                continue
            if kind not in kinds:
                raise ValueError(f"{cls}: неизвестный тип {kind}")
            if len(values) != len(LEVELS):
                raise ValueError(f"{cls}.{kind}: нужно {len(LEVELS)} значения силы")
            cells += [Cell(cls, kind, level, float(v)) for level, v in zip(LEVELS, values)]
    if cfg.get("none"):
        cells.append(Cell("none", "none", "-", float("nan")))
    return cells


def layout(cfg: dict) -> tuple[int, int]:
    """Точка смены τ и длина ряда N, при которых все окна сетки d помещаются в ряд."""
    w, v = cfg["windows"], cfg["variants"]
    longest = w["context"] + max([0, *v["s7_tail_k"]])  # самый длинный вариант контекста
    tau = longest - 1 - w["d_min"]  # самое раннее окно t = τ + d_min начинается с индекса 0
    length = tau + w["d_max"] + max(w["horizons"]) + 1  # самое позднее окно: будущее до N − 1
    return tau, length


def series_rng(seed: int, key: str, i: int) -> np.random.Generator:
    """Генератор случайных чисел ряда; key — обычно cell_id.

    crc32 (а не hash) не меняется между запусками Python; добавление новых ячеек
    в конфиг не меняет ряды уже существующих.
    """
    return np.random.default_rng([seed, zlib.crc32(key.encode()), i])


def generate(cell: Cell, i: int, cfg: dict, rng: np.random.Generator | None = None) -> tuple[np.ndarray, dict]:
    """i-й ряд ячейки: (ряд длины N, параметры ряда до смены)."""
    tau, n = layout(cfg)
    rng = series_rng(cfg["seed"], cell.cell_id, i) if rng is None else rng
    b = cfg["base"]

    # Параметры ряда до смены. Порядок вызовов rng не менять: от него зависят ряды.
    phi0 = rng.uniform(*b["ar_phi"])
    amp0 = rng.uniform(*b["season_amp"])
    period0 = int(rng.choice(b["season_period"]))
    theta0 = rng.uniform(0, 2 * np.pi)
    scale = 10 ** rng.uniform(*b["log10_scale"])
    offset = scale * rng.uniform(*b["offset"])
    eps = rng.standard_normal(n)
    sign = rng.choice([-1.0, 1.0])  # знак выброса

    # Доля пути от старого значения параметра к новому: 0 до τ, 1 после смены.
    t = np.arange(n)
    if cell.cls == "change":
        ramp = (t >= tau).astype(float)
    elif cell.cls == "drift":
        ramp = np.clip((t - tau + 1) / cfg["drift"]["duration"], 0.0, 1.0)
    else:
        ramp = np.zeros(n)

    level, std, slope, bias = np.zeros(n), np.ones(n), np.zeros(n), np.zeros(n)
    phi, amp, freq = np.full(n, phi0), np.full(n, amp0), np.full(n, 1.0 / period0)
    k = cell.value
    if cell.cls in ("change", "drift"):
        if cell.kind == "mean":
            level = k * ramp
        elif cell.kind == "variance":
            std = 1 + (k - 1) * ramp
        elif cell.kind == "trend":
            slope = k / TREND_SPAN * ramp
        elif cell.kind == "ar":
            phi = phi0 - k * ramp
        elif cell.kind == "seasonality":
            amp = amp0 + k * ramp
        elif cell.kind == "frequency":
            freq = (1 + (k - 1) * ramp) / period0
        elif cell.kind == "sensor_bias":
            bias = k * ramp

    # AR(1) вокруг уровня; старт сразу из стационарного распределения.
    u = np.empty(n)
    u[0] = level[0] + std[0] * eps[0]
    for j in range(1, n):
        u[j] = phi[j] * u[j - 1] + (1 - phi[j]) * level[j] + std[j] * np.sqrt(1 - phi[j] ** 2) * eps[j]

    theta = theta0 + 2 * np.pi * np.cumsum(freq)
    x = u + amp * np.sin(theta) + np.cumsum(slope) + bias

    if cell.cls == "anomaly":
        if cell.kind == "spike":
            x[tau] += sign * k
        elif cell.kind == "freeze":
            x[tau : tau + int(k)] = x[tau - 1]

    params = {"phi": phi0, "season_amp": amp0, "season_period": period0, "scale": scale, "offset": offset}
    return offset + scale * x, params
