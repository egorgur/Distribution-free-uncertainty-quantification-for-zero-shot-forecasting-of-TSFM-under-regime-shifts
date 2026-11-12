"""Синтетические контроли мощности исходного экрана; газ сюда не поступает."""

from statistics import NormalDist

import numpy as np
import pandas as pd

from src.gas_reference import BLOCK_SECONDS, SENSORS, passes

PHIS = (0., .5, .9, .98)
CONDITIONS = ("stationary", "linear_0.5", "linear_2", "mean_step_1", "scale_step_2")
PANELS = 100
DATA_SEED = 20261113
BOOTSTRAP_SEED = 20261114
CRITERIA = {"primary": (.25, 1.5), "sensitivity": (.5, 2.)}


def stationary_ar(phi, panel, phi_index, length=120, channels=16):
    """Точное стационарное начальное распределение N(0,1), без burn-in."""
    if not 0 <= phi < 1 or length < 2 or channels < 1:
        raise ValueError("require 0<=phi<1, length>=2, channels>=1")
    rng = np.random.default_rng(np.random.SeedSequence([DATA_SEED, phi_index, panel]))
    innovations = rng.normal(size=(length, channels))
    x = np.empty_like(innovations)
    x[0] = innovations[0]
    for t in range(1, length):
        x[t] = phi * x[t - 1] + np.sqrt(1 - phi * phi) * innovations[t]
    return x


def transform(base, condition):
    """Эффект выражен в исходной std=1; входной массив не меняется."""
    x = np.array(base, dtype=float, copy=True)
    if x.shape != (120, 16):
        raise ValueError("expected (120,16) panel")
    if condition == "stationary":
        pass
    elif condition in ("linear_0.5", "linear_2"):
        magnitude = .5 if condition == "linear_0.5" else 2.
        x += np.linspace(0, magnitude, len(x))[:, None]
    elif condition == "mean_step_1":
        x[60:] += 1
    elif condition == "scale_step_2":
        x[60:] *= 2
    else:
        raise ValueError(condition)
    return x


def channel_outcomes(diagnostics):
    """Обе длины обязательны, решения по каналам, без ложной независимости."""
    if set(diagnostics.sensor) != set(SENSORS):
        raise ValueError("all 16 channels required")
    rows = []
    for sensor, group in diagnostics.groupby("sensor", sort=True):
        if sorted(group.block_s.tolist()) != list(BLOCK_SECONDS):
            raise ValueError("both block lengths required exactly once")
        regular = bool((group.status == "ok").all())
        for criterion, (mean_margin, variance_ratio) in CRITERIA.items():
            accepted = all(passes(r.mean_upper, r.log_variance_upper,
                                  mean_margin, variance_ratio) for r in group.itertuples())
            point_only = all(passes(r.mean_spread, r.log_variance_spread,
                                    mean_margin, variance_ratio) for r in group.itertuples())
            rows.append({"sensor": sensor, "criterion": criterion,
                         "accepted": accepted, "point_only": point_only,
                         "mean_not_confirmed": not (regular and (group.mean_upper <= mean_margin).all()),
                         "variance_not_confirmed": not (regular and (group.log_variance_upper <= np.log(variance_ratio)).all()),
                         "undefined": not regular})
    return pd.DataFrame(rows)


def panel_interval(fractions):
    """Интервал MC по независимым панелям, с безопасными крайними случаями."""
    x = np.asarray(fractions, dtype=float)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all() or ((x < 0) | (x > 1)).any():
        raise ValueError("need >=2 finite panel fractions in [0,1]")
    estimate = float(x.mean())
    if (x == 0).all():
        # P(принят отдельный канал) <= P(в панели принят хотя бы один).
        return estimate, 0., float(1 - .025 ** (1 / len(x))), "panel_event_exact_zero"
    if (x == 1).all():
        return estimate, float(.025 ** (1 / len(x))), 1., "panel_event_exact_one"
    radius = NormalDist().inv_cdf(.975) * float(x.std(ddof=1)) / np.sqrt(len(x))
    return estimate, max(0., estimate - radius), min(1., estimate + radius), "panel_mean_normal"


def summarize(outcomes):
    """Основные и диагностические доли, все интервалы по панелям."""
    keys = ["phi", "condition", "criterion"]
    measures = ("accepted", "point_only", "mean_not_confirmed", "variance_not_confirmed", "undefined")
    panels = outcomes.groupby([*keys, "panel"], as_index=False)[list(measures)].mean()
    rows = []
    for identity, group in panels.groupby(keys, sort=True):
        for measure in measures:
            estimate, low, high, method = panel_interval(group[measure])
            rows.append(dict(zip(keys, identity), measure=measure, rate=estimate,
                             low=low, high=high, interval=method, panels=len(group)))
    return pd.DataFrame(rows)


def suitability(summary):
    """Заранее заданный ориентир, не подбор нового метода."""
    rows = []
    main = summary.loc[summary.measure == "accepted"]
    for (phi, criterion), group in main.groupby(["phi", "criterion"], sort=True):
        by_condition = group.set_index("condition")
        stable = bool(by_condition.loc["stationary", "low"] >= .8)
        strong = bool((by_condition.loc[["linear_2", "mean_step_1", "scale_step_2"], "high"] <= .05).all())
        rows.append({"phi": phi, "criterion": criterion, "stable_acceptance_ok": stable,
                     "strong_alternatives_ok": strong, "both_ok": stable and strong})
    return pd.DataFrame(rows)


def ar_mean_variance(n, phi):
    """Точная дисперсия среднего n стационарных AR(1) отсчётов с Var=1."""
    if n < 1 or not 0 <= phi < 1:
        raise ValueError("positive n and 0<=phi<1 required")
    return (n * (1 + phi) / (1 - phi)
            - 2 * phi * (1 - phi ** n) / (1 - phi) ** 2) / n ** 2


def mean_only_planning():
    """Иллюстрация планирования, НЕ новая длина хвоста или гарантия мощности.

    Известная Var=1, истинная разность=0; только среднее, без требования
    дисперсии. Три парных сравнения, Bonferroni97.5%; sqrt(2*Var(mean))
    консервативно игнорирует положительную ковариацию соседних блоков.
    """
    z = NormalDist().inv_cdf(1 - .025 / 6)
    rows = []
    for phi in PHIS:
        for margin in (.25, .5):
            low, high = 1, 1_000_000
            while low < high:
                mid = (low + high) // 2
                if z * np.sqrt(2 * ar_mean_variance(mid, phi)) <= margin:
                    high = mid
                else:
                    low = mid + 1
            rows.append({"phi": phi, "mean_margin": margin,
                         "illustrative_points_per_block": low,
                         "illustrative_total_points": 3 * low,
                         "known_variance": True, "not_a_power_guarantee": True})
    return pd.DataFrame(rows)