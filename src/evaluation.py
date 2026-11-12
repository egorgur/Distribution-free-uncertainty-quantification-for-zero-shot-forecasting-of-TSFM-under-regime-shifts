"""Общая оценка H1/H3 по утверждённому протоколу v1 (2026-11-12).

Эталон будущей ошибки применяется только при оценке, не при вычислении сигналов.
Логарифмы сохраняют исторический floor 1e-6 для геометрического среднего при
нулевом числителе. Сами отношения не обрезаются, знаменатели не заменяются epsilon.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import warnings

import numpy as np
import pandas as pd

PROTOCOL_VERSION = "h1_h3_v1"
LOG_FLOOR = 1e-6
REFERENCE_END = 640
SEARCH_GRID = np.arange(0, 505, 8)
PERCENTILES = (95, 90)
STATUSES = ("no_degradation", "recovered", "not_confirmed")


def safe_log(values: np.ndarray) -> np.ndarray:
    """Исторический log(max(r, 1e-6)); нечисловые/отрицательные r остаются NaN."""
    a = np.asarray(values, dtype=float)
    return np.log(np.maximum(np.where(np.isfinite(a) & (a >= 0), a, np.nan), LOG_FLOOR))


def _divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    valid = np.isfinite(denominator) & (denominator > 0) & np.isfinite(numerator) & (numerator >= 0)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        result = numerator / denominator.where(valid)
    return result.where(np.isfinite(result))


def add_ratios(df: pd.DataFrame, w: pd.DataFrame, L_main: int) -> pd.DataFrame:
    """Добавить общие H1/H3 отношения по ПОЛНОЙ сетке до фильтрации оценки.

    Вход совместим с прежним stage3.add_ratios: row/L/horizon/mae/crps/width/scale,
    метаданные окон в w. Эталоны раздельны по ряду/L/H (и model, если он есть).
    Новый эталон: L_main <= d <= 640 у change/drift; старый: d < -H.
    У none/anomaly старый эталон остаётся на всей сетке. pos используется и у none.
    """
    out = df.merge(w[["row", "cell", "cls", "kind", "level", "series_idx", "pos"]],
                   on="row", validate="many_to_one").rename(columns={"pos": "d"})
    keys = [c for c in ("model", "model_id") if c in out] + ["series_idx", "L", "horizon"]
    groupers = [out[k] for k in keys]
    changes = out.cls.isin(["change", "drift"])
    old_band = out.d < -out.horizon
    new_band = out.d.between(L_main, REFERENCE_END)
    p = ((out.d + out.horizon + 1) / out.horizon).clip(0, 1).where(changes, 0.0)
    out["reference_approximation"] = out.cls.eq("drift")
    out["reference_band_start"], out["reference_band_end"] = L_main, REFERENCE_END
    for metric in ("mae", "crps", "width"):
        old_values = out[metric].where(old_band)
        new_values = out[metric].where(new_band)
        old = old_values.groupby(groupers).transform("median")
        new = new_values.groupby(groupers).transform("median").where(changes, old)
        out[f"{metric}_ref_old"], out[f"{metric}_ref_new"] = old, new
        out[f"{metric}_ref_old_n"] = old_values.groupby(groupers).transform("count")
        out[f"{metric}_ref_new_n"] = new_values.groupby(groupers).transform("count").where(
            changes, out[f"{metric}_ref_old_n"])
        # Не позволяем невалидному компоненту участвовать даже в положительной смеси.
        old_ok = old.where(np.isfinite(old) & (old > 0))
        new_ok = new.where(np.isfinite(new) & (new > 0))
        baseline = ((1 - p) * old_ok + p * new_ok).where(p != 0, old_ok).where(p != 1, new_ok)
        out[f"{metric}_reference"] = baseline
        out[f"{metric}_ratio"] = _divide(out[metric], baseline)
    out["mase"] = _divide(out.mae, out.scale)
    for metric in ("mae_ratio", "crps_ratio", "width_ratio", "mase"):
        out[f"log_{metric}"] = safe_log(out[metric].to_numpy())
    return out


def reference_audit(df: pd.DataFrame) -> pd.DataFrame:
    """Одна строка на ряд/L/H/метрику: эталоны, объём и явные исключения."""
    keys = [c for c in ("model", "model_id") if c in df] + [
        "cell", "cls", "kind", "level", "series_idx", "L", "horizon"]
    rows = []
    for metric in ("mae", "crps", "width"):
        audit = df.groupby(keys, sort=False, dropna=False).agg(
            ref_old=(f"{metric}_ref_old", "first"), ref_new=(f"{metric}_ref_new", "first"),
            ref_old_n=(f"{metric}_ref_old_n", "first"), ref_new_n=(f"{metric}_ref_new_n", "first"),
            reference_approximation=("reference_approximation", "first"),
            n_windows=("row", "size"), n_valid=(f"{metric}_ratio", "count"),
        ).reset_index()
        audit["metric"] = metric
        audit["n_excluded"] = audit.n_windows - audit.n_valid
        for side in ("old", "new"):
            audit[f"invalid_{side}"] = ~np.isfinite(audit[f"ref_{side}"]) | (audit[f"ref_{side}"] <= 0)
        rows.append(audit)
    return pd.concat(rows, ignore_index=True)


def smooth(a: np.ndarray, k: int = 3) -> np.ndarray:
    """Среднее соседей по последней оси; края — только по имеющимся точкам."""
    a = np.asarray(a, dtype=float)
    if k <= 0 or k % 2 != 1:
        raise ValueError("k must be a positive odd integer")
    radius = k // 2
    pad = np.pad(a, [(0, 0)] * (a.ndim - 1) + [(radius, radius)], constant_values=np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.stack([pad[..., i:i + a.shape[-1]] for i in range(k)]), axis=0)


def smoothed_search(log_values: np.ndarray, dgrid: np.ndarray) -> np.ndarray:
    """Сначала вырезать 0..504, затем сгладить: нет утечки соседей из эталона."""
    dgrid = np.asarray(dgrid)
    region = (dgrid >= 0) & (dgrid <= 504)
    if not np.array_equal(dgrid[region], SEARCH_GRID):
        raise ValueError("protocol v1 requires the complete grid 0,8,...,504")
    selected = np.asarray(log_values)[..., region]
    # Сглаживание не должно превращать отсутствующую точку в подтверждение восстановления.
    return np.where(np.isfinite(selected), smooth(selected), np.nan)


def control_thresholds(control: pd.DataFrame, expected_n: int = 10) -> pd.DataFrame:
    """Q95/Q90 сглаженной геометрической кривой none (не по отдельным окнам).

    Получает add_ratios по полной сетке seed1. Ни тип, ни сила, ни seed0 не участвуют.
    Отсутствующие точки/ряды/эталоны — ошибка, а не уменьшение эффективного n контроля.
    """
    rows = []
    none = control[control.cls == "none"]
    if none.empty:
        raise ValueError("no none control series")
    for (length, horizon), g in none.groupby(["L", "horizon"], sort=True):
        p = g.pivot(index="series_idx", columns="d", values="mae_ratio")
        region = (p.columns >= 0) & (p.columns <= 504)
        a = safe_log(p.to_numpy())
        if len(p) != expected_n or not np.isfinite(a[:, region]).all():
            raise ValueError("control must contain 10 complete finite series per L/H")
        curve = np.exp(smoothed_search(a.mean(axis=0), p.columns.to_numpy()))
        for percentile in PERCENTILES:
            raw = float(np.percentile(curve, percentile, method="linear"))
            rows.append({"L": length, "horizon": horizon, "percentile": percentile,
                         "role": "primary" if percentile == 95 else "sensitivity",
                         "raw_quantile": raw, "threshold": max(1.0, raw),
                         "n_series": len(p), "n_d": len(curve), "control_seed": 1,
                         "search_start": 0, "search_end": 504, "smooth": 3})
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class Recovery:
    status: str
    d_star: float = float("nan")
    peak_d: float = float("nan")
    reason: str = ""


def recovery_v1(log_curve: np.ndarray, dgrid: np.ndarray, threshold: float) -> Recovery:
    """Ретроспективное d* по УЖЕ сглаженной log-кривой; три разных статуса.

    Поиск строго после последнего максимума; подтверждение 9 точками (span64),
    на всём суффиксе <= floor(n/10) превышений и не более двух подряд.
    NaN/inf кривой не считаются свидетельством отсутствия деградации.
    """
    a, d = np.asarray(log_curve, dtype=float), np.asarray(dgrid)
    if a.ndim != 1 or a.shape != d.shape or not np.array_equal(d, SEARCH_GRID):
        raise ValueError("recovery requires one curve on 0,8,...,504")
    if not np.isfinite(threshold) or threshold < 1:
        raise ValueError("threshold must be finite and >= 1")
    if not np.isfinite(a).all():
        return Recovery("not_confirmed", reason="incomplete_curve")
    peak = int(np.flatnonzero(a == a.max())[-1])
    above = a > np.log(threshold)
    if not above.any():
        return Recovery("no_degradation", peak_d=float(d[peak]))
    for candidate in range(peak + 1, len(d) - 8):
        suffix = above[candidate:]
        if suffix[:9].any() or suffix.sum() > len(suffix) // 10:
            continue
        if np.any(suffix[:-2] & suffix[1:-1] & suffix[2:]):
            continue
        return Recovery("recovered", float(d[candidate]), float(d[peak]))
    return Recovery("not_confirmed", peak_d=float(d[peak]), reason="no_confirmed_candidate")


def bootstrap_recovery(log_series: np.ndarray, dgrid: np.ndarray, threshold: float,
                       n_boot: int = 1000, seed: int = 0,
                       indices: np.ndarray | None = None) -> tuple[Recovery, dict, pd.DataFrame]:
    """Бутстреп рядов целиком; порог фиксирован, CI условен на recovered-повторах.

    indices позволяет использовать одни выборки для всех конфигураций ячейки.
    Возвращает точечный исход, сводку частот/условного CI и каждый бутстреп-исход.
    """
    a = np.asarray(log_series, dtype=float)
    if a.ndim != 2 or not len(a) or n_boot < 1:
        raise ValueError("need series x d and n_boot >= 1")
    if indices is None:
        indices = np.random.default_rng(seed).integers(0, len(a), (n_boot, len(a)))
    if indices.shape != (n_boot, len(a)):
        raise ValueError("bootstrap indices have wrong shape")
    # Одинаковая агрегация для контроля и ячейки. Частичные исключения видны в аудите.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        point = np.nanmean(a, axis=0)
        boot = np.nanmean(a[indices], axis=1)
    result = recovery_v1(smoothed_search(point, dgrid), SEARCH_GRID, threshold)
    outcomes = pd.DataFrame([asdict(recovery_v1(curve, SEARCH_GRID, threshold))
                             for curve in smoothed_search(boot, dgrid)])
    values = outcomes.loc[outcomes.status == "recovered", "d_star"].to_numpy()
    lo, hi = np.percentile(values, [2.5, 97.5], method="linear") if len(values) else (np.nan, np.nan)
    summary = {"d_star_lo": float(lo), "d_star_hi": float(hi),
               "n_boot": n_boot, "n_boot_recovered": len(values),
               "ci_condition": "recovered_only; fixed_seed1_control"}
    for status in STATUSES:
        count = int((outcomes.status == status).sum())
        summary[f"boot_{status}_n"] = count
        summary[f"boot_{status}_share"] = count / n_boot
    return result, summary, outcomes