"""H1/H2 по готовому кешу, утверждённый docs/protocol_h1_h3.md v1.

python -m scripts.stage3_failures --model amazon/chronos-bolt-small
python -m scripts.stage3_failures --model amazon/chronos-2

Нового инференса нет. Пороги Q95 (основной) и Q90 (чувствительность) берутся
ТОЛЬКО по none seed1 отдельно для model/L/H. d* — только для MAE, три статуса,
CI условен на recovered-повторах и фиксированном контроле. Старые результаты
results/stage3/ не перезаписываются; новый путь results/protocol_h1_h3_v1/stage3/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
import warnings
import zlib
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # без окна: только сохранение в файл
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullLocator

from scripts.plot_synth import CLASS_TITLES, COLORS, TITLES
from src import metrics
from src.cache import load_dataset_meta, load_forecasts, load_series, load_windows, model_dir
from src.evaluation import (  # add_ratios re-exported for historical stage4 imports
    LOG_FLOOR, PERCENTILES, PROTOCOL_VERSION, SEARCH_GRID, add_ratios,
    bootstrap_recovery, control_thresholds, reference_audit, safe_log, smoothed_search,
)
from src.models import QUANTILE_LEVELS

SEED = 0
N_BOOT = 1000
SMOOTH = 3  # d* ищется по скользящему среднему log-кривой по стольким соседним d
L_VARIANTS = ("L512", "L256", "L128", "L64")  # варианты кеша с другой длиной контекста (S4)
CURVE_METRICS = ("mae_ratio", "crps_ratio", "width_ratio", "mase", "coverage")
L_COLORS = {512: "black", 256: "tab:purple", 128: "tab:green", 64: "tab:olive"}
I_LO, I_MED, I_HI = (QUANTILE_LEVELS.index(q) for q in (0.1, 0.5, 0.9))


# ---------- метрики окон ----------


def window_metrics(root: Path, model_id: str, w: pd.DataFrame, x: np.ndarray, ds: dict) -> pd.DataFrame:
    """Метрики каждого окна для вариантов L* и всех горизонтов; векторно по всем окнам сразу.

    Результат сверяется с функциями src/metrics.py на случайных окнах.
    """
    t, s = w.t.to_numpy(), w.series_idx.to_numpy()
    levels = np.asarray(QUANTILE_LEVELS)
    check = np.random.default_rng(SEED).choice(len(w), min(5, len(w)), replace=False)
    frames = []
    for h in ds["horizons"]:
        fut = x[s[:, None], t[:, None] + 1 + np.arange(h)]  # (окна, H)
        q_all = load_forecasts(root, model_id, h, variants=[ds["variants"].index(v) for v in L_VARIANTS])
        if len(w) != ds["n_windows"]:  # для контроля нужны только none, но вся его временная сетка
            q_all = q_all[w.row.to_numpy()]
        if not np.isfinite(q_all).all():
            raise ValueError(f"в кеше {model_id} есть непосчитанные окна (H = {h})")
        for j, v in enumerate(L_VARIANTS):
            L = int(v[1:])
            q = q_all[:, j].astype(np.float64)  # (окна, 9, H)
            ctx = x[s[:, None], t[:, None] - L + 1 + np.arange(L)]
            scale = np.abs(np.diff(ctx, axis=1)).mean(axis=1)  # знаменатель обычного MASE
            pinball = np.zeros(len(w))
            for k, lv in enumerate(levels):  # по одному квантилю: меньше памяти
                diff = fut - q[:, k]
                pinball += np.maximum(lv * diff, (lv - 1) * diff).mean(axis=1)
            f = pd.DataFrame({
                "row": w.row.to_numpy(), "L": L, "horizon": h,
                "mae": np.abs(fut - q[:, I_MED]).mean(axis=1),
                "crps": 2 * pinball / len(levels),
                "coverage": ((fut >= q[:, I_LO]) & (fut <= q[:, I_HI])).mean(axis=1),
                "width": (q[:, I_HI] - q[:, I_LO]).mean(axis=1),
                "scale": scale,
            })
            for i in check:
                assert np.isclose(scale[i], metrics.seasonal_naive_scale(ctx[i]))
                if np.isfinite(scale[i]) and scale[i] > 0:  # freeze может иметь нулевой масштаб
                    assert np.isclose(f.mae[i] / scale[i], metrics.mase(fut[i], q[i, I_MED], scale[i]))
                    assert np.isclose(f.crps[i] / scale[i], metrics.scaled_crps(fut[i], q[i], QUANTILE_LEVELS, scale[i]))
                assert np.isclose(f.coverage[i], metrics.interval_coverage(fut[i], q[i, I_LO], q[i, I_HI]))
            frames.append(f)
        del q_all
    return pd.concat(frames, ignore_index=True)


# ---------- кривые по d и d* ----------


def peak(dgrid: np.ndarray, point: np.ndarray, boot: np.ndarray, mask: np.ndarray) -> tuple[float, float, float, float]:
    """Максимум log-кривой в зоне mask: (d, значение, 95% интервал в той же точке d)."""
    mask = mask & np.isfinite(point)
    if not mask.any():
        return (float("nan"),) * 4
    i = np.flatnonzero(mask)[np.argmax(point[mask])]
    lo, hi = np.nanpercentile(boot[:, i], [2.5, 97.5])
    return int(dgrid[i]), float(np.exp(point[i])), float(np.exp(lo)), float(np.exp(hi))


def curves_and_dstar(df: pd.DataFrame, L_main: int, thresholds: pd.DataFrame,
                     n_boot: int = N_BOOT, boot_seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    curve_frames, star_rows = [], []
    for (cell, L, h), g in df.groupby(["cell", "L", "horizon"], sort=False):
        first = g.iloc[0]
        base = {"cell": cell, "cls": first.cls, "kind": first.kind, "level": first.level, "L": L, "horizon": h,
            "reference_approximation": first.cls == "drift"}
        n_series = g.series_idx.nunique()
        if n_series != 10:
            raise ValueError(f"{cell}: v1 control aggregation is matched to 10 series, got {n_series}")
        # Одни и те же выборки рядов для всех L, H и метрик ячейки: сравнения между ними парные.
        idx = np.random.default_rng([boot_seed, zlib.crc32(cell.encode())]).integers(0, n_series, (n_boot, n_series))
        refs = g.groupby("series_idx")[["mae_ref_old", "mae_ref_new"]].first()
        valid = np.isfinite(refs).all(axis=1) & (refs > 0).all(axis=1)
        new_old = float(np.exp(safe_log((refs.mae_ref_new / refs.mae_ref_old)[valid].to_numpy()).mean()))
        for m in CURVE_METRICS:
            P = g.pivot(index="series_idx", columns="d", values=m)
            dgrid = P.columns.to_numpy()
            geo = m != "coverage"  # отношения усредняем геометрически, доли — обычным средним
            A = safe_log(P.to_numpy()) if geo else P.to_numpy()
            if np.isnan(A).any():  # невалидные эталоны/нулевой масштаб: объём исключений сохраняется в аудите
                with warnings.catch_warnings():  # если MASE нет ни у одного ряда, точка кривой = NaN
                    warnings.simplefilter("ignore", RuntimeWarning)
                    point, boot = np.nanmean(A, axis=0), np.nanmean(A[idx], axis=1)
                    lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
            else:
                point, boot = A.mean(axis=0), A[idx].mean(axis=1)
                lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
            back = np.exp if geo else (lambda a: a)
            curve_frames.append(pd.DataFrame(
                {**base, "metric": m, "d": dgrid, "d_over_L": dgrid / L,
                 "value": back(point), "lo": back(lo), "hi": back(hi),
                 "n_series": n_series, "n_valid": np.isfinite(A).sum(axis=0)}))
            if m != "mae_ratio":  # CRPS описателен: пороги MAE к нему неприменимы.
                continue
            in_h = (dgrid >= -h) & (dgrid < 0)
            post = (dgrid >= 0) & (dgrid <= 504)
            ph = peak(dgrid, point, boot, in_h)
            pp = peak(dgrid, point, boot, post)
            sm_point = smoothed_search(point, dgrid)
            for threshold_row in thresholds[(thresholds.L == L) & (thresholds.horizon == h)].itertuples():
                result, stats, _ = bootstrap_recovery(A, dgrid, threshold_row.threshold,
                                                     n_boot=n_boot, indices=idx)
                star_rows.append({
                    **base, "metric": m, "percentile": threshold_row.percentile,
                    "role": threshold_row.role, "threshold": threshold_row.threshold,
                    "raw_quantile": threshold_row.raw_quantile, "new_old": new_old,
                    "peak_h_d": ph[0], "peak_h": ph[1], "peak_h_lo": ph[2], "peak_h_hi": ph[3],
                    "peak_d": pp[0], "peak": pp[1], "peak_lo": pp[2], "peak_hi": pp[3],
                    "smooth_peak_d": result.peak_d, "smooth_peak": np.exp(np.max(sm_point)),
                    "status": result.status, "reason": result.reason, "d_star": result.d_star,
                    "d_star_over_L": result.d_star / L, "d_star_over_L_lo": stats["d_star_lo"] / L,
                    "d_star_over_L_hi": stats["d_star_hi"] / L,
                    "search_end": 504, "last_candidate": 440, "confirmation_span": 64,
                    "n_series": n_series, **stats,
                })
    return pd.concat(curve_frames, ignore_index=True), pd.DataFrame(star_rows)


# ---------- покрытие по зонам d ----------


def zone_edges(L_main: int) -> list[int]:
    return [e for e in (0, 64, 128, 256) if e < L_main] + [L_main]


def zone_names(L_main: int) -> list[str]:
    e = zone_edges(L_main)
    return ["до смены", "смена в горизонте"] + [f"{a}–{b - 1}" for a, b in zip(e, e[1:])] + [f"≥{L_main}"]


def assign_zone(d: pd.Series, h: pd.Series, L_main: int) -> pd.Series:
    """Зона d для каждого окна: d < −H, −H ≤ d < 0, интервалы d ≥ 0, d ≥ L."""
    names, e = zone_names(L_main), zone_edges(L_main)
    z = pd.Series(names[-1], index=d.index)
    z[d < -h] = names[0]
    z[(d >= -h) & (d < 0)] = names[1]
    for name, a, b in zip(names[2:], e, e[1:]):
        z[(d >= a) & (d < b)] = name
    return z


def zone_stats(dm: pd.DataFrame, groups: dict[str, list[str]], L_main: int,
               n_boot: int = N_BOOT, boot_seed: int = SEED) -> pd.DataFrame:
    """Покрытие (с бутстрепом по рядам), рост ошибки, ширины и обычный MASE по зонам d; только L = L_main."""
    rng = np.random.default_rng(boot_seed)
    rows = []
    for h, dh in dm.groupby("horizon"):
        for gname, cells in groups.items():
            dg = dh[dh.cell.isin(cells)]
            for zone in ["все окна", *zone_names(L_main)]:
                dz = dg if zone == "все окна" else dg[dg.zone == zone]
                per = dz.groupby("series_idx").agg(
                    coverage=("coverage", "mean"), lmae=("log_mae_ratio", "mean"),
                    lwidth=("log_width_ratio", "mean"), lmase=("log_mase", "mean"))
                cov = per.coverage.to_numpy()
                if not len(cov):
                    continue
                boot = cov[rng.integers(0, len(cov), (n_boot, len(cov)))].mean(axis=1)
                rows.append({
                    "group": gname, "horizon": h, "zone": zone, "n_series": len(cov), "n_windows": len(dz),
                    "coverage": cov.mean(), "coverage_lo": np.percentile(boot, 2.5),
                    "coverage_hi": np.percentile(boot, 97.5),
                    "mae_ratio": np.exp(per.lmae.mean()), "width_ratio": np.exp(per.lwidth.mean()),
                    "mase": np.exp(per.lmase.mean()),
                })
    return pd.DataFrame(rows)


# ---------- графики ----------


def panels(c: pd.DataFrame) -> list[tuple[str, str]]:
    return list(dict.fromkeys(zip(c.cls, c.kind)))  # порядок ячеек сетки


def log_y(ax) -> None:
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator([0.25, 0.5, 1, 2, 4, 8]))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.yaxis.set_minor_locator(NullLocator())


def mark_zones(ax, h: int, L: int) -> None:
    ax.axvspan(-h, 0, color="0.92", lw=0)  # смена внутри горизонта
    ax.axvline(0, color="k", lw=0.8)
    ax.axvline(L - 1, color="k", lw=0.8, ls="--")  # старого режима в контексте больше нет


def level_label(level: str) -> str:
    return "без изменений" if level == "-" else level


def grid_figure(curves: pd.DataFrame, metric: str, h: int, L: int, path: Path, title: str, ylabel: str,
                ref: float, thr: float | None = None, log: bool = True) -> None:
    c = curves[(curves.metric == metric) & (curves.horizon == h) & (curves.L == L)]
    fig, axes = plt.subplots(3, 4, figsize=(16, 9.5), sharex=True, sharey=True)
    for ax, (cls, kind) in zip(axes.flat, panels(c)):
        p = c[(c.cls == cls) & (c.kind == kind)]
        mark_zones(ax, h, L)
        ax.axhline(ref, color="gray", lw=0.8)
        if thr is not None:
            ax.axhline(thr, color="gray", lw=0.8, ls=":")
        for level, g in p.groupby("level", sort=False):
            color = COLORS.get(level, "gray")
            ax.plot(g.d, g.value, color=color, lw=1.3, label=level_label(level))
            ax.fill_between(g.d, g.lo, g.hi, color=color, alpha=0.15, lw=0)
        if log:
            log_y(ax)
        ax.set_title(f"{CLASS_TITLES[cls]}: {TITLES[kind]}" + (" (≈ эталон)" if cls == "drift" else ""), fontsize=10)
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(title)
    fig.supxlabel("d = t − τ (серая полоса — смена в горизонте; пунктир — d = L − 1)")
    fig.supylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def context_figure(curves: pd.DataFrame, h: int, path: Path, thresholds: pd.DataFrame) -> None:
    """Отношение ошибки от d / L при разных L, сила strong: совпадают ли кривые по оси d / L."""
    c = curves[(curves.metric == "mae_ratio") & (curves.horizon == h) & curves.level.isin(["strong", "-"])]
    fig, axes = plt.subplots(3, 4, figsize=(16, 9.5), sharex=True, sharey=True)
    for ax, (cls, kind) in zip(axes.flat, panels(c)):
        p = c[(c.cls == cls) & (c.kind == kind)]
        ax.axvline(0, color="k", lw=0.8)
        ax.axvline(1, color="k", lw=0.8, ls="--")
        ax.axhline(1, color="gray", lw=0.8)
        for L, g in p.groupby("L"):
            ax.plot(g.d / L, g.value, color=L_COLORS.get(L, "gray"), lw=1.2, label=f"L = {L}")
            thr = thresholds[(thresholds.L == L) & (thresholds.horizon == h) & (thresholds.percentile == 95)]
            ax.axhline(thr.iloc[0].threshold, color=L_COLORS.get(L, "gray"), lw=0.6, ls=":")
        log_y(ax)
        ax.set_xlim(-0.3, 2.5)
        ax.set_title(f"{CLASS_TITLES[cls]}: {TITLES[kind]}" + (" (≈ эталон)" if cls == "drift" else ""), fontsize=10)
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(f"H1 v1: MAE при разных L, strong, H = {h}; цветной пунктир: Q95 контроля для каждого L")
    fig.supxlabel("d / L (пунктир — старого режима в контексте больше нет)")
    fig.supylabel("ошибка / ошибка в устойчивом режиме")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def width_figure(curves: pd.DataFrame, h: int, L: int, path: Path) -> None:
    """Рост ошибки против роста ширины интервала, сила strong."""
    c = curves[(curves.horizon == h) & (curves.L == L) & curves.level.isin(["strong", "-"])]
    fig, axes = plt.subplots(3, 4, figsize=(16, 9.5), sharex=True, sharey=True)
    for ax, (cls, kind) in zip(axes.flat, panels(c)):
        mark_zones(ax, h, L)
        ax.axhline(1, color="gray", lw=0.8)
        for metric, color, label in (("mae_ratio", "tab:red", "ошибка"), ("width_ratio", "tab:blue", "ширина 80%-интервала")):
            g = c[(c.cls == cls) & (c.kind == kind) & (c.metric == metric)]
            ax.plot(g.d, g.value, color=color, lw=1.3, label=label)
            ax.fill_between(g.d, g.lo, g.hi, color=color, alpha=0.15, lw=0)
        log_y(ax)
        ax.set_title(f"{CLASS_TITLES[cls]}: {TITLES[kind]}" + (" (≈ эталон)" if cls == "drift" else ""), fontsize=10)
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(f"H2: во сколько раз выросли ошибка и ширина интервала, сила strong, L = {L}, H = {h}")
    fig.supxlabel("d = t − τ (серая полоса — смена в горизонте; пунктир — d = L − 1)")
    fig.supylabel("значение / значение в устойчивом режиме")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------- сводка ----------


def md_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows])


def summarize(star: pd.DataFrame, zones: pd.DataFrame, thresholds: pd.DataFrame,
              audit: pd.DataFrame, meta: dict, L_main: int) -> str:
    """Таблицы с явными статусами и условностью интервалов, без подбора порога."""
    out = [f"# Этап 3 — {meta['model']}, протокол v1", "",
           f"Дата: {meta['date']}. Протокол: `{meta['protocol']}`, SHA256 `{meta['protocol_sha256']}`.",
           f"Оценка: `{meta['cache']}`, seed {meta['evaluation_seed']}; контроль: `{meta['control_cache']}`, seed 1.",
           "", "## Определения и ограничения", "",
           f"- MAE / медианная MAE того же ряда/model/L/H. Старый эталон d < −H; новый {L_main} ≤ d ≤ 640.",
           "- В горизонте p = clip((d+H+1)/H,0,1), смесь (1−p)b_old+p b_new. У none/anomaly всегда старый эталон.",
           "- Drift: полоса 512–640 — приближение, не строго устойчивый контекст; ниже есть анализ без drift.",
           f"- Геометрическое среднее по 10 рядам. Для log при нулевом отношении floor {LOG_FLOOR}; "
           "невалидные знаменатели исключаются, не заменяются epsilon. Сами отношения не обрезаются.",
           "- Контроль и оценка: log-сглаживание по 3 соседям строго внутри 0,8,…,504. Q95 основной, Q90 чувствительность; "
           "линейный процентиль, threshold=max(1,Q). Нет выбора по seed0, типу или силе.",
           "- d* строго после последнего максимума сглаженной MAE: 9 значений ≤ порога (span64), "
           "на остатке ≤floor(n/10) превышений и максимум два подряд.",
           "- no_degradation: превышений нет, численное d* не назначается. recovered: найдено d*. "
           "not_confirmed: подтверждения нет. Поиск до 504, последний кандидат 440. not_confirmed НЕ означает d* ≥ 512.",
           f"- Бутстреп {meta['n_boot']} раз по рядам целиком. CI d* условен только на recovered-повторах и фиксированном "
           "контроле seed1, неопределённость контроля в него не входит. Частоты всех статусов приведены рядом.",
           "- CRPS, ширина и MASE — описательные кривые, d* для CRPS с порогами MAE не вычисляется.",
           "- Обе синтетические выборки уже изучались; пересчёт не является новым независимым подтверждением H1.",
           "", "## Фиксированные пороги (отдельно по L/H)", "",
           md_table(["L", "H", "Q", "Сырой Q", "Порог"],
                    [[str(r.L), str(r.horizon), str(r.percentile), f"{r.raw_quantile:.6f}", f"{r.threshold:.6f}"]
                     for r in thresholds.itertuples()])]
    for (h, percentile), s in star.groupby(["horizon", "percentile"], sort=False):
        out += ["", f"## H1: H={h}, Q{percentile} ({'основной' if percentile == 95 else 'чувствительность'})", "",
                "CI в таблице — только по recovered-повторам. Drift помечен `≈`.", ""]
        rows = []
        for r in s.itertuples():
            numeric = f"{r.d_star:g}" if np.isfinite(r.d_star) else "—"
            interval = f"[{r.d_star_lo:g}; {r.d_star_hi:g}]" if r.n_boot_recovered else "—"
            rows.append([r.cell + (" ≈" if r.reference_approximation else ""), str(r.L),
                         f"{r.smooth_peak:.2f} @ {r.smooth_peak_d:g}", r.status, numeric,
                         f"{r.d_star_over_L:.3f}" if np.isfinite(r.d_star) else "—", interval,
                         str(r.boot_no_degradation_n), str(r.n_boot_recovered), str(r.boot_not_confirmed_n)])
        out.append(md_table(["Ячейка", "L", "Сглаженный пик @ d", "Статус", "d*", "d*/L", "Условный CI95 d*",
                             "boot: нет дегр.", "boot: recovered", "boot: не подтв."], rows))
    out += ["", "## Чувствительность: без drift", "",
            "Состав основной оценки не меняется; таблица ниже исключает только приближённый эталон drift.", ""]
    counts = star[star.cls != "drift"].groupby(["L", "horizon", "percentile", "status"]).size()
    out.append(md_table(["L", "H", "Q", "Статус", "Ячеек"],
                        [[str(v) for v in (*key, count)] for key, count in counts.items()]))
    for h, z in zones.groupby("horizon"):
        out += ["", f"## H2: покрытие 80%-интервала, L={L_main}, H={h}", ""]
        rows = [[r.group, r.zone, f"{r.coverage:.3f} [{r.coverage_lo:.3f}; {r.coverage_hi:.3f}]",
                 f"{r.mae_ratio:.3f}", f"{r.width_ratio:.3f}"] for r in z.itertuples()
                if r.group in ("все ячейки", "без drift", "change", "drift", "anomaly", "none")]
        out.append(md_table(["Набор", "Зона d", "Coverage [CI95]", "MAE ×", "Ширина ×"], rows))
    out += ["", "## Аудит и файлы", "",
            "`reference_audit.csv`: эталоны и число окон по каждому ряду/L/H/метрике. "
            "`exclusion_audit.csv`: объём исключений. `thresholds.csv`: сырой и ограниченный порог.",
            "`curves.csv.gz`: описательные кривые/CI и n_valid; `d_star.csv`: все статусы, частоты бутстрепа, "
            "условные CI и d*/L. `coverage_zones.csv`, `coverage_zones_no_drift.csv`: H2.",
            f"Исключено отношений (по метрикам суммарно): {int(audit.n_excluded.sum())}. "
            "Окна перекрываются; это не число независимых наблюдений.",
            f"Время расчёта: {meta['elapsed_seconds']:.1f} с (только кеш, без инференса)."]
    return "\n".join(out) + "\n"


# ---------- запуск ----------


def checked_cache(root: Path, model: str) -> tuple[dict, dict, dict, pd.DataFrame]:
    """Сверить сохранённый хеш конфига, модель, размер и полную сетку кеша."""
    ds = load_dataset_meta(root)
    cfg = yaml.safe_load((root / "config.yaml").read_text())
    # Та же канонизация, что scripts.cache_forecasts.dataset_hash; не импортируем генератор/модель.
    data = {k: v for k, v in cfg.items() if k != "models"}
    digest = hashlib.sha1(json.dumps(data, sort_keys=True).encode()).hexdigest()[:12]
    if ds["hash"] != digest:
        raise ValueError(f"{root}: config hash mismatch ({ds['hash']} != {digest})")
    meta = json.loads((model_dir(root, model) / "meta.json").read_text())
    if meta.get("model") != model:
        raise ValueError(f"{root}: cached model does not match {model}")
    if cfg["n_per_cell"] != 10 or cfg["windows"]["context"] != 512:
        raise ValueError("protocol v1 requires n_per_cell=10, main context=512")
    win = cfg["windows"]
    if win["d_step"] != 8 or win["d_min"] > -128 or win["d_max"] < 640:
        raise ValueError("protocol v1 requires step8, full old references and band512..640")
    if ds["horizons"] != win["horizons"] or not set(L_VARIANTS).issubset(ds["variants"]):
        raise ValueError(f"{root}: incompatible horizons/variants")
    w = load_windows(root)
    if len(w) != ds["n_windows"] or not np.array_equal(w.row.to_numpy(), np.arange(len(w))):
        raise ValueError(f"{root}: windows metadata/row order mismatch")
    if w.series_idx.nunique() != ds["n_series"] or w.cell.nunique() != ds["n_cells"]:
        raise ValueError(f"{root}: series/cell metadata mismatch")
    if not w.groupby("cell").series_idx.nunique().eq(10).all():
        raise ValueError(f"{root}: every cell must contain 10 series")
    expected_d = np.arange(win["d_min"], win["d_max"] + 1, win["d_step"])
    for _, g in w.groupby("series_idx", sort=False):
        if not np.array_equal(g.pos.to_numpy(), expected_d):
            raise ValueError(f"{root}: incomplete/unsorted per-series grid")
        if not np.array_equal(g.t.to_numpy() - ds["tau"], expected_d):
            raise ValueError(f"{root}: inconsistent forecast-origin indices")
    return ds, cfg, meta, w


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path, default=Path("results/cache/synth_v1"))
    parser.add_argument("--control-cache", type=Path, default=Path("results/cache/synth_v1_seed1"))
    parser.add_argument("--model", default="amazon/chronos-bolt-small")
    parser.add_argument("--out", type=Path, default=None, help="по умолчанию results/protocol_h1_h3_v1/stage3/<модель>")
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--boot-seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.n_boot < 1:
        parser.error("--n-boot must be >= 1")
    base = Path("results/protocol_h1_h3_v1/stage3")
    if args.cache.name != "synth_v1":
        base /= args.cache.name
    out = args.out or base / model_dir(args.cache, args.model).name
    historical = Path("results/stage3").resolve()
    if out.resolve().is_relative_to(historical):
        parser.error("historical results/stage3 must not be overwritten")
    t0 = time.perf_counter()

    ds, cfg, cache_meta, w = checked_cache(args.cache, args.model)
    cds, ccfg, control_meta, cw = checked_cache(args.control_cache, args.model)
    if ccfg["seed"] != 1:
        raise ValueError("control must be seed1")
    comparable = lambda c: {k: v for k, v in c.items() if k not in ("seed", "models")}
    if comparable(cfg) != comparable(ccfg):
        raise ValueError("evaluation/control configurations differ beyond seed/models")
    L_main, win = cfg["windows"]["context"], cfg["windows"]
    cx, _ = load_series(args.control_cache)
    cw = cw[cw.cls == "none"].reset_index(drop=True)
    print("Пороги по none seed1, полная сетка контроля …")
    control = add_ratios(window_metrics(args.control_cache, args.model, cw, cx, cds), cw, L_main)
    thresholds = control_thresholds(control)
    thresholds["model"] = args.model
    thresholds["control_cache"] = str(args.control_cache)
    thresholds["control_dataset_hash"] = cds["hash"]
    control_audit = reference_audit(control)
    control_curves = []
    for (length, horizon), g in control.groupby(["L", "horizon"]):
        p = g.pivot(index="series_idx", columns="d", values="mae_ratio")
        logs = safe_log(p.to_numpy()).mean(axis=0)
        region = (p.columns >= 0) & (p.columns <= 504)
        control_curves.append(pd.DataFrame({"model": args.model, "L": length, "horizon": horizon,
                                           "d": SEARCH_GRID, "value": np.exp(logs[region]),
                                           "smoothed_value": np.exp(smoothed_search(logs, p.columns.to_numpy())),
                                           "n_series": len(p), "control_seed": 1}))
    del cx, control

    x, _ = load_series(args.cache)
    print(f"Метрики окон: {len(w)} окон × {len(L_VARIANTS)} длины контекста × {len(ds['horizons'])} горизонта …")
    df = add_ratios(window_metrics(args.cache, args.model, w, x, ds), w, L_main)
    del x
    print(f"  {time.perf_counter() - t0:.0f} с; сверка с src/metrics.py пройдена")
    audit = reference_audit(df)
    curves, star = curves_and_dstar(df, L_main, thresholds, args.n_boot, args.boot_seed)
    dm = df[df.L == L_main].assign(zone=lambda a: assign_zone(a.d, a.horizon, L_main))
    cell_cls = w.drop_duplicates("cell").set_index("cell").cls
    groups = {"все ячейки": list(cell_cls.index)}
    groups["без drift"] = list(cell_cls.index[cell_cls != "drift"])
    groups |= {c: list(cell_cls.index[cell_cls == c]) for c in ("change", "drift", "anomaly", "none")}
    groups |= {cell: [cell] for cell in cell_cls.index if cell != "none"}
    zones = zone_stats(dm, groups, L_main, args.n_boot, args.boot_seed)

    protocol = Path(__file__).resolve().parents[1] / "docs/protocol_h1_h3.md"
    sha256 = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    meta = {
        "protocol": PROTOCOL_VERSION, "protocol_file": str(protocol), "protocol_sha256": sha256(protocol),
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"), "model": args.model, "cache": str(args.cache),
        "control_cache": str(args.control_cache), "control_dataset_hash": cds["hash"],
        "evaluation_seed": cfg["seed"], "control_seed": ccfg["seed"], "source_config": cfg,
        "control_config": ccfg, "source_config_sha256": sha256(args.cache / "config.yaml"),
        "control_config_sha256": sha256(args.control_cache / "config.yaml"),
        "source_dataset_meta_sha256": sha256(args.cache / "dataset.json"),
        "control_dataset_meta_sha256": sha256(args.control_cache / "dataset.json"),
        "source_windows_sha256": sha256(args.cache / "windows.csv"),
        "control_windows_sha256": sha256(args.control_cache / "windows.csv"),
        "source_model_meta_sha256": sha256(model_dir(args.cache, args.model) / "meta.json"),
        "control_model_meta_sha256": sha256(model_dir(args.control_cache, args.model) / "meta.json"),
        "source_code_sha256": {str(p): sha256(p) for p in (
            Path(__file__).resolve(), Path(__file__).resolve().parents[1] / "src/evaluation.py")},
        "dataset_hash": ds["hash"], "n_cells": ds["n_cells"], "n_series": ds["n_series"],
        "n_windows": ds["n_windows"], "n_per_cell": ds["n_series"] // ds["n_cells"], "tau": ds["tau"],
        "d_min": win["d_min"], "d_max": win["d_max"], "d_step": win["d_step"],
        "drift_duration": cfg["drift"]["duration"], "L_main": L_main, "L_variants": L_VARIANTS,
        "horizons": ds["horizons"], "threshold_percentiles": PERCENTILES, "smooth": SMOOTH,
        "n_boot": args.n_boot, "boot_seed": args.boot_seed, "log_floor": LOG_FLOOR,
        "search_grid": SEARCH_GRID.tolist(), "last_candidate": 440,
        "confirmation_values": 9, "confirmation_span": 64, "max_suffix_exceedance_fraction": 0.1,
        "max_exceedance_run": 2, "bootstrap_control_fixed": True, "ci_recovered_only": True,
        "reference_band": [L_main, 640], "drift_reference_approximation": True,
        "env": {"python": platform.python_version(), "platform": platform.platform(),
            **{p: version(p) for p in ("numpy", "pandas", "matplotlib", "PyYAML")}},
        "cache_meta": cache_meta, "control_cache_meta": control_meta,
    }
    out.mkdir(parents=True, exist_ok=True)
    cols = ["row", "cell", "series_idx", "d", "L", "horizon", "mae", "crps", "coverage", "width", "scale",
            "mase", "mae_ratio", "crps_ratio", "width_ratio"]
    cols += ["reference_approximation", "mae_ref_old", "mae_ref_new", "mae_reference"]
    df[cols].to_csv(out / "window_metrics.csv.gz", index=False, float_format="%.12g")
    curves.to_csv(out / "curves.csv.gz", index=False, float_format="%.12g")
    star.to_csv(out / "d_star.csv", index=False, float_format="%.12g")
    star[star.cls != "drift"].to_csv(out / "d_star_no_drift.csv", index=False, float_format="%.12g")
    zones.to_csv(out / "coverage_zones.csv", index=False, float_format="%.12g")
    zones[zones.group == "без drift"].to_csv(out / "coverage_zones_no_drift.csv", index=False, float_format="%.12g")
    thresholds.to_csv(out / "thresholds.csv", index=False, float_format="%.17g")
    pd.concat(control_curves, ignore_index=True).to_csv(out / "control_curves.csv", index=False, float_format="%.17g")
    audit.to_csv(out / "reference_audit.csv", index=False, float_format="%.12g")
    control_audit.to_csv(out / "control_reference_audit.csv", index=False, float_format="%.12g")
    audit.groupby(["cls", "L", "horizon", "metric"], sort=False)[["n_windows", "n_valid", "n_excluded"]].sum().to_csv(
        out / "exclusion_audit.csv")

    for h in ds["horizons"]:
        thr = float(thresholds[(thresholds.L == L_main) & (thresholds.horizon == h) &
                               (thresholds.percentile == 95)].iloc[0].threshold)
        grid_figure(curves, "mae_ratio", h, L_main, out / f"fig_h1_error_H{h}.png",
                    f"H1 v1: MAE, L = {L_main}, H = {h}; пунктир Q95 seed1 = {thr:.3f} (d* по log-сглаживанию)",
                    "ошибка / ошибка в устойчивом режиме", ref=1.0, thr=thr)
        context_figure(curves, h, out / f"fig_h1_context_H{h}.png", thresholds)
        grid_figure(curves, "coverage", h, L_main, out / f"fig_h2_coverage_H{h}.png",
                    f"H2: покрытие 80%-интервала, L = {L_main}, H = {h}", "доля факта внутри [q0.1, q0.9]",
                    ref=0.8, log=False)
        width_figure(curves, h, L_main, out / f"fig_h2_width_H{h}.png")

    meta["elapsed_seconds"] = time.perf_counter() - t0
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    summary = summarize(star, zones, thresholds, audit, meta, L_main)
    (out / "summary.md").write_text(summary)
    g = zones[(zones.group == "все ячейки") & (zones.zone == "все окна")]
    print("  глобальное покрытие: " + ", ".join(f"H = {r.horizon}: {r.coverage:.3f}" for r in g.itertuples()))
    print("  пороги Q95/Q90 зафиксированы по seed1 для каждого L/H → thresholds.csv")
    print(f"Готово за {time.perf_counter() - t0:.0f} с → {out}/")


if __name__ == "__main__":
    main()
