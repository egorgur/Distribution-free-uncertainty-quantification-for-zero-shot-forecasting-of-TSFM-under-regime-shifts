"""Этап 3: где и насколько ломается zero-shot прогноз при смене режима (H1, H2).

Запуск из корня репозитория (нужен кеш прогнозов, см. scripts/cache_forecasts.py):
    python -m scripts.stage3_failures                                     # chronos-bolt-small
    python -m scripts.stage3_failures --model amazon/chronos-2
    python -m scripts.stage3_failures --cache results/cache/synth_v1_n1   # пробный кеш

Что считается. Для каждого окна кеша (ячейка × ряд × d) и каждой длины контекста
L ∈ {512, 256, 128, 64} на горизонтах H = 16 и 64: ошибка медианы (MAE), CRPS по
9 квантилям, покрытие интервала [q0.1, q0.9] (номинал 0.8) и его ширина.

H1 — кривая ошибки от d. Ошибку окна делим на типичную ошибку той же модели
на том же ряду в устойчивом режиме, который она прогнозирует:
    d < −H        → медиана по окнам d < −H (старый режим);
    d ≥ 0         → медиана по окнам d ≥ 512 (в контексте только новый режим);
    −H ≤ d < 0    → смесь двух уровней пропорционально доле горизонта после τ.
У аномалий и контроля процесс после τ не меняется, поэтому эталон везде — d < −H.
1 = как в устойчивом режиме, 2 = вдвое хуже. Обычный MASE здесь не годится:
его знаменатель (масштаб наивного прогноза по контексту) сам меняется с d, когда
в контекст входит новый режим. При смене дисперсии MASE «восстанавливается» к d ≈ L,
хотя ошибка модели после смены почти не меняется (таблица в summary.md).

Кривая по d — геометрическое среднее отношения по рядам ячейки. 95% полоса —
бутстреп по рядам целиком: окна одного ряда зависимы, бутстреп по окнам занизил бы разброс.

d* — время восстановления: первое d ≥ 0, начиная с которого сглаженная кривая
(среднее логарифма по 3 соседним d) не выше порога вплоть до последнего окна перед
эталонной зоной (d = 504). Если кривая выше порога и там, пишем d* ≥ 512: ошибка
не вернулась к норме, пока в контексте оставался старый режим. Пороги 1.1 и 1.25;
основной для каждого H — наименьший, при котором контроль без изменений даёт d* = 0 при всех L.

H2 — покрытие: глобальное (все окна, как в бенчмарках) против условного по зонам d.
Рядом — во сколько раз выросли ошибка и ширина интервала: если ошибка растёт
сильнее ширины, интервал переуверен.

Результаты: results/stage3/<модель>/
    summary.md             — таблицы (выводы — в CLAUDE.md; файл перезаписывается);
    d_star.csv             — пики ошибки и d* по ячейкам, L, H, метрикам и порогам;
    coverage_zones.csv     — покрытие, ошибка и ширина по зонам d;
    curves.csv.gz          — все кривые по d с бутстреп-полосами;
    window_metrics.csv.gz  — метрики каждого окна (~10 МБ, в git не попадает);
    fig_*.png              — графики;
    meta.json              — версии и параметры.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
import warnings
import zlib
from datetime import datetime
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
from src.models import QUANTILE_LEVELS

SEED = 0
N_BOOT = 1000
THRESHOLDS = (1.1, 1.25)  # порог «ошибка вернулась к норме» для d*
SMOOTH = 3  # d* ищется по скользящему среднему log-кривой по стольким соседним d
L_VARIANTS = ("L512", "L256", "L128", "L64")  # варианты кеша с другой длиной контекста (S4)
RATIO_METRICS = ("mae", "crps", "width")
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
    check = np.random.default_rng(SEED).choice(len(w), 5, replace=False)
    frames = []
    for h in ds["horizons"]:
        fut = x[s[:, None], t[:, None] + 1 + np.arange(h)]  # (окна, H)
        q_all = load_forecasts(root, model_id, h, variants=[ds["variants"].index(v) for v in L_VARIANTS])
        if np.isnan(q_all).any():
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
                assert np.isclose(f.mae[i] / scale[i], metrics.mase(fut[i], q[i, I_MED], scale[i]))
                assert np.isclose(f.crps[i] / scale[i], metrics.scaled_crps(fut[i], q[i], QUANTILE_LEVELS, scale[i]))
                assert np.isclose(f.coverage[i], metrics.interval_coverage(fut[i], q[i, I_LO], q[i, I_HI]))
            frames.append(f)
        del q_all
    return pd.concat(frames, ignore_index=True)


def add_ratios(df: pd.DataFrame, w: pd.DataFrame, L_main: int) -> pd.DataFrame:
    """Отношения ошибки и ширины к уровню того же ряда в устойчивом режиме (см. начало файла)."""
    df = df.merge(w[["row", "cell", "cls", "kind", "level", "series_idx", "pos"]], on="row")
    df = df.rename(columns={"pos": "d"})  # pos = t − τ; у контроля d в windows.csv пустой
    d, h = df.d, df.horizon
    changes = df.cls.isin(["change", "drift"])
    keys = [df.series_idx, df.L, df.horizon]
    frac_new = np.clip((d + h + 1) / h, 0, 1).where(changes, 0.0)  # доля горизонта после τ
    for m in RATIO_METRICS:
        ref_old = df[m].where(d < -h).groupby(keys).transform("median")
        ref_new = df[m].where(d >= L_main).groupby(keys).transform("median").where(changes, ref_old)
        if m == "mae":
            df["mae_ref_old"], df["mae_ref_new"] = ref_old, ref_new
        df[f"{m}_ratio"] = df[m] / ((1 - frac_new) * ref_old + frac_new * ref_new)
    # Если контекст целиком внутри залипания (L = 64, freeze 128), наивный масштаб = 0 и MASE не определён.
    df["mase"] = (df.mae / df.scale).where(df.scale > 0)
    for m in ("mae_ratio", "width_ratio", "mase"):
        df[f"log_{m}"] = np.log(np.maximum(df[m], 1e-6))  # при залипании ошибка бывает почти 0
    return df


# ---------- кривые по d и d* ----------


def smooth(a: np.ndarray, k: int = SMOOTH) -> np.ndarray:
    """Скользящее среднее по последней оси (соседние d); на краях — по имеющимся точкам."""
    r = k // 2
    pad = np.pad(a, [(0, 0)] * (a.ndim - 1) + [(r, r)], constant_values=np.nan)
    return np.nanmean(np.stack([pad[..., i : i + a.shape[-1]] for i in range(k)]), axis=0)


def recovery_d(log_curve: np.ndarray, d_search: np.ndarray, thr: float) -> np.ndarray:
    """d* для каждой строки (…, d): первое d, после которого кривая ≤ thr до конца; inf — не восстановилась."""
    above = log_curve > np.log(thr)
    n = above.shape[-1]
    last = np.where(above.any(axis=-1), n - 1 - np.argmax(above[..., ::-1], axis=-1), -1)
    return np.where(last >= n - 1, np.inf, d_search[np.minimum(last + 1, n - 1)])


def peak(dgrid: np.ndarray, point: np.ndarray, boot: np.ndarray, mask: np.ndarray) -> tuple[int, float, float, float]:
    """Максимум log-кривой в зоне mask: (d, значение, 95% интервал в той же точке d)."""
    i = np.flatnonzero(mask)[np.argmax(point[mask])]
    lo, hi = np.percentile(boot[:, i], [2.5, 97.5])
    return int(dgrid[i]), float(np.exp(point[i])), float(np.exp(lo)), float(np.exp(hi))


def curves_and_dstar(df: pd.DataFrame, L_main: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    curve_frames, star_rows = [], []
    for (cell, L, h), g in df.groupby(["cell", "L", "horizon"], sort=False):
        first = g.iloc[0]
        base = {"cell": cell, "cls": first.cls, "kind": first.kind, "level": first.level, "L": L, "horizon": h}
        n_series = g.series_idx.nunique()
        # Одни и те же выборки рядов для всех L, H и метрик ячейки: сравнения между ними парные.
        idx = np.random.default_rng([SEED, zlib.crc32(cell.encode())]).integers(0, n_series, (N_BOOT, n_series))
        refs = g.groupby("series_idx")[["mae_ref_old", "mae_ref_new"]].first()
        new_old = float(np.exp(np.log(refs.mae_ref_new / refs.mae_ref_old).mean()))
        for m in CURVE_METRICS:
            P = g.pivot(index="series_idx", columns="d", values=m)
            dgrid = P.columns.to_numpy()
            geo = m != "coverage"  # отношения усредняем геометрически, доли — обычным средним
            A = np.log(np.maximum(P.to_numpy(), 1e-6)) if geo else P.to_numpy()
            if np.isnan(A).any():  # только обычный MASE при нулевом масштабе: среднее по рядам, где он есть
                with warnings.catch_warnings():  # если MASE нет ни у одного ряда, точка кривой = NaN
                    warnings.simplefilter("ignore", RuntimeWarning)
                    point, boot = np.nanmean(A, axis=0), np.nanmean(A[idx], axis=1)
                    lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
            else:
                point, boot = A.mean(axis=0), A[idx].mean(axis=1)
                lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
            back = np.exp if geo else (lambda a: a)
            curve_frames.append(pd.DataFrame(
                {**base, "metric": m, "d": dgrid, "value": back(point), "lo": back(lo), "hi": back(hi)}))
            if m not in ("mae_ratio", "crps_ratio"):
                continue
            in_h = (dgrid >= -h) & (dgrid < 0)
            post = (dgrid >= 0) & (dgrid < L_main)
            ph = peak(dgrid, point, boot, in_h)
            pp = peak(dgrid, point, boot, post)
            sm_point, sm_boot = smooth(point[post]), smooth(boot[:, post])
            for thr in THRESHOLDS:
                db = recovery_d(sm_boot, dgrid[post], thr)
                d_lo, d_hi = np.percentile(db, [2.5, 97.5], method="inverted_cdf")  # inf допустим
                star_rows.append({
                    **base, "metric": m, "threshold": thr, "new_old": new_old,
                    "peak_h_d": ph[0], "peak_h": ph[1], "peak_h_lo": ph[2], "peak_h_hi": ph[3],
                    "peak_d": pp[0], "peak": pp[1], "peak_lo": pp[2], "peak_hi": pp[3],
                    "d_star": float(recovery_d(sm_point, dgrid[post], thr)), "d_star_lo": float(d_lo),
                    "d_star_hi": float(d_hi), "censored_share": float(np.isinf(db).mean()),
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


def zone_stats(dm: pd.DataFrame, groups: dict[str, list[str]], L_main: int) -> pd.DataFrame:
    """Покрытие (с бутстрепом по рядам), рост ошибки, ширины и обычный MASE по зонам d; только L = L_main."""
    rng = np.random.default_rng(SEED)
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
                boot = cov[rng.integers(0, len(cov), (N_BOOT, len(cov)))].mean(axis=1)
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
        ax.set_title(f"{CLASS_TITLES[cls]}: {TITLES[kind]}", fontsize=10)
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(title)
    fig.supxlabel("d = t − τ (серая полоса — смена в горизонте; пунктир — d = L − 1)")
    fig.supylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def context_figure(curves: pd.DataFrame, h: int, path: Path, thr: float | None) -> None:
    """Отношение ошибки от d / L при разных L, сила strong: совпадают ли кривые по оси d / L."""
    c = curves[(curves.metric == "mae_ratio") & (curves.horizon == h) & curves.level.isin(["strong", "-"])]
    fig, axes = plt.subplots(3, 4, figsize=(16, 9.5), sharex=True, sharey=True)
    for ax, (cls, kind) in zip(axes.flat, panels(c)):
        p = c[(c.cls == cls) & (c.kind == kind)]
        ax.axvline(0, color="k", lw=0.8)
        ax.axvline(1, color="k", lw=0.8, ls="--")
        ax.axhline(1, color="gray", lw=0.8)
        if thr is not None:
            ax.axhline(thr, color="gray", lw=0.8, ls=":")
        for L, g in p.groupby("L"):
            ax.plot(g.d / L, g.value, color=L_COLORS.get(L, "gray"), lw=1.2, label=f"L = {L}")
        log_y(ax)
        ax.set_xlim(-0.3, 2.5)
        ax.set_title(f"{CLASS_TITLES[cls]}: {TITLES[kind]}", fontsize=10)
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(f"H1: отношение ошибки при разной длине контекста, сила strong, H = {h}")
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
        ax.set_title(f"{CLASS_TITLES[cls]}: {TITLES[kind]}", fontsize=10)
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


def fmt_d(v: float, L_main: int) -> str:
    return f"≥{L_main}" if np.isinf(v) else str(int(v))


def fmt_peak(v: float, lo: float, hi: float, d: int) -> str:
    return f"{v:.2f} [{lo:.2f}; {hi:.2f}], d = {d}"


def fmt_dstar(r, L_main: int) -> str:
    return f"{fmt_d(r.d_star, L_main)} [{fmt_d(r.d_star_lo, L_main)}; {fmt_d(r.d_star_hi, L_main)}]"


def main_threshold(star: pd.DataFrame, h: int) -> float | None:
    """Наименьший порог, при котором контроль без изменений даёт d* = 0 при всех L."""
    c = star[(star.cell == "none") & (star.horizon == h) & (star.metric == "mae_ratio")]
    for thr in THRESHOLDS:
        if (c[c.threshold == thr].d_star == 0).all():
            return thr
    return None


def summarize(star: pd.DataFrame, zones: pd.DataFrame, curves: pd.DataFrame, meta: dict, L_main: int) -> str:
    horizons = sorted(star.horizon.unique())
    cells = list(dict.fromkeys(star.cell))
    thr_main = {h: main_threshold(star, h) for h in horizons}
    zn = zone_names(L_main)
    out = [
        f"# Этап 3: провалы прогноза при смене режима — {meta['model']}",
        "",
        f"Дата: {meta['date']}. Скрипт: `python -m scripts.stage3_failures --model {meta['model']}`.",
        f"Кеш `{meta['cache']}` (хеш конфига {meta['dataset_hash']}): {meta['n_cells']} ячеек, "
        f"{meta['n_series']} рядов, {meta['n_windows']} окон, τ = {meta['tau']}, d от {meta['d_min']} до "
        f"{meta['d_max']} с шагом {meta['d_step']}. Основной контекст L = {L_main}.",
        f"В квадратных скобках — 95% бутстреп-интервал по рядам ячейки ({N_BOOT} повторов).",
        "",
        "## Как читать",
        "",
        "- **Отношение ошибки** — MAE медианы окна, делённая на типичную MAE той же модели на том же ряду "
        "в устойчивом режиме, который она прогнозирует (до смены — окна d < −H, после — окна "
        f"d ≥ {L_main}, где в контексте только новый режим). 1 — как в устойчивом режиме, 2 — вдвое хуже. "
        "Геометрическое среднее по рядам ячейки. Определения — в начале `scripts/stage3_failures.py`.",
        "- **Новый / старый** — во сколько раз типичная ошибка в новом режиме больше, чем в старом. "
        "Это трудность самого нового режима, а не провал перехода.",
        f"- **d*** — время восстановления: первое d ≥ 0, начиная с которого сглаженная кривая не выше порога "
        f"до d = {L_main - meta['d_step']}. «≥{L_main}» — не восстановилась, пока в контексте был старый режим.",
        "- **Покрытие** — доля точек горизонта внутри [q0.1, q0.9]; номинал 0.80.",
        "",
        "## Контроль без изменений и выбор порога d*",
        "",
    ]
    rows = []
    for h in horizons:
        c = star[(star.cell == "none") & (star.horizon == h) & (star.metric == "mae_ratio")]
        for thr in THRESHOLDS:
            cc = c[c.threshold == thr].sort_values("L")
            rows.append([str(h), str(thr)] + [fmt_d(v, L_main) for v in cc.d_star]
                        + [f"{cc.peak.max():.2f}"])
    Ls = sorted(star.L.unique())
    out.append(md_table(["H", "Порог"] + [f"d* контроля, L = {L}" for L in Ls] + ["Макс. кривой контроля при d ≥ 0"],
                        rows))
    out += ["", "У контроля смены нет, поэтому любое d* > 0 — ложное: кривая поднялась над порогом от шума. "
            "Основной порог для каждого H — наименьший, при котором ложных d* нет: "
            + "; ".join(f"H = {h} — {thr_main[h] if thr_main[h] else 'ни один из ' + str(THRESHOLDS)}"
                        for h in horizons) + "."]

    for h in horizons:
        thr = thr_main[h] or THRESHOLDS[-1]
        s = star[(star.horizon == h) & (star.L == L_main)]
        rows = []
        for cell in cells:
            r = s[(s.cell == cell) & (s.metric == "mae_ratio")].set_index("threshold")
            rc = s[(s.cell == cell) & (s.metric == "crps_ratio")].iloc[0]
            r0 = r.iloc[0]
            rows.append([cell, f"{r0.new_old:.2f}",
                         fmt_peak(r0.peak_h, r0.peak_h_lo, r0.peak_h_hi, r0.peak_h_d),
                         fmt_peak(r0.peak, r0.peak_lo, r0.peak_hi, r0.peak_d),
                         fmt_peak(rc.peak, rc.peak_lo, rc.peak_hi, rc.peak_d)]
                        + [fmt_dstar(r.loc[t], L_main) for t in THRESHOLDS])
        out += ["", f"## H1: пик ошибки и время восстановления, L = {L_main}, H = {h}", "",
                "Пики — максимум несглаженной кривой в зоне; максимум по многим шумным точкам смещён вверх "
                "(для сравнения — строка none).", "",
                md_table(["Ячейка", "Новый / старый", "Пик: смена в горизонте", "Пик при d ≥ 0",
                          "Пик CRPS при d ≥ 0"]
                         + [f"d* (порог {t}{', осн.' if t == thr else ''})" for t in THRESHOLDS], rows)]

    for h in horizons:
        thr = thr_main[h] or THRESHOLDS[-1]
        s = star[(star.horizon == h) & (star.metric == "mae_ratio") & (star.threshold == thr)]
        rows = [[cell] + [fmt_dstar(s[(s.cell == cell) & (s.L == L)].iloc[0], L_main) for L in Ls]
                + [f"{s[(s.cell == cell) & (s.L == L)].iloc[0].peak:.2f}" for L in Ls] for cell in cells]
        out += ["", f"## H1: d* при разной длине контекста, H = {h}, порог {thr}", "",
                "Если d* растёт пропорционально L, ошибка держится, пока старый режим в контексте; если d* от L "
                "почти не зависит, модель подстраивается по последним точкам. Эталон у каждого L свой "
                f"(окна d ≥ {L_main} с тем же L).", "",
                md_table(["Ячейка"] + [f"d*, L = {L}" for L in Ls] + [f"Пик, L = {L}" for L in Ls], rows)]

    groups = ["все ячейки", "change", "drift", "anomaly", "none"]
    for h in horizons:
        z = zones[(zones.horizon == h)]
        rows = []
        for zone in ["все окна", *zn]:
            row = [zone]
            for g in groups:
                r = z[(z.group == g) & (z.zone == zone)]
                row.append("—" if r.empty or r.iloc[0].n_series == 0 else
                           f"{r.iloc[0].coverage:.2f} [{r.iloc[0].coverage_lo:.2f}; {r.iloc[0].coverage_hi:.2f}]")
            rows.append(row)
        out += ["", f"## H2: глобальное и условное покрытие 80%-интервала, L = {L_main}, H = {h}", "",
                "Строка «все окна» — глобальная оценка, как в бенчмарках. Остальные — по зонам d.", "",
                md_table(["Зона d", *groups], rows)]

    for h in horizons:
        z = zones[(zones.horizon == h)]
        mins = curves[(curves.metric == "coverage") & (curves.horizon == h) & (curves.L == L_main)]
        rows = []
        for cell in cells:
            zc = z[z.group == cell].set_index("zone")
            m = mins[mins.cell == cell]
            i = m.value.idxmin()
            rows.append([cell] + [f"{zc.loc[name].coverage:.2f}" for name in zn]
                        + [f"{m.value[i]:.2f} (d = {int(m.d[i])})",
                           f"{zc.loc['0–63'].mae_ratio:.2f}", f"{zc.loc['0–63'].width_ratio:.2f}"])
        out += ["", f"## H2: покрытие по ячейкам, L = {L_main}, H = {h}", "",
                "Среднее по окнам зоны; интервалы — в coverage_zones.csv. Последние два столбца — во сколько "
                "раз ошибка и ширина интервала в зоне 0 ≤ d < 64 больше, чем в устойчивом режиме: если ошибка "
                "выросла сильнее ширины, интервал переуверен.", "",
                md_table(["Ячейка", *zn, "Мин. по d", "Ошибка ×, 0–63", "Ширина ×, 0–63"], rows)]

    h = max(horizons)
    show = [c for c in ("change-variance-strong", "change-ar-strong", "change-seasonality-strong",
                        "anomaly-freeze-strong", "none") if c in cells]
    zsel = [zn[0], zn[2], zn[-2], zn[-1]]
    z = zones[zones.horizon == h]
    rows = []
    for cell in show:
        zc = z[z.group == cell].set_index("zone")
        rows.append([cell, "обычный MASE"] + [f"{zc.loc[n].mase:.2f}" for n in zsel])
        rows.append(["", "отношение ошибки"] + [f"{zc.loc[n].mae_ratio:.2f}" for n in zsel])
    out += ["", f"## Почему не обычный MASE, L = {L_main}, H = {h}", "",
            "Геометрическое среднее по окнам зоны. У обычного MASE знаменатель — ошибка наивного прогноза "
            "по контексту — меняется, пока в контекст входит новый режим. Поэтому MASE «восстанавливается» "
            "к d ≈ L даже там, где ошибка модели после смены почти не меняется (дисперсия), опускается ниже "
            "уровня до смены (AR) или растёт после залипания, когда в контексте много нулевых разностей.", "",
            md_table(["Ячейка", "Метрика", *zsel], rows)]

    out += [
        "",
        "## Ограничения",
        "",
        f"- {meta['n_per_cell']} рядов на ячейку: интервалы по ячейке широкие, при H = 16 кривые шумнее "
        "(горизонт короче, ошибка окна изменчивее).",
        f"- Эталон нового режима — окна d ≥ {L_main}. У дрейфа новый режим наступает через "
        f"{meta['drift_duration']} шагов после τ, поэтому при L = {L_main} в контексте эталонных окон ещё есть "
        "часть дрейфа (полностью новый контекст — с d = L + duration − 1).",
        "- d* зависит от порога и сглаживания; показаны оба порога и контроль без изменений.",
        "- Синтетика близка к рядам KernelSynth из предобучения Chronos; на реальных данных величины могут быть другими.",
    ]
    return "\n".join(out) + "\n"


# ---------- запуск ----------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path, default=Path("results/cache/synth_v1"))
    parser.add_argument("--model", default="amazon/chronos-bolt-small")
    parser.add_argument("--out", type=Path, default=None, help="по умолчанию results/stage3/<модель>")
    args = parser.parse_args()
    out = args.out or Path("results/stage3") / model_dir(args.cache, args.model).name
    t0 = time.perf_counter()

    ds = load_dataset_meta(args.cache)
    cfg = yaml.safe_load((args.cache / "config.yaml").read_text())
    L_main, win = cfg["windows"]["context"], cfg["windows"]
    if win["d_max"] < L_main:
        raise ValueError(f"нет окон d ≥ L = {L_main}: эталон нового режима не построить")
    w = load_windows(args.cache)
    assert (w.row.to_numpy() == np.arange(len(w))).all()
    x, _ = load_series(args.cache)

    print(f"Метрики окон: {len(w)} окон × {len(L_VARIANTS)} длины контекста × {len(ds['horizons'])} горизонта …")
    df = add_ratios(window_metrics(args.cache, args.model, w, x, ds), w, L_main)
    print(f"  {time.perf_counter() - t0:.0f} с; сверка с src/metrics.py пройдена")
    curves, star = curves_and_dstar(df, L_main)
    dm = df[df.L == L_main].assign(zone=lambda a: assign_zone(a.d, a.horizon, L_main))
    cell_cls = w.drop_duplicates("cell").set_index("cell").cls
    groups = {"все ячейки": list(cell_cls.index)}
    groups |= {c: list(cell_cls.index[cell_cls == c]) for c in ("change", "drift", "anomaly", "none")}
    groups |= {cell: [cell] for cell in cell_cls.index if cell != "none"}
    zones = zone_stats(dm, groups, L_main)

    meta = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"), "model": args.model, "cache": str(args.cache),
        "dataset_hash": ds["hash"], "n_cells": ds["n_cells"], "n_series": ds["n_series"],
        "n_windows": ds["n_windows"], "n_per_cell": ds["n_series"] // ds["n_cells"], "tau": ds["tau"],
        "d_min": win["d_min"], "d_max": win["d_max"], "d_step": win["d_step"],
        "drift_duration": cfg["drift"]["duration"], "L_main": L_main, "L_variants": L_VARIANTS,
        "horizons": ds["horizons"], "thresholds": THRESHOLDS, "smooth": SMOOTH, "n_boot": N_BOOT, "seed": SEED,
        "env": f"python {platform.python_version()}, numpy {np.__version__}, pandas {pd.__version__}",
        "cache_meta": json.loads((model_dir(args.cache, args.model) / "meta.json").read_text()).get("env"),
    }
    out.mkdir(parents=True, exist_ok=True)
    cols = ["row", "cell", "series_idx", "d", "L", "horizon", "mae", "crps", "coverage", "width", "scale",
            "mase", "mae_ratio", "crps_ratio", "width_ratio"]
    df[cols].to_csv(out / "window_metrics.csv.gz", index=False, float_format="%.6g")
    curves.to_csv(out / "curves.csv.gz", index=False, float_format="%.5g")
    star.to_csv(out / "d_star.csv", index=False, float_format="%.5g")
    zones.to_csv(out / "coverage_zones.csv", index=False, float_format="%.5g")
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    for h in ds["horizons"]:
        thr = main_threshold(star, h)
        grid_figure(curves, "mae_ratio", h, L_main, out / f"fig_h1_error_H{h}.png",
                    f"H1: отношение ошибки к устойчивому режиму, L = {L_main}, H = {h}",
                    "ошибка / ошибка в устойчивом режиме", ref=1.0, thr=thr)
        context_figure(curves, h, out / f"fig_h1_context_H{h}.png", thr)
        grid_figure(curves, "coverage", h, L_main, out / f"fig_h2_coverage_H{h}.png",
                    f"H2: покрытие 80%-интервала, L = {L_main}, H = {h}", "доля факта внутри [q0.1, q0.9]",
                    ref=0.8, log=False)
        width_figure(curves, h, L_main, out / f"fig_h2_width_H{h}.png")

    summary = summarize(star, zones, curves, meta, L_main)
    (out / "summary.md").write_text(summary)
    g = zones[(zones.group == "все ячейки") & (zones.zone == "все окна")]
    print("  глобальное покрытие: " + ", ".join(f"H = {r.horizon}: {r.coverage:.3f}" for r in g.itertuples()))
    print("  основной порог d*: " + ", ".join(f"H = {h}: {main_threshold(star, h)}" for h in ds["horizons"]))
    print(f"Готово за {time.perf_counter() - t0:.0f} с → {out}/")


if __name__ == "__main__":
    main()
