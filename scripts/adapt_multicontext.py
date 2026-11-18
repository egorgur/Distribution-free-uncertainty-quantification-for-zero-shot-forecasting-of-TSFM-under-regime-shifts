"""Пилот Multi-Context (H7) по готовому кешу: протокол docs/protocol_v3_multicontext.md, v1.

    python -m scripts.adapt_multicontext                                   # основная: Bolt, H = 64
    python -m scripts.adapt_multicontext --model amazon/chronos-bolt-small --horizon 16
    python -m scripts.adapt_multicontext --model amazon/chronos-2 --horizon 64

Нового инференса нет. Порядок:
  1) seed 1 (results/cache/synth_v1_seed1): температура T = c · медиана признака на none,
     c из C_GRID — по минимуму средней по рядам log(MAE метода / MAE Full);
  2) seed 0 (results/cache/synth_v1): все методы с зафиксированными T, обрезки из cascade.npz,
     TG/SH/WIS/покрытие/d*, критерий H7 для sel_hind и mix_hind.
Прогноз «назад» берётся из окна pos − 16 того же ряда: его контекст кончается за 16 точек до t.
Результат: results/adapt_multicontext_v1/<модель>/H<H>/; summary.md перезаписывается, выводов в нём нет.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
import zlib
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from src import cascade as cas
from src import signals
from src.adaptation import evaluate as ev
from src.adaptation import multicontext as mc
from src.cache import load_dataset_meta, load_forecasts, load_series, load_windows, model_dir, signals_path
from src.evaluation import add_ratios, bootstrap_recovery, control_thresholds, recovery_v1, safe_log, smoothed_search
from src.models import QUANTILE_LEVELS

PROTOCOL = "multicontext_v1"
TUNE = Path("results/cache/synth_v1_seed1")
REPORT = Path("results/cache/synth_v1")
OUT = Path("results/adapt_multicontext_v1")
STAGE5_META = Path("results/stage5/meta.json")
THRESHOLDS_V1 = Path("results/protocol_h1_h3_v1/stage3")
H0 = 16  # длина прогноза «назад»
D_MIN_EVAL = -112
C_GRID = (0.25, 0.5, 1.0, 2.0, 4.0)
SH_TOL = 0.01
N_BOOT = 1000
SEED = 0
I_LO, I_MED, I_HI = (QUANTILE_LEVELS.index(v) for v in (0.1, 0.5, 0.9))
SCORES = ("hind", "width", "cons")
MAIN_METHODS = ("sel_hind", "mix_hind")
TRIMS = ("det_trim", "oracle_trim", "rag")
ZONES = [("pre", -np.inf, -64), ("in_h", -64, 0), ("0-64", 0, 64), ("64-128", 64, 128),
         ("128-512", 128, 512), ("512+", 512, np.inf)]


# ---------- данные ----------


def load_seed(root: Path, model_id: str, H: int) -> dict:
    """Окна, ряды, квантили четырёх длин, прогноз «назад», масштаб и будущее."""
    ds = load_dataset_meta(root)
    w = load_windows(root)
    assert (w.row.to_numpy() == np.arange(len(w))).all()
    x, _ = load_series(root)
    vi = [ds["variants"].index(f"L{L}") for L in mc.LENGTHS]
    q = load_forecasts(root, model_id, H, variants=vi)
    # Прогноз на 16 шагов: у Bolt — начало H64 (load_forecasts делает это сам), у Chronos-2 — свой вызов.
    med16 = load_forecasts(root, model_id, H0, variants=vi)[:, :, I_MED, :]
    if not (np.isfinite(q).all() and np.isfinite(med16).all()):
        raise ValueError(f"{root}: в кеше {model_id} есть непосчитанные окна")
    s, t, pos = w.series_idx.to_numpy(), w.t.to_numpy(), w.pos.to_numpy()
    L = max(mc.LENGTHS)
    scale = np.empty(len(w))
    for a in range(0, len(w), 4096):
        sl = slice(a, a + 4096)
        scale[sl] = signals.naive_scale(x[s[sl, None], t[sl, None] - L + 1 + np.arange(L)])
    prev = mc.previous_rows(s, pos, w.row.to_numpy(), H0)
    med_prev = np.where((prev >= 0)[:, None, None], med16[np.maximum(prev, 0)], np.nan)
    actual = x[s[:, None], t[:, None] - H0 + 1 + np.arange(H0)]
    fut = x[s[:, None], t[:, None] + 1 + np.arange(H)]
    return {"root": root, "ds": ds, "w": w, "q": q, "med_prev": med_prev, "actual": actual,
            "scale": scale, "fut": fut, "prev": prev}


def scores_of(d: dict) -> dict[str, np.ndarray]:
    """Три признака надёжности (окна, 4); неопределённые (нет окна t−16, s(t) = 0) → NaN."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = {"hind": mc.hindcast_scores(d["med_prev"], d["actual"], d["scale"]),
               "width": mc.width_scores(d["q"], d["scale"], I_LO, I_HI),
               "cons": mc.consistency_scores(d["q"], d["scale"])}
    return {k: np.where(np.isfinite(v), v, np.nan) for k, v in out.items()}


def window_metrics(q: np.ndarray, fut: np.ndarray) -> dict[str, np.ndarray]:
    """MAE медианы, WIS (2 · средняя pinball, как crps в src/cascade.py), покрытие и ширина 80%."""
    return cas.forecast_metrics(q, fut, QUANTILE_LEVELS)


def reference(d: dict, full: dict, H: int) -> pd.DataFrame:
    """Эталон Full L512 по протоколу v1 (полная сетка до фильтрации) — общий знаменатель всех методов."""
    f = pd.DataFrame({"row": d["w"].row, "L": 512, "horizon": H, "mae": full["mae"], "crps": full["crps"],
                      "width": full["width"], "scale": d["scale"]})
    r = add_ratios(f, d["w"], 512).sort_values("row").reset_index(drop=True)
    assert (r.row.to_numpy() == np.arange(len(f))).all()
    return r


def eval_mask(d: dict, mae_full: np.ndarray, ref: pd.DataFrame) -> np.ndarray:
    return ((d["w"].pos.to_numpy() >= D_MIN_EVAL) & (mae_full > 0)
            & np.isfinite(ref.mae_reference.to_numpy()) & (ref.mae_reference.to_numpy() > 0))


# ---------- методы ----------


def mae_only(qm: np.ndarray, fut: np.ndarray) -> np.ndarray:
    """MAE по медиане; qm — (окна, 1, H) только медиана или (окна, 9, H)."""
    med = qm[:, 0] if qm.shape[1] == 1 else qm[:, I_MED]
    return np.abs(fut - med).mean(axis=1)


def build_methods(d: dict, sc: dict, temps: dict, median_only: bool, reduce) -> tuple[dict, dict, dict]:
    """Все методы: reduce(квантили метода) — сразу сводится к метрикам, чтобы не держать квантили в памяти.

    Возвращает (метрики методов, выбор/веса для сохранения, время расчёта каждого метода, с).
    """
    q, levels = d["q"], ([0.5] if median_only else QUANTILE_LEVELS)
    qsel = q[:, :, [I_MED]] if median_only else q
    methods, extra, cost = {}, {"crossed": 0}, {}
    for j, L in enumerate(mc.LENGTHS):
        methods["full" if L == 512 else f"L{L}"] = reduce(qsel[:, j])
    n = len(q)
    for name, w in [("eq", np.full((n, len(mc.LENGTHS)), 1 / len(mc.LENGTHS)))] + [
            (x, None) for x in SCORES]:
        if w is None:
            t0 = time.perf_counter()
            idx = mc.select_index(sc[name])
            qm = mc.pick(qsel, idx)
            cost[f"sel_{name}"] = time.perf_counter() - t0
            methods[f"sel_{name}"] = reduce(qm)
            extra[f"sel_{name}_L"] = np.asarray(mc.LENGTHS)[idx]
        for obj in ("mix", "vin"):
            key = f"{obj}_{name}"
            t0 = time.perf_counter()
            ww = w if w is not None else mc.softmax_weights(sc[name], temps[key])
            if obj == "mix":
                qm, extra["crossed"] = mc.mixture_quantiles(q, ww, QUANTILE_LEVELS, levels)
            else:
                qm = mc.vincent(qsel, ww)
            cost[key] = time.perf_counter() - t0
            methods[key] = reduce(qm)
            if w is None:
                extra[f"{key}_w"] = ww
    return methods, extra, cost


def tune(d: dict, sc: dict, mae_full: np.ndarray, mask: np.ndarray) -> tuple[dict, pd.DataFrame]:
    """Подбор c на seed 1 (протокол, 4.1): минимум средней по рядам log(MAE/MAE Full), при равенстве — бо́льшее c."""
    none = (d["w"].cls.to_numpy() == "none") & mask
    series, rows, temps = d["w"].series_idx.to_numpy(), [], {}
    for x in SCORES:
        m = float(np.nanmedian(sc[x][none]))
        if not np.isfinite(m) or m <= 0:
            raise ValueError(f"медиана признака {x} на none не положительна: {m}")
        for obj in ("mix", "vin"):
            values = []
            for c in C_GRID:
                w = mc.softmax_weights(sc[x], c * m)
                if obj == "mix":
                    qm, _ = mc.mixture_quantiles(d["q"], w, QUANTILE_LEVELS, [0.5])
                else:
                    qm = mc.vincent(d["q"][:, :, [I_MED]], w)
                # Цель: среднее по рядам средних по окнам log(MAE метода / MAE Full).
                _, per_series = ev.series_means(ev.log_ratio(mae_only(qm, d["fut"]), mae_full), series, mask)
                values.append(float(per_series.mean()))
            # Минимум; при равенстве (до округления) — бо́льшее c.
            chosen = max(c for c, v in zip(C_GRID, values) if v <= min(values) + 1e-12)
            temps[f"{obj}_{x}"] = chosen * m
            rows += [{"method": f"{obj}_{x}", "score": x, "c": c, "median_none": m, "T": c * m,
                      "objective_mean_log_ratio": v, "chosen": c == chosen} for c, v in zip(C_GRID, values)]
    return temps, pd.DataFrame(rows)


def trims(root: Path, model_id: str, H: int, full: dict, n: int) -> dict[str, dict]:
    """Метрики обрезок этапа 6 (квантилей нет): П2 (det_trim), П4 (oracle_trim), П3 (rag)."""
    casc = np.load(signals_path(root, model_id, "cascade"))
    cmeta = json.loads(signals_path(root, model_id, "cascade").with_suffix(".json").read_text())
    hs = casc["horizon"] == H

    def col(z, key, sel):
        rows = z["row"][sel]
        assert np.array_equal(np.sort(rows), np.arange(n)), key
        out = np.empty(n, dtype=z[key].dtype)
        out[rows] = z[key][sel]
        return out

    h_loc = None if None in cas.STAGE_B[model_id] else H
    cfg, level = cas.STAGE_B[model_id][h_loc]
    loc_m = np.load(signals_path(root, model_id, "localization"))
    sel = np.ones(len(loc_m["row"]), bool) if h_loc is None else loc_m["horizon"] == h_loc
    score = col(loc_m, f"rbf_score_{level}", sel)
    thr = json.loads(STAGE5_META.read_text())["thresholds"][cfg][f"lvl:{level}"]
    key = "all" if "all" in cmeta["levels"] else f"H{H}"
    assert np.isclose(cmeta["levels"][key]["threshold"], thr, rtol=1e-12), "П3 посчитан с другим порогом"
    use = {"det_trim": ("own", score > thr),
           "oracle_trim": ("oracle", col(casc, "start_oracle", hs) != cas.NO_CROP),
           "rag": ("rag", col(casc, "start_rag", hs) != cas.NO_CROP)}
    out = {}
    for name, (src, crop) in use.items():
        out[name] = {m: np.where(crop, col(casc, f"{m}_{src}", hs).astype(float), full[m])
                     for m in ("mae", "crps", "coverage", "width")}
        out[name]["crop_share"] = float(crop.mean())
    return out


# ---------- сводки ----------


def group_masks(w: pd.DataFrame, mask: np.ndarray) -> dict[str, np.ndarray]:
    cls, pos, kind = w.cls.to_numpy(), w.pos.to_numpy(), w.kind.to_numpy()
    chg = (cls == "change") & mask
    g = {"change_0_64": chg & (pos >= 0) & (pos < 64), "change_0_128": chg & (pos >= 0) & (pos < 128),
         "none_all": (cls == "none") & mask, "change_pre": chg & (pos < -64), "change_all": chg,
         "all": mask, "drift_0_64": (cls == "drift") & mask & (pos >= 0) & (pos < 64),
         "anomaly_all": (cls == "anomaly") & mask}
    for k in sorted(set(kind[cls == "change"])):
        g[f"change_{k}_0_64"] = chg & (kind == k) & (pos >= 0) & (pos < 64)
    return g


def draws_for(series: np.ndarray, groups: dict) -> dict[str, np.ndarray]:
    """Одни и те же бутстреп-выборки рядов группы для всех методов (парность)."""
    return {g: ev.bootstrap_draws(len(np.unique(series[m])), N_BOOT, [SEED, zlib.crc32(g.encode())])
            for g, m in groups.items() if m.any()}


def summaries(res: dict, full: dict, ref: pd.DataFrame, w: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    """TG/SH по MAE и WIS для всех методов и групп, с 95% интервалами."""
    series = w.series_idx.to_numpy()
    groups = group_masks(w, mask)
    draws = draws_for(series, groups)
    rows = []
    for name, m in res.items():
        for metric in ("mae", "crps"):
            lr = ev.log_ratio(m[metric], full[metric])
            for g, gm in groups.items():
                if not gm.any():
                    continue
                point, boot = ev.group_mean(lr, series, gm, draws[g])
                tg = ev.gain(point, boot)
                sh = ev.harm(point, boot)
                rows.append({"method": name, "metric": "mae" if metric == "mae" else "wis", "group": g,
                             "n_series": draws[g].shape[1], "n_windows": int(gm.sum()), "mean_log_ratio": point,
                             "gain": tg[0], "gain_lo": tg[1], "gain_hi": tg[2],
                             "harm": sh[0], "harm_lo": sh[1], "harm_hi": sh[2]})
    return pd.DataFrame(rows)


def versus_trims(res: dict, w: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    """Парное mean log(MAE метода / MAE обрезки): change 0 ≤ d < 64 и все окна оценки (описательно)."""
    series = w.series_idx.to_numpy()
    groups = {k: v for k, v in group_masks(w, mask).items() if k in ("change_0_64", "all")}
    draws = draws_for(series, groups)
    rows = []
    for name in res:
        if name in TRIMS:
            continue
        for trim in ("det_trim", "oracle_trim"):
            lr = ev.log_ratio(res[name]["mae"], res[trim]["mae"])
            for g, gm in groups.items():
                point, boot = ev.group_mean(lr, series, gm, draws[g])
                lo, hi = ev.ci(boot)
                rows.append({"method": name, "versus": trim, "group": g, "ratio": np.exp(point),
                             "ratio_lo": np.exp(lo), "ratio_hi": np.exp(hi)})
    return pd.DataFrame(rows)


def zone_of(pos: np.ndarray) -> np.ndarray:
    z = np.full(len(pos), "", dtype=object)
    for name, a, b in ZONES:
        z[(pos >= a) & (pos < b)] = name
    return z


def zone_table(res: dict, ref: pd.DataFrame, w: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    """По зонам d (change) и none: отн. MAE и WIS к эталону Full, покрытие, отн. ширина (по рядам, затем среднее)."""
    series, cls = w.series_idx.to_numpy(), w.cls.to_numpy()
    zone = np.where(cls == "none", "none", zone_of(w.pos.to_numpy()))
    rows = []
    for name, m in res.items():
        vals = {"rel_mae": safe_log(m["mae"] / ref.mae_reference.to_numpy()),
                "rel_wis": safe_log(m["crps"] / ref.crps_reference.to_numpy()),
                "coverage": m["coverage"],
                "rel_width": safe_log(m["width"] / ref.width_reference.to_numpy())}
        for z in [n for n, _, _ in ZONES] + ["none"]:
            gm = mask & (zone == z) & np.isin(cls, ["change", "none"])
            if not gm.any():
                continue
            row = {"method": name, "zone": z, "n_windows": int(gm.sum())}
            for k, v in vals.items():
                ok = gm & np.isfinite(v)
                _, sm = ev.series_means(v, series, ok)
                row[k] = float(sm.mean()) if k == "coverage" else float(np.exp(sm.mean()))
            rows.append(row)
    return pd.DataFrame(rows)


def d_star(res: dict, ref: pd.DataFrame, w: pd.DataFrame, thresholds: dict, own_thr: dict) -> pd.DataFrame:
    """d* по протоколу v1 для change/drift; бутстреп — только для основного порога (Q95 Full)."""
    rows = []
    sub = w.cls.isin(["change", "drift"]).to_numpy()
    for name, m in res.items():
        lr = safe_log(m["mae"] / ref.mae_reference.to_numpy())
        df = pd.DataFrame({"cell": w.cell, "cls": w.cls, "kind": w.kind, "level": w.level,
                           "series": w.series_idx, "d": w.pos, "v": lr})[sub]
        for cell, g in df.groupby("cell", sort=False):
            P = g.pivot(index="series", columns="d", values="v")
            A, dgrid = P.to_numpy(), P.columns.to_numpy()
            idx = np.random.default_rng([SEED, zlib.crc32(cell.encode())]).integers(0, len(P), (N_BOOT, len(P)))
            base = {"method": name, "cell": cell, "cls": g.cls.iloc[0], "kind": g.kind.iloc[0],
                    "level": g.level.iloc[0]}
            result, stats, _ = bootstrap_recovery(A, dgrid, thresholds[95], n_boot=N_BOOT, indices=idx)
            row = {**base, "status": result.status, "d_star": result.d_star, "peak_d": result.peak_d,
                   "d_star_lo": stats["d_star_lo"], "d_star_hi": stats["d_star_hi"],
                   "boot_recovered_share": stats["boot_recovered_share"]}
            curve = smoothed_search(np.nanmean(A, axis=0), dgrid)
            for label, thr in (("q90_full", thresholds[90]), ("q95_own", own_thr.get(name, np.nan))):
                if np.isfinite(thr):
                    r = recovery_v1(curve, np.arange(0, 505, 8), thr)
                    row[f"status_{label}"], row[f"d_star_{label}"] = r.status, r.d_star
            rows.append(row)
    return pd.DataFrame(rows)


def own_control_thresholds(d1: dict, mae1: dict, ref1: pd.DataFrame, H: int) -> dict:
    """Q95 по контролю самого метода на seed 1 (чувствительность d*)."""
    out = {}
    w = d1["w"]
    for name, mae in mae1.items():
        ctrl = pd.DataFrame({"series_idx": w.series_idx, "d": w.pos, "L": 512, "horizon": H, "cls": w.cls,
                             "mae_ratio": mae / ref1.mae_reference.to_numpy()})
        t = control_thresholds(ctrl)
        out[name] = float(t.loc[t.percentile == 95, "threshold"].iloc[0])
    return out


def fmt(v, nd=3):
    return "—" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def summary_md(model_id, H, crit, summ, vt, zones, dstar, tuning, meta) -> str:
    lines = [f"# Multi-Context: {model_id}, H = {H}", "",
             "Создано `scripts/adapt_multicontext.py` по `docs/protocol_v3_multicontext.md` (v1). Файл "
             "перезаписывается; выводы — в `docs/results/adapt_multicontext.md`. Synth_v1 seed 0/1 уже изучены: "
             "результаты разведочные.", "",
             f"Окон оценки: {meta['n_eval']} из {meta['n_windows']}; окон без признака (fallback L512): "
             f"{meta['n_fallback_eval']} среди оценки; окон с упорядоченными квантилями в смеси: {meta['crossed_windows']}.",
             "", "## Критерий H7 (основной)", "",
             "| метод | TG64 [95%] | SH none [95%] | TG64 > 0 | SH_hi ≤ 1% | выполнен |", "|---|---|---|---|---|---|"]
    for r in crit.itertuples():
        lines.append(f"| {r.method} | {fmt(r.tg64)} [{fmt(r.tg64_lo)}; {fmt(r.tg64_hi)}] | {fmt(r.sh)} "
                     f"[{fmt(r.sh_lo)}; {fmt(r.sh_hi)}] | {r.tg_ok} | {r.sh_ok} | {r.passed} |")
    lines += ["", "## TG и SH по MAE (все методы)", "",
              "| метод | TG64 | TG128 | SH none | SH до смены | log r все окна |", "|---|---|---|---|---|---|"]
    s = summ[summ.metric == "mae"].set_index(["method", "group"])
    for mname in summ.method.unique():
        def g(group, col):
            return s.loc[(mname, group), col] if (mname, group) in s.index else np.nan
        lines.append(f"| {mname} | {fmt(g('change_0_64', 'gain'))} [{fmt(g('change_0_64', 'gain_lo'))}; "
                     f"{fmt(g('change_0_64', 'gain_hi'))}] | {fmt(g('change_0_128', 'gain'))} | "
                     f"{fmt(g('none_all', 'harm'))} [{fmt(g('none_all', 'harm_lo'))}; {fmt(g('none_all', 'harm_hi'))}] | "
                     f"{fmt(g('change_pre', 'harm'))} | {fmt(g('all', 'mean_log_ratio'), 4)} |")
    lines += ["", "## TG64 по WIS", "", "| метод | TG64 WIS | SH none WIS |", "|---|---|---|"]
    s2 = summ[summ.metric == "wis"].set_index(["method", "group"])
    for mname in summ.method.unique():
        lines.append(f"| {mname} | {fmt(s2.loc[(mname, 'change_0_64'), 'gain'])} | "
                     f"{fmt(s2.loc[(mname, 'none_all'), 'harm'])} |")
    lines += ["", "## Против обрезок (MAE метода / MAE обрезки, геом. среднее)", "",
              "| метод | vs det_trim 0–64 | vs oracle_trim 0–64 | vs det_trim все | vs oracle_trim все |",
              "|---|---|---|---|---|"]
    v = vt.set_index(["method", "versus", "group"])
    for mname in vt.method.unique():
        cells = [f"{fmt(v.loc[(mname, t, g), 'ratio'])} [{fmt(v.loc[(mname, t, g), 'ratio_lo'])}; "
                 f"{fmt(v.loc[(mname, t, g), 'ratio_hi'])}]"
                 for g in ("change_0_64", "all") for t in ("det_trim", "oracle_trim")]
        lines.append(f"| {mname} | " + " | ".join(cells) + " |")
    lines += ["", "## TG64 по типам смены (MAE)", ""]
    kinds = sorted(g[len("change_"):-len("_0_64")] for g in summ.group.unique()
                   if g.startswith("change_") and g.endswith("_0_64") and g != "change_0_64")
    lines += ["| метод | " + " | ".join(kinds) + " |", "|---" * (len(kinds) + 1) + "|"]
    for mname in summ.method.unique():
        lines.append(f"| {mname} | " + " | ".join(fmt(s.loc[(mname, f'change_{k}_0_64'), 'gain']) for k in kinds) + " |")
    lines += ["", "## Покрытие 80% по зонам (change; none)", ""]
    zn = [n for n, _, _ in ZONES] + ["none"]
    zt = zones.set_index(["method", "zone"])
    lines += ["| метод | " + " | ".join(zn) + " |", "|---" * (len(zn) + 1) + "|"]
    for mname in zones.method.unique():
        lines.append(f"| {mname} | " + " | ".join(fmt(zt.loc[(mname, z), 'coverage']) if (mname, z) in zt.index
                                                  else "—" for z in zn) + " |")
    lines += ["", "## Отн. MAE к эталону Full по зонам", "", "| метод | " + " | ".join(zn) + " |",
              "|---" * (len(zn) + 1) + "|"]
    for mname in zones.method.unique():
        lines.append(f"| {mname} | " + " | ".join(fmt(zt.loc[(mname, z), 'rel_mae']) if (mname, z) in zt.index
                                                  else "—" for z in zn) + " |")
    strong = dstar[dstar.level == "strong"]
    if len(strong):
        lines += ["", "## d* (основной порог Q95 Full), сильные смены", ""]
        cells = list(strong.cell.unique())
        lines += ["| метод | " + " | ".join(c.replace("change-", "").replace("-strong", "") for c in cells) + " |",
                  "|---" * (len(cells) + 1) + "|"]
        ds_ = strong.set_index(["method", "cell"])
        for mname in strong.method.unique():
            vals = []
            for c in cells:
                r = ds_.loc[(mname, c)]
                vals.append(str(int(r.d_star)) if r.status == "recovered" else
                            ("нет дегр." if r.status == "no_degradation" else "не подтв."))
            lines.append(f"| {mname} | " + " | ".join(vals) + " |")
    lines += ["", "## Подбор температуры (seed 1)", "", "| метод | c | T | цель |", "|---|---|---|---|"]
    for r in tuning[tuning.chosen].itertuples():
        lines.append(f"| {r.method} | {r.c} | {fmt(r.T, 4)} | {fmt(r.objective_mean_log_ratio, 5)} |")
    lines += ["", "## Стоимость", "",
              "Прогнозов модели на окно в онлайн-работе: full/L* — 1; sel/mix/vin по ширине или согласованности — 4; "
              "по hindcast — 4 (+4 из t−16, уже сделанных онлайн); обрезки — 1 + детектор.", "",
              "| метод | время расчёта по кешу, с |", "|---|---|"]
    for k, val in meta["seconds_method"].items():
        lines.append(f"| {k} | {val:.1f} |")
    return "\n".join(lines) + "\n"


# ---------- главный сценарий ----------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="amazon/chronos-bolt-small", choices=sorted(cas.STAGE_B))
    parser.add_argument("--horizon", type=int, default=64, choices=(16, 64))
    args = parser.parse_args()
    H, model_id = args.horizon, args.model
    t_start = time.perf_counter()
    out = OUT / model_dir(OUT, model_id).name / f"H{H}"
    out.mkdir(parents=True, exist_ok=True)

    # 1) настройка на seed 1
    print(f"seed 1: {model_id}, H = {H} …", flush=True)
    d1 = load_seed(TUNE, model_id, H)
    sc1 = scores_of(d1)
    full1 = window_metrics(d1["q"][:, mc.FULL], d1["fut"])
    ref1 = reference(d1, full1, H)
    mask1 = eval_mask(d1, full1["mae"], ref1)
    temps, tuning = tune(d1, sc1, full1["mae"], mask1)
    print("  T:", {k: round(v, 4) for k, v in temps.items()}, f"{time.perf_counter() - t_start:.0f} с", flush=True)
    mae1, _, _ = build_methods(d1, sc1, temps, median_only=True, reduce=lambda qm: mae_only(qm, d1["fut"]))
    assert np.allclose(mae1["full"], full1["mae"])
    own_thr = own_control_thresholds(d1, mae1, ref1, H)
    tune_hash = d1["ds"]["hash"]
    del d1, sc1

    # 2) отчёт на seed 0
    print("seed 0 …", flush=True)
    d0 = load_seed(REPORT, model_id, H)
    sc0 = scores_of(d0)
    res, extra, cost = build_methods(d0, sc0, temps, median_only=False,
                                     reduce=lambda qm: window_metrics(qm, d0["fut"]))
    full = res["full"]
    ref = reference(d0, full, H)
    mask = eval_mask(d0, full["mae"], ref)
    n = len(d0["w"])
    tr = trims(REPORT, model_id, H, full, n)
    crop_share = {k: v.pop("crop_share") for k, v in tr.items()}
    res.update(tr)
    print(f"  методы готовы, {time.perf_counter() - t_start:.0f} с", flush=True)

    w = d0["w"]
    summ = summaries(res, full, ref, w, mask)
    vt = versus_trims(res, w, mask)
    zones = zone_table(res, ref, w, mask)
    thr_v1 = pd.read_csv(THRESHOLDS_V1 / model_dir(OUT, model_id).name / "thresholds.csv")
    tv = thr_v1[(thr_v1.L == 512) & (thr_v1.horizon == H)].set_index("percentile").threshold.to_dict()
    print("  d* …", flush=True)
    dstar = d_star(res, ref, w, tv, own_thr)

    # критерий H7
    s = summ[summ.metric == "mae"].set_index(["method", "group"])
    crit = []
    for mname in MAIN_METHODS:
        a, b = s.loc[(mname, "change_0_64")], s.loc[(mname, "none_all")]
        crit.append({"method": mname, "tg64": a.gain, "tg64_lo": a.gain_lo, "tg64_hi": a.gain_hi,
                     "sh": b.harm, "sh_lo": b.harm_lo, "sh_hi": b.harm_hi,
                     "tg_ok": bool(a.gain_lo > 0), "sh_ok": bool(b.harm_hi <= SH_TOL)})
        crit[-1]["passed"] = crit[-1]["tg_ok"] and crit[-1]["sh_ok"]
    crit = pd.DataFrame(crit)
    primary = (model_id == "amazon/chronos-bolt-small") and H == 64

    # сохранение
    wm = w[["row", "cell", "cls", "kind", "level", "series_idx", "pos"]].copy()
    wm["eval"] = mask
    wm["scale"] = d0["scale"]
    wm["mae_reference"], wm["crps_reference"] = ref.mae_reference, ref.crps_reference
    for k, v in sc0.items():
        for j, L in enumerate(mc.LENGTHS):
            wm[f"score_{k}_L{L}"] = v[:, j]
    for k in [k for k in extra if k.startswith("sel_")]:
        wm[k] = extra[k]
    for k in [k for k in extra if k.endswith("_w")]:
        for j, L in enumerate(mc.LENGTHS):
            wm[f"{k}_L{L}"] = extra[k][:, j]
    for name, m in res.items():
        for metric, v in m.items():
            wm[f"{metric}_{name}"] = v
    wm.to_csv(out / "window_metrics.csv.gz", index=False, float_format="%.7g")
    tuning.to_csv(out / "tuning.csv", index=False)
    summ.to_csv(out / "summary_gain_harm.csv", index=False)
    vt.to_csv(out / "versus_trims.csv", index=False)
    zones.to_csv(out / "zones.csv", index=False)
    dstar.to_csv(out / "d_star.csv", index=False)
    crit.to_csv(out / "criterion.csv", index=False)
    fallback = ~mc.valid_rows(sc0["hind"])
    meta = {
        "protocol": PROTOCOL, "date": datetime.now().strftime("%Y-%m-%d %H:%M"), "model": model_id, "horizon": H,
        "primary_configuration": primary, "tune_cache": str(TUNE), "report_cache": str(REPORT),
        "tune_hash": tune_hash, "report_hash": d0["ds"]["hash"], "n_windows": n, "n_eval": int(mask.sum()),
        "n_fallback_eval": int((fallback & mask).sum()), "n_fallback_all": int(fallback.sum()),
        "crossed_windows": int(extra["crossed"]), "temperatures": temps, "c_grid": list(C_GRID),
        "thresholds_v1": tv, "own_control_q95": own_thr, "trim_crop_share": crop_share,
        "sh_tolerance": SH_TOL, "n_boot": N_BOOT, "seed": SEED, "criterion": crit.to_dict("records"),
        "seconds_method": cost, "seconds_total": round(time.perf_counter() - t_start, 1),
        "env": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                "chronos-forecasting": version("chronos-forecasting")},
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=float))
    (out / "summary.md").write_text(summary_md(model_id, H, crit, summ, vt, zones, dstar, tuning, meta))
    print(crit.to_string(index=False))
    print(f"Готово: {out}, {meta['seconds_total']} с", flush=True)


if __name__ == "__main__":
    main()
