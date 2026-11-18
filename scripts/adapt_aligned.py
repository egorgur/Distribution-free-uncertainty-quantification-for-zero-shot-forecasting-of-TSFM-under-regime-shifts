"""Оценка Regime-Aligned Context (H8): протокол docs/protocol_v3_aligned.md, v1.

    python -m scripts.adapt_aligned                                   # основная: Bolt, H = 64
    python -m scripts.adapt_aligned --horizon 16
    python -m scripts.adapt_aligned --model amazon/chronos-2 --horizon 64

Нужны: кеш synth_v1 (L512/L64/L128), прогнозы по выровненным контекстам
(scripts/cache_aligned.py → <кеш>/<модель>/aligned_v1/), cascade.npz/localization.npz (обрезки),
результаты пилота H7 (mix_hind для сравнения). Нового инференса нет.

Выбор кандидата в окне pos — по ошибке «назад»: медиана прогноза того же кандидата из окна pos − 16
на x[t − 15 … t]. Окно pos − 16 построено по истории до t − 16, скрытые точки в него не входят.
Результат: results/adapt_aligned_v1/<модель>/H<H>/; summary.md перезаписывается, выводов в нём нет.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import adapt_multicontext as amc
from scripts.cache_aligned import SUBDIR, VARIANTS, variant_name
from src import cascade as cas
from src.adaptation import alignment as al
from src.adaptation import evaluate as ev
from src.cache import model_dir
from src.models import QUANTILE_LEVELS

PROTOCOL = "aligned_v1"
REPORT = amc.REPORT
OUT = Path("results/adapt_aligned_v1")
P1 = Path("results/adapt_multicontext_v1")
TIE_REL = 1e-6
METRICS = ("mae", "crps", "coverage", "width")
NAMES = [variant_name(k, a) for k, a in VARIANTS]


def load_aligned(root: Path, model_id: str, H: int, n: int) -> dict:
    """Метрики (окна, 12), медианы первых 16 шагов (окна, 12, 16), полные квантили α = 1 (окна, 3, 9, H)."""
    d = model_dir(root, model_id) / SUBDIR
    files = sorted(f for f in d.glob("*.npz"))
    out = {"med16": np.full((n, len(VARIANTS), 16), np.nan), "q_full": None,
           "fallback": np.zeros(n, bool), **{m: np.full((n, len(VARIANTS)), np.nan) for m in METRICS}}
    seen = np.zeros(n, bool)
    for f in files:
        with np.load(f) as z:
            r = z["row"]
            seen[r] = True
            out["med16"][r] = z["med16"]
            out["fallback"][r] = z["fallback"]
            for m in METRICS:
                out[m][r] = z[f"{'cov' if m == 'coverage' else m}{H}"]
            # У Bolt короткий горизонт — начало прогноза H64, отдельно не хранится.
            qf = z[f"q{H}_full"] if f"q{H}_full" in z else z["q64_full"][..., :H]
            if out["q_full"] is None:
                out["q_full"] = np.full((n, *qf.shape[1:]), np.nan, dtype=np.float32)
            out["q_full"][r] = qf
    if not seen.all():
        raise ValueError(f"{d}: посчитаны не все окна ({seen.sum()} из {n})")
    return out


def gather(cands: list[dict], idx: np.ndarray) -> dict[str, np.ndarray]:
    """Метрики выбранного кандидата в каждом окне."""
    return {m: np.stack([c[m] for c in cands], axis=1)[np.arange(len(idx)), idx] for m in METRICS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="amazon/chronos-bolt-small", choices=sorted(cas.STAGE_B))
    parser.add_argument("--horizon", type=int, default=64, choices=(16, 64))
    args = parser.parse_args()
    H, model_id = args.horizon, args.model
    t_start = time.perf_counter()
    mname = model_dir(OUT, model_id).name
    out = OUT / mname / f"H{H}"
    out.mkdir(parents=True, exist_ok=True)

    d = amc.load_seed(REPORT, model_id, H)
    w, n = d["w"], len(d["w"])
    q = d["q"]  # (окна, 4: L64, L128, L256, L512, 9, H)
    full = amc.window_metrics(q[:, 3], d["fut"])
    short = amc.window_metrics(q[:, 0], d["fut"])
    ref = amc.reference(d, full, H)
    mask = amc.eval_mask(d, full["mae"], ref)
    a = load_aligned(REPORT, model_id, H, n)
    # Сверка: метрики α = 1 из сохранённых квантилей совпадают с сохранёнными метриками.
    for j in range(len(al.KINDS)):
        m = amc.window_metrics(a["q_full"][:, j], d["fut"])
        assert np.allclose(m["mae"], a["mae"][:, j], rtol=1e-4, atol=1e-6), "метрики aligned_v1 не совпадают"
    aligned = [{m: a[m][:, j].astype(float) for m in METRICS} for j in range(len(VARIANTS))]

    # Ошибки «назад» всех кандидатов: Original/Short — кеш (окно pos − 16), выровненные — aligned_v1 того же окна.
    prev, act, s = d["prev"], d["actual"], d["scale"]
    has = prev >= 0
    pp = np.maximum(prev, 0)

    def back(med16_prev):  # (окна, 16) → (окна,)
        return np.where(has, np.abs(med16_prev - act).mean(axis=1), np.nan)

    err = {"Original": back(d["med_prev"][:, 3]), "Short": back(d["med_prev"][:, 0])}
    for j, name in enumerate(NAMES):
        err[name] = back(a["med16"][pp, j])

    cand = {"Original": full, "Short": short, **{nm: aligned[j] for j, nm in enumerate(NAMES)}}
    sets = {
        "sel_align": ["Original", "level", "scale", "trend", "Short"],
        "sel_align_noshort": ["Original", "level", "scale", "trend"],
        "sel_partial": ["Original"] + [variant_name(k, al_) for al_ in (0.25, 0.5, 0.75, 1.0) for k in al.KINDS]
                       + ["Short"],
    }
    res = {"full": full, "L64": short, "L128": amc.window_metrics(q[:, 1], d["fut"])}
    choice = {}
    for meth, names in sets.items():
        idx = al.choose(np.stack([err[k] for k in names], axis=1), s, TIE_REL)
        res[meth] = gather([cand[k] for k in names], idx)
        choice[meth] = np.asarray(names, dtype=object)[idx]
    for j, nm in enumerate(NAMES):
        res[f"fix_{nm}"] = aligned[j]
    tr = amc.trims(REPORT, model_id, H, full, n)
    crop_share = {k: v.pop("crop_share") for k, v in tr.items()}
    res.update({k: tr[k] for k in ("det_trim", "oracle_trim")})
    p1 = pd.read_csv(P1 / mname / f"H{H}" / "window_metrics.csv.gz",
                     usecols=["row"] + [f"{m}_mix_hind" for m in METRICS] + ["mae_full"])
    assert (p1.row.to_numpy() == np.arange(n)).all() and np.allclose(p1.mae_full, full["mae"], rtol=1e-5)
    res["mix_hind_p1"] = {m: p1[f"{m}_mix_hind"].to_numpy() for m in METRICS}

    summ = amc.summaries(res, full, ref, w, mask)
    vt = amc.versus_trims(res, w, mask)
    zones = amc.zone_table(res, ref, w, mask)
    thr_v1 = pd.read_csv(amc.THRESHOLDS_V1 / mname / "thresholds.csv")
    tv = thr_v1[(thr_v1.L == 512) & (thr_v1.horizon == H)].set_index("percentile").threshold.to_dict()
    main_methods = ["full", "L64", "sel_align", "sel_align_noshort", "sel_partial", "fix_level", "fix_scale",
                    "fix_trend", "det_trim", "oracle_trim", "mix_hind_p1"]
    dstar = amc.d_star({k: res[k] for k in main_methods}, ref, w, tv, {})

    # Критерий H8: часть 1 (как H7) и часть 2 (против удаления истории).
    sm = summ[summ.metric == "mae"].set_index(["method", "group"])
    g1, g2 = sm.loc[("sel_align", "change_0_64")], sm.loc[("sel_align", "none_all")]
    series = w.series_idx.to_numpy()
    groups = amc.group_masks(w, mask)
    draws = amc.draws_for(series, {"change_0_64": groups["change_0_64"]})["change_0_64"]
    part2 = {}
    for other in ("L64", "det_trim"):
        point, boot = ev.group_mean(ev.log_ratio(res["sel_align"]["mae"], res[other]["mae"]), series,
                                    groups["change_0_64"], draws)
        lo, hi = ev.ci(boot)
        part2[other] = {"ratio": float(np.exp(point)), "lo": float(np.exp(lo)), "hi": float(np.exp(hi)),
                        "ok": bool(np.exp(hi) < 1)}
    crit = {"tg64": g1.gain, "tg64_lo": g1.gain_lo, "tg64_hi": g1.gain_hi, "sh": g2.harm, "sh_lo": g2.harm_lo,
            "sh_hi": g2.harm_hi, "tg_ok": bool(g1.gain_lo > 0), "sh_ok": bool(g2.harm_hi <= amc.SH_TOL),
            "vs_L64": part2["L64"], "vs_det_trim": part2["det_trim"]}
    crit["part1"] = crit["tg_ok"] and crit["sh_ok"]
    crit["part2"] = part2["L64"]["ok"] and part2["det_trim"]["ok"]
    crit["passed"] = crit["part1"] and crit["part2"]

    # Частоты выбора по зонам.
    zone = np.where(w.cls.to_numpy() == "none", "none", amc.zone_of(w.pos.to_numpy()))
    sub = mask & np.isin(w.cls.to_numpy(), ["change", "none"])
    freq = pd.concat({m: pd.crosstab(zone[sub], c[sub], normalize="index") for m, c in choice.items()})
    freq.to_csv(out / "choice_by_zone.csv")

    wm = pd.concat([w[["row", "cell", "cls", "kind", "level", "series_idx", "pos"]],
                    pd.DataFrame({"eval": mask, "scale": s, "mae_reference": ref.mae_reference,
                                  "fallback_scale": a["fallback"]}),
                    pd.DataFrame({f"err_{k}": v for k, v in err.items()}),
                    pd.DataFrame({f"choice_{k}": v for k, v in choice.items()}),
                    pd.DataFrame({f"{m}_{k}": v for k, r in res.items() for m, v in r.items()})], axis=1)
    wm.to_csv(out / "window_metrics.csv.gz", index=False, float_format="%.7g")
    summ.to_csv(out / "summary_gain_harm.csv", index=False)
    vt.to_csv(out / "versus_trims.csv", index=False)
    zones.to_csv(out / "zones.csv", index=False)
    dstar.to_csv(out / "d_star.csv", index=False)
    meta = {"protocol": PROTOCOL, "date": datetime.now().strftime("%Y-%m-%d %H:%M"), "model": model_id,
            "horizon": H, "report_hash": d["ds"]["hash"], "n_windows": n, "n_eval": int(mask.sum()),
            "fallback_scale_windows": int(a["fallback"].sum()),
            "fallback_scale_eval": int((a["fallback"] & mask).sum()),
            "no_hindcast_eval": int((~(np.isfinite(err["Original"]) & (s > 0)) & mask).sum()),
            "trim_crop_share": crop_share, "criterion": crit, "thresholds_v1": tv, "n_boot": amc.N_BOOT,
            "seconds_total": round(time.perf_counter() - t_start, 1)}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=float))
    (out / "summary.md").write_text(summary_md(model_id, H, crit, summ, vt, zones, dstar, freq, meta))
    print(json.dumps(crit, ensure_ascii=False, default=float, indent=1))
    print(f"Готово: {out}, {meta['seconds_total']} с")


def summary_md(model_id, H, crit, summ, vt, zones, dstar, freq, meta) -> str:
    f = amc.fmt
    L = [f"# Regime-Aligned Context: {model_id}, H = {H}", "",
         "Создано `scripts/adapt_aligned.py` по `docs/protocol_v3_aligned.md` (v1). Файл перезаписывается; выводы — "
         "в `docs/results/adapt_aligned.md`. Synth_v1 seed 0 уже изучен: результаты разведочные.", "",
         f"Окон оценки: {meta['n_eval']}; scale заменён на level: {meta['fallback_scale_eval']} окон оценки; "
         f"без ошибки «назад» (→ Original): {meta['no_hindcast_eval']}.", "", "## Критерий H8 (sel_align)", "",
         f"- Часть 1: TG64 = {f(crit['tg64'])} [{f(crit['tg64_lo'])}; {f(crit['tg64_hi'])}], "
         f"SH none = {f(crit['sh'])} [{f(crit['sh_lo'])}; {f(crit['sh_hi'])}] → {crit['part1']}",
         f"- Часть 2 (0 ≤ d < 64): MAE/MAE(L64) = {f(crit['vs_L64']['ratio'])} [{f(crit['vs_L64']['lo'])}; "
         f"{f(crit['vs_L64']['hi'])}], MAE/MAE(det_trim) = {f(crit['vs_det_trim']['ratio'])} "
         f"[{f(crit['vs_det_trim']['lo'])}; {f(crit['vs_det_trim']['hi'])}] → {crit['part2']}",
         f"- **Выполнен: {crit['passed']}**", "", "## TG и SH по MAE", "",
         "| метод | TG64 [95%] | TG128 | SH none [95%] | SH до смены | TG64 WIS |", "|---|---|---|---|---|---|"]
    s = summ[summ.metric == "mae"].set_index(["method", "group"])
    sw = summ[summ.metric == "wis"].set_index(["method", "group"])
    for m in summ.method.unique():
        L.append(f"| {m} | {f(s.loc[(m, 'change_0_64'), 'gain'])} [{f(s.loc[(m, 'change_0_64'), 'gain_lo'])}; "
                 f"{f(s.loc[(m, 'change_0_64'), 'gain_hi'])}] | {f(s.loc[(m, 'change_0_128'), 'gain'])} | "
                 f"{f(s.loc[(m, 'none_all'), 'harm'])} [{f(s.loc[(m, 'none_all'), 'harm_lo'])}; "
                 f"{f(s.loc[(m, 'none_all'), 'harm_hi'])}] | {f(s.loc[(m, 'change_pre'), 'harm'])} | "
                 f"{f(sw.loc[(m, 'change_0_64'), 'gain'])} |")
    kinds = sorted(g[len("change_"):-len("_0_64")] for g in summ.group.unique()
                   if g.startswith("change_") and g.endswith("_0_64") and g != "change_0_64")
    L += ["", "## TG64 по типам смены (MAE)", "", "| метод | " + " | ".join(kinds) + " |",
          "|---" * (len(kinds) + 1) + "|"]
    for m in summ.method.unique():
        L.append(f"| {m} | " + " | ".join(f(s.loc[(m, f'change_{k}_0_64'), 'gain']) for k in kinds) + " |")
    zn = [z for z, _, _ in amc.ZONES] + ["none"]
    zt = zones.set_index(["method", "zone"])
    L += ["", "## Покрытие 80% по зонам", "", "| метод | " + " | ".join(zn) + " |", "|---" * (len(zn) + 1) + "|"]
    for m in zones.method.unique():
        L.append(f"| {m} | " + " | ".join(f(zt.loc[(m, z), 'coverage']) if (m, z) in zt.index else "—"
                                         for z in zn) + " |")
    L += ["", "## Против обрезок (MAE / MAE обрезки)", "",
          "| метод | det_trim 0–64 | oracle_trim 0–64 | det_trim все | oracle_trim все |", "|---|---|---|---|---|"]
    v = vt.set_index(["method", "versus", "group"])
    for m in vt.method.unique():
        L.append(f"| {m} | " + " | ".join(
            f"{f(v.loc[(m, t, g), 'ratio'])} [{f(v.loc[(m, t, g), 'ratio_lo'])}; {f(v.loc[(m, t, g), 'ratio_hi'])}]"
            for g in ("change_0_64", "all") for t in ("det_trim", "oracle_trim")) + " |")
    fr = freq.loc["sel_align"].dropna(axis=1, how="all").round(3)
    L += ["", "## Частоты выбора sel_align по зонам", "", "| зона | " + " | ".join(fr.columns) + " |",
          "|---" * (len(fr.columns) + 1) + "|"]
    L += [f"| {z} | " + " | ".join(f(v) for v in row) + " |" for z, row in fr.iterrows()]
    strong = dstar[dstar.level == "strong"]
    cells = list(strong.cell.unique())
    L += ["", "## d* (Q95 Full), сильные смены", "",
          "| метод | " + " | ".join(c.replace("change-", "").replace("-strong", "") for c in cells) + " |",
          "|---" * (len(cells) + 1) + "|"]
    ds_ = strong.set_index(["method", "cell"])
    for m in strong.method.unique():
        vals = [str(int(ds_.loc[(m, c)].d_star)) if ds_.loc[(m, c)].status == "recovered" else
                ("нет дегр." if ds_.loc[(m, c)].status == "no_degradation" else "не подтв.") for c in cells]
        L.append(f"| {m} | " + " | ".join(vals) + " |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
