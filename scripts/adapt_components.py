"""Оценка Regime-Component Adaptation (H9): протокол docs/protocol_v3_component.md, v1.

    python -m scripts.adapt_components                                    # основная: Bolt, H = 64
    python -m scripts.adapt_components --horizon 16
    python -m scripts.adapt_components --model amazon/chronos-2 --horizon 64

Нового инференса нет: реакции берутся из основного кеша (Full, L64, L128) и кеша aligned_v1 (H8).
Пороги компонент — по рядам none seed 1 (только сырые ряды), отчёт — seed 0.
Результат: results/adapt_components_v1/<модель>/H<H>/; summary.md перезаписывается, выводов в нём нет.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import adapt_aligned as aa
from scripts import adapt_multicontext as amc
from src import cascade as cas
from src import signals
from src.adaptation import components as cp
from src.adaptation import evaluate as ev
from src.cache import load_series, load_windows, model_dir

PROTOCOL = "component_v1"
OUT = Path("results/adapt_components_v1")
H8 = Path("results/adapt_aligned_v1")
L = 512
CHUNK = 4096
METRICS = aa.METRICS


def component_stats(root: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Окна и z всех компонент по контексту L512 каждого окна (без модели)."""
    w = load_windows(root)
    x, _ = load_series(root)
    s, t = w.series_idx.to_numpy(), w.t.to_numpy()
    z = {k: np.empty(len(w)) for k in cp.STATS}
    for a in range(0, len(w), CHUNK):
        sl = slice(a, a + CHUNK)
        ctx = x[s[sl, None], t[sl, None] - L + 1 + np.arange(L)]
        zz = cp.statistics(ctx, signals.naive_scale(ctx))
        for k in cp.STATS:
            z[k][sl] = zz[k]
    return w, z


def react(reactions: dict[int, dict], comp: np.ndarray) -> dict[str, np.ndarray]:
    """Метрики окна при реакции, выбранной компонентой comp (0…4)."""
    out = {}
    for m in METRICS:
        stack = np.stack([reactions[c][m] for c in range(len(cp.COMPONENTS))], axis=1)
        out[m] = stack[np.arange(len(comp)), comp]
    return out


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

    # 1) пороги по none seed 1 (окна pos ≥ −112, как в оценке)
    w1, z1 = component_stats(amc.TUNE)
    m1 = (w1.cls.to_numpy() == "none") & (w1.pos.to_numpy() >= amc.D_MIN_EVAL)
    thr = cp.thresholds(z1, m1)

    # 2) seed 0: статистики, правило, истинный тип и момент
    d = amc.load_seed(amc.REPORT, model_id, H)
    w, n = d["w"], len(d["w"])
    _, z0 = component_stats(amc.REPORT)
    detected = cp.decide(z0, thr)
    true = cp.true_component(w.cls.to_numpy(), w.kind.to_numpy())
    pos, cls = w.pos.to_numpy(), w.cls.to_numpy()
    in_context = np.isin(cls, ["change", "drift"]) & (pos >= 0) & (pos < L - 1)

    # 3) реакции: метрики окна для каждой компоненты
    q = d["q"]
    full = amc.window_metrics(q[:, 3], d["fut"])
    l64 = amc.window_metrics(q[:, 0], d["fut"])
    l128 = amc.window_metrics(q[:, 1], d["fut"])
    ref = amc.reference(d, full, H)
    mask = amc.eval_mask(d, full["mae"], ref)
    a = aa.load_aligned(amc.REPORT, model_id, H, n)
    idx = {nm: j for j, nm in enumerate(aa.NAMES)}
    al_m = {nm: {m: a[m][:, j].astype(float) for m in METRICS} for nm, j in idx.items()}
    main_r = {0: full, 1: al_m["level"], 2: al_m["scale"], 3: al_m["trend"], 4: l64}
    h8_r = {0: full, 1: al_m["level_0.5"], 2: al_m["scale_0.5"], 3: al_m["trend_0.5"], 4: l128}

    zero = np.zeros(n, int)
    res = {"full": full, "L64": l64}
    for tag, rr in (("", main_r), ("_h8", h8_r)):
        res[f"comp_detected{tag}"] = react(rr, detected)
        res[f"comp_oracle_type{tag}"] = react(rr, np.where(detected > 0, true, 0))
        res[f"comp_oracle{tag}"] = react(rr, np.where(in_context, true, 0))
    for c, name in enumerate(cp.COMPONENTS[1:], start=1):
        uni = "short" if name == "freq" else name
        res[f"uni_{uni}"] = react(main_r, np.where(in_context, c, zero))
    tr = amc.trims(amc.REPORT, model_id, H, full, n)
    tr = {k: v for k, v in tr.items() if k == "oracle_trim"}
    for v in tr.values():
        v.pop("crop_share")
    res.update(tr)
    for label, path, col in (("sel_align_h8", H8 / mname / f"H{H}" / "window_metrics.csv.gz", "sel_align"),
                             ("mix_hind_h7", amc.OUT / mname / f"H{H}" / "window_metrics.csv.gz", "mix_hind")):
        prev = pd.read_csv(path, usecols=["row", "mae_full"] + [f"{m}_{col}" for m in METRICS])
        assert (prev.row.to_numpy() == np.arange(n)).all() and np.allclose(prev.mae_full, full["mae"], rtol=1e-5)
        res[label] = {m: prev[f"{m}_{col}"].to_numpy() for m in METRICS}

    summ = amc.summaries(res, full, ref, w, mask)
    vt = amc.versus_trims({**res, "det_trim": res["full"]}, w, mask)  # det_trim не нужен: только oracle_trim
    vt = vt[vt.versus == "oracle_trim"]
    zones = amc.zone_table(res, ref, w, mask)
    thr_v1 = pd.read_csv(amc.THRESHOLDS_V1 / mname / "thresholds.csv")
    tv = thr_v1[(thr_v1.L == 512) & (thr_v1.horizon == H)].set_index("percentile").threshold.to_dict()
    dstar = amc.d_star({k: res[k] for k in ("full", "comp_oracle", "comp_oracle_type", "comp_detected",
                                             "uni_level", "oracle_trim")}, ref, w, tv, {})

    # 4) основной критерий: comp_oracle против лучшей единой стратегии на change 0 ≤ d < 64
    series = w.series_idx.to_numpy()
    groups = amc.group_masks(w, mask)
    g = groups["change_0_64"]
    draws = amc.draws_for(series, {"change_0_64": g})["change_0_64"]
    uni_names = ["uni_level", "uni_scale", "uni_trend", "uni_short"]
    uni_mean = {u: ev.group_mean(ev.log_ratio(res[u]["mae"], full["mae"]), series, g, draws)[0] for u in uni_names}
    best_uni = min(uni_mean, key=uni_mean.get)
    pairs = {}
    for a_, b_ in (("comp_oracle", best_uni), ("comp_oracle", "comp_oracle_type"),
                   ("comp_oracle_type", "comp_detected"), ("comp_oracle_h8", "comp_oracle"),
                   ("comp_detected", "sel_align_h8")):
        point, boot = ev.group_mean(ev.log_ratio(res[a_]["mae"], res[b_]["mae"]), series, g, draws)
        lo, hi = ev.ci(boot)
        pairs[f"{a_}/{b_}"] = {"ratio": float(np.exp(point)), "lo": float(np.exp(lo)), "hi": float(np.exp(hi))}
    sm = summ[summ.metric == "mae"].set_index(["method", "group"])
    det_tg, det_sh = sm.loc[("comp_detected", "change_0_64")], sm.loc[("comp_detected", "none_all")]
    crit = {"best_uniform": best_uni, "uniform_mean_log_ratio": uni_mean,
            "primary": pairs[f"comp_oracle/{best_uni}"],
            "primary_passed": bool(pairs[f"comp_oracle/{best_uni}"]["hi"] < 1),
            "detected_tg64": [det_tg.gain, det_tg.gain_lo, det_tg.gain_hi],
            "detected_sh": [det_sh.harm, det_sh.harm_lo, det_sh.harm_hi],
            "detected_passed": bool(det_tg.gain_lo > 0 and det_sh.harm_hi <= amc.SH_TOL), "pairs": pairs}

    # 5) диагностика правила: матрица «истинная → найденная компонента» по зонам
    zone = np.where(cls == "none", "none", amc.zone_of(pos))
    diag = pd.DataFrame({"cls": cls, "kind": w.kind, "zone": zone, "true": np.asarray(cp.COMPONENTS)[true],
                         "detected": np.asarray(cp.COMPONENTS)[detected]})[mask]
    conf = diag[diag.zone.isin(["0-64", "64-128", "none", "pre"])].groupby(["zone", "kind"]).detected \
        .value_counts(normalize=True).unstack(fill_value=0).round(3)
    conf.to_csv(out / "detector_confusion.csv")
    far_none = float((diag[diag.cls == "none"].detected != "none").mean())

    wm = pd.concat([w[["row", "cell", "cls", "kind", "level", "series_idx", "pos"]],
                    pd.DataFrame({"eval": mask, "mae_reference": ref.mae_reference, "true": true,
                                  "detected": detected, "in_context": in_context}),
                    pd.DataFrame({f"z_{k}": v for k, v in z0.items()}),
                    pd.DataFrame({f"{m}_{k}": v for k, r in res.items() for m, v in r.items()})], axis=1)
    wm.to_csv(out / "window_metrics.csv.gz", index=False, float_format="%.7g")
    summ.to_csv(out / "summary_gain_harm.csv", index=False)
    vt.to_csv(out / "versus_oracle_trim.csv", index=False)
    zones.to_csv(out / "zones.csv", index=False)
    dstar.to_csv(out / "d_star.csv", index=False)
    meta = {"protocol": PROTOCOL, "date": datetime.now().strftime("%Y-%m-%d %H:%M"), "model": model_id,
            "horizon": H, "thresholds_seed1_none_q95": thr, "n_threshold_windows": int(m1.sum()),
            "false_alarm_none_seed0": far_none, "n_eval": int(mask.sum()), "criterion": crit,
            "seconds_total": round(time.perf_counter() - t_start, 1)}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=float))
    (out / "summary.md").write_text(summary_md(model_id, H, crit, summ, zones, conf, meta))
    print(json.dumps({k: crit[k] for k in ("best_uniform", "primary", "primary_passed", "detected_tg64",
                                           "detected_sh", "detected_passed")}, ensure_ascii=False, default=float))
    print(f"Готово: {out}, {meta['seconds_total']} с")


def summary_md(model_id, H, crit, summ, zones, conf, meta) -> str:
    f = amc.fmt
    p = crit["primary"]
    L_ = [f"# Regime-Component: {model_id}, H = {H}", "",
          "Создано `scripts/adapt_components.py` по `docs/protocol_v3_component.md` (v1). Файл перезаписывается; "
          "выводы — в `docs/results/adapt_components.md`. Разведка на synth_v1 seed 0.", "",
          f"Пороги (Q95 none seed 1, {meta['n_threshold_windows']} окон): "
          + ", ".join(f"{k} {v:.3f}" for k, v in meta["thresholds_seed1_none_q95"].items())
          + f". Доля окон none seed 0 с адаптацией: {f(meta['false_alarm_none_seed0'])}.", "",
          "## Критерий H9", "",
          f"- Лучшая единая стратегия на change 0–64: **{crit['best_uniform']}**",
          f"- MAE(comp_oracle) / MAE({crit['best_uniform']}) = {f(p['ratio'])} [{f(p['lo'])}; {f(p['hi'])}] "
          f"→ **{crit['primary_passed']}**",
          f"- comp_detected: TG64 {f(crit['detected_tg64'][0])} [{f(crit['detected_tg64'][1])}; "
          f"{f(crit['detected_tg64'][2])}], SH {f(crit['detected_sh'][0])} [{f(crit['detected_sh'][1])}; "
          f"{f(crit['detected_sh'][2])}] → {crit['detected_passed']}", "", "## Разложение (change 0–64, MAE-отношение)", ""]
    L_ += [f"- {k}: {f(v['ratio'])} [{f(v['lo'])}; {f(v['hi'])}]" for k, v in crit["pairs"].items()]
    L_ += ["", "## TG и SH по MAE", "", "| метод | TG64 [95%] | TG128 | SH none [95%] | TG64 WIS |", "|---|---|---|---|---|"]
    s = summ[summ.metric == "mae"].set_index(["method", "group"])
    sw = summ[summ.metric == "wis"].set_index(["method", "group"])
    for m in summ.method.unique():
        L_.append(f"| {m} | {f(s.loc[(m, 'change_0_64'), 'gain'])} [{f(s.loc[(m, 'change_0_64'), 'gain_lo'])}; "
                  f"{f(s.loc[(m, 'change_0_64'), 'gain_hi'])}] | {f(s.loc[(m, 'change_0_128'), 'gain'])} | "
                  f"{f(s.loc[(m, 'none_all'), 'harm'])} [{f(s.loc[(m, 'none_all'), 'harm_lo'])}; "
                  f"{f(s.loc[(m, 'none_all'), 'harm_hi'])}] | {f(sw.loc[(m, 'change_0_64'), 'gain'])} |")
    kinds = sorted(g[len("change_"):-len("_0_64")] for g in summ.group.unique()
                   if g.startswith("change_") and g.endswith("_0_64") and g != "change_0_64")
    L_ += ["", "## TG64 по типам смены (MAE) [95%]", "", "| метод | " + " | ".join(kinds) + " |",
           "|---" * (len(kinds) + 1) + "|"]
    for m in summ.method.unique():
        L_.append(f"| {m} | " + " | ".join(
            f"{f(s.loc[(m, f'change_{k}_0_64'), 'gain'])} [{f(s.loc[(m, f'change_{k}_0_64'), 'gain_lo'], 2)}; "
            f"{f(s.loc[(m, f'change_{k}_0_64'), 'gain_hi'], 2)}]" for k in kinds) + " |")
    zn = [z for z, _, _ in amc.ZONES] + ["none"]
    zt = zones.set_index(["method", "zone"])
    L_ += ["", "## Покрытие 80% по зонам", "", "| метод | " + " | ".join(zn) + " |", "|---" * (len(zn) + 1) + "|"]
    for m in zones.method.unique():
        L_.append(f"| {m} | " + " | ".join(f(zt.loc[(m, z), 'coverage']) if (m, z) in zt.index else "—"
                                          for z in zn) + " |")
    L_ += ["", "## Что находит правило (доли по зоне и типу)", "", "| зона | тип | " + " | ".join(conf.columns) + " |",
           "|---|---" + "|---" * len(conf.columns) + "|"]
    L_ += [f"| {z} | {k} | " + " | ".join(f(v) for v in row) + " |" for (z, k), row in conf.iterrows()]
    return "\n".join(L_) + "\n"


if __name__ == "__main__":
    main()
