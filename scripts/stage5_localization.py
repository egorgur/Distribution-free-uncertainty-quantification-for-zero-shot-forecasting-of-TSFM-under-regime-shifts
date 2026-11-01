"""Этап B (H4): точность локализации точки смены по уровням представления TSFM и на сырых значениях.

Запуск из корня репозитория, после scripts/cache_localization.py для обоих seed и всех моделей:
    python -m scripts.stage5_localization
    python -m scripts.stage5_localization --n-boot 200      # быстрее, интервалы грубее

Определения и критерий — журнал решений docs/decisions.md (2026-11-01). Коротко:
- окна «со сменой» — change и drift при 0 ≤ d < L − 1; «без смены» — весь контекст в одном устойчивом
  режиме (src/localization.py, window_zones); аномалии — отдельная таблица;
- порог каждого представления — 95-й перцентиль счёта на окнах «без смены» seed 1 (FAR 5%),
  на seed 0 не меняется;
- попадание — |τ′ − τ| ≤ δ (у drift τ′ ∈ [τ − δ, τ + 128 + δ]); срабатывание мимо — и ложная тревога,
  и пропуск; основной δ = 32;
- seed 1 — пороги и выбор лучшего уровня модели и лучшего сырого метода по F1; seed 0 — отчёт и критерий:
  F1(лучший уровень) − F1(лучший сырой метод) > 0 и 95% интервал парного бутстрепа по рядам не включает 0.

Результат — results/stage5/: summary.md (все таблицы; перезаписывается, выводы — в docs/results/stage5_localization.md),
metrics.csv (длинный формат), differences.csv, cells.csv, hit_rate_by_d.csv.gz, signed_error.csv,
anomaly.csv, fig_profile.png, fig_hit_rate_by_d.png, fig_recall_by_type.png, meta.json.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src import localization as loc
from src.cache import dataset_signals_path, load_dataset_meta, load_windows, signals_path
from src.metrics import f1_from_counts
from src.synth import CHANGE_KINDS, LEVELS

TUNE_CACHE = Path("results/cache/synth_v1_seed1")  # пороги и выбор
REPORT_CACHE = Path("results/cache/synth_v1")  # отчёт и критерий
OUT = Path("results/stage5")
SEED = 0
N_BOOT = 1000
FAR = 0.05
DELTA = 32  # основной допуск
DELTAS = (16, 32, 64)
D_MAX = 504  # последнее d окна «со сменой» в сетке synth_v1
SUBSETS = ("all", "change", "drift")
SUBSET_LABELS = {"all": "change и drift", "change": "change", "drift": "drift"}

# Конфигурации моделей: (имя весов, горизонт прохода, участвует ли в критерии).
CONFIGS = {
    "bolt-small": ("amazon/chronos-bolt-small", None, True),
    "chronos-2": ("amazon/chronos-2", 64, True),
    "chronos-2-H16": ("amazon/chronos-2", 16, False),
}
CFG_TITLES = {"bolt-small": "Chronos-Bolt-small", "chronos-2": "Chronos-2, проход H = 64",
              "chronos-2-H16": "Chronos-2, проход H = 16 (отдельная проверка)", "raw": "сырые значения"}
STAGE3 = {"bolt-small": "chronos-bolt-small", "chronos-2": "chronos-2"}  # d* этапа 3 (H = 64)
RAW_MAIN = ("p1", "p2", "p3", "p4")  # из них выбирается лучший сырой метод
RAW_LABELS = {
    "p1": "р1 значения патча", "p2": "р2 статистики патча", "p3": "р3 PELT по точкам", "p4": "р4 CUSUM по точкам",
    "p1:l2": "р1 значения патча (l2)", "p2:l2": "р2 статистики патча (l2)",
}
RAW_STYLE = {"p1": ("tab:blue", "--"), "p2": ("tab:green", "--"), "p3": ("tab:red", ":"), "p4": ("tab:brown", ":")}


@dataclass
class Rep:
    """Одно представление с одним детектором: счёт и τ′ (k, в точках контекста) для каждого окна."""

    key: str
    label: str
    depth: float  # нормированная глубина; NaN у сырых
    score: np.ndarray
    k: np.ndarray | None = None
    pelt: tuple[np.ndarray, np.ndarray] | None = None  # (cost, last) у PELT

    def decide(self, thr: float) -> tuple[np.ndarray, np.ndarray]:
        """(сработал ли детектор, τ′) при пороге thr."""
        if self.pelt is not None:
            _, k = loc.pelt_decision(*self.pelt, thr)
            return self.score > thr, k
        return self.score > thr, self.k


def level_label(name: str) -> str:
    if name == "input":
        return "вход энкодера"
    if name == "final":
        return "final (после layer norm)"
    return f"блок {name.removeprefix('block')}"


def _align(rows: np.ndarray, values: np.ndarray, n: int) -> np.ndarray:
    out = np.full(n, np.nan) if values.dtype.kind == "f" else np.full(n, -1, dtype=values.dtype)
    out[rows] = values
    return out


def load_seed(root: Path) -> dict:
    """Окна, зоны и все представления одного кеша."""
    cfg = yaml.safe_load((root / "config.yaml").read_text())
    L, dur = cfg["windows"]["context"], cfg["drift"]["duration"]
    w = load_windows(root)
    assert (w.row.to_numpy() == np.arange(len(w))).all()
    w["zone"] = loc.window_zones(w, L, dur)
    n = len(w)
    reps: dict[str, dict[str, Rep]] = {}

    z = np.load(dataset_signals_path(root, "localization_raw"))
    rows = z["row"]
    assert len(rows) == n, f"{root}: детекторы на сырых значениях посчитаны не для всех окон"
    raw = {}
    for p in ("p1", "p2"):
        for det, suffix in (("rbf", ""), ("l2", ":l2")):
            raw[p + suffix] = Rep(p + suffix, RAW_LABELS[p + suffix], np.nan,
                                  _align(rows, z[f"{det}_score_{p}"], n), _align(rows, z[f"{det}_k_{p}"], n))
    cost = np.full((n, z["pelt_cost"].shape[1]), np.nan)
    last = np.zeros((n, z["pelt_last"].shape[1]), dtype=np.int64)
    cost[rows], last[rows] = z["pelt_cost"], z["pelt_last"]
    raw["p3"] = Rep("p3", RAW_LABELS["p3"], np.nan, loc.pelt_critical_penalty(cost), pelt=(cost, last))
    raw["p4"] = Rep("p4", RAW_LABELS["p4"], np.nan, _align(rows, z["cusum_score"], n), _align(rows, z["cusum_k"], n))
    reps["raw"] = raw

    meta = {"L": L, "drift_duration": dur, "hash": load_dataset_meta(root)["hash"], "levels": {}}
    for name, (model_id, horizon, _) in CONFIGS.items():
        path = signals_path(root, model_id, "localization")
        info = json.loads(path.with_suffix(".json").read_text())
        z = np.load(path)
        sel = np.ones(len(z["row"]), bool) if "horizon" not in z.files else z["horizon"] == horizon
        assert (horizon is None) == ("horizon" not in z.files), (name, z.files[:3])
        rows = z["row"][sel]
        assert len(rows) == n, f"{path}: посчитаны не все окна"
        out = {}
        for lv, depth in zip(info["layers"], info["depths"]):
            out[f"lvl:{lv}"] = Rep(f"lvl:{lv}", level_label(lv), depth,
                                   _align(rows, z[f"rbf_score_{lv}"][sel], n), _align(rows, z[f"rbf_k_{lv}"][sel], n))
            out[f"lvl:{lv}:l2"] = Rep(f"lvl:{lv}:l2", level_label(lv) + " (l2)", depth,
                                      _align(rows, z[f"l2_score_{lv}"][sel], n), _align(rows, z[f"l2_k_{lv}"][sel], n))
        reps[name] = out
        meta["levels"][name] = list(info["layers"])
    return {"w": w, "reps": reps, "meta": meta}


# ---------- метрики ----------


def bootstrap_counts(n_series: int, n_boot: int, seed) -> np.ndarray:
    """Сколько раз каждый ряд попал в бутстреп-выборку: (выборки, ряды)."""
    draws = np.random.default_rng(seed).integers(0, n_series, (n_boot, n_series))
    flat = draws + (np.arange(n_boot) * n_series)[:, None]
    return np.bincount(flat.ravel(), minlength=n_boot * n_series).reshape(n_boot, n_series).astype(np.float64)


def weighted_median(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Нижняя медиана values (ряды,) с весами (выборки, ряды) → (выборки,). inf — «не найдено»."""
    order = np.argsort(values, kind="stable")
    v, cw = values[order], np.cumsum(weights[:, order], axis=1)
    idx = (cw < cw[:, -1:] / 2).sum(axis=1)
    return v[np.minimum(idx, len(v) - 1)]


def ci(boot: np.ndarray) -> tuple[float, float]:
    """95% перцентильный интервал; method lower / higher — чтобы inf не превращался в NaN."""
    return float(np.percentile(boot, 2.5, method="lower")), float(np.percentile(boot, 97.5, method="higher"))


def subset_metrics(ctx: dict, sub: np.ndarray, detected: np.ndarray, hit: np.ndarray, C: np.ndarray | None) -> dict:
    """F1, FAR, полнота, задержка и доля рядов без попадания для окон «со сменой» sub и всех окон «без смены».

    Возвращает {метрика: (значение, бутстреп-выборки или None)}.
    """
    codes, ns, neg, d = ctx["series"], ctx["n_series"], ctx["neg"], ctx["d"]
    tp_s = np.bincount(codes, weights=sub & hit, minlength=ns)
    fp_s = np.bincount(codes, weights=(neg & detected) | (sub & detected & ~hit), minlength=ns)
    p_s = np.bincount(codes, weights=sub, minlength=ns)
    fa_s = np.bincount(codes, weights=neg & detected, minlength=ns)
    n_s = np.bincount(codes, weights=neg, minlength=ns)
    # Задержка: первое d ∈ [0, D_MAX] с попаданием, по рядам с окнами «со сменой» из sub.
    pos_series = np.unique(codes[sub])
    first = np.full(ns, np.inf)
    ok = sub & hit & (d <= D_MAX)
    np.minimum.at(first, codes[ok], d[ok])
    delay = first[pos_series]
    unit = np.ones((1, len(pos_series)))
    out = {
        "f1": (float(f1_from_counts(tp_s.sum(), fp_s.sum(), p_s.sum())), None),
        "far": (float(fa_s.sum() / n_s.sum()), None),
        "recall": (float(tp_s.sum() / p_s.sum()), None),
        "delay": (float(weighted_median(delay, unit)[0]), None),
        "not_found": (float(np.isinf(delay).mean()), None),
    }
    if C is not None:
        tp, fp, p = C @ tp_s, C @ fp_s, C @ p_s
        wd = C[:, pos_series]
        boot = {"f1": f1_from_counts(tp, fp, p), "far": (C @ fa_s) / (C @ n_s), "recall": tp / p,
                "delay": weighted_median(delay, wd), "not_found": (wd * np.isinf(delay)).sum(axis=1) / wd.sum(axis=1)}
        out = {m: (v[0], boot[m]) for m, v in out.items()}
    return out


def context(data: dict) -> dict:
    w = data["w"]
    zone, cls = w.zone.to_numpy(), w.cls.to_numpy()
    return {
        "series": w.series_idx.to_numpy(), "n_series": int(w.series_idx.max()) + 1,
        "neg": zone == loc.NEGATIVE, "pos": zone == loc.POSITIVE, "cls": cls,
        "d": w.d.fillna(-1e9).to_numpy(),  # у none d нет; такие окна в «со сменой» не входят
        "subsets": {"all": zone == loc.POSITIVE, "change": (zone == loc.POSITIVE) & (cls == "change"),
                    "drift": (zone == loc.POSITIVE) & (cls == "drift")},
    }


def evaluate(tune: dict, report: dict, n_boot: int) -> tuple[pd.DataFrame, dict, dict, dict]:
    """Пороги по seed 1, метрики на обоих seed. Возвращает (metrics, пороги, бутстреп F1 на seed 0
    при δ = 32 по всем окнам со сменой, решения на seed 0 при пороге: {(cfg, rep): (detected, k)})."""
    ctxs = {"seed1": context(tune), "seed0": context(report)}
    C0 = bootstrap_counts(ctxs["seed0"]["n_series"], n_boot, [SEED, 5])
    records, thresholds, boot_f1, decisions = [], {}, {}, {}
    L, dur = report["meta"]["L"], report["meta"]["drift_duration"]
    for cfg in tune["reps"]:
        thresholds[cfg] = {}
        for key, rep_t in tune["reps"][cfg].items():
            neg_scores = rep_t.score[ctxs["seed1"]["neg"]]
            thr = float(np.quantile(neg_scores, 1 - FAR))
            thresholds[cfg][key] = thr
            for seed, data in (("seed1", tune), ("seed0", report)):
                c = ctxs[seed]
                detected, k = data["reps"][cfg][key].decide(thr)
                if seed == "seed0":
                    decisions[(cfg, key)] = (detected, k)
                for delta in DELTAS:
                    hit = detected & loc.is_hit(k, c["d"], c["cls"], L, delta, dur)
                    for sub_name, sub in c["subsets"].items():
                        boot = seed == "seed0" and (delta == DELTA or sub_name == "all")
                        res = subset_metrics(c, sub, detected, hit, C0 if boot else None)
                        for m, (v, b) in res.items():
                            if m == "far" and sub_name != "all":
                                continue
                            lo, hi = ci(b) if b is not None else (np.nan, np.nan)
                            records.append((cfg, key, seed, delta, sub_name, m, v, lo, hi))
                            if b is not None and m == "f1" and delta == DELTA and sub_name == "all":
                                boot_f1[(cfg, key)] = b
    cols = ["cfg", "rep", "seed", "delta", "subset", "metric", "value", "lo", "hi"]
    return pd.DataFrame.from_records(records, columns=cols), thresholds, boot_f1, decisions


def select(met: pd.DataFrame) -> dict:
    """Выбор по seed 1 (F1, δ = 32, change и drift) и проверка критерия на seed 0."""
    f1 = met[(met.metric == "f1") & (met.delta == DELTA) & (met.subset == "all")].set_index(["cfg", "rep", "seed"]).value
    raw = {s: f1.loc[[("raw", r, s) for r in RAW_MAIN]].droplevel([0, 2]) for s in ("seed1", "seed0")}
    best_raw = raw["seed1"].idxmax()
    out = {"best_raw": best_raw, "best_raw_seed0": raw["seed0"].idxmax()}
    for cfg in CONFIGS:
        lv = {s: f1.loc[cfg].xs(s, level="seed") for s in ("seed1", "seed0")}
        lv = {s: v[~v.index.str.endswith(":l2")] for s, v in lv.items()}
        best = lv["seed1"].idxmax()
        out[cfg] = {"best_level": best, "best_level_seed0": lv["seed0"].idxmax(),
                    "f1_level_seed0": float(lv["seed0"][best]), "f1_raw_seed0": float(raw["seed0"][best_raw])}
    return out


def differences(sel: dict, boot_f1: dict, met: pd.DataFrame) -> pd.DataFrame:
    """F1 на seed 0 (δ = 32): каждый уровень и каждый сырой метод минус лучший сырой метод (выбран по seed 1)."""
    f1 = met[(met.metric == "f1") & (met.delta == DELTA) & (met.subset == "all") & (met.seed == "seed0")]
    f1 = f1.set_index(["cfg", "rep"]).value
    best = sel["best_raw"]
    rows = []
    for (cfg, key), b in boot_f1.items():
        if (cfg, key) == ("raw", best):
            continue
        diff = b - boot_f1[("raw", best)]
        lo, hi = ci(diff)
        rows.append((cfg, key, best, float(f1[(cfg, key)] - f1[("raw", best)]), lo, hi))
    return pd.DataFrame(rows, columns=["cfg", "rep", "versus", "value", "lo", "hi"])


# ---------- таблицы по ячейкам, по d, ошибка положения, аномалии ----------


def cells_table(report: dict, decisions: dict) -> pd.DataFrame:
    """Полнота и задержка по ячейкам (seed 0, δ = 32)."""
    w, L, dur = report["w"], report["meta"]["L"], report["meta"]["drift_duration"]
    pos = (w.zone == loc.POSITIVE).to_numpy()
    d, cls, cell, series = w.d.to_numpy(), w.cls.to_numpy(), w.cell.to_numpy(), w.series_idx.to_numpy()
    rows = []
    for (cfg, key), (detected, k) in decisions.items():
        hit = detected & loc.is_hit(k, d, cls, L, DELTA, dur) & pos
        df = pd.DataFrame({"cell": cell[pos], "series": series[pos], "d": d[pos], "hit": hit[pos]})
        first = df[df.hit & (df.d <= D_MAX)].groupby("series").d.min()
        per_series = df.groupby("series").cell.first().to_frame().join(first.rename("delay"))
        per_series["delay"] = per_series.delay.fillna(np.inf)
        g = per_series.groupby("cell").delay
        agg = pd.DataFrame({"recall": df.groupby("cell").hit.mean(),
                            "delay": g.apply(lambda v: weighted_median(v.to_numpy(), np.ones((1, len(v))))[0]),
                            "not_found": g.apply(lambda v: np.isinf(v).mean())})
        agg["cfg"], agg["rep"] = cfg, key
        rows.append(agg.reset_index())
    out = pd.concat(rows, ignore_index=True)
    meta = w.drop_duplicates("cell").set_index("cell")[["cls", "kind", "level"]]
    return out.join(meta, on="cell")


def hit_rate_by_d(report: dict) -> pd.DataFrame:
    """Доля попаданий без порога (τ′ — лучшее разбиение при любом счёте) по d, seed 0, δ = 32. Без PELT."""
    w, L, dur = report["w"], report["meta"]["L"], report["meta"]["drift_duration"]
    pos = (w.zone == loc.POSITIVE).to_numpy()
    base = w.loc[pos, ["cls", "kind", "level", "d"]].reset_index(drop=True)
    d, cls = w.d.to_numpy()[pos], w.cls.to_numpy()[pos]
    rows = []
    for cfg, reps in report["reps"].items():
        for key, rep in reps.items():
            if rep.pelt is not None:
                continue
            hit = loc.is_hit(rep.k[pos], d, cls, L, DELTA, dur)
            g = base.assign(hit=hit).groupby(["cls", "kind", "level", "d"]).hit.mean().reset_index()
            g["cfg"], g["rep"] = cfg, key
            rows.append(g)
    return pd.concat(rows, ignore_index=True)


def signed_error(report: dict, decisions: dict) -> pd.DataFrame:
    """τ′ − τ у сработавших окон change «со сменой» (seed 0, при пороге)."""
    w, L = report["w"], report["meta"]["L"]
    pos = ((w.zone == loc.POSITIVE) & (w.cls == "change")).to_numpy()
    d = w.d.to_numpy()
    rows = []
    for (cfg, key), (detected, k) in decisions.items():
        sel = pos & detected
        off = loc.change_offset(k[sel], d[sel], L)
        if len(off) == 0:
            continue
        q1, med, q3 = np.percentile(off, [25, 50, 75])
        rows.append((cfg, key, int(sel.sum()), med, q1, q3, float((off < -DELTA).mean()), float((off > DELTA).mean())))
    return pd.DataFrame(rows, columns=["cfg", "rep", "n", "median", "q1", "q3", "early", "late"])


def edge_table(report: dict) -> pd.DataFrame:
    """Где лучшее разбиение (без порога) оказывается в окнах без смены, seed 0: доля τ′ на крайних точках сетки
    (k = 32 — левый край, k = L − 32 — правый, у последних патчей). При равномерной τ′ на 29 точках сетки — по 1/29.
    Только rbf и CUSUM (у PELT нет одного лучшего разбиения)."""
    w, L = report["w"], report["meta"]["L"]
    neg = (w.zone == loc.NEGATIVE).to_numpy()
    rows = []
    for cfg, reps in report["reps"].items():
        for key, rep in reps.items():
            if rep.pelt is not None or rep.k is None:
                continue
            k = rep.k[neg]
            rows.append((cfg, key, rep.label, rep.depth, float((k == L - loc.MIN_POINTS).mean()),
                         float((k == loc.MIN_POINTS).mean()), float(np.median(k))))
    return pd.DataFrame(rows, columns=["cfg", "rep", "label", "depth", "right_edge", "left_edge", "median_k"])


def anomaly_table(report: dict, decisions: dict) -> pd.DataFrame:
    """Окна с аномалией в контексте (seed 0): доля срабатываний; где τ′ — у начала аномалии (|τ′ − τ| ≤ δ)
    или, у freeze, у её конца (|τ′ − (τ + k)| ≤ δ; только окна, где конец уже в контексте, d ≥ k)."""
    w, L = report["w"], report["meta"]["L"]
    an = (w.zone == loc.ANOMALY).to_numpy()
    d, value, cell, kind = w.d.to_numpy(), w.value.to_numpy(), w.cell.to_numpy(), w.kind.to_numpy()
    rows = []
    for (cfg, key), (detected, k) in decisions.items():
        off = loc.change_offset(k, d, L)
        for c in np.unique(cell[an]):
            m = an & (cell == c)
            det = m & detected
            freeze = kind[m][0] == "freeze"
            end_in = det & (d >= value) if freeze else np.zeros_like(det)
            rows.append((cfg, key, c, int(m.sum()), float(det.sum() / m.sum()),
                         float((det & (np.abs(off) <= DELTA)).sum() / max(det.sum(), 1)),
                         float((end_in & (np.abs(off - value) <= DELTA)).sum() / max(end_in.sum(), 1)) if freeze else np.nan,
                         int(end_in.sum()) if freeze else 0))
    return pd.DataFrame(rows, columns=["cfg", "rep", "cell", "n", "detected", "near_start", "near_end", "n_end_in"])


def stage3_dstar() -> dict[str, pd.Series]:
    """d* этапа 3 при общем пороге 1.25 (отношение MAE, L = 512, H = 64), по ячейкам."""
    out = {}
    for cfg, folder in STAGE3.items():
        path = Path("results/stage3") / folder / "d_star.csv"
        if not path.exists():
            continue
        t = pd.read_csv(path)
        t = t[(t.metric == "mae_ratio") & (t.threshold == 1.25) & (t.L == 512) & (t.horizon == 64)]
        out[cfg] = t.set_index("cell").d_star.astype(float)
    return out


# ---------- графики ----------


def _depth_x(rep: Rep, n_blocks: int) -> float:
    """final рисуется чуть правее блока N (у них одна глубина)."""
    return rep.depth + (0.5 / n_blocks if rep.key == "lvl:final" else 0.0)


def profile_figure(report: dict, met: pd.DataFrame, sel: dict, path: Path) -> None:
    m = met[(met.seed == "seed0") & (met.delta == DELTA) & (met.subset == "all")].set_index(["cfg", "rep", "metric"])
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True)
    for col, (cfg, extra) in enumerate((("bolt-small", None), ("chronos-2", "chronos-2-H16"))):
        for row, metric in enumerate(("f1", "delay")):
            ax = axes[row, col]
            for r in RAW_MAIN:
                v = m.loc[("raw", r, metric)].value
                color, ls = RAW_STYLE[r]
                ax.axhline(min(v, 520), color=color, ls=ls, lw=1.3, label=RAW_LABELS[r])
            for c, lw, alpha, label in ((cfg, 2.0, 0.2, CFG_TITLES[cfg]), (extra, 1.0, 0.0, CFG_TITLES.get(extra))):
                if c is None:
                    continue
                reps = [r for k, r in report["reps"][c].items() if not k.endswith(":l2")]
                nb = len(reps) - 2
                xs = np.array([_depth_x(r, nb) for r in reps])
                ys = np.array([m.loc[(c, r.key, metric)].value for r in reps])
                lo = np.array([m.loc[(c, r.key, metric)].lo for r in reps])
                hi = np.array([m.loc[(c, r.key, metric)].hi for r in reps])
                cl = (lambda a: np.minimum(a, 520)) if metric == "delay" else (lambda a: a)
                ax.plot(xs, cl(ys), "o-", color="black" if c == cfg else "tab:purple", lw=lw, ms=4, label=label)
                if alpha:
                    ax.fill_between(xs, cl(lo), cl(hi), color="black", alpha=alpha)
                if c in sel:
                    best = report["reps"][c][sel[c]["best_level"]]
                    ax.plot(_depth_x(best, nb), min(m.loc[(c, best.key, metric)].value, 520), "*",
                            color="tab:orange" if c == cfg else "tab:purple", ms=14, zorder=5)
            ax.grid(alpha=0.3)
            if metric == "delay":
                ax.set_ylim(0, 530)
                ax.set_ylabel("медианная задержка, точек (520 — больше 504)")
                ax.set_xlabel("нормированная глубина уровня (0 — вход энкодера; правее 1 — final)")
            else:
                ax.set_ylabel(f"F1, δ = {DELTA}")
                ax.set_title(CFG_TITLES[cfg])
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].legend(fontsize=8)
    fig.suptitle("Этап B: локализация смены по уровням энкодера (seed 0; пороги — FAR 5% на seed 1; "
                 "звезда — уровень, выбранный по seed 1)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def hit_rate_figure(hr: pd.DataFrame, sel: dict, path: Path) -> None:
    panels = [("change", k) for k in CHANGE_KINDS] + [("drift", "mean"), ("drift", "frequency")]
    lines = [("bolt-small", sel["bolt-small"]["best_level"], "black", "-"),
             ("bolt-small", "lvl:final", "black", ":"),
             ("chronos-2", sel["chronos-2"]["best_level"], "tab:purple", "-"),
             ("chronos-2", "lvl:final", "tab:purple", ":"),
             ("raw", "p1", *RAW_STYLE["p1"]), ("raw", "p2", *RAW_STYLE["p2"]), ("raw", "p4", *RAW_STYLE["p4"])]
    h = hr[hr.level == "strong"]
    fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True, sharey=True)
    for ax, (cls, kind) in zip(axes.flat, panels):
        p = h[(h.cls == cls) & (h.kind == kind)]
        for cfg, key, color, ls in lines:
            g = p[(p.cfg == cfg) & (p.rep == key)]
            name = RAW_LABELS[key] if cfg == "raw" else f"{CFG_TITLES[cfg].split(',')[0]}: {level_label(key[4:])}"
            ax.plot(g.d, g.hit, color=color, ls=ls, lw=1.4, label=name)
        ax.set_title(f"{cls}: {kind}, strong", fontsize=10)
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=7)
    fig.supxlabel("d = t − τ (в контексте d + 1 точек нового режима)")
    fig.supylabel(f"доля окон, где лучшее разбиение попало в ±{DELTA}")
    fig.suptitle("Попадание лучшего разбиения без порога, seed 0", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def recall_figure(report: dict, cells: pd.DataFrame, path: Path) -> None:
    order = [f"change-{k}-strong" for k in CHANGE_KINDS] + ["drift-mean-strong", "drift-frequency-strong"]
    widths = [sum(not k.endswith(":l2") for k in report["reps"][cfg]) + len(RAW_MAIN) for cfg in ("bolt-small", "chronos-2")]
    fig, axes = plt.subplots(1, 2, figsize=(17, 5.5), gridspec_kw={"width_ratios": widths})
    for ax, cfg in zip(axes, ("bolt-small", "chronos-2")):
        cols = [(cfg, k) for k in report["reps"][cfg] if not k.endswith(":l2")] + [("raw", r) for r in RAW_MAIN]
        c = cells.set_index(["cfg", "rep", "cell"]).recall
        mat = np.array([[c.get((cf, k, cell), np.nan) for cf, k in cols] for cell in order])
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if mat[i, j] < 0.5 else "black")
        labels = [level_label(k[4:]) if cf != "raw" else RAW_LABELS[k].split()[0] for cf, k in cols]
        ax.set_xticks(range(len(cols)), labels, rotation=60, ha="right", fontsize=7.5)
        ax.set_yticks(range(len(order)), [o.removesuffix("-strong") for o in order], fontsize=8)
        ax.set_title(CFG_TITLES[cfg], fontsize=10)
    fig.colorbar(im, ax=axes, shrink=0.8, label=f"полнота (попадание ±{DELTA}), seed 0")
    fig.suptitle("Полнота по типам смены, сила strong; порог — FAR 5% на seed 1", fontsize=11)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------- сводка ----------


def md_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows])


def fv(v: float, metric: str) -> str:
    if np.isnan(v):
        return "—"
    if metric in ("delay",):
        return "> 504" if np.isinf(v) else f"{v:.0f}"
    if metric in ("not_found", "far", "recall", "early", "late", "detected", "near_start", "near_end"):
        return f"{100 * v:.0f}%"
    return f"{round(v, 2) + 0.0:.2f}"  # + 0.0 убирает «-0.00»


def fci(r, metric: str) -> str:
    s = fv(r.value, metric)
    return s if np.isnan(r.lo) else f"{s} [{fv(r.lo, metric)}; {fv(r.hi, metric)}]"


def rep_label(report: dict, cfg: str, key: str) -> str:
    return report["reps"][cfg][key].label


def summarize(report, tune, met, diffs, sel, cells, se, an, edge, dstar, meta) -> str:
    m = met.set_index(["cfg", "rep", "seed", "delta", "subset", "metric"]).sort_index()
    dif = diffs.set_index(["cfg", "rep"])
    get = lambda cfg, rep, metric, seed="seed0", delta=DELTA, subset="all": m.loc[(cfg, rep, seed, delta, subset, metric)]
    best_raw = sel["best_raw"]
    out = [
        "# Этап B: локализация точки смены по уровням представления TSFM (H4)",
        "",
        f"Дата: {meta['date']}. Скрипт: `python -m scripts.stage5_localization`.",
        f"Пороги и выбор — seed 1 (`{TUNE_CACHE}`, хеш {meta['tune_hash']}); отчёт и критерий — seed 0 "
        f"(`{REPORT_CACHE}`, хеш {meta['report_hash']}). Контекст L = {meta['L']}, детектор ищет одну точку смены.",
        f"Окон на seed: со сменой {meta['n_pos']}, без смены {meta['n_neg']}, с аномалией {meta['n_anomaly']}, "
        f"вне оценки {meta['n_excluded']}. В квадратных скобках — 95% бутстреп-интервал по рядам seed 0 "
        f"({meta['n_boot']} повторов; одни выборки для всех представлений, сравнение парное; порог в бутстрепе "
        "не пересчитывается).",
        "",
        "## Как читать",
        "",
        "- **Окно со сменой** — change и drift при 0 ≤ d < L − 1 (в контексте есть точка смены). **Без смены** — весь "
        "контекст в одном устойчивом режиме: d < 0 любого класса, контроль без изменений, change при d ≥ L − 1, drift "
        "после конца подъёма, аномалия, уже вышедшая из контекста.",
        "- **Порог** каждого представления — 95-й перцентиль счёта на окнах без смены seed 1 (FAR 5%); на seed 0 тот же.",
        f"- **Попадание** — детектор сработал и |τ′ − τ| ≤ δ (у drift τ′ ∈ [τ − δ, τ + 128 + δ]); основной δ = {DELTA}. "
        "Срабатывание в окне со сменой, но мимо — и ложная тревога, и пропуск. **F1** = 2·TP / (2·TP + FP + FN) "
        "по окнам со сменой и всем окнам без смены вместе. **FAR** — доля окон без смены, где детектор сработал. "
        "**Полнота** — доля окон со сменой с попаданием.",
        "- **Задержка** — первое d ≥ 0 с попаданием в ряду, медиана по рядам; в скобках — доля рядов без попадания "
        "до d = 504. «> 504» — больше половины рядов без попадания.",
        "- **Глубина** — номер блока / число блоков; вход энкодера — 0, final — выход последнего блока после "
        "итоговой layer norm (глубина 1, как у последнего блока).",
        f"- Лучший сырой метод по seed 1: **{RAW_LABELS[best_raw]}** (на seed 0 лучший — {RAW_LABELS[sel['best_raw_seed0']]}).",
        "",
    ]
    for cfg in CONFIGS:
        s = sel[cfg]
        crit = CONFIGS[cfg][2]
        out += [f"## {CFG_TITLES[cfg]}", ""]
        rows = []
        for key, rep in report["reps"][cfg].items():
            if key.endswith(":l2"):
                continue
            mark = " **(выбран по seed 1)**" if key == s["best_level"] else ""
            d = dif.loc[(cfg, key)]
            rows.append([rep.label + mark, f"{rep.depth:.2f}", fv(get(cfg, key, "f1", "seed1").value, "f1"),
                         fci(get(cfg, key, "f1"), "f1"), fv(get(cfg, key, "far").value, "far"),
                         fv(get(cfg, key, "recall").value, "recall"),
                         f"{fci(get(cfg, key, 'delay'), 'delay')} ({fv(get(cfg, key, 'not_found').value, 'not_found')})",
                         fci(d, "f1")])
        for r in RAW_MAIN:
            mark = " **(лучший сырой по seed 1)**" if r == best_raw else ""
            d = "—" if r == best_raw else fci(dif.loc[("raw", r)], "f1")
            rows.append([RAW_LABELS[r] + mark, "—", fv(get("raw", r, "f1", "seed1").value, "f1"),
                         fci(get("raw", r, "f1"), "f1"), fv(get("raw", r, "far").value, "far"),
                         fv(get("raw", r, "recall").value, "recall"),
                         f"{fci(get('raw', r, 'delay'), 'delay')} ({fv(get('raw', r, 'not_found').value, 'not_found')})", d])
        out.append(md_table(["Представление", "Глубина", "F1, seed 1", f"F1, seed 0", "FAR, seed 0", "Полнота, seed 0",
                             "Задержка, seed 0 (без попадания)", f"F1 − F1({RAW_LABELS[best_raw].split()[0]}), seed 0"],
                            rows))
        d = dif.loc[(cfg, s["best_level"])]
        verdict = d.value > 0 and d.lo > 0
        label = rep_label(report, cfg, s["best_level"])
        out += ["", (f"**Критерий H4** (записан до расчёта): F1(«{label}») − F1(«{RAW_LABELS[best_raw]}») на seed 0 = "
                     f"{fci(d, 'f1')} → **{'выполнен' if verdict else 'не выполнен'}**.") if crit else
                (f"Описание (критерий записан только для прохода H = 64): F1(«{label}») − F1(«{RAW_LABELS[best_raw]}») "
                 f"на seed 0 = {fci(d, 'f1')}."),
                f"Лучший уровень на seed 0: «{rep_label(report, cfg, s['best_level_seed0'])}» "
                f"({'тот же, что по seed 1' if s['best_level_seed0'] == s['best_level'] else 'не тот, что по seed 1'}).", ""]

    # Абляция l2, δ, change и drift.
    out += ["## Абляция: стоимость l2 вместо rbf (F1 seed 0, δ = 32)", ""]
    rows = []
    for cfg in ("bolt-small", "chronos-2"):
        for key, rep in report["reps"][cfg].items():
            if not key.endswith(":l2"):
                rows.append([CFG_TITLES[cfg].split(",")[0], rep.label, fci(get(cfg, key, "f1"), "f1"),
                             fci(get(cfg, key + ":l2", "f1"), "f1")])
    for r in ("p1", "p2"):
        rows.append(["сырые", RAW_LABELS[r], fci(get("raw", r, "f1"), "f1"), fci(get("raw", r + ":l2", "f1"), "f1")])
    out += [md_table(["Модель", "Представление", "rbf", "l2"], rows), ""]

    chosen = [("bolt-small", sel["bolt-small"]["best_level"]), ("bolt-small", "lvl:input"), ("bolt-small", "lvl:final"),
              ("chronos-2", sel["chronos-2"]["best_level"]), ("chronos-2", "lvl:input"), ("chronos-2", "lvl:final"),
              *[("raw", r) for r in RAW_MAIN]]
    chosen = list(dict.fromkeys(chosen))
    name = lambda cfg, key: RAW_LABELS[key] if cfg == "raw" else f"{CFG_TITLES[cfg].split(',')[0]}: {rep_label(report, cfg, key)}"
    # Короткие названия столбцов широких таблиц: B — Bolt-small, C2 — Chronos-2 (проход H = 64).
    short_lv = lambda key: {"lvl:input": "вход", "lvl:final": "final"}.get(key, f"блок {key.removeprefix('lvl:block')}")
    short = lambda cfg, key: RAW_LABELS[key].split()[0] if cfg == "raw" else \
        f"{'B' if cfg == 'bolt-small' else 'C2'} {short_lv(key)}"
    legend = ("Столбцы: B — Chronos-Bolt-small, C2 — Chronos-2 (проход H = 64); «вход» — вход энкодера, «final» — "
              "выход после итоговой layer norm; уровни «блок …» — выбранные по seed 1; "
              + "; ".join(f"{RAW_LABELS[r].split(' ', 1)[0]} — {RAW_LABELS[r].split(' ', 1)[1]}" for r in RAW_MAIN) + ".")
    out += ["## Допуск δ и класс изменения (F1 seed 0)", ""]
    rows = [[name(c, k)] + [fv(get(c, k, "f1", delta=dl).value, "f1") for dl in DELTAS]
            + [fci(get(c, k, "f1", subset=sb), "f1") for sb in ("change", "drift")]
            + [fv(get(c, k, "far", delta=dl).value, "far") for dl in (DELTA,)] for c, k in chosen]
    out += [md_table(["Представление", *[f"δ = {dl}" for dl in DELTAS], "только change", "только drift", "FAR"], rows),
            "", "FAR от δ не зависит. F1 «только change» и «только drift» — окна этого класса со сменой и все окна без смены.", ""]

    # Полнота и задержка по ячейкам.
    cidx = cells.set_index(["cfg", "rep", "cell"])
    cval = lambda cf, k, c, col: cidx.loc[(cf, k, c), col] if (cf, k, c) in cidx.index else np.nan
    cell_order = [f"{cls}-{k}-{lv}" for cls, kinds in (("change", CHANGE_KINDS), ("drift", ("mean", "frequency")))
                  for k in kinds for lv in LEVELS]
    cell_order = [c for c in cell_order if c in set(cells.cell)]
    out += ["## Полнота по типу и силе смены (seed 0, δ = 32)", ""]
    rows = [[c] + [fv(cval(cf, k, c, "recall"), "recall") for cf, k in chosen] for c in cell_order]
    out += [md_table(["Ячейка", *[short(cf, k) for cf, k in chosen]], rows), "", legend, ""]

    out += ["## Задержка по типу смены, сила strong (seed 0, δ = 32), и d* этапа 3", "",
            "Медиана по 10 рядам ячейки; в скобках — доля рядов без попадания до d = 504. d* — первое d, начиная "
            "с которого ошибка прогноза не выше 1.25 × ошибки в устойчивом режиме (этап 3, H = 64, L = 512). "
            "Задержка меньше d* значит, что смену находят раньше, чем прогноз восстанавливается сам. "
            "d* = ∞ — сглаженная кривая ошибки не опускается ниже порога до конца сетки d.", ""]
    rows = []
    fd = lambda v: "∞" if np.isinf(v) else ("—" if np.isnan(v) else f"{v:.0f}")
    for c in [x for x in cell_order if x.endswith("strong")]:
        r = [c] + [fd(dstar[cf].get(c, np.nan)) if cf in dstar else "—" for cf in ("bolt-small", "chronos-2")]
        r += [f"{fv(cval(cf, k, c, 'delay'), 'delay')} ({fv(cval(cf, k, c, 'not_found'), 'not_found')})"
              for cf, k in chosen]
        rows.append(r)
    out += [md_table(["Ячейка", "d* Bolt", "d* Chronos-2", *[short(cf, k) for cf, k in chosen]], rows), "", legend, ""]

    out += ["## Ошибка положения τ′ − τ (сработавшие окна change со сменой, seed 0, при пороге)", "",
            "Меньше 0 — τ′ раньше настоящей смены (обрезка оставит в контексте часть старого режима), больше 0 — "
            "позже (обрезка выбросит часть нового).", ""]
    sei = se.set_index(["cfg", "rep"])
    rows = [[name(c, k), str(int(sei.loc[(c, k)].n)), f"{sei.loc[(c, k)]['median']:.0f}",
             f"[{sei.loc[(c, k)].q1:.0f}; {sei.loc[(c, k)].q3:.0f}]",
             fv(sei.loc[(c, k)].early, "early"), fv(sei.loc[(c, k)].late, "late")] for c, k in chosen if (c, k) in sei.index]
    out += [md_table(["Представление", "Окон", "Медиана", "Квартили", f"< −{DELTA}", f"> +{DELTA}"], rows), ""]

    out += ["## Аномалии в контексте (seed 0, при пороге)", "",
            "Аномалия — не смена режима, поэтому в F1 эти окна не входят. «Сработал» — доля окон ячейки с аномалией "
            f"в контексте. «У начала» — доля сработавших окон, где |τ′ − τ| ≤ {DELTA}. «У конца» (freeze) считается "
            "только по сработавшим окнам, где конец залипания уже в контексте (d ≥ длины): доля тех, где "
            f"|τ′ − (τ + длина)| ≤ {DELTA}. Знаменатели «у начала» и «у конца» разные, поэтому их сумма может "
            "быть больше 100%.", ""]
    ani = an.set_index(["cfg", "rep", "cell"])
    acells = sorted(an.cell.unique(), key=lambda c: (c.split("-")[1], LEVELS.index(c.split("-")[2])))
    rows = []
    for c in acells:
        r = [c, str(int(ani.loc[(chosen[0][0], chosen[0][1], c)].n))]
        for cf, k in chosen:
            a = ani.loc[(cf, k, c)]
            s = f"{fv(a.detected, 'detected')} / {fv(a.near_start, 'near_start')}"
            if "freeze" in c:
                s += f" / {fv(a.near_end, 'near_end')}"
            r.append(s)
        rows.append(r)
    out += [md_table(["Ячейка", "Окон", *[short(cf, k) for cf, k in chosen]], rows),
            "", "Формат: сработал / у начала (/ у конца — для freeze). " + legend, ""]

    n_grid = (report["meta"]["L"] - 2 * loc.MIN_POINTS) // loc.PATCH + 1
    out += ["## Где разбиение без порога в окнах без смены (seed 0)", "",
            f"Доля окон без смены, где лучшее разбиение (без порога) — на краю сетки: правый край k = "
            f"{report['meta']['L'] - loc.MIN_POINTS} (отделены последние 2 патча) и левый край k = {loc.MIN_POINTS}. "
            f"Если бы τ′ была равномерной на {n_grid} точках сетки, у каждого края было бы {1 / n_grid:.1%}. "
            f"Разбиение k = {report['meta']['L'] - loc.MIN_POINTS} засчитывается как попадание при любом "
            f"d ≤ {loc.MIN_POINTS - 1 + DELTA}, поэтому у представлений, которые часто отделяют последние патчи и без "
            "смены, доля попаданий без порога при малых d завышена.", ""]
    rows = [[CFG_TITLES.get(r.cfg, r.cfg), r.label, "—" if np.isnan(r.depth) else f"{r.depth:.2f}",
             f"{r.right_edge:.0%}", f"{r.left_edge:.0%}", f"{r.median_k:.0f}"]
            for r in edge.itertuples() if r.cfg in ("bolt-small", "chronos-2", "raw") and not r.rep.endswith(":l2")]
    out += [md_table(["Модель", "Представление", "Глубина", f"k = {report['meta']['L'] - loc.MIN_POINTS}",
                      f"k = {loc.MIN_POINTS}", "Медиана k"], rows), ""]

    out += ["## Ограничения", "",
            "- Синтетика synth_v1, 10 рядов на ячейку, одна точка смены на ряд; состав окон задан сеткой d.",
            "- При FAR 5% часть попаданий случайна: случайная τ′ ∈ [32, 480] попадает в ±32 от τ примерно в 13% случаев. "
            "Поэтому полнота при малых d и задержки меньше ~32 — с оговоркой. У drift окно попадания шире "
            "([τ − 32, τ + 160]), и случайная τ′ попадает примерно в 34% случаев (39–40% при d ≥ 128).",
            "- Минимальный сегмент 32 точки: при d < 31 правый сегмент не помещается, и попасть можно только в пределах δ.",
            "- Порог перенесён с seed 1 без пересчёта; интервалы не учитывают разброс порога.",
            "- Профиль по слоям (кроме выбранного уровня) — описание, без поправки на множественные сравнения.", ""]
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tune-cache", type=Path, default=TUNE_CACHE)
    parser.add_argument("--report-cache", type=Path, default=REPORT_CACHE)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    args = parser.parse_args()
    t0 = datetime.now()

    print("Загрузка …", flush=True)
    tune, report = load_seed(args.tune_cache), load_seed(args.report_cache)
    for k in ("L", "drift_duration"):
        assert tune["meta"][k] == report["meta"][k]
    assert tune["meta"]["levels"] == report["meta"]["levels"]
    print("Метрики и бутстреп …", flush=True)
    met, thresholds, boot_f1, decisions = evaluate(tune, report, args.n_boot)
    sel = select(met)
    diffs = differences(sel, boot_f1, met)
    print("Таблицы по ячейкам и по d …", flush=True)
    cells = cells_table(report, decisions)
    hr = hit_rate_by_d(report)
    se = signed_error(report, decisions)
    an = anomaly_table(report, decisions)
    edge = edge_table(report)
    dstar = stage3_dstar()

    args.out.mkdir(parents=True, exist_ok=True)
    zone = report["w"].zone
    meta = {
        "date": t0.strftime("%Y-%m-%d %H:%M"), "tune_cache": str(args.tune_cache), "report_cache": str(args.report_cache),
        "tune_hash": tune["meta"]["hash"], "report_hash": report["meta"]["hash"], "L": report["meta"]["L"],
        "n_boot": args.n_boot, "seed": SEED, "far_target": FAR, "delta": DELTA, "deltas": DELTAS, "d_max": D_MAX,
        "n_pos": int((zone == loc.POSITIVE).sum()), "n_neg": int((zone == loc.NEGATIVE).sum()),
        "n_anomaly": int((zone == loc.ANOMALY).sum()), "n_excluded": int((zone == loc.EXCLUDED).sum()),
        "thresholds": thresholds, "selection": sel, "configs": {k: list(v[:2]) for k, v in CONFIGS.items()},
    }
    met.to_csv(args.out / "metrics.csv", index=False)
    diffs.to_csv(args.out / "differences.csv", index=False)
    cells.to_csv(args.out / "cells.csv", index=False)
    hr.to_csv(args.out / "hit_rate_by_d.csv.gz", index=False)
    se.to_csv(args.out / "signed_error.csv", index=False)
    an.to_csv(args.out / "anomaly.csv", index=False)
    edge.to_csv(args.out / "edge.csv", index=False)
    profile_figure(report, met, sel, args.out / "fig_profile.png")
    hit_rate_figure(hr, sel, args.out / "fig_hit_rate_by_d.png")
    recall_figure(report, cells, args.out / "fig_recall_by_type.png")
    (args.out / "summary.md").write_text(summarize(report, tune, met, diffs, sel, cells, se, an, edge, dstar, meta))
    meta["seconds"] = round((datetime.now() - t0).total_seconds())
    (args.out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str))
    print(f"Готово за {meta['seconds']} с → {args.out}")


if __name__ == "__main__":
    main()
