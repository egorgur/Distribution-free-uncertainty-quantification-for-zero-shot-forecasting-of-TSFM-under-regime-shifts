"""Каскад (H5) на синтетике: надёжность (этап A) → локализация (этап B) → обрезка контекста.

Запуск из корня репозитория, после scripts.cache_cascade для отчётного кеша (seed 0) и scripts.stage5_localization:
    python -m scripts.stage6_cascade                              # Bolt, этап A — S4б → results/stage6/chronos-bolt-small/
    python -m scripts.stage6_cascade --model amazon/chronos-2     # → results/stage6/chronos-2/
    python -m scripts.stage6_cascade --signal s2                  # этап A — S2 → results/stage6/s2/<модель>/
    python -m scripts.stage6_cascade --n-boot 200                 # быстрее, интервалы грубее

Определения, политики, метрики и критерий — журнал решений docs/decisions.md (2026-11-02). Коротко:
- этап A — S4б (или S2) окна; отмечено, если сигнал больше перцентиля 1 − q на окнах оценки seed 1 того же H,
  q = 20%; этап B — уровень, выбранный в H4 (Bolt блок 6, Chronos-2 блок 8 того же H), порог FAR 5% с seed 1;
- П1 (каскад): отмечено и этап B сработал — прогноз по [τ′, t]; отмечено и не сработал — отказ, он считается
  прогнозом П0; П2 — этап B на каждом окне; П2м — то же с порогом, дающим на seed 1 столько же обрезок, сколько у
  П1; П3 — reset-and-grow (приближение); П4 — обрезка по истинной τ; П5 — отметка этапа A + истинная τ;
- ошибка окна r = MAE политики / MAE П0, сводная — геометрическое среднее; ложная обрезка — по разметке и
  допуску этапа B (δ = 32);
- критерий (Bolt, H = 64, seed 0): К1 — геом. среднее r(П1) на отмеченных окнах < 1, 95% интервал не включает 1;
  К2 — доля ложных обрезок П1 − П2м < 0, интервал не включает 0. Остальное — описание.
Бутстреп по рядам (1000 повторов), одни выборки для всех политик и обоих горизонтов; пороги в бутстрепе
не пересчитываются.

Результат — results/stage6/[s2/]<модель>/: summary.md (все таблицы; перезаписывается, выводы — в
docs/results/stage6_cascade.md), metrics.csv (длинный формат), criteria.csv, zones.csv, types.csv, r_by_d.csv,
q_curve.csv, wilcoxon.csv, fig_r_by_d.png, fig_q_curve.png, meta.json.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import wilcoxon

from scripts.stage3_failures import assign_zone, md_table, zone_names
from scripts.stage5_localization import bootstrap_counts
from src import cascade as cas
from src import localization as loc
from src import signals
from src.cache import (dataset_signals_path, load_dataset_meta, load_forecasts, load_series, load_windows,
                       model_dir, signals_path)
from src.models import QUANTILE_LEVELS

TUNE_CACHE = Path("results/cache/synth_v1_seed1")  # пороги
REPORT_CACHE = Path("results/cache/synth_v1")  # отчёт и критерий
STAGE5_META = Path("results/stage5/meta.json")
OUT = Path("results/stage6")
SEED = 0
N_BOOT = 1000
Q_MAIN = 0.20  # доля окон, которые отмечает этап A
Q_CURVE = (0.05, 0.10, 0.20, 0.30, 0.50, 1.00)
FAR = 0.05
DELTA = 32
HARM = 1.10  # вредная обрезка: MAE выросла больше чем на 10%
LOG_FLOOR = 1e-6  # нижняя граница r перед логарифмом
MIN_CROP_ABL = 64  # абляция: обрезка не короче 64 точек
S2_H0 = 16
S4_VARIANTS = ("L512", "L256", "L128", "L64")
CRITERION = ("amazon/chronos-bolt-small", 64)
SIGNALS = {"s4b": "S4б", "s2": "S2"}
CROPS = ("own", "p2", "oracle", "rag")
METRICS = ("mae", "crps", "coverage", "width")
CONTROL = "контроль"
I_MED = QUANTILE_LEVELS.index(0.5)

POLICY_LABELS = {
    "P0": "П0 полный контекст",
    "P1": "П1 каскад",
    "P2": "П2 детекция на каждом окне",
    "P2m": "П2м то же, обрезок как у П1",
    "P3": "П3 reset-and-grow (приближение)",
    "P4": "П4 оракул τ",
    "P5": "П5 этап A + оракул τ",
    "P1_p2": "П1 с детектором р2",
    "P2_p2": "П2 с детектором р2",
    "P1_min64": "П1, обрезка не короче 64 точек",
}
REFUSING = ("P1", "P1_p2", "P1_min64")  # политики с отказом
ZONE_POLICIES = ("P0", "P1", "P2", "P2m", "P3", "P4", "P5")
CURVE_POLICIES = ("P1", "P2", "P2m", "P3", "P4", "P5")
CURVE_STYLE = {"P1": ("tab:red", "-"), "P2": ("tab:blue", "-"), "P2m": ("tab:blue", "--"),
               "P3": ("tab:green", "-"), "P4": ("black", ":"), "P5": ("tab:red", ":")}
SUBSETS = {"flagged": "отмечены этапом A", "all": "все окна оценки", "positive": "«со сменой»",
           "negative": "«без смены»", "anomaly": "аномалия в контексте", "no_anomaly": "все, кроме аномалий"}


# ---------- данные ----------


def aligned(z, key: str, sel: np.ndarray, n: int) -> np.ndarray:
    """Столбец key файла кеша в порядке windows.csv; sel — нужные строки файла (например, один горизонт)."""
    rows = z["row"][sel]
    assert np.array_equal(np.sort(rows), np.arange(n)), f"{key}: посчитаны не все окна"
    out = np.empty(n, dtype=z[key].dtype)
    out[rows] = z[key][sel]
    return out


def stage_a(root: Path, model_id: str, w: pd.DataFrame, x: np.ndarray, ds: dict, L: int, with_p0: bool) -> dict:
    """S4б, S2 и метрики П0 каждого окна для каждого H — так же, как на этапе 4 (scripts/stage4_signals.py)."""
    vi = {v: i for i, v in enumerate(ds["variants"])}
    n, t, s = len(w), w.t.to_numpy(), w.series_idx.to_numpy()
    scale = np.empty(n)
    for a in range(0, n, 4096):
        sl = slice(a, a + 4096)
        scale[sl] = signals.naive_scale(x[s[sl, None], t[sl, None] - L + 1 + np.arange(L)])
    # S2: прогноз окна t − 16 (L512) из кеша, его первые 16 шагов — прогноз «назад».
    h_short = min(ds["horizons"])
    q_short = load_forecasts(root, model_id, h_short, variants=[vi[f"L{L}"]])[:, 0]
    keys = w[["series_idx", "pos"]]
    prev = (keys.assign(pos=keys.pos - S2_H0)
            .merge(w[["series_idx", "pos", "row"]], how="left", on=["series_idx", "pos"])["row"].to_numpy())
    ok = ~np.isnan(prev.astype(float))
    p = np.where(ok, prev, 0).astype(int)
    actual = x[s[:, None], t[:, None] - S2_H0 + 1 + np.arange(S2_H0)]
    s2 = np.where(ok, signals.backcast_error(q_short[p, I_MED, :S2_H0], actual, scale), np.nan)
    del q_short
    out = {"scale": scale}
    for h in ds["horizons"]:
        q = load_forecasts(root, model_id, h, variants=[vi[v] for v in S4_VARIANTS])
        if np.isnan(q).any():
            raise ValueError(f"{root}: в кеше {model_id} есть непосчитанные окна (H = {h})")
        out[h] = {"s4b": signals.quantile_distance([q[:, j] for j in range(len(S4_VARIANTS))], scale), "s2": s2}
        if with_p0:
            fut = x[s[:, None], t[:, None] + 1 + np.arange(h)]
            out[h]["p0"] = cas.forecast_metrics(q[:, 0], fut, QUANTILE_LEVELS)
        del q
    return out


def load_seed(root: Path, model_id: str, report: bool) -> dict:
    """Окна, сигналы этапов A и B и (для отчётного кеша) метрики П0 и обрезок; таблица на каждый H."""
    ds = load_dataset_meta(root)
    cfg = yaml.safe_load((root / "config.yaml").read_text())
    L, dur, tau = cfg["windows"]["context"], cfg["drift"]["duration"], ds["tau"]
    w = load_windows(root)
    assert (w.row.to_numpy() == np.arange(len(w))).all()
    x, _ = load_series(root)
    n = len(w)
    zone_b = loc.window_zones(w, L, dur)
    a = stage_a(root, model_id, w, x, ds, L, with_p0=report)
    loc_m = np.load(signals_path(root, model_id, "localization"))
    raw = np.load(dataset_signals_path(root, "localization_raw"))
    all_raw = np.ones(len(raw["row"]), bool)
    casc = np.load(signals_path(root, model_id, "cascade")) if report else None
    frames = {}
    for h in ds["horizons"]:
        h_loc = None if None in cas.STAGE_B[model_id] else h
        _, level = cas.STAGE_B[model_id][h_loc]
        sel = np.ones(len(loc_m["row"]), bool) if h_loc is None else loc_m["horizon"] == h_loc
        df = pd.DataFrame({
            "row": w.row, "series": w.series_idx, "t": w.t, "cell": w.cell, "cls": w.cls, "kind": w.kind,
            "level": w.level, "pos": w.pos, "d": w.d, "zone_b": zone_b,
            "s4b": a[h]["s4b"], "s2": a[h]["s2"], "scale": a["scale"],
            "score_m": aligned(loc_m, f"rbf_score_{level}", sel, n), "k_m": aligned(loc_m, f"rbf_k_{level}", sel, n),
            "score_p2": aligned(raw, f"rbf_score_{cas.RAW_DETECTOR}", all_raw, n),
            "k_p2": aligned(raw, f"rbf_k_{cas.RAW_DETECTOR}", all_raw, n),
        })
        df["zone_d"] = assign_zone(df.pos, pd.Series(h, index=df.index), L).where(df.cls != "none", CONTROL)
        if report:
            for m, v in a[h]["p0"].items():
                df[f"{m}0"] = v
            hs = casc["horizon"] == h
            for c in CROPS:
                df[f"start_{c}"] = aligned(casc, f"start_{c}", hs, n)
                for m in METRICS:
                    df[f"{m}_{c}"] = aligned(casc, f"{m}_{c}", hs, n).astype(np.float64)
            own = cas.crop_start(df.t.to_numpy(), df.k_m.to_numpy(), L)
            assert np.array_equal(df.start_own.to_numpy(), own), "обрезки в cascade.npz сняты с другого уровня"
            assert np.array_equal(df.start_p2.to_numpy(), cas.crop_start(df.t.to_numpy(), df.k_p2.to_numpy(), L))
        frames[h] = df
    out = {"frames": frames, "L": L, "dur": dur, "tau": tau, "hash": ds["hash"], "horizons": list(ds["horizons"]),
           "d_min": cfg["windows"]["d_min"]}
    if report:
        out["cascade_meta"] = json.loads(signals_path(root, model_id, "cascade").with_suffix(".json").read_text())
    return out


# ---------- пороги и политики ----------


def eval_mask(df: pd.DataFrame, d_min: int, report: bool) -> np.ndarray:
    """Окна оценки: d ≥ −112 (у none — pos), как на этапе 4; в отчёте ещё MAE П0 > 0."""
    m = df.pos.to_numpy() >= d_min + S2_H0
    if report:
        m &= df.mae0.to_numpy() > 0
    return m


def flag_threshold(sig_tune: np.ndarray, q: float) -> float:
    """Порог этапа A: перцентиль 1 − q сигнала на окнах оценки seed 1; при q = 100% отмечены все окна."""
    return -np.inf if q >= 1 else float(np.quantile(sig_tune, 1 - q))


def matched_threshold(score_tune: np.ndarray, n_crops: int) -> float:
    """Порог П2м: на seed 1 счёт выше него ровно у n_crops окон оценки (при отсутствии совпадений счёта)."""
    v = np.sort(score_tune)[::-1]
    return float(v[n_crops]) if n_crops < len(v) else -np.inf


def stage_b_thresholds(tune: dict, s5: dict, model_id: str) -> dict:
    """Пороги этапа B (FAR 5% на окнах «без смены» seed 1) — пересчёт и сверка с results/stage5/meta.json."""
    out = {}
    for h, df in tune["frames"].items():
        h_loc = None if None in cas.STAGE_B[model_id] else h
        cfg_name, level = cas.STAGE_B[model_id][h_loc]
        neg = df.zone_b.to_numpy() == loc.NEGATIVE
        thr_m = float(np.quantile(df.score_m.to_numpy()[neg], 1 - FAR))
        thr_p2 = float(np.quantile(df.score_p2.to_numpy()[neg], 1 - FAR))
        ref_m, ref_p2 = s5["thresholds"][cfg_name][f"lvl:{level}"], s5["thresholds"]["raw"][cas.RAW_DETECTOR]
        assert np.isclose(thr_m, ref_m, rtol=1e-12) and np.isclose(thr_p2, ref_p2, rtol=1e-12), (thr_m, ref_m)
        out[h] = {"b": thr_m, "p2": thr_p2, "stage5_config": cfg_name, "level": level}
    return out


def policy_rules(df: pd.DataFrame, flagged: np.ndarray, thr: dict) -> dict:
    """Для каждой политики: (обрезать ли окно, какой обрезкой, отказ ли)."""
    det = df.score_m.to_numpy() > thr["b"]
    det_m = df.score_m.to_numpy() > thr["bm"]
    det_p2 = df.score_p2.to_numpy() > thr["p2"]
    has = {c: df[f"start_{c}"].to_numpy() != cas.NO_CROP for c in ("oracle", "rag")}
    long = df.t.to_numpy() - df.start_own.to_numpy() + 1 >= MIN_CROP_ABL
    no = np.zeros(len(df), bool)
    F = flagged
    return {
        "P0": (no, "own", no),
        "P1": (F & det, "own", F & ~det),
        "P2": (det, "own", no),
        "P2m": (det_m, "own", no),
        "P3": (has["rag"], "rag", no),
        "P4": (has["oracle"], "oracle", no),
        "P5": (F & has["oracle"], "oracle", no),
        "P1_p2": (F & det_p2, "p2", F & ~det_p2),
        "P2_p2": (det_p2, "p2", no),
        "P1_min64": (F & det & long, "own", F & ~(det & long)),
    }


def apply_policy(df: pd.DataFrame, crop: np.ndarray, src: str, refused: np.ndarray, rep: dict) -> dict:
    """Метрики окон при политике: обрезанные окна берут метрики обрезки src, остальные — П0."""
    out = {"crop": crop, "refused": refused,
           "start": np.where(crop, df[f"start_{src}"].to_numpy(), cas.NO_CROP)}
    for m in METRICS:
        v = np.where(crop, df[f"{m}_{src}"].to_numpy(), df[f"{m}0"].to_numpy())
        if not np.isfinite(v).all():
            raise ValueError(f"нет метрики {m} обрезки {src} у части обрезанных окон")
        out[m] = v
    mae0 = df.mae0.to_numpy()
    out["r"] = np.divide(out["mae"], mae0, out=np.full(len(df), np.nan), where=mae0 > 0)
    # При обрезке по залипанию прогноз бывает почти точным (r → 0): log r ограничен снизу, как log MASE на этапе 3.
    out["logr"] = np.log(np.maximum(out["r"], LOG_FLOOR))
    out["r_crps"] = out["crps"] / df.crps0.to_numpy()
    w0 = df.width0.to_numpy()
    out["r_width"] = np.divide(out["width"], w0, out=np.full(len(df), np.nan), where=w0 > 0)
    out["outcome"] = cas.crop_outcome(crop, out["start"], df.zone_b.to_numpy(), df.cls.to_numpy(),
                                      rep["tau"], DELTA, rep["dur"])
    return out


# ---------- бутстреп ----------


class Boot:
    """Отношения сумм по окнам с бутстрепом по рядам: веса выборки — сколько раз ряд попал в неё."""

    def __init__(self, codes: np.ndarray, n_series: int, n_boot: int):
        self.codes, self.ns = codes, n_series
        self.C = bootstrap_counts(n_series, n_boot, [SEED, 6])

    def ratio(self, num: np.ndarray, den: np.ndarray) -> tuple[float, np.ndarray]:
        a = np.bincount(self.codes, weights=np.asarray(num, dtype=np.float64), minlength=self.ns)
        b = np.bincount(self.codes, weights=np.asarray(den, dtype=np.float64), minlength=self.ns)
        point = a.sum() / b.sum() if b.sum() > 0 else np.nan
        A, B = self.C @ a, self.C @ b
        return float(point), np.divide(A, B, out=np.full_like(A, np.nan), where=B > 0)

    def gmean(self, logv: np.ndarray, mask: np.ndarray) -> tuple[float, np.ndarray]:
        """Геометрическое среднее по окнам mask: exp(среднее log)."""
        p, b = self.ratio(np.where(mask, logv, 0.0), mask)
        return float(np.exp(p)), np.exp(b)


def ci(boot: np.ndarray) -> tuple[float, float]:
    if boot is None or np.isnan(boot).all():
        return np.nan, np.nan
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


# ---------- оценка ----------


def evaluate_h(df: pd.DataFrame, ev: np.ndarray, flagged: np.ndarray, thr: dict, rep: dict, boot: Boot, h: int):
    """Все политики на одном H. Возвращает (политики, записи метрик, бутстреп-выборки для критерия)."""
    pol = {p: apply_policy(df, *rule, rep) for p, rule in policy_rules(df, flagged, thr).items()}
    zone_b = df.zone_b.to_numpy()
    subsets = {"flagged": ev & flagged, "all": ev, "positive": ev & (zone_b == loc.POSITIVE),
               "negative": ev & (zone_b == loc.NEGATIVE), "anomaly": ev & (zone_b == loc.ANOMALY),
               "no_anomaly": ev & (zone_b != loc.ANOMALY)}
    lm = np.log(np.where(df.scale.to_numpy() > 0, df.mae0.to_numpy() / df.scale.to_numpy(), np.nan))
    records, boots = [], {}

    def rec(p, sub, metric, point, b=None, n=None):
        lo, hi = ci(b)
        records.append({"H": h, "policy": p, "subset": sub, "metric": metric, "value": point, "lo": lo, "hi": hi,
                        "n": n})
        if b is not None:
            boots[(p, sub, metric)] = b

    for p, o in pol.items():
        logr, logc = o["logr"], np.log(np.maximum(o["r_crps"], LOG_FLOOR))
        for sub, m in subsets.items():
            rec(p, sub, "gm_r", *boot.gmean(logr, m), n=int(m.sum()))
            rec(p, sub, "gm_r_crps", *boot.gmean(logc, m), n=int(m.sum()))
            rec(p, sub, "crop_rate", *boot.ratio(m & o["crop"], m), n=int((m & o["crop"]).sum()))
            rec(p, sub, "coverage", *boot.ratio(np.where(m, o["coverage"], 0.0), m), n=int(m.sum()))
        out = o["outcome"]
        cor, fal = ev & (out == cas.CORRECT), ev & (out == cas.FALSE)
        rec(p, "all", "false_share", *boot.ratio(fal, cor | fal), n=int((cor | fal).sum()))
        crop = ev & o["crop"]
        rec(p, "all", "harmful_share", *boot.ratio(crop & (o["r"] > HARM), crop), n=int(crop.sum()))
        for kind, label in ((cas.FALSE, "n_false"), (cas.CORRECT, "n_correct"), (cas.ANOMALY, "n_anomaly"),
                            (cas.EXCLUDED, "n_excluded")):
            rec(p, "all", label, float((ev & (out == kind)).sum()))
        if p in REFUSING:
            fl = ev & flagged
            rec(p, "flagged", "refusal_rate", *boot.ratio(fl & o["refused"], fl), n=int((fl & o["refused"]).sum()))
            ref, unf = ev & o["refused"] & np.isfinite(lm), ev & ~flagged & np.isfinite(lm)
            pr, br = boot.ratio(np.where(ref, lm, 0.0), ref)
            pu, bu = boot.ratio(np.where(unf, lm, 0.0), unf)
            rec(p, "flagged", "refused_vs_unflagged_mase", float(np.exp(pr - pu)), np.exp(br - bu), n=int(ref.sum()))
    # К2: разность долей ложных обрезок П1 − П2м (парный бутстреп).
    p1, p2m = (records_value(records, h, x, "all", "false_share") for x in ("P1", "P2m"))
    diff_b = boots[("P1", "all", "false_share")] - boots[("P2m", "all", "false_share")]
    rec("P1-P2m", "all", "false_share_diff", p1 - p2m, diff_b)
    return pol, records, boots


def records_value(records: list, h: int, p: str, sub: str, metric: str) -> float:
    return next(r["value"] for r in records if r["H"] == h and r["policy"] == p and r["subset"] == sub
                and r["metric"] == metric)


def criteria(met: pd.DataFrame, model_id: str) -> pd.DataFrame:
    rows = []
    for h in sorted(met.H.unique(), reverse=True):
        g = met[met.H == h]
        k1 = g[(g.policy == "P1") & (g.subset == "flagged") & (g.metric == "gm_r")].iloc[0]
        k2 = g[(g.policy == "P1-P2m") & (g.metric == "false_share_diff")].iloc[0]
        ok1, ok2 = bool(k1.value < 1 and k1.hi < 1), bool(k2.value < 0 and k2.hi < 0)
        rows.append({"H": h, "K1_value": k1.value, "K1_lo": k1.lo, "K1_hi": k1.hi, "K1": ok1,
                     "K2_value": k2.value, "K2_lo": k2.lo, "K2_hi": k2.hi, "K2": ok2,
                     "is_criterion": (model_id, int(h)) == CRITERION})
    return pd.DataFrame(rows)


def gmean_pos(v: np.ndarray) -> float:
    """Геометрическое среднее конечных положительных значений (у залипания ширина интервала бывает 0)."""
    v = v[np.isfinite(v) & (v > 0)]
    return float(np.exp(np.log(v).mean())) if len(v) else np.nan


def zone_table(df, ev, pol, h, L) -> pd.DataFrame:
    """По зонам d — только change, drift и контроль: у окон с залипанием r бывает почти 0, и они
    определяли бы геометрическое среднее зоны (аномалии — отдельным столбцом таблицы политик)."""
    rows = []
    keep = ev & df.cls.isin(["change", "drift", "none"]).to_numpy()
    for z in [*zone_names(L), CONTROL]:
        m = keep & (df.zone_d.to_numpy() == z)
        if not m.any():
            continue
        for p in ZONE_POLICIES:
            o = pol[p]
            rows.append({"H": h, "zone": z, "policy": p, "n": int(m.sum()),
                         "gm_r": float(np.exp(o["logr"][m].mean())),
                         "coverage": float(o["coverage"][m].mean()),
                         "gm_width_ratio": gmean_pos(o["r_width"][m]),
                         "crop_rate": float(o["crop"][m].mean())})
    return pd.DataFrame(rows)


def type_table(df, ev, flagged, pol, h) -> pd.DataFrame:
    """r по ячейкам change и drift в окнах «со сменой» (все d в контексте и переход 0 ≤ d < 128)."""
    rows = []
    pos = ev & (df.zone_b.to_numpy() == loc.POSITIVE)
    d = df.pos.to_numpy()
    for cell in df.cell[pos].unique():
        c = pos & (df.cell.to_numpy() == cell)
        for sub, m in (("в контексте", c), ("0–127", c & (d < 128))):
            r0 = df[c].iloc[0]
            row = {"H": h, "cell": cell, "cls": r0.cls, "kind": r0.kind, "level": r0.level, "subset": sub,
                   "n": int(m.sum()), "flagged_share": float(flagged[m].mean())}
            for p in CURVE_POLICIES:
                row[f"gm_r_{p}"] = float(np.exp(pol[p]["logr"][m].mean()))
                row[f"crop_{p}"] = float(pol[p]["crop"][m].mean())
            rows.append(row)
    return pd.DataFrame(rows)


def r_by_d(df, ev, pol, h) -> pd.DataFrame:
    m = ev & df.cls.isin(["change", "drift"]).to_numpy()
    rows = []
    for d in np.unique(df.pos.to_numpy()[m]):
        md = m & (df.pos.to_numpy() == d)
        for p in CURVE_POLICIES:
            rows.append({"H": h, "d": int(d), "policy": p, "gm_r": float(np.exp(pol[p]["logr"][md].mean())),
                         "crop_rate": float(pol[p]["crop"][md].mean()), "n": int(md.sum())})
    return pd.DataFrame(rows)


def series_mean(v: np.ndarray, m: np.ndarray, codes: np.ndarray, ns: int) -> tuple[np.ndarray, np.ndarray]:
    s = np.bincount(codes, weights=np.where(m, v, 0.0), minlength=ns)
    c = np.bincount(codes, weights=m.astype(float), minlength=ns)
    return np.divide(s, c, out=np.zeros(ns), where=c > 0), c > 0


def wilcoxon_rows(pol, ev, flagged, codes, ns, h) -> list[dict]:
    """Критерий Уилкоксона по средним log r рядов (дополнительно к бутстрепу)."""
    lr = {p: pol[p]["logr"] for p in ("P1", "P2", "P2m", "P3")}
    tests = [("П1 против П0, отмеченные окна", *series_mean(lr["P1"], ev & flagged, codes, ns))]
    for other in ("P2", "P2m", "P3"):
        a, ok = series_mean(lr["P1"] - lr[other], ev, codes, ns)
        tests.append((f"П1 против {POLICY_LABELS[other].split()[0]}, все окна оценки", a, ok))
    rows = []
    for name, v, ok in tests:
        v = v[ok]
        nz = int((v != 0).sum())
        stat, pval = wilcoxon(v, zero_method="wilcox") if nz > 0 else (np.nan, np.nan)
        rows.append({"H": h, "test": name, "n_series": int(ok.sum()), "n_nonzero": nz,
                     "mean_log_r": float(np.mean(v)), "share_below_0": float((v < 0).mean()),
                     "share_above_0": float((v > 0).mean()), "statistic": float(stat), "p_value": float(pval)})
    return rows


def q_curve(tune_df, tune_ev, df, ev, sig, thr, rep, boot, h) -> pd.DataFrame:
    """П1 и П2м при разной доле отмеченных окон q (пороги — по seed 1)."""
    rows = []
    det_t = tune_df.score_m.to_numpy() > thr["b"]
    for q in Q_CURVE:
        th_a = flag_threshold(tune_df[sig].to_numpy()[tune_ev], q)
        F_t = tune_df[sig].to_numpy() > th_a
        n1 = int((F_t & det_t & tune_ev).sum())
        thr_q = {**thr, "bm": matched_threshold(tune_df.score_m.to_numpy()[tune_ev], n1)}
        F = df[sig].to_numpy() > th_a
        rules = policy_rules(df, F, thr_q)
        for p in ("P1", "P2m"):
            o = apply_policy(df, *rules[p], rep)
            lr = o["logr"]
            out = o["outcome"]
            cor, fal = ev & (out == cas.CORRECT), ev & (out == cas.FALSE)
            g_all, b_all = boot.gmean(lr, ev)
            g_fl, b_fl = boot.gmean(lr, ev & F) if p == "P1" else (np.nan, None)
            fs, b_fs = boot.ratio(fal, cor | fal)
            rows.append({"H": h, "q": q, "policy": p, "flagged_share": float(F[ev].mean()),
                         "n_crops": int((ev & o["crop"]).sum()), "gm_r_all": g_all, "gm_r_all_lo": ci(b_all)[0],
                         "gm_r_all_hi": ci(b_all)[1], "gm_r_flagged": g_fl, "gm_r_flagged_lo": ci(b_fl)[0],
                         "gm_r_flagged_hi": ci(b_fl)[1], "false_share": fs, "false_share_lo": ci(b_fs)[0],
                         "false_share_hi": ci(b_fs)[1]})
    return pd.DataFrame(rows)


# ---------- графики ----------


def r_by_d_figure(curves: pd.DataFrame, L: int, path: Path) -> None:
    hs = sorted(curves.H.unique(), reverse=True)
    fig, axes = plt.subplots(1, len(hs), figsize=(6.5 * len(hs), 4.2), sharey=True)
    for ax, h in zip(np.atleast_1d(axes), hs):
        for p in CURVE_POLICIES:
            c = curves[(curves.H == h) & (curves.policy == p)]
            col, ls = CURVE_STYLE[p]
            ax.plot(c.d, c.gm_r, color=col, ls=ls, lw=1.4, label=POLICY_LABELS[p])
        ax.axhline(1, color="gray", lw=0.8)
        for e in (0, L - 1):
            ax.axvline(e, color="gray", lw=0.6, ls=":")
        ax.set_yscale("log")
        ax.set_title(f"H = {h}: change и drift, все силы")
        ax.set_xlabel("d = t − τ")
    np.atleast_1d(axes)[0].set_ylabel("геом. среднее MAE / MAE П0")
    np.atleast_1d(axes)[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def q_curve_figure(qc: pd.DataFrame, path: Path) -> None:
    hs = sorted(qc.H.unique(), reverse=True)
    fig, axes = plt.subplots(len(hs), 2, figsize=(11, 3.6 * len(hs)), squeeze=False)
    for i, h in enumerate(hs):
        for p, col in (("P1", "tab:red"), ("P2m", "tab:blue")):
            c = qc[(qc.H == h) & (qc.policy == p)]
            axes[i, 0].errorbar(c.q * 100, c.gm_r_all, yerr=[c.gm_r_all - c.gm_r_all_lo, c.gm_r_all_hi - c.gm_r_all],
                                color=col, marker="o", capsize=3, label=POLICY_LABELS[p])
            axes[i, 1].errorbar(c.q * 100, c.false_share * 100,
                                yerr=[(c.false_share - c.false_share_lo) * 100, (c.false_share_hi - c.false_share) * 100],
                                color=col, marker="o", capsize=3, label=POLICY_LABELS[p])
        axes[i, 0].axhline(1, color="gray", lw=0.8)
        axes[i, 0].set_ylabel(f"H = {h}: r по всем окнам")
        axes[i, 1].set_ylabel("доля ложных обрезок, %")
        for ax in axes[i]:
            ax.set_xlabel("q — доля окон, отмеченных этапом A, %")
        axes[i, 1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------- summary.md ----------


def fv(v, nd=3):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def fci(r, nd=3, pct=False):
    if r is None or np.isnan(r.value):
        return "—"
    k = 100 if pct else 1
    nd_ = 1 if pct else nd
    s = f"{r.value * k:.{nd_}f}{'%' if pct else ''}"
    if not np.isnan(r.lo):
        s += f" [{r.lo * k:.{nd_}f}; {r.hi * k:.{nd_}f}]"
    return s


def get(met, h, p, sub, metric):
    g = met[(met.H == h) & (met.policy == p) & (met.subset == sub) & (met.metric == metric)]
    return None if g.empty else g.iloc[0]


def summarize(model_id, sig, met, crit, zones, types, qc, wil, meta, L) -> str:
    hs = sorted(met.H.unique(), reverse=True)
    sl = SIGNALS[sig]
    out = [
        f"# Этап 6: каскад (H5), {model_id}, этап A — {sl}",
        "",
        f"Создано {meta['date']} скриптом `scripts/stage6_cascade.py`; файл перезаписывается при запуске, выводы — в "
        "`docs/results/stage6_cascade.md`. Определения, политики и критерий — `docs/decisions.md` (2026-11-02).",
        "",
        f"Пороги — по `{meta['tune_cache']}` (seed 1), отчёт — `{meta['report_cache']}` (seed 0). Окна оценки — "
        f"d ≥ {meta['d_eval_min']} (у none — pos), {meta['n_eval']} окон на H. r — MAE медианы политики / MAE П0 на "
        "том же окне, сводная — геометрическое среднее; отказ считается прогнозом П0 (r = 1). В квадратных скобках — "
        f"95% бутстреп-интервал по рядам ({meta['n_boot']} повторов, одни выборки для всех политик; пороги в бутстрепе "
        "не пересчитываются).",
        "",
        "## Пороги",
        "",
    ]
    rows = []
    for h in hs:
        t = meta["thresholds"][str(h)]
        rows.append([str(h), fv(t["a"], 4), f"{t['flagged_tune'] * 100:.1f}%", f"{t['flagged_report'] * 100:.1f}%",
                     fv(t["b"], 4), fv(t["bm"], 4), str(t["crops_p1_tune"]), str(t["crops_p2m_tune"]),
                     fv(t["p2"], 4)])
    out += [md_table(["H", f"θ_A ({sl}, q = {Q_MAIN:.0%})", "отмечено, seed 1", "отмечено, seed 0", "θ_B",
                      "θ_B′ (П2м)", "обрезок П1, seed 1", "обрезок П2м, seed 1", "θ_B р2"], rows), ""]

    out += ["## Критерий H5", "",
            "К1 — геометрическое среднее r(П1) на окнах, отмеченных этапом A: < 1 и интервал не включает 1. "
            "К2 — доля ложных обрезок П1 минус П2м: < 0 и интервал не включает 0. Критерий — только Bolt при H = 64; "
            "остальные строки — описание.", ""]
    rows = []
    for _, r in crit.iterrows():
        status = ("выполнен" if r.K1 and r.K2 else "не выполнен") if r.is_criterion else "описание"
        rows.append([str(r.H), f"{r.K1_value:.3f} [{r.K1_lo:.3f}; {r.K1_hi:.3f}]", "да" if r.K1 else "нет",
                     f"{r.K2_value * 100:+.1f} п. п. [{r.K2_lo * 100:+.1f}; {r.K2_hi * 100:+.1f}]", "да" if r.K2 else "нет",
                     status])
    out += [md_table(["H", "К1: r(П1), отмеченные", "К1", "К2: ложные П1 − П2м", "К2", "H5"], rows), ""]

    for h in hs:
        out += [f"## Политики, H = {h}", "",
                "Обрезок — число обрезок в окнах оценки; доли ложных и вредных (r > 1.1) — среди обрезок; r — по "
                "подмножествам окон; покрытие 80%-интервала — по всем окнам оценки.", ""]
        rows = []
        for p in POLICY_LABELS:
            n_crop = get(met, h, p, "all", "crop_rate")
            refusal = get(met, h, p, "flagged", "refusal_rate")
            rows.append([POLICY_LABELS[p], str(int(n_crop.n)), fci(get(met, h, p, "all", "false_share"), pct=True),
                         fci(get(met, h, p, "all", "harmful_share"), pct=True),
                         fci(refusal, pct=True) if refusal is not None else "—",
                         fci(get(met, h, p, "flagged", "gm_r")), fci(get(met, h, p, "all", "gm_r")),
                         fci(get(met, h, p, "positive", "gm_r")), fci(get(met, h, p, "negative", "gm_r")),
                         fci(get(met, h, p, "anomaly", "gm_r")), fci(get(met, h, p, "no_anomaly", "gm_r")),
                         fci(get(met, h, p, "all", "gm_r_crps")),
                         fv(get(met, h, p, "all", "coverage").value)])
        out += [md_table(["Политика", "Обрезок", "Ложных", "Вредных", "Отказов среди отмеченных",
                          "r, отмеченные", "r, все", "r, «со сменой»", "r, «без смены»", "r, аномалия",
                          "r, все без аномалий", "CRPS / CRPS П0, все", "Покрытие"], rows), ""]
        rows = []
        for p in ("P1", "P2", "P2m", "P3", "P1_p2", "P2_p2"):
            g = {m: get(met, h, p, "all", m).value for m in ("n_correct", "n_false", "n_anomaly", "n_excluded")}
            rows.append([POLICY_LABELS[p], *(str(int(g[m])) for m in ("n_correct", "n_false", "n_anomaly",
                                                                       "n_excluded"))])
        out += [f"Исходы обрезок, H = {h} (аномалия и «вне оценки» в долю ложных не входят):", "",
                md_table(["Политика", "верных", "ложных", "в окнах с аномалией", "«вне оценки»"], rows), ""]

    out += ["## Отказы", "",
            "Отношение геометрических средних MASE П0 в окнах с отказом и в неотмеченных окнах: больше 1 — отказ "
            "приходится на окна, где полный контекст ошибается сильнее.", ""]
    rows = []
    for h in hs:
        for p in REFUSING:
            rr = get(met, h, p, "flagged", "refused_vs_unflagged_mase")
            rows.append([str(h), POLICY_LABELS[p], str(int(rr.n)), fci(get(met, h, p, "flagged", "refusal_rate"), pct=True),
                         fci(rr)])
    out += [md_table(["H", "Политика", "Отказов", "Доля среди отмеченных", "MASE П0: отказ / неотмеченные"], rows), ""]

    out += ["## По зонам d (описание)", "",
            "Точечные оценки по окнам change, drift и контроля (без аномалий). r и ширина интервала — "
            "геометрические средние отношения к П0; покрытие — номинал 80%.", ""]
    for h in hs:
        z = zones[zones.H == h]
        rows = []
        for zn in dict.fromkeys(z.zone):
            g = z[z.zone == zn].set_index("policy")
            rows.append([zn, str(int(g.n.iloc[0])), *(fv(g.loc[p, "gm_r"]) for p in ZONE_POLICIES[1:]),
                         *(fv(g.loc[p, "coverage"], 2) for p in ("P0", "P1", "P2", "P3", "P4")),
                         fv(g.loc["P1", "crop_rate"], 2)])
        head = ["Зона", "Окон", *(f"r {POLICY_LABELS[p].split()[0]}" for p in ZONE_POLICIES[1:]),
                *(f"покр. {POLICY_LABELS[p].split()[0]}" for p in ("P0", "P1", "P2", "P3", "P4")), "обрезок П1"]
        out += [f"H = {h}:", "", md_table(head, rows), ""]

    out += ["## По типам смены, strong (описание)", "",
            "Окна «со сменой» (смена в контексте); «0–127» — переход. Точечные оценки r.", ""]
    for h in hs:
        t = types[(types.H == h) & (types.level == "strong")]
        rows = []
        for _, r in t.iterrows():
            rows.append([f"{r.cls}-{r.kind}", r.subset, str(r.n), f"{r.flagged_share * 100:.0f}%",
                         f"{r.crop_P1 * 100:.0f}%", *(fv(r[f"gm_r_{p}"]) for p in CURVE_POLICIES)])
        head = ["Тип", "d", "Окон", "отмечено", "обрезок П1", *(f"r {POLICY_LABELS[p].split()[0]}" for p in CURVE_POLICIES)]
        out += [f"H = {h}:", "", md_table(head, rows), ""]

    out += ["## Кривая по q (описание)", "",
            "Для каждого q пороги этапа A и П2м — по seed 1. q = 100% у П1 — то же, что П2.", ""]
    for h in hs:
        rows = []
        for q in Q_CURVE:
            a = qc[(qc.H == h) & (qc.q == q) & (qc.policy == "P1")].iloc[0]
            b = qc[(qc.H == h) & (qc.q == q) & (qc.policy == "P2m")].iloc[0]
            rows.append([f"{q:.0%}", f"{a.flagged_share * 100:.1f}%", str(a.n_crops),
                         f"{a.gm_r_flagged:.3f} [{a.gm_r_flagged_lo:.3f}; {a.gm_r_flagged_hi:.3f}]",
                         f"{a.gm_r_all:.3f} [{a.gm_r_all_lo:.3f}; {a.gm_r_all_hi:.3f}]",
                         f"{b.gm_r_all:.3f} [{b.gm_r_all_lo:.3f}; {b.gm_r_all_hi:.3f}]",
                         f"{a.false_share * 100:.1f}% [{a.false_share_lo * 100:.1f}; {a.false_share_hi * 100:.1f}]",
                         f"{b.false_share * 100:.1f}% [{b.false_share_lo * 100:.1f}; {b.false_share_hi * 100:.1f}]"])
        out += [f"H = {h}:", "", md_table(["q", "отмечено, seed 0", "обрезок П1", "r П1, отмеченные", "r П1, все",
                                           "r П2м, все", "ложных П1", "ложных П2м"], rows), ""]

    out += ["## Критерий Уилкоксона (дополнительно)", "",
            "По средним log r рядов (для «П1 против …» — средним log r(П1) − log r(другой)); ряды с нулевой "
            "разностью (ни одной обрезки) не учитываются (zero_method = wilcox). Меньше 0 — у П1 ошибка меньше.", ""]
    rows = [[str(r.H), r.test, str(r.n_series), str(r.n_nonzero), f"{r.mean_log_r:+.4f}",
             f"{r.share_below_0 * 100:.0f}% / {r.share_above_0 * 100:.0f}%", fv(r.statistic, 1), f"{r.p_value:.2g}"]
            for _, r in wil.iterrows()]
    out += [md_table(["H", "Сравнение", "Рядов", "Ненулевых", "Среднее по рядам", "Рядов < 0 / > 0", "W", "p"],
                     rows), ""]

    out += ["## Ограничения", "",
            "- Синтетика synth_v1, 10 рядов на ячейку; состав окон задан сеткой (одно окно на каждое d с шагом 8).",
            "- П3 — приближение reset-and-grow: статья «Breaking context inertia» не прочитана; детектор П3 видит полное "
            "окно 512, а не обрезанный контекст; окна идут с шагом 8.",
            "- Обрезка по τ′ этапа B кратна 16 точкам (патч), по истинной τ — нет.",
            "- В окнах с залипанием обрезка по его началу даёт постоянный контекст и почти точный прогноз (r → 0, "
            f"log r ограничен снизу log {LOG_FLOOR:g}). Такие окна сильно сдвигают геометрическое среднее "
            "столбца «все»; столбцы «со сменой», «без смены», «все без аномалий» и таблица по зонам их не "
            "содержат. Столбец «все без аномалий» и ограничение таблицы по зонам добавлены после первого "
            "расчёта (описание).",
            "- Пороги перенесены с seed 1; интервалы их разброс не учитывают.",
            f"- Сигнал этапа A ({sl}) и ожидание «польза — при смене частоты и тренда» выбраны по результатам seed 0 "
            "(этапы 4 и B).", ""]
    return "\n".join(out)


# ---------- main ----------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="amazon/chronos-bolt-small", choices=sorted(cas.STAGE_B))
    parser.add_argument("--signal", default="s4b", choices=sorted(SIGNALS))
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    args = parser.parse_args()
    t0 = time.perf_counter()
    base = OUT if args.signal == "s4b" else OUT / args.signal
    out = base / model_dir(REPORT_CACHE, args.model).name
    out.mkdir(parents=True, exist_ok=True)

    s5 = json.loads(STAGE5_META.read_text())
    print(f"Чтение кешей ({args.model}, этап A — {SIGNALS[args.signal]}) …", flush=True)
    tune = load_seed(TUNE_CACHE, args.model, report=False)
    rep = load_seed(REPORT_CACHE, args.model, report=True)
    assert (tune["hash"], rep["hash"]) == (s5["tune_hash"], s5["report_hash"])
    thr_b = stage_b_thresholds(tune, s5, args.model)
    cm = rep["cascade_meta"]
    for key, lv in cm["levels"].items():  # П3 в cascade.npz посчитан с тем же порогом этапа B
        h = None if key == "all" else int(key[1:])
        hs_ = rep["horizons"] if h is None else [h]
        assert all(np.isclose(lv["threshold"], thr_b[x]["b"], rtol=1e-12) for x in hs_), (key, lv)
    print(f"  {time.perf_counter() - t0:.0f} с", flush=True)

    first = rep["frames"][rep["horizons"][0]]
    codes = first.series.to_numpy()
    boot = Boot(codes, int(codes.max()) + 1, args.n_boot)
    records, zones, types, curves, qcs, wil, thresholds = [], [], [], [], [], [], {}
    n_eval = None
    for h in rep["horizons"]:
        tdf, df = tune["frames"][h], rep["frames"][h]
        tev = eval_mask(tdf, tune["d_min"], report=False)
        ev = eval_mask(df, rep["d_min"], report=True)
        if np.isnan(tdf[args.signal].to_numpy()[tev]).any() or np.isnan(df[args.signal].to_numpy()[ev]).any():
            raise ValueError("пропуски в сигнале этапа A на окнах оценки")
        n_eval = int(ev.sum())
        th_a = flag_threshold(tdf[args.signal].to_numpy()[tev], Q_MAIN)
        F_t, F = tdf[args.signal].to_numpy() > th_a, df[args.signal].to_numpy() > th_a
        det_t = tdf.score_m.to_numpy() > thr_b[h]["b"]
        n1 = int((F_t & det_t & tev).sum())
        thr = {**thr_b[h], "bm": matched_threshold(tdf.score_m.to_numpy()[tev], n1)}
        thresholds[str(h)] = {
            "a": th_a, "b": thr["b"], "bm": thr["bm"], "p2": thr["p2"], "level": thr["level"],
            "flagged_tune": float(F_t[tev].mean()), "flagged_report": float(F[ev].mean()),
            "crops_p1_tune": n1, "crops_p2m_tune": int(((tdf.score_m.to_numpy() > thr["bm"]) & tev).sum()),
            "n_mae0_zero": int((df.pos.to_numpy() >= rep["d_min"] + S2_H0).sum() - ev.sum()),
        }
        print(f"H = {h}: θ_A = {th_a:.4g}, отмечено {F[ev].mean():.1%} (seed 0); θ_B = {thr['b']:.4g}, "
              f"θ_B′ = {thr['bm']:.4g}", flush=True)
        pol, rec, _ = evaluate_h(df, ev, F, thr, rep, boot, h)
        records += rec
        zones.append(zone_table(df, ev, pol, h, rep["L"]))
        types.append(type_table(df, ev, F, pol, h))
        curves.append(r_by_d(df, ev, pol, h))
        wil += wilcoxon_rows(pol, ev, F, codes, boot.ns, h)
        qcs.append(q_curve(tdf, tev, df, ev, args.signal, thr, rep, boot, h))

    met = pd.DataFrame(records)
    crit = criteria(met, args.model)
    zones, types, curves, qc = (pd.concat(x, ignore_index=True) for x in (zones, types, curves, qcs))
    wil = pd.DataFrame(wil)
    meta = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"), "model": args.model, "signal": args.signal,
        "tune_cache": str(TUNE_CACHE), "report_cache": str(REPORT_CACHE), "tune_hash": tune["hash"],
        "report_hash": rep["hash"], "q": Q_MAIN, "q_curve": Q_CURVE, "far": FAR, "delta": DELTA, "harm": HARM,
        "min_crop_ablation": MIN_CROP_ABL, "s2_h0": S2_H0, "s4_variants": S4_VARIANTS, "n_boot": args.n_boot,
        "seed": SEED, "d_eval_min": rep["d_min"] + S2_H0, "n_eval": n_eval, "thresholds": thresholds,
        "criterion": CRITERION, "criteria": crit.to_dict(orient="records"),
        "cascade_cache": {k: cm[k] for k in ("date", "n_forecasts", "crops", "checks", "device", "seconds")},
    }
    met.to_csv(out / "metrics.csv", index=False)
    crit.to_csv(out / "criteria.csv", index=False)
    zones.to_csv(out / "zones.csv", index=False)
    types.to_csv(out / "types.csv", index=False)
    curves.to_csv(out / "r_by_d.csv", index=False)
    qc.to_csv(out / "q_curve.csv", index=False)
    wil.to_csv(out / "wilcoxon.csv", index=False)
    r_by_d_figure(curves, rep["L"], out / "fig_r_by_d.png")
    q_curve_figure(qc, out / "fig_q_curve.png")
    meta["seconds"] = round(time.perf_counter() - t0)
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=float))
    (out / "summary.md").write_text(summarize(args.model, args.signal, met, crit, zones, types, qc, wil, meta,
                                              rep["L"]))
    print(crit.to_string(index=False))
    print(f"Готово за {meta['seconds']} с → {out}/")


if __name__ == "__main__":
    main()
