"""Этап 4: сигналы надёжности прогноза и их сравнение (H3) — только по кешу, без новых прогонов модели.

Запуск из корня репозитория (нужен кеш прогнозов, см. scripts/cache_forecasts.py):
    python -m scripts.stage4_signals                               # chronos-bolt-small
    python -m scripts.stage4_signals --model amazon/chronos-2
    python -m scripts.stage4_signals --n-boot 200                  # быстрее, интервалы грубее
    python -m scripts.stage4_signals --cache results/cache/synth_v1_seed1   # → results/stage4/synth_v1_seed1/<модель>/

Сигналы (определения — src/signals.py; большой сигнал = прогнозу доверять нельзя):
    S1   ширина 80%-интервала основного прогноза                              0 доп. прогонов
    S2   ошибка прогноза «назад» на последних 16 известных точках             1
    S3   CUSUM по всему контексту; KS: последние 64 точки против остальных    0, без модели
    S4   расхождение прогнозов при L = 512, 256, 128, 64:                     3
         (а) разброс медиан, (б) расстояние между квантилями
    S7   то же при длине L + k, k = ±3, ±9, и слабом шуме (одна реализация)   5
    S5   сдвиг эмбеддингов последних 4 патчей относительно прежних,        0, тот же проход
         выход энкодера: (а) косинус, (б) диагональный Махаланобис          энкодера
Бюджет этапа A — не больше 5 доп. прогонов (CLAUDE.md, раздел 4). S7 по всем 6 вариантам кеша
превышает его и показан только в абляции. S6 — этап 7.

S5 читается из <кеш>/<модель>/signals/s5.npz, его считает scripts/cache_s5.py (нужен прогон
энкодера: кеш хранит только квантили). S5 дополнительный сигнал: в сравнение H3 и в сочетания
не входит, показан в основных таблицах и абляциях.

S2 берётся из кеша. Окна идут с шагом 8, поэтому прогноз окна t − 16 уже посчитан; его первые
16 шагов — прогноз «назад» на x[t − 15 … t]. В работе это и есть один дополнительный прогон.
У самых ранних окон (d < −112) окна t − 16 нет, поэтому они исключены из оценки у всех сигналов.

Ошибка окна (то, что сигнал должен предсказать) — MASE медианы основного прогноза (L = 512),
как в плане. Для проверки устойчивости выводов — ещё CRPS / масштаб; MAE / масштаб ряда до смены
(один знаменатель на ряд, а не масштаб контекста окна, как у сигналов); отношение ошибки к ошибке
в устойчивом режиме из этапа 3 (есть только на синтетике).

Метрики (src/metrics.py): Спирмен; AURC и выигрыш AURC (0 — как случайный отказ, 1 — как оракул,
знающий ошибку); снижение средней ошибки при отказе от 10% и 20% окон; AUROC и AUPRC для
«окно в 10% худших». Наборы окон: все; без зоны смены в горизонте (−H ≤ d < 0), которую по
контексту не предсказать; только переход (0 ≤ d < 128, без контроля). 95% интервалы — бутстреп
по рядам целиком, одни и те же выборки для всех сигналов; поэтому есть интервалы и для разности
двух сигналов (S4, S7 минус S1, S2, S3 — проверка H3).

Гиперпараметры сигналов (16 точек у S2, 64 у KS, состав вариантов S4 и S7) выбраны до запуска.
Другие значения показаны в абляциях; лучший вариант задним числом не выбирается.

Сочетания сигналов (COMBOS) зафиксированы до расчёта в журнале решений CLAUDE.md: среднее или
максимум рангов, среднее геометрическое, отношение. Весов нет; обученное сочетание — S6 (этап 7).

Результаты: results/stage4/<модель>/ (для кеша, отличного от synth_v1, — results/stage4/<кеш>/<модель>/)
    summary.md             — таблицы (выводы — в CLAUDE.md; файл перезаписывается);
    metrics.csv            — все метрики: сигнал × горизонт × ошибка × набор окон;
    h3_differences.csv     — парные разности метрик (S4, S7 минус S1, S2, S3; сочетания минус S1, S4б);
    rejected_zones.csv     — куда попадают отказы (по зонам d);
    spearman_groups.csv    — Спирмен внутри зон d, типов и сил смены;
    transition_response.csv — во сколько раз сигнал и ошибка сразу после смены больше, чем до неё;
    risk_coverage.csv      — кривые риск–покрытие;
    signal_curves.csv.gz   — сигналы и ошибка в зависимости от d по ячейкам;
    window_signals.csv.gz  — сигналы и ошибки каждого окна (~20 МБ, в git не попадает);
    fig_*.png, meta.json.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # без окна: только сохранение в файл
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullLocator

from scripts.plot_synth import CLASS_TITLES, TITLES
from scripts.cache_s5 import s5_keys
from scripts.stage3_failures import add_ratios, assign_zone, mark_zones, md_table, panels, zone_names
from src import metrics, signals
from src.cache import load_dataset_meta, load_forecasts, load_series, load_windows, model_dir, signals_path
from src.models import QUANTILE_LEVELS

DEFAULT_CACHE = Path("results/cache/synth_v1")
SEED = 0
N_BOOT = 1000
BOOT_CHUNK = 200  # бутстреп-выборок за раз: память ~ BOOT_CHUNK × окна × 8 байт на массив
CTX_CHUNK = 4096  # окон за раз при расчёте S3
WORST = 0.10  # «плохое окно» — 10% окон с наибольшей ошибкой
COVERAGES = (0.9, 0.8)  # отказ от 10% и 20% окон
REJECT = 0.10  # доля отказов для таблицы «куда попадают отказы»
RESPONSE_ZONE = 64  # отклик на переход считается по окнам 0 ≤ d < RESPONSE_ZONE
S2_H0 = (16, 8)  # первое значение — основной вариант
KS_W = (64, 32, 128)  # первое значение — основной вариант
I_LO, I_MED, I_HI = (QUANTILE_LEVELS.index(q) for q in (0.1, 0.5, 0.9))
CONTROL = "контроль"  # зона для рядов без изменений

# Варианты контекста для S4 и S7 (имена — как в кеше). Основной прогноз L512 входит в каждый набор.
SPREAD_SETS = {
    "s4": ["L512", "L256", "L128", "L64"],
    "s4_k1_256": ["L512", "L256"],
    "s4_k1_128": ["L512", "L128"],
    "s4_k1_64": ["L512", "L64"],
    "s4_k2": ["L512", "L256", "L128"],
    "s7": ["L512", "tail-9", "tail-3", "tail+3", "tail+9", "noise1"],
    "s7_tail3": ["L512", "tail-3", "tail+3"],
    "s7_tails": ["L512", "tail-9", "tail-3", "tail+3", "tail+9"],
    "s7_noise1": ["L512", "noise1"],
    "s7_noise2": ["L512", "noise1", "noise2"],
    "s7_all": ["L512", "tail-9", "tail-3", "tail+3", "tail+9", "noise1", "noise2"],
}

# Сигналы: ключ → (подпись, доп. прогонов модели на окно).
MAIN = {
    "s1": ("S1 ширина интервала", "0"),
    "s2_16": ("S2 ошибка «назад», 16 точек", "1"),
    "s3_cusum": ("S3 CUSUM по контексту", "0, без модели"),
    "s3_ks64": ("S3 KS, последние 64 точки", "0, без модели"),
    "s4_a": ("S4а разброс медиан по L", "3"),
    "s4_b": ("S4б расстояние квантилей по L", "3"),
    "s7_a": ("S7а разброс медиан при возмущениях", "5"),
    "s7_b": ("S7б расстояние квантилей при возмущениях", "5"),
    "s5_a": ("S5а сдвиг эмбеддингов, косинус", "0, тот же проход"),
    "s5_b": ("S5б сдвиг эмбеддингов, Махаланобис", "0, тот же проход"),
}
S5_MAIN = {"s5_a": "s5_final_m4_a", "s5_b": "s5_final_m4_b"}  # основной вариант S5 (scripts/cache_s5.py)
_S5_LAYERS = {"final": "выход энкодера", "input": "вход энкодера"}
S5_ABLATIONS = {
    k: (f"S5, {_S5_LAYERS[k.split('_')[1]]}, m = {k.split('_')[2][1:]} ({'а' if k.endswith('a') else 'б'})",
        "0, тот же проход")
    for k in s5_keys() if k not in S5_MAIN.values()
}
_SET_LABELS = {
    "s4_k1_256": ("S4, L 512 + 256", "1"),
    "s4_k1_128": ("S4, L 512 + 128", "1"),
    "s4_k1_64": ("S4, L 512 + 64", "1"),
    "s4_k2": ("S4, L 512 + 256 + 128", "2"),
    "s7_tail3": ("S7, хвост ±3", "2"),
    "s7_tails": ("S7, хвост ±3, ±9", "4"),
    "s7_noise1": ("S7, шум × 1", "1"),
    "s7_noise2": ("S7, шум × 2", "2"),
    "s7_all": ("S7, хвост ±3, ±9 + шум × 2", "6, сверх бюджета"),
}
ABLATIONS = {
    "s2_8": ("S2, 8 точек", "1"),
    "s3_ks32": ("S3 KS, 32 точки", "0, без модели"),
    "s3_ks128": ("S3 KS, 128 точек", "0, без модели"),
    **{f"{key}_{m}": (f"{label} ({'а' if m == 'a' else 'б'})", runs)
       for key, (label, runs) in _SET_LABELS.items() for m in "ab"},
    **S5_ABLATIONS,
}
REFERENCE = {"random": ("случайный отказ", "—"), "oracle": ("оракул (знает ошибку)", "—")}
# Сочетания сигналов зафиксированы до расчёта (CLAUDE.md, журнал решений, 2026-11-01). Весов нет:
# обученное сочетание — это S6 (этап 7). Ранги считаются в add_combos.
COMBOS = {
    "c_rank_s1_s4": ("среднее рангов S1 + S4б", "3"),
    "c_geo_s1_s4": ("√(S1 · S4б)", "3"),
    "c_max_s1_s4": ("максимум рангов S1, S4б", "3"),
    "c_rank_s1_s2": ("среднее рангов S1 + S2", "1"),
    "c_rank_s1_s2_s4": ("среднее рангов S1 + S2 + S4б", "4"),
    "c_rank_s1_s4k1": ("среднее рангов S1 + S4б (L 512 + 256)", "1"),
    "c_ratio_s4_s1": ("S4б / S1", "3"),
    "c_rank_all": ("среднее рангов S1 + S2 + S4б + S7б", "9, сверх бюджета"),
}
COMBO_VERSUS = ("s1", "s4_b")  # с чем сравнивать сочетания
FIG_EXTRA = ("c_rank_s1_s4",)  # сочетания на графике риск–покрытие
LABELS = {**REFERENCE, **MAIN, **COMBOS, **ABLATIONS}

TARGETS = {
    "mase": "MASE",
    "crps_s": "CRPS / масштаб",
    "mae_fixed": "MAE / масштаб ряда до смены",
    "mae_ratio": "отношение к устойчивому режиму",
}
MAIN_TARGET = "mase"
ALL = "все окна"
TRANSITION = "переход, 0 ≤ d < 128"
SUBSETS = {  # наборы окон для оценки; g — таблица окон одного горизонта
    ALL: lambda g: np.ones(len(g), bool),
    "без смены в горизонте": lambda g: ~((g.d >= -g.horizon) & (g.d < 0)).to_numpy(),
    TRANSITION: lambda g: ((g.cls != "none") & (g.d >= 0) & (g.d < 128)).to_numpy(),
}
H3_ROWS = ("s4_a", "s4_b", "s7_a", "s7_b")  # предлагаемые сигналы
H3_BASELINES = ("s1", "s2_16", "s3_cusum", "s3_ks64")
DIFF_PAIRS = [(a, b) for a in H3_ROWS for b in H3_BASELINES] + [(c, b) for c in COMBOS for b in COMBO_VERSUS]
METRIC_NAMES = ("spearman", "aurc", "gain", "drop10", "drop20", "auroc", "ap")
COLUMNS = {
    "spearman": "Спирмен", "aurc": "AURC ↓", "gain": "Выигрыш AURC",
    "drop10": "Ошибка ↓ при отказе от 10%", "drop20": "Ошибка ↓ при отказе от 20%",
    "auroc": "AUROC", "ap": "AUPRC",
}
STYLE = {
    "s1": ("tab:gray", "-"), "s2_16": ("tab:blue", "-"), "s3_cusum": ("tab:brown", "-"),
    "s3_ks64": ("tab:olive", "-"), "s4_a": ("tab:purple", "--"), "s4_b": ("tab:purple", "-"),
    "s7_a": ("tab:green", "--"), "s7_b": ("tab:green", "-"), "s5_a": ("tab:orange", "--"), "s5_b": ("tab:orange", "-"),
    "random": ("0.6", ":"), "oracle": ("black", ":"), "mase": ("black", "-"), "c_rank_s1_s4": ("tab:red", "-"),
}
CURVE_KEYS = ("s1", "s2_16", "s3_cusum", "s3_ks64", "s4_b", "s7_b", "s5_b")


# ---------- ошибки и сигналы окон ----------


def build_table(root: Path, model_id: str, w: pd.DataFrame, x: np.ndarray, ds: dict, L: int) -> pd.DataFrame:
    """Ошибки и сигналы каждого окна на всех горизонтах; основной контекст L."""
    vi = {v: i for i, v in enumerate(ds["variants"])}
    missing = {v for vs in SPREAD_SETS.values() for v in vs} - set(vi)
    if missing:
        raise ValueError(f"в кеше нет вариантов контекста {sorted(missing)}")
    n, t, s = len(w), w.t.to_numpy(), w.series_idx.to_numpy()

    # Масштаб и S3 — по самому контексту; частями, чтобы не держать все контексты в памяти.
    scale = np.empty(n)
    common = {"s3_cusum": np.empty(n), **{f"s3_ks{k}": np.empty(n) for k in KS_W}}
    for a in range(0, n, CTX_CHUNK):
        sl = slice(a, a + CTX_CHUNK)
        ctx = x[s[sl, None], t[sl, None] - L + 1 + np.arange(L)]
        scale[sl] = signals.naive_scale(ctx)
        common["s3_cusum"][sl] = signals.cusum_statistic(ctx)
        for k in KS_W:
            common[f"s3_ks{k}"][sl] = signals.ks_recent(ctx, k)

    # S2: прогноз окна t − h0 (L512) уже есть в кеше; его первые h0 шагов — прогноз «назад».
    h_short = min(ds["horizons"])
    q_short = load_forecasts(root, model_id, h_short, variants=[vi[f"L{L}"]])[:, 0]
    keys = w[["series_idx", "pos"]]
    for h0 in S2_H0:
        if h0 > h_short:
            raise ValueError(f"S2: h0 = {h0} больше самого короткого горизонта кеша {h_short}")
        prev = (keys.assign(pos=keys.pos - h0)
                .merge(w[["series_idx", "pos", "row"]], how="left", on=["series_idx", "pos"])["row"].to_numpy())
        ok = ~np.isnan(prev.astype(float))
        p = np.where(ok, prev, 0).astype(int)
        actual = x[s[:, None], t[:, None] - h0 + 1 + np.arange(h0)]
        common[f"s2_{h0}"] = np.where(ok, signals.backcast_error(q_short[p, I_MED, :h0], actual, scale), np.nan)
    del q_short

    check = np.random.default_rng(SEED).choice(n, 5, replace=False)
    frames = []
    for h in ds["horizons"]:
        q = load_forecasts(root, model_id, h)  # (окна, варианты, 9, H), float32
        if np.isnan(q).any():
            raise ValueError(f"в кеше {model_id} есть непосчитанные окна (H = {h})")
        fut = x[s[:, None], t[:, None] + 1 + np.arange(h)]
        base = q[:, vi[f"L{L}"]].astype(np.float64)
        pinball = np.zeros(n)
        for k, lv in enumerate(QUANTILE_LEVELS):
            diff = fut - base[:, k]
            pinball += np.maximum(lv * diff, (lv - 1) * diff).mean(axis=1)
        f = {
            "row": w.row.to_numpy(), "L": L, "horizon": h,
            "mae": np.abs(fut - base[:, I_MED]).mean(axis=1),
            "crps": 2 * pinball / len(QUANTILE_LEVELS),
            "width": (base[:, I_HI] - base[:, I_LO]).mean(axis=1),
            "scale": scale,
            "s1": signals.interval_width(base[:, I_LO], base[:, I_HI], scale),
        }
        for i in check:
            assert np.isclose(f["crps"][i] / scale[i], metrics.scaled_crps(fut[i], base[i], QUANTILE_LEVELS, scale[i]))
        for name, variants in SPREAD_SETS.items():
            qs = [q[:, vi[v]] for v in variants]
            f[f"{name}_a"] = signals.median_spread([qq[:, I_MED] for qq in qs], scale)
            f[f"{name}_b"] = signals.quantile_distance(qs, scale)
        frames.append(pd.DataFrame(f))
        del q, base

    df = add_ratios(pd.concat(frames, ignore_index=True), w, L)  # + mase, mae_ratio, ячейка, d
    df["crps_s"] = df.crps / df.scale
    # Масштаб ряда — один на ряд: наивный масштаб отрезка длины L перед τ. Знаменатель не зависит от окна.
    tau = ds["tau"]
    series_scale = signals.naive_scale(x[:, tau - L : tau])
    df["mae_fixed"] = df.mae / series_scale[df.series_idx.to_numpy()]
    df = df.merge(pd.DataFrame({"row": w.row.to_numpy(), **common}), on="row")

    # S5 от горизонта не зависит: одно значение на окно.
    s5_file = signals_path(root, model_id, "s5")
    if not s5_file.exists():
        raise SystemExit(f"нет {s5_file}: сначала python -m scripts.cache_s5 --cache {root} --model {model_id}")
    with np.load(s5_file) as z:
        s5 = pd.DataFrame({"row": z["row"], **{k: z[k] for k in s5_keys()}})
    s5 = s5.rename(columns={v: k for k, v in S5_MAIN.items()})
    df = df.merge(s5, on="row", how="left")
    df["zone"] = assign_zone(df.d, df.horizon, L).where(df.cls != "none", CONTROL)
    return df


# ---------- метрики ----------


def add_combos(df: pd.DataFrame) -> pd.DataFrame:
    """Сочетания сигналов (COMBOS). Ранг — доля окон того же H с сигналом не больше данного,
    по всем окнам оценки; ошибка не используется. √(S1 · S4б) и S4б / S1 считаются по одному окну."""
    eps = 1e-12
    r = df.groupby("horizon")[["s1", "s2_16", "s4_b", "s4_k1_256_b", "s7_b"]].rank(pct=True)
    out = df.assign(
        c_rank_s1_s4=(r.s1 + r.s4_b) / 2,
        c_geo_s1_s4=np.sqrt(np.maximum(df.s1, eps) * np.maximum(df.s4_b, eps)),
        c_max_s1_s4=np.maximum(r.s1, r.s4_b),
        c_rank_s1_s2=(r.s1 + r.s2_16) / 2,
        c_rank_s1_s2_s4=(r.s1 + r.s2_16 + r.s4_b) / 3,
        c_rank_s1_s4k1=(r.s1 + r.s4_k1_256_b) / 2,
        c_ratio_s4_s1=df.s4_b / np.maximum(df.s1, eps),
        c_rank_all=(r.s1 + r.s2_16 + r.s4_b + r.s7_b) / 4,
    )
    assert set(COMBOS) <= set(out.columns)
    return out


def bootstrap_weights(series_idx: np.ndarray, n_boot: int, seed) -> list[np.ndarray]:
    """Веса окон в бутстреп-выборках по рядам: (выборки, окна), частями по BOOT_CHUNK."""
    codes, uniq = pd.factorize(series_idx)
    draws = np.random.default_rng(seed).integers(0, len(uniq), (n_boot, len(uniq)))
    counts = np.stack([np.bincount(r, minlength=len(uniq)) for r in draws])
    return [counts[a : a + BOOT_CHUNK][:, codes].astype(np.float64) for a in range(0, n_boot, BOOT_CHUNK)]


def signal_metrics(score, error, label, weights, order, error_order, mean_error, oracle_aurc) -> dict:
    a = metrics.aurc(score, error, weights, order)
    r = metrics.risk_at_coverage(score, error, COVERAGES, weights, order)
    return {
        "spearman": metrics.spearman(score, error, weights, order, error_order),
        "aurc": a,
        "gain": (mean_error - a) / (mean_error - oracle_aurc),
        "drop10": 1 - r[..., 0] / mean_error,
        "drop20": 1 - r[..., 1] / mean_error,
        "auroc": metrics.auroc(score, label, weights, order),
        "ap": metrics.average_precision(score, label, weights, order),
    }


def evaluate(df: pd.DataFrame, n_boot: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Все метрики всех сигналов и парные разности для H3.

    Интервалы — только для MASE: все сигналы на всех окнах, основные — и на остальных наборах окон.
    Для других ошибок — точечные оценки.
    """
    rows, diffs = [], []
    for h, dh in df.groupby("horizon"):
        for j, (subset, keep) in enumerate(SUBSETS.items()):
            t0 = time.perf_counter()
            dsub = dh[keep(dh)]
            chunks = bootstrap_weights(dsub.series_idx.to_numpy(), n_boot, [SEED, int(h), j])
            base = {"horizon": h, "subset": subset, "n_windows": len(dsub), "n_series": dsub.series_idx.nunique()}
            for target in TARGETS:
                e = dsub[target].to_numpy(np.float64)
                label = e >= np.quantile(e, 1 - WORST)
                e_order = metrics.rank_order(e)
                with_ci = target == MAIN_TARGET
                refs = [(e.mean(), metrics.aurc(e, e, None, e_order))]
                if with_ci:
                    refs += [((c @ e) / c.sum(axis=1), metrics.aurc(e, e, c, e_order)) for c in chunks]
                key_base = {**base, "target": target}
                random = {"spearman": 0.0, "aurc": e.mean(), "gain": 0.0, "drop10": 0.0, "drop20": 0.0,
                          "auroc": 0.5, "ap": label.mean()}
                rows += [{**key_base, "signal": "random", "metric": m, "value": v, "lo": np.nan, "hi": np.nan}
                         for m, v in random.items()]
                points, boots = {}, {}
                for key in ["oracle", *MAIN, *COMBOS, *ABLATIONS]:
                    score = e if key == "oracle" else dsub[key].to_numpy(np.float64)
                    order = e_order if key == "oracle" else metrics.rank_order(score)
                    point = signal_metrics(score, e, label, None, order, e_order, *refs[0])
                    boot = None
                    if with_ci and (subset == ALL or key not in ABLATIONS):
                        parts = [signal_metrics(score, e, label, c, order, e_order, *r) for c, r in zip(chunks, refs[1:])]
                        boot = {m: np.concatenate([p[m] for p in parts]) for m in METRIC_NAMES}
                    for m in METRIC_NAMES:
                        lo, hi = np.percentile(boot[m], [2.5, 97.5]) if boot else (np.nan, np.nan)
                        rows.append({**key_base, "signal": key, "metric": m, "value": float(point[m]), "lo": lo, "hi": hi})
                    points[key], boots[key] = point, boot
                if with_ci:  # сигнал минус сигнал на одних и тех же выборках (H3 и сочетания)
                    for a, b in DIFF_PAIRS:
                        for m in ("gain", "auroc", "spearman"):
                            lo, hi = np.percentile(boots[a][m] - boots[b][m], [2.5, 97.5])
                            diffs.append({**key_base, "signal": a, "versus": b, "metric": m,
                                          "value": float(points[a][m] - points[b][m]), "lo": lo, "hi": hi})
            del chunks
            print(f"  H = {h}, {subset}: {time.perf_counter() - t0:.0f} с")
    return pd.DataFrame(rows), pd.DataFrame(diffs)


def rejected_zones(df: pd.DataFrame, zones: list[str]) -> pd.DataFrame:
    """Куда попадают окна, отброшенные по сигналу, при отказе от REJECT окон (все окна, ошибка MASE)."""
    rows = []
    for h, dh in df.groupby("horizon"):
        zone = dh.zone.to_numpy()
        n_rej = int(round(REJECT * len(dh)))
        share = lambda idx: {z: float(np.mean(zone[idx] == z)) for z in zones}  # noqa: E731
        rows.append({"horizon": h, "signal": "random", **share(np.arange(len(dh)))})
        for key in [*MAIN, *COMBOS, "oracle"]:
            score = dh["mase" if key == "oracle" else key].to_numpy()
            rows.append({"horizon": h, "signal": key, **share(metrics.rank_order(score)[-n_rej:])})
    return pd.DataFrame(rows)


def spearman_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Спирмен сигнала с MASE внутри зон d, типов и сил смены (только основные сигналы)."""
    df = df.assign(kind_key=np.where(df.cls == "none", "none", df.cls + "-" + df.kind))
    rows = []
    for h, dh in df.groupby("horizon"):
        for by in ("zone", "kind_key", "level"):
            for g, dg in dh.groupby(by, sort=False):
                e = dg.mase.to_numpy()
                for key in MAIN:
                    rows.append({"horizon": h, "by": by, "group": g, "signal": key, "n": len(dg),
                                 "spearman": float(metrics.spearman(dg[key].to_numpy(), e))})
    return pd.DataFrame(rows)


def transition_response(df: pd.DataFrame) -> pd.DataFrame:
    """Во сколько раз сигнал и ошибка сразу после смены (0 ≤ d < 64) больше, чем в старом режиме (d < −H).

    Для каждого ряда — отношение медиан по окнам двух зон, затем геометрическое среднее по рядам ячейки.
    Ошибка — MAE / масштаб ряда до смены: её знаменатель один на ряд, поэтому отношение показывает
    рост самой ошибки. Хороший сигнал растёт там, где растёт ошибка, и не растёт там, где она не выросла.
    """
    keys = ["mae_fixed", *MAIN]
    rows = []
    for (h, cell), g in df.groupby(["horizon", "cell"], sort=False):
        old = g[g.d < -h].groupby("series_idx")[keys].median()
        new = g[(g.d >= 0) & (g.d < RESPONSE_ZONE)].groupby("series_idx")[keys].median()
        ratio = np.exp(np.log(np.maximum(new, 1e-12) / np.maximum(old, 1e-12)).mean())
        first = g.iloc[0]
        rows.append({"horizon": h, "cell": cell, "cls": first.cls, "kind": first.kind, "level": first.level,
                     **ratio.to_dict()})
    return pd.DataFrame(rows)


def risk_coverage_table(df: pd.DataFrame) -> pd.DataFrame:
    grid = np.round(np.arange(0.5, 1.0001, 0.01), 2)
    out = []
    for h, dh in df.groupby("horizon"):
        e = dh.mase.to_numpy()
        for key in [*MAIN, *FIG_EXTRA, "oracle"]:
            score = e if key == "oracle" else dh[key].to_numpy()
            out.append(pd.DataFrame({"horizon": h, "signal": key, "coverage": grid,
                                     "risk": metrics.risk_at_coverage(score, e, grid)}))
        out.append(pd.DataFrame({"horizon": h, "signal": "random", "coverage": grid, "risk": e.mean()}))
    return pd.concat(out, ignore_index=True)


def signal_curves(df: pd.DataFrame) -> pd.DataFrame:
    """Сигналы и MASE по d: геометрическое среднее по рядам ячейки, делённое на медиану
    того же сигнала на контроле без изменений (1 — как на контроле)."""
    out = []
    for h, dh in df.groupby("horizon"):
        ctrl = dh[dh.cls == "none"]
        for key in (*MAIN, "mase"):
            v = np.log(np.maximum(dh[key].to_numpy() / ctrl[key].median(), 1e-12))
            g = dh.assign(v=v).groupby(["cell", "cls", "kind", "level", "d"], sort=False).v.mean()
            out.append(np.exp(g).rename("value").reset_index().assign(horizon=h, signal=key))
    return pd.concat(out, ignore_index=True)


# ---------- графики ----------


def risk_coverage_figure(rc: pd.DataFrame, h: int, L: int, path: Path) -> None:
    c = rc[rc.horizon == h]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for key in ["random", *MAIN, *FIG_EXTRA, "oracle"]:
        g = c[c.signal == key]
        color, ls = STYLE[key]
        ax.plot(g.coverage, g.risk, color=color, ls=ls, lw=1.6, label=LABELS[key][0])
    ax.set_xlabel("покрытие — доля принятых окон (остальные отброшены по наибольшему сигналу)")
    ax.set_ylabel("средний MASE принятых окон")
    ax.set_title(f"Кривые риск–покрытие, все окна, L = {L}, H = {h}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def curves_figure(curves: pd.DataFrame, h: int, L: int, path: Path) -> None:
    c = curves[(curves.horizon == h) & curves.level.isin(["strong", "-"])]
    fig, axes = plt.subplots(3, 4, figsize=(16, 9.5), sharex=True, sharey=True)
    for ax, (cls, kind) in zip(axes.flat, panels(c)):
        p = c[(c.cls == cls) & (c.kind == kind)]
        mark_zones(ax, h, L)
        ax.axhline(1, color="gray", lw=0.8)
        for key in (*CURVE_KEYS, "mase"):
            g = p[p.signal == key]
            color, ls = STYLE[key]
            ax.plot(g.d, g.value, color=color, ls=ls, lw=2.4 if key == "mase" else 1.2,
                    label="ошибка (MASE)" if key == "mase" else LABELS[key][0])
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(FixedLocator([0.25, 0.5, 1, 2, 4, 8, 16, 32]))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_title(f"{CLASS_TITLES[cls]}: {TITLES[kind]}", fontsize=10)
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle(f"Сигналы и ошибка в зависимости от d, сила strong, L = {L}, H = {h}")
    fig.supxlabel("d = t − τ (серая полоса — смена в горизонте; пунктир — d = L − 1)")
    fig.supylabel("значение / медиана на контроле без изменений")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------- сводка ----------


def fmt(r, m: str) -> str:
    pct = m in ("drop10", "drop20")
    k, digits = (100, 0) if pct else (1, 3 if m == "aurc" else 2)
    unit = "%" if pct else ""
    v = f"{k * r.value:.{digits}f}{unit}"
    return v if np.isnan(r.lo) else f"{v} [{k * r.lo:.{digits}f}; {k * r.hi:.{digits}f}]"


def metric_table(met: pd.DataFrame, h: int, subset: str, target: str, keys: list[str], cols: list[str]) -> str:
    m = met[(met.horizon == h) & (met.subset == subset) & (met.target == target)].set_index(["signal", "metric"])
    rows = [[LABELS[k][0], LABELS[k][1]] + [fmt(m.loc[(k, c)], c) for c in cols] for k in keys]
    return md_table(["Сигнал", "Доп. прогонов", *[COLUMNS[c] for c in cols]], rows)


def group_table(groups: pd.DataFrame, h: int, by: str, order: list[str]) -> str:
    g = groups[(groups.horizon == h) & (groups.by == by)]
    order = [o for o in order if o in set(g.group)]
    rho = g.set_index(["group", "signal"]).spearman
    n = g.drop_duplicates("group").set_index("group").n
    rows = [["окон в группе"] + [str(n[o]) for o in order]]
    rows += [[LABELS[k][0]] + [f"{rho[(o, k)]:.2f}" for o in order] for k in MAIN]
    return md_table(["Сигнал", *order], rows)


def summarize(met, diffs, zones_df, groups, response, meta, zones, kinds, L) -> str:
    hs = sorted(met.horizon.unique())
    main_rows = ["random", *MAIN, "oracle"]
    n_all = met[(met.subset == ALL)].drop_duplicates("horizon").set_index("horizon")
    out = [
        f"# Этап 4: сигналы надёжности прогноза — {meta['model']}",
        "",
        f"Дата: {meta['date']}. Скрипт: `python -m scripts.stage4_signals --model {meta['model']}`.",
        f"Кеш `{meta['cache']}` (хеш конфига {meta['dataset_hash']}). Основной контекст L = {L}. "
        f"Окон в оценке: {n_all.n_windows.iloc[0]} на горизонт (d ≥ {meta['d_eval_min']}), рядов {n_all.n_series.iloc[0]}.",
        f"В квадратных скобках — 95% бутстреп-интервал по рядам ({meta['n_boot']} повторов; одни и те же "
        "выборки для всех сигналов, поэтому сравнение парное).",
        "",
        "## Как читать",
        "",
        "- **Ошибка окна** — MASE медианы основного прогноза. **Плохое окно** — 10% окон с наибольшей ошибкой "
        "(отдельно для каждого H и набора окон).",
        "- **Спирмен** — ранговая связь сигнала с ошибкой: 0 — связи нет, 1 — окна по сигналу упорядочены так же, "
        "как по ошибке.",
        "- **AURC** — средняя ошибка принятых окон, усреднённая по всем долям отказа; меньше — лучше. "
        "**Выигрыш AURC** = (AURC случайного отказа − AURC сигнала) / (AURC случайного отказа − AURC оракула): "
        "0 — сигнал не лучше случайного отказа, 1 — как оракул, который знает ошибку заранее, меньше 0 — хуже случайного.",
        "- **Ошибка ↓ при отказе от 10% (20%)** — на сколько процентов падает средний MASE принятых окон, если "
        "отбросить 10% (20%) окон с наибольшим сигналом. Случайный отказ в среднем даёт 0%.",
        "- **AUROC** — вероятность, что у плохого окна сигнал больше, чем у хорошего (0.5 — случайно). "
        "**AUPRC** — средняя доля плохих среди отброшенных (случайно — 0.10).",
        "- **Доп. прогонов** — сколько раз сигнал дополнительно вызывает модель на окно; бюджет этапа A — 5. "
        "Эти прогоны уже есть в кеше, скрипт модель не запускает.",
    ]
    cols = list(COLUMNS)
    for h in hs:
        out += ["", f"## Основная таблица: все окна, H = {h}", "",
                metric_table(met, h, ALL, MAIN_TARGET, main_rows, cols)]
    captions = {
        "без смены в горизонте": "Смену внутри горизонта (−H ≤ d < 0) по контексту не увидеть; здесь эти окна исключены.",
        TRANSITION: "Только окна сразу после смены, дрейфа или аномалии (0 ≤ d < 128), без контроля. Плохое окно — 10% "
                    "худших среди них: проверяется, упорядочивает ли сигнал окна внутри переходов («особенно в переходах» в H3).",
    }
    for subset, caption in captions.items():
        for h in hs:
            out += ["", f"## Набор окон «{subset}», H = {h}", "", caption, "",
                    metric_table(met, h, subset, MAIN_TARGET, main_rows, ["spearman", "gain", "drop10", "auroc", "ap"])]

    def diff_table(h: int, subset: str) -> str:
        d = diffs[(diffs.horizon == h) & (diffs.subset == subset) & (diffs.metric == "gain")].set_index(["signal", "versus"])
        rows = [[LABELS[a][0]] + [fmt(d.loc[(a, b)], "gain") for b in H3_BASELINES] for a in H3_ROWS]
        return md_table(["Сигнал", *[f"− {LABELS[b][0]}" for b in H3_BASELINES]], rows)

    for h in hs:
        out += ["", f"## H3: разность выигрыша AURC предлагаемых и базовых сигналов, H = {h}", "",
                "Выигрыш строки минус выигрыш столбца на одних и тех же бутстреп-выборках. Больше 0 — предлагаемый "
                "сигнал лучше; если интервал включает 0, разница не отличается от случайной. Разности AUROC и Спирмена — "
                "в h3_differences.csv.", "",
                f"Все окна:", "", diff_table(h, ALL), "", f"Переход, 0 ≤ d < 128:", "", diff_table(h, TRANSITION)]

    def combo_table(h: int) -> str:
        m = met[(met.horizon == h) & (met.target == MAIN_TARGET)].set_index(["subset", "signal", "metric"])
        d = diffs[(diffs.horizon == h) & (diffs.metric == "gain")].set_index(["subset", "signal", "versus"])
        rows = []
        for k in [*COMBO_VERSUS, *COMBOS]:
            row = [LABELS[k][0], LABELS[k][1], fmt(m.loc[(ALL, k, "gain")], "gain"),
                   fmt(m.loc[(TRANSITION, k, "gain")], "gain"),
                   f"{m.loc[(ALL, k, 'auroc')].value:.2f}", f"{m.loc[(ALL, k, 'ap')].value:.2f}"]
            row += [fmt(d.loc[(s, k, b)], "gain") if (s, k, b) in d.index else "—"
                    for s in (ALL, TRANSITION) for b in COMBO_VERSUS]
            rows.append(row)
        vs = [LABELS[b][0].split()[0] for b in COMBO_VERSUS]
        header = ["Сигнал", "Доп. прогонов", "Выигрыш AURC: все окна", "Выигрыш AURC: переход", "AUROC", "AUPRC"]
        header += [f"− {v}, {s}" for s in ("все окна", "переход") for v in vs]
        return md_table(header, rows)

    for h in hs:
        out += ["", f"## Сочетания сигналов (зафиксированы до расчёта), H = {h}", "",
                f"Восемь сочетаний из журнала решений CLAUDE.md (2026-11-01), без весов. Ранг — доля окон того же H "
                "с меньшим сигналом по всем окнам оценки. Первые две строки — S1 и S4б для сравнения. "
                "Последние четыре столбца — разность выигрыша AURC с S1 и S4б на одних и тех же бутстреп-выборках: "
                "больше 0 — сочетание лучше. Показаны все сочетания, лучшее задним числом не выбирается. "
                "Сами сочетания выбраны по результатам этого же этапа, поэтому вывод разведочный.", "",
                combo_table(h)]

    rcols = ("spearman", "gain")
    rows = []
    for h in hs:
        m = met[(met.horizon == h) & (met.subset == ALL)].set_index(["target", "signal", "metric"]).value
        for k in MAIN:
            rows.append([str(h), LABELS[k][0]] + [f"{m[(t, k, c)]:.2f}" for t in TARGETS for c in rcols])
    header = ["H", "Сигнал"] + [f"{TARGETS[t]}: {COLUMNS[c]}" for t in TARGETS for c in rcols]
    out += ["", "## Устойчивость к выбору ошибки, все окна", "",
            "Точечные оценки. MASE и CRPS делятся на масштаб контекста окна — тот же, что и S1, S2, S4, S7. "
            "«MAE / масштаб ряда до смены» делится на один масштаб на ряд (наивный прогноз на отрезке "
            f"длины {L} перед τ): если связь с ней такая же, как с MASE, она не объясняется общим знаменателем. "
            "«Отношение к устойчивому режиму» (этап 3) делит ошибку на обычную ошибку той же модели на том же "
            "ряду в том же режиме. Так убирается разница в трудности между рядами и режимами, и остаётся только "
            "избыточная ошибка окна (переход плюс случайный разброс отдельного окна).", "",
            md_table(header, rows)]

    for h in hs:
        z = zones_df[zones_df.horizon == h].set_index("signal")
        rows = [[LABELS[k][0]] + [f"{100 * z.loc[k, name]:.0f}%" for name in zones]
                for k in ["random", *MAIN, *COMBOS, "oracle"]]
        out += ["", f"## Куда попадают отказы: отброшено {REJECT:.0%} окон, все окна, H = {h}", "",
                "Доля отброшенных окон в каждой зоне d. Строка «случайный отказ» — доля зоны среди всех окон, "
                "«оракул» — где на самом деле лежат 10% худших окон. «Контроль» — ряды без изменений.", "",
                md_table(["Сигнал", *zones], rows)]

    for h in hs:
        out += ["", f"## Спирмен с MASE внутри зон d, H = {h}", "",
                "Насколько сигнал упорядочивает окна по ошибке внутри одной зоны.", "",
                group_table(groups, h, "zone", zones)]
    h = max(hs)
    out += ["", f"## Спирмен с MASE по типам изменений, H = {h}", "",
            "Все окна ячеек этого типа (все силы и d). H = 16 — в spearman_groups.csv.", "",
            group_table(groups, h, "kind_key", kinds),
            "", f"## Спирмен с MASE по силе изменения, H = {h}", "",
            group_table(groups, h, "level", ["weak", "medium", "strong", "-"])]

    h = max(hs)
    r = response[(response.horizon == h) & response.level.isin(["strong", "-"])]
    rows = [[c.cell] + [f"{getattr(c, k):.2f}" for k in ("mae_fixed", *MAIN)] for c in r.itertuples()]
    out += ["", f"## Отклик сигналов на переход: сила strong, H = {h}", "",
            f"Во сколько раз медиана по окнам 0 ≤ d < {RESPONSE_ZONE} больше, чем по окнам до смены (d < −H), "
            "геометрическое среднее по рядам. Первый столбец — ошибка (MAE / масштаб ряда до смены). "
            "Хороший сигнал растёт вместе с ошибкой и не растёт там, где ошибка не выросла (выброс, залипание, "
            "контроль). Все ячейки и H — в transition_response.csv.", "",
            md_table(["Ячейка", "Ошибка ×", *[f"{LABELS[k][0]} ×" for k in MAIN]], rows)]

    abl = ["s4_a", "s4_b", *[k for k in ABLATIONS if k.startswith("s4")],
           "s7_a", "s7_b", *[k for k in ABLATIONS if k.startswith("s7")],
           "s2_16", "s2_8", "s3_ks64", "s3_ks32", "s3_ks128", "s5_a", "s5_b", *S5_ABLATIONS]
    for h in hs:
        out += ["", f"## Абляции, все окна, H = {h}", "",
                "Основные варианты (без пометки) выбраны до запуска; остальные — для чувствительности. "
                "У S7 есть только один уровень шума (α = 0.1 std разностей, как в кеше).", "",
                metric_table(met, h, ALL, MAIN_TARGET, abl, ["spearman", "gain", "drop10", "auroc"])]

    out += [
        "",
        "## Ограничения",
        "",
        f"- Синтетика: {meta['n_per_cell']} рядов на ячейку. Состав окон задан сеткой d, зоны представлены "
        "неравномерно (строка «случайный отказ» в таблице отказов), и от этого зависят AURC и AUPRC.",
        "- S1, S2, S4, S7 и MASE делятся на один и тот же масштаб контекста. Если он у окна мал, вырастут и сигнал, "
        "и ошибка. Проверка — ошибка «MAE / масштаб ряда до смены» с одним знаменателем на ряд.",
        "- S2 взят из кеша: прогноз окна t − 16 (окна идут с шагом 8). В работе это один дополнительный прогон.",
        "- Шум S7 — только α = 0.1; другие уровни требуют новых прогонов. S6 (обучаемый предсказатель ошибки) — этап 7.",
        "- S5 — только вход и выход энкодера; промежуточные слои — этап B (H4). Дополнительный сигнал: "
        "в сравнение H3 и сочетания не входит.",
        "- AURC и AUPRC при бутстреп-весах приближённые (повторы окна принимаются вместе); расхождение — доли процента.",
        "- Ранги для сочетаний посчитаны один раз по всем окнам оценки, а не заново в каждой бутстреп-выборке. "
        "В работе ранг считался бы по эталонному набору окон; √(S1 · S4б) и S4б / S1 от других окон не зависят.",
    ]
    return "\n".join(out) + "\n"


# ---------- запуск ----------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--model", default="amazon/chronos-bolt-small")
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--out", type=Path, default=None,
                        help="по умолчанию results/stage4/<модель>; для другого кеша — results/stage4/<кеш>/<модель>")
    args = parser.parse_args()
    # Результаты по другому кешу (например, другой seed) не должны затирать основные.
    base = Path("results/stage4") if args.cache == DEFAULT_CACHE else Path("results/stage4") / args.cache.name
    out = args.out or base / model_dir(args.cache, args.model).name
    t0 = time.perf_counter()

    ds = load_dataset_meta(args.cache)
    cfg = yaml.safe_load((args.cache / "config.yaml").read_text())
    L, win = cfg["windows"]["context"], cfg["windows"]
    w = load_windows(args.cache)
    assert (w.row.to_numpy() == np.arange(len(w))).all()
    x, _ = load_series(args.cache)

    print(f"Сигналы и ошибки: {len(w)} окон × {len(ds['horizons'])} горизонта …")
    df = build_table(args.cache, args.model, w, x, ds, L)
    d_eval_min = win["d_min"] + max(S2_H0)
    pop = add_combos(df[df.d >= d_eval_min].reset_index(drop=True))
    sig_cols = [*MAIN, *COMBOS, *ABLATIONS]
    if pop[sig_cols + list(TARGETS)].isna().any().any():
        raise ValueError("пропуски в сигналах или ошибках окон оценки")
    print(f"  {time.perf_counter() - t0:.0f} с; окон в оценке: {pop.row.nunique()} (d ≥ {d_eval_min})")

    print("Метрики и бутстреп …")
    met, diffs = evaluate(pop, args.n_boot)
    zones = [*zone_names(L), CONTROL]
    zones_df = rejected_zones(pop, zones)
    groups = spearman_groups(pop)
    response = transition_response(pop)
    rc = risk_coverage_table(pop)
    curves = signal_curves(pop)
    kinds = list(dict.fromkeys(np.where(pop.cls == "none", "none", pop.cls + "-" + pop.kind)))

    meta = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"), "model": args.model, "cache": str(args.cache),
        "dataset_hash": ds["hash"], "n_per_cell": ds["n_series"] // ds["n_cells"], "L": L,
        "horizons": ds["horizons"], "d_eval_min": d_eval_min, "n_boot": args.n_boot, "seed": SEED,
        "worst": WORST, "coverages": COVERAGES, "s2_h0": S2_H0, "ks_w": KS_W, "spread_sets": SPREAD_SETS,
        "env": f"python {platform.python_version()}, numpy {np.__version__}, pandas {pd.__version__}",
        "cache_env": json.loads((model_dir(args.cache, args.model) / "meta.json").read_text()).get("env"),
    }
    out.mkdir(parents=True, exist_ok=True)
    keep = ["row", "cell", "cls", "kind", "level", "series_idx", "d", "zone", "horizon", *TARGETS, *sig_cols]
    pop[keep].to_csv(out / "window_signals.csv.gz", index=False, float_format="%.6g")
    met.to_csv(out / "metrics.csv", index=False, float_format="%.5g")
    diffs.to_csv(out / "h3_differences.csv", index=False, float_format="%.5g")
    zones_df.to_csv(out / "rejected_zones.csv", index=False, float_format="%.4g")
    groups.to_csv(out / "spearman_groups.csv", index=False, float_format="%.4g")
    response.to_csv(out / "transition_response.csv", index=False, float_format="%.4g")
    rc.to_csv(out / "risk_coverage.csv", index=False, float_format="%.5g")
    curves.to_csv(out / "signal_curves.csv.gz", index=False, float_format="%.5g")
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    for h in ds["horizons"]:
        risk_coverage_figure(rc, h, L, out / f"fig_risk_coverage_H{h}.png")
        curves_figure(curves, h, L, out / f"fig_signals_by_d_H{h}.png")
    (out / "summary.md").write_text(summarize(met, diffs, zones_df, groups, response, meta, zones, kinds, L))

    g = met[(met.subset == ALL) & (met.target == MAIN_TARGET) & (met.metric == "gain")].set_index(["horizon", "signal"])
    for h in ds["horizons"]:
        print(f"  выигрыш AURC, H = {h}: " + ", ".join(f"{k} {g.loc[(h, k)].value:.2f}" for k in [*MAIN, *COMBOS]))
    print(f"Готово за {time.perf_counter() - t0:.0f} с → {out}/")


if __name__ == "__main__":
    main()
