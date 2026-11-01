"""Каскад (H5): прогнозы по обрезанным контекстам для всех окон кеша; сохраняются метрики окна каждой обрезки.

Запуск из корня репозитория (нужны кеш прогнозов, localization*.npz этапа B и results/stage5/meta.json):
    python -m scripts.cache_cascade --model amazon/chronos-bolt-small
    python -m scripts.cache_cascade --model amazon/chronos-2 --device cuda

Обрезки окна — начало контекста; определения — docs/decisions.md (2026-11-02) и src/cascade.py:
    own     — по τ′ уровня модели, выбранного в H4 (Bolt блок 6; Chronos-2 блок 8 из прохода того же H), при
              любом счёте детектора: тогда любой порог этапов A и B оценивается без новых прогонов;
    p2      — по τ′ детектора р2 (статистики патча), тоже при любом счёте;
    oracle  — по истинной τ (П4, П5): change и drift при 31 ≤ d < L − 1;
    rag     — reset-and-grow (П3) с порогом этапа B из results/stage5/meta.json.
Совпадающие контексты считаются один раз. Сохраняются только метрики окна (MAE, CRPS, покрытие и ширина
80%-интервала, в единицах ряда), сами прогнозы — нет.

У Chronos-Bolt прогноз на 16 шагов — начало прогноза на 64, а τ′ от горизонта не зависит: один проход на H = 64.
У Chronos-2 τ′ своя для каждого H (уровни сняты с прохода прогноза на этот H), и каждый H — отдельный проход.

Сверки (при провале — ошибка, файл не пишется): прогноз по полному контексту, посчитанный в одном батче
с обрезанными контекстами, совпадает с кешем (L512); обрезанный контекст в одиночку — как в общем батче.

Результат: results/cache/<name>/<модель>/signals/cascade.npz (+ .json), длинный формат: строка на (окно, горизонт);
столбцы start_<обрезка> (−1 — обрезки нет) и <метрика>_<обрезка>. Читает scripts/stage6_cascade.py.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime
from pathlib import Path

import chronos
import numpy as np
import torch
import transformers
import yaml

from src import cascade as cas
from src.cache import dataset_signals_path, load_dataset_meta, load_forecasts, load_series, load_windows, signals_path
from src.models import QUANTILE_LEVELS, load_forecaster

SEED = 0
STAGE5_META = Path("results/stage5/meta.json")
FORECAST_TOL = 1e-4  # в std контекста, как в cache_forecasts
CROPS = ("own", "p2", "oracle", "rag")
METRICS = ("mae", "crps", "coverage", "width")
LEVELS = cas.STAGE_B
RAW_KEY = cas.RAW_DETECTOR


def env_string() -> str:
    # Не из cache_localization: при импорте он ставит OMP_NUM_THREADS=1, и torch стал бы однопоточным.
    return (f"python {platform.python_version()}, numpy {np.__version__}, torch {torch.__version__}, "
            f"transformers {transformers.__version__}, chronos-forecasting {chronos.__version__}")


def aligned(z, key: str, rows: np.ndarray, sel: np.ndarray, n: int) -> np.ndarray:
    """Столбец key файла этапа B в порядке windows.csv (rows — номера окон строк файла, sel — нужные строки)."""
    out = np.empty(n, dtype=z[key].dtype)
    r = rows[sel]
    assert np.array_equal(np.sort(r), np.arange(n)), f"{key}: посчитаны не все окна"
    out[r] = z[key][sel]
    return out


def predict_sorted(model, ctxs: list[np.ndarray], horizon: int, batch_size: int) -> np.ndarray:
    """predict_quantiles по контекстам, отсортированным по длине (меньше дополнения NaN в батче),
    в исходном порядке."""
    order = np.argsort([len(c) for c in ctxs], kind="stable")
    q = model.predict_quantiles([ctxs[i] for i in order], horizon=horizon, batch_size=batch_size)
    out = np.empty_like(q)
    out[order] = q
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path, default=Path("results/cache/synth_v1"))
    parser.add_argument("--model", default="amazon/chronos-bolt-small", choices=sorted(LEVELS))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--verify", type=int, default=64, help="сколько окон сверять")
    args = parser.parse_args()
    t_start = time.perf_counter()
    torch.manual_seed(SEED)

    ds = load_dataset_meta(args.cache)
    cfg = yaml.safe_load((args.cache / "config.yaml").read_text())
    L, dur, tau = cfg["windows"]["context"], cfg["drift"]["duration"], ds["tau"]
    w = load_windows(args.cache)
    assert (w.row.to_numpy() == np.arange(len(w))).all()
    x, _ = load_series(args.cache)
    n = len(w)
    t, s = w.t.to_numpy(), w.series_idx.to_numpy()
    cls, d = w.cls.to_numpy(), w.d.to_numpy()

    # Пороги и выбранные уровни этапа B (подобраны по seed 1); проверяем, что они от этих данных.
    s5 = json.loads(STAGE5_META.read_text())
    if ds["hash"] not in (s5["report_hash"], s5["tune_hash"]):
        raise SystemExit(f"{STAGE5_META} посчитан для других данных (хеш {ds['hash']} не совпал)")
    thr_raw = s5["thresholds"]["raw"][RAW_KEY]

    raw = np.load(dataset_signals_path(args.cache, "localization_raw"))
    all_rows = np.ones(len(raw["row"]), bool)
    k_p2 = aligned(raw, f"rbf_k_{RAW_KEY}", raw["row"], all_rows, n)
    loc_m = np.load(signals_path(args.cache, args.model, "localization"))

    model = load_forecaster(args.model, device=args.device)
    passes = {h: list(ds["horizons"]) if h is None else [h] for h in LEVELS[args.model]}
    starts: dict[int, dict[str, np.ndarray]] = {}
    values: dict[int, dict[str, np.ndarray]] = {}
    info: dict = {"thresholds": {RAW_KEY: thr_raw}, "levels": {}, "n_forecasts": {}, "crops": {}}
    checks = {"full_vs_cache_max_std": 0.0, "single_vs_batch_max_std": 0.0}
    rng = np.random.default_rng(SEED)
    for h_loc, horizons in passes.items():
        cfg_name, level = LEVELS[args.model][h_loc]
        assert s5["selection"][cfg_name]["best_level"] == f"lvl:{level}", (cfg_name, s5["selection"][cfg_name])
        thr = s5["thresholds"][cfg_name][f"lvl:{level}"]
        sel = np.ones(len(loc_m["row"]), bool) if h_loc is None else loc_m["horizon"] == h_loc
        score_m = aligned(loc_m, f"rbf_score_{level}", loc_m["row"], sel, n)
        k_m = aligned(loc_m, f"rbf_k_{level}", loc_m["row"], sel, n)
        own = cas.crop_start(t, k_m, L)
        st = {
            "own": own,
            "p2": cas.crop_start(t, k_p2, L),
            "oracle": cas.oracle_start(cls, d, tau, L),
            "rag": cas.reset_and_grow_starts(s, t, score_m > thr, own, L),
        }
        key = f"H{h_loc}" if h_loc else "all"
        info["levels"][key] = {"stage5_config": cfg_name, "level": level, "threshold": thr}
        info["crops"][key] = {c: int((v != cas.NO_CROP).sum()) for c, v in st.items()}

        # Уникальные контексты (ряд, начало, конец) по всем обрезкам.
        keys = np.concatenate([np.stack([s, v, t], axis=1)[v != cas.NO_CROP] for v in st.values()])
        uniq, inv = np.unique(keys, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        ctxs = [x[a, b : c + 1] for a, b, c in uniq]
        h_max = max(horizons)
        print(f"{args.model}, проход H = {h_max}: {len(uniq)} уникальных контекстов "
              f"(обрезок: {info['crops'][key]})", flush=True)
        info["n_forecasts"][key] = int(len(uniq))
        t0 = time.perf_counter()
        q = predict_sorted(model, ctxs, h_max, args.batch_size)
        sec = time.perf_counter() - t0
        info.setdefault("seconds_forecast", {})[key] = round(sec, 1)
        print(f"  {sec:.0f} с, {sec / len(uniq) * 1e3:.2f} мс на прогноз", flush=True)

        # Метрики каждого уникального контекста и каждого горизонта прохода → по обрезкам окон.
        for h in horizons:
            fut = x[uniq[:, 0, None], uniq[:, 2, None] + 1 + np.arange(h)]
            m = cas.forecast_metrics(q[:, :, :h], fut, QUANTILE_LEVELS)
            starts[h], values[h] = {}, {}
            offset = 0
            for c, v in st.items():
                has = v != cas.NO_CROP
                idx = inv[offset : offset + has.sum()]
                offset += has.sum()
                starts[h][c] = v
                for name, arr in m.items():
                    col = np.full(n, np.nan, dtype=np.float32)
                    col[has] = arr[idx]
                    values[h][f"{name}_{c}"] = col

        # Сверка 1: полный контекст в одном батче с обрезанными — против кеша (L512).
        vi = rng.choice(n, args.verify, replace=False)
        crop_i = rng.choice(len(uniq), args.verify, replace=False)
        full = [x[s[i], t[i] - L + 1 : t[i] + 1] for i in vi]
        qv = predict_sorted(model, full + [ctxs[j] for j in crop_i], h_max, args.batch_size)
        for h in horizons:
            ref = load_forecasts(args.cache, args.model, h, variants=[ds["variants"].index(f"L{L}")])[vi, 0]
            sd = np.array([c.std() for c in full])[:, None, None]
            diff = float(np.max(np.abs(qv[: len(vi), :, :h] - ref) / sd))
            checks["full_vs_cache_max_std"] = max(checks["full_vs_cache_max_std"], diff)
        # Сверка 2: обрезанный контекст в одиночку — как в общем батче.
        for j in crop_i[:8]:
            one = model.predict_quantiles([ctxs[j]], horizon=h_max, batch_size=1)
            a, _, c = uniq[j]
            sd = x[a, c - L + 1 : c + 1].std()
            checks["single_vs_batch_max_std"] = max(checks["single_vs_batch_max_std"],
                                                    float(np.max(np.abs(one[0] - q[j])) / sd))
        del q, ctxs

    print("Сверки:", json.dumps(checks))
    if max(checks.values()) > FORECAST_TOL:
        raise RuntimeError(f"сверка не прошла (порог {FORECAST_TOL} std контекста)")

    hs = list(ds["horizons"])
    save = {"row": np.tile(w.row.to_numpy(), len(hs)), "horizon": np.repeat(hs, n)}
    for c in CROPS:
        save[f"start_{c}"] = np.concatenate([starts[h][c] for h in hs]).astype(np.int64)
        for name in METRICS:
            save[f"{name}_{c}"] = np.concatenate([values[h][f"{name}_{c}"] for h in hs])
    path = signals_path(args.cache, args.model, "cascade")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **save)
    meta = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"), "model": args.model, "dataset_hash": ds["hash"],
        "stage5_meta": str(STAGE5_META), "L": L, "min_points": cas.MIN_POINTS, "n_windows": n, **info,
        "seconds": round(time.perf_counter() - t_start), "device": args.device, "batch_size": args.batch_size,
        "checks": checks, "env": env_string(),
    }
    path.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"Готово за {time.perf_counter() - t_start:.0f} с → {path}")


if __name__ == "__main__":
    main()
