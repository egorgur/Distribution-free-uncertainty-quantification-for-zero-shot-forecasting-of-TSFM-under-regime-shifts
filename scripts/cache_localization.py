"""Этап B (H4): детекторы точки смены для всех окон кеша — счёт и τ′ каждого представления.

Запуск из корня репозитория (нужны ряды и окна кеша, см. scripts/cache_forecasts.py):
    python -m scripts.cache_localization --model raw                       # сырые значения, без модели
    python -m scripts.cache_localization --model amazon/chronos-bolt-small
    python -m scripts.cache_localization --model amazon/chronos-2 --device cuda
    python -m scripts.cache_localization --cache results/cache/synth_v1_seed1 --model raw
Пробный прогон на части ячеек, результат — в отдельную папку:
    python -m scripts.cache_localization --model raw --cells change-mean-strong,none --out /tmp/stage5_pilot

Определения — журнал решений docs/decisions.md (2026-11-01) и src/localization.py. Для каждого окна берётся
основной контекст (L = 512) и для каждого представления сохраняется:
    <детектор>_score_<представление>  — счёт (чем больше, тем увереннее «смена есть»);
    <детектор>_k_<представление>      — τ′: первая точка правого сегмента, в точках контекста (0 … 511).
Уровни модели — layer_names обёртки (input, block1 … blockN, final), детекторы rbf (основной) и l2
(абляция). У Chronos-2 выход энкодера зависит от горизонта: уровни считаются для каждого горизонта кеша
из прохода прогноза на этот горизонт, в файле тогда есть столбец horizon (длинный формат, как в s5.npz).
Сырые представления (--model raw): р1 — нормированные значения патча, р2 — статистики патча
(rbf и l2), р3 — PELT с rbf по точкам (pelt_cost, pelt_last: стоимость и последняя точка лучшего
разбиения с m = 0 … 15 точками смены; ответ при любом штрафе — localization.pelt_decision),
р4 — CUSUM среднего (cusum_score, cusum_k).

Эмбеддинги не сохраняются (у Chronos-2 это ~42 ГБ на seed): детекторы считаются сразу в проходе.
Если детектор поменяется, проход нужно повторить.

Сверки в конце: одно разбиение — с ruptures.Binseg, PELT — с ruptures.Pelt (на нескольких окнах);
окно в одиночку — как в общем батче; у Chronos-2 — прогноз из того же прохода против кеша
(или predict_quantiles, если прогнозов этой модели в кеше нет).

Результат: results/cache/<name>/<модель>/signals/localization.npz (+ .json) или, для raw,
results/cache/<name>/signals/localization_raw.npz (+ .json). Читает scripts/stage5_localization.py.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

# Один поток BLAS / OpenMP: процессы PELT не делят ядра между потоками. Действует и на torch (он читает
# переменную при импорте), поэтому энкодер на CPU тоже работает в один поток — так посчитан кеш synth_v1.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import ruptures as rpt
import yaml

from src import localization as loc
from src.cache import dataset_signals_path, load_dataset_meta, load_forecasts, load_series, load_windows, signals_path

SEED = 0
FORECAST_TOL = 1e-4  # прогноз из прохода против кеша, в std контекста (как в cache_forecasts, cache_s5)
SCORE_TOL = 1e-4  # отн. разница счёта: окно в одиночку против батча, наш детектор против ruptures
RAW_SPLITS = ("p1", "p2")  # представления р1, р2 — для них rbf и l2; р3 — PELT, р4 — CUSUM


def contexts(x: np.ndarray, series_idx: np.ndarray, t: np.ndarray, L: int) -> np.ndarray:
    """Основные контексты окон: (окна, L), последняя точка — t."""
    return x[series_idx[:, None], t[:, None] - L + 1 + np.arange(L)]


def raw_tokens(z: np.ndarray) -> dict[str, np.ndarray]:
    return {"p1": loc.raw_patches(z), "p2": loc.patch_stats(z)}


def raw_splits(ctx: np.ndarray) -> dict[str, np.ndarray]:
    """Детекторы одного разбиения на сырых значениях (всё, кроме PELT)."""
    z = loc.normalize_context(ctx)
    out = {}
    for name, tok in raw_tokens(z).items():
        out[f"rbf_score_{name}"], out[f"rbf_k_{name}"] = loc.split_rbf(tok)
        out[f"l2_score_{name}"], out[f"l2_k_{name}"] = loc.split_l2(tok)
    out["cusum_score"], out["cusum_k"] = loc.split_cusum(z)
    return out


def level_splits(layers: np.ndarray, names: list[str]) -> dict[str, np.ndarray]:
    """rbf и l2 на каждом уровне: layers (окна, уровни, патчи, D)."""
    out = {}
    for li, name in enumerate(names):
        tok = layers[:, li]
        out[f"rbf_score_{name}"], out[f"rbf_k_{name}"] = loc.split_rbf(tok)
        out[f"l2_score_{name}"], out[f"l2_k_{name}"] = loc.split_l2(tok)
    return out


def binseg_diff(tokens: np.ndarray, score: dict, k: dict, i: int, name: str) -> tuple[float, int]:
    """Наш детектор против ruptures.Binseg (одно разбиение) на окне i: (макс. отн. разница счёта, |Δk|)."""
    n, dim = tokens.shape
    unit = 512 // n
    worst_s, worst_k = 0.0, 0
    for det, sig, norm in (("rbf", tokens, n), ("l2", loc.standardize_columns(tokens[None])[0], n * dim)):
        bkp, gain = rpt.Binseg(model=det, min_size=loc.MIN_POINTS // unit, jump=1).fit(sig).single_bkp(0, n)
        ours = score[f"{det}_score_{name}"][i]
        worst_s = max(worst_s, abs(gain / norm - ours) / max(abs(ours), 1e-12))
        worst_k = max(worst_k, abs(bkp * unit - int(k[f"{det}_k_{name}"][i])))
    return worst_s, worst_k


def env_string() -> str:
    pk = ["numpy", "ruptures", "scipy"]
    s = f"python {platform.python_version()}, " + ", ".join(f"{p} {version(p)}" for p in pk)
    try:
        import torch

        s += f", torch {torch.__version__}, transformers {version('transformers')}, "
        s += f"chronos-forecasting {version('chronos-forecasting')}"
    except ImportError:
        pass
    return s


def run_raw(ctx: np.ndarray, args, verify_idx: np.ndarray) -> tuple[dict, dict]:
    t0 = time.perf_counter()
    out: dict[str, list] = {}
    for a in range(0, len(ctx), args.chunk):
        for key, v in raw_splits(ctx[a : a + args.chunk]).items():
            out.setdefault(key, []).append(v)
    out = {key: np.concatenate(v) for key, v in out.items()}
    print(f"  р1, р2, р4: {time.perf_counter() - t0:.0f} с")
    t1 = time.perf_counter()
    out["pelt_cost"], out["pelt_last"] = loc.pelt_path_many(loc.normalize_context(ctx), processes=args.processes)
    print(f"  р3 (PELT, {args.processes} проц.): {time.perf_counter() - t1:.0f} с")

    # Сверки на нескольких окнах.
    checks = {"binseg_max_rel": 0.0, "binseg_max_dk": 0, "pelt_mismatch": 0, "pelt_compared": 0}
    z = loc.normalize_context(ctx[verify_idx])
    for j, i in enumerate(verify_idx):
        for name, tok in raw_tokens(z[j : j + 1]).items():
            s_, k_ = binseg_diff(tok[0].astype(np.float64), out, out, i, name)
            checks["binseg_max_rel"] = max(checks["binseg_max_rel"], s_)
            checks["binseg_max_dk"] = max(checks["binseg_max_dk"], k_)
    beta = loc.pelt_critical_penalty(out["pelt_cost"])
    for i in verify_idx[: args.verify_pelt]:
        zi = loc.normalize_context(ctx[i : i + 1])[0]
        for f in (0.99, 1.01):
            m, k = loc.pelt_decision(out["pelt_cost"][i : i + 1], out["pelt_last"][i : i + 1], beta[i] * f)
            bk = rpt.Pelt(model="rbf", min_size=loc.MIN_POINTS, jump=1).fit(zi).predict(pen=beta[i] * f)
            ref_m, ref_k = len(bk) - 1, (bk[-2] if len(bk) > 1 else 0)
            checks["pelt_compared"] += 1
            checks["pelt_mismatch"] += int(ref_m != m[0] or ref_k != k[0])
    # Окно в одиночку — как в общем расчёте.
    single = max(
        abs(float(raw_splits(ctx[i : i + 1])[key][0]) - float(out[key][i])) / max(abs(float(out[key][i])), 1e-12)
        for i in verify_idx for key in out if key.endswith("score") or "_score_" in key
    )
    checks["single_vs_batch_max_rel"] = single
    return out, checks


def run_model(ctx: np.ndarray, args, verify_idx: np.ndarray, ds: dict, rows: np.ndarray) -> tuple[dict, dict, dict]:
    import torch

    from src.models import load_forecaster

    torch.manual_seed(SEED)
    torch.set_grad_enabled(False)
    model = load_forecaster(args.model, device=args.device)
    names = model.layer_names
    horizons = list(ds["horizons"]) if model.embeddings_depend_on_horizon else [None]
    n, t0 = len(ctx), time.perf_counter()
    out = {}
    for hi, h in enumerate(horizons):
        for a in range(0, n, args.chunk):
            sl = slice(a, a + args.chunk)
            layers = model.encode_layers(ctx[sl].astype(np.float32), batch_size=args.batch_size, horizon=h)
            for key, v in level_splits(layers, names).items():
                if key not in out:
                    out[key] = np.empty((len(horizons), n), dtype=np.float64 if "score" in key else np.int64)
                out[key][hi, sl] = v
            done = hi * n + min(a + args.chunk, n)
            el = time.perf_counter() - t0
            print(f"  {'' if h is None else f'H = {h}: '}{min(a + args.chunk, n)}/{n} окон, {el:.0f} с; "
                  f"осталось ~{el / done * (len(horizons) * n - done) / 60:.1f} мин", flush=True)

    checks = {"binseg_max_rel": 0.0, "binseg_max_dk": 0, "single_vs_batch_max_rel": 0.0, "single_vs_batch_dk": 0,
              "forecast_max_diff_std": None, "input_final_vs_encode_patches": 0.0}
    vctx = ctx[verify_idx].astype(np.float32)
    for hi, h in enumerate(horizons):
        # Окно в одиночку (батч 1) — как в общем расчёте; заодно сверка с ruptures.Binseg на всех уровнях.
        for j, i in enumerate(verify_idx):
            one = model.encode_layers(vctx[j : j + 1], batch_size=1, horizon=h)
            res = level_splits(one, names)
            for key, v in res.items():
                ref = out[key][hi, i]
                if "score" in key:
                    rel = abs(float(v[0]) - float(ref)) / max(abs(float(ref)), 1e-12)
                    checks["single_vs_batch_max_rel"] = max(checks["single_vs_batch_max_rel"], rel)
                else:
                    checks["single_vs_batch_dk"] = max(checks["single_vs_batch_dk"], abs(int(v[0]) - int(ref)))
            cur = {key: v[hi] for key, v in out.items()}
            for li, name in enumerate(names):
                s_, k_ = binseg_diff(one[0, li].astype(np.float64), cur, cur, i, name)
                checks["binseg_max_rel"] = max(checks["binseg_max_rel"], s_)
                checks["binseg_max_dk"] = max(checks["binseg_max_dk"], k_)
        # Уровни input и final — те же, что encode_patches (по ним считался S5).
        lay = model.encode_layers(vctx, batch_size=args.batch_size, horizon=h)
        inp, fin = model.encode_patches(vctx, batch_size=args.batch_size, horizon=h)
        diff = max(np.abs(lay[:, 0] - inp).max(), np.abs(lay[:, -1] - fin).max())
        checks["input_final_vs_encode_patches"] = max(checks["input_final_vs_encode_patches"], float(diff))
        # Chronos-2: прогноз из того же прохода — против кеша или predict_quantiles.
        if h is not None:
            _, q = model.encode_layers_with_forecast(vctx, h, batch_size=args.batch_size)
            sd = vctx.std(axis=1)[:, None, None]
            meta_path = Path(args.cache) / args.model.split("/")[-1] / "meta.json"
            if meta_path.exists():
                vi = ds["variants"].index("L512")
                ref = load_forecasts(args.cache, args.model, h, variants=[vi])[rows[verify_idx], 0]
                checks["forecast_reference"] = "кеш"
            else:
                ref = model.predict_quantiles(list(vctx), h)
                checks["forecast_reference"] = "predict_quantiles"
            d = float(np.max(np.abs(q - ref) / sd))
            checks["forecast_max_diff_std"] = max(checks["forecast_max_diff_std"] or 0.0, d)
    levels = {"layers": names, "depths": model.layer_depths, "horizons": None if horizons == [None] else horizons,
              "n_blocks": model.n_blocks}
    return out, checks, levels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path, default=Path("results/cache/synth_v1"))
    parser.add_argument("--model", default="raw", help="raw (сырые значения) или имя весов, например amazon/chronos-2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--chunk", type=int, default=1024, help="окон за раз")
    parser.add_argument("--processes", type=int, default=max(1, (os.cpu_count() or 2) - 2), help="процессов для PELT")
    parser.add_argument("--cells", default=None, help="только эти ячейки (через запятую) — для пробного прогона")
    parser.add_argument("--out", type=Path, default=None, help="папка для результата вместо кеша")
    parser.add_argument("--verify", type=int, default=4, help="сколько окон сверять")
    parser.add_argument("--verify-pelt", type=int, default=2, help="на скольких из них сверять PELT (~1.3 с на вызов)")
    args = parser.parse_args()
    t0 = time.perf_counter()

    ds = load_dataset_meta(args.cache)
    L = yaml.safe_load((args.cache / "config.yaml").read_text())["windows"]["context"]
    assert L == 512, "детекторы рассчитаны на L = 512 (32 патча)"
    w = load_windows(args.cache)
    assert (w.row.to_numpy() == np.arange(len(w))).all()
    if args.cells:
        w = w[w.cell.isin(args.cells.split(","))]
        if w.empty:
            raise ValueError(f"нет окон для ячеек {args.cells}")
    x, _ = load_series(args.cache)
    rows = w.row.to_numpy()
    ctx = contexts(x, w.series_idx.to_numpy(), w.t.to_numpy(), L)
    verify_idx = np.random.default_rng(SEED).choice(len(w), min(args.verify, len(w)), replace=False)
    print(f"Этап B: {len(w)} окон, L = {L}, {args.model}, кеш {args.cache}")

    if args.model == "raw":
        out, checks = run_raw(ctx, args, verify_idx)
        levels = {"splits": list(RAW_SPLITS), "pelt_max_changes": loc.MAX_CHANGES}
        default_path = dataset_signals_path(args.cache, "localization_raw")
        save = {"row": rows, **out}
    else:
        out, checks, levels = run_model(ctx, args, verify_idx, ds, rows)
        default_path = signals_path(args.cache, args.model, "localization")
        if levels["horizons"] is None:
            save = {"row": rows, **{key: v[0] for key, v in out.items()}}
        else:  # длинный формат: строка на (окно, горизонт)
            nh = len(levels["horizons"])
            save = {"row": np.tile(rows, nh), "horizon": np.repeat(levels["horizons"], len(rows)),
                    **{key: v.reshape(-1) for key, v in out.items()}}

    # Итог сверок; при провале — ошибка, файл не пишется.
    print("Сверки:", json.dumps(checks, ensure_ascii=False))
    fails = []
    if checks["binseg_max_rel"] > SCORE_TOL or checks["binseg_max_dk"] > 0:
        fails.append("детектор одного разбиения не совпал с ruptures.Binseg")
    if checks.get("pelt_mismatch"):
        fails.append("PELT не совпал с ruptures.Pelt")
    if checks["single_vs_batch_max_rel"] > SCORE_TOL:
        fails.append("счёт окна зависит от состава батча")
    if checks.get("forecast_max_diff_std") is not None and checks["forecast_max_diff_std"] > FORECAST_TOL:
        fails.append("уровни сняты не с того прохода, что прогноз")
    if checks.get("input_final_vs_encode_patches", 0.0) > 0:
        fails.append("уровни input / final не совпали с encode_patches")
    bad = [key for key, v in out.items() if not np.isfinite(v).all()]
    if bad:
        fails.append(f"нечисловые значения: {bad}")
    if fails:
        raise RuntimeError("; ".join(fails))

    path = default_path if args.out is None else args.out / default_path.relative_to(args.cache)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **save)
    meta = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"), "model": args.model, "dataset_hash": ds["hash"],
        "L": L, "min_points": loc.MIN_POINTS, "n_windows": len(rows), "cells": args.cells, **levels,
        "seconds": round(time.perf_counter() - t0), "device": args.device, "checks": checks, "env": env_string(),
    }
    path.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"Готово за {time.perf_counter() - t0:.0f} с → {path}")


if __name__ == "__main__":
    main()
