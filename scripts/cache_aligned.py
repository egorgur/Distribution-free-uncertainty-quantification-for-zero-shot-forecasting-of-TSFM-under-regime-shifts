"""Regime-Aligned Context: прогнозы по выровненным контекстам для всех окон кеша (docs/protocol_v3_aligned.md).

    python -m scripts.cache_aligned                                            # Bolt, seed 0, CPU
    python -m scripts.cache_aligned --model amazon/chronos-2 --device cuda
    python -m scripts.cache_aligned --cells change-mean-strong --limit 64 --out <папка>   # пробный прогон

Для каждого окна строятся 12 контекстов L512: преобразования level/scale/trend (src/adaptation/alignment.py)
с α ∈ {0.25, 0.5, 0.75, 1}; недавние 64 точки не меняются. Статистики — только по самому контексту
(история до t), поэтому прогноз окна pos − 16 служит честным прогнозом «назад» для окна pos.

Сохраняется в <кеш>/<модель>/aligned_v1/<ячейка>.npz (подпапка не читается load_forecasts):
    row          — строки windows.csv;
    q{H}_full    — (окна, 3 преобразования при α = 1, 9, H) квантили (H = 64; у Chronos-2 и 16);
    med16        — (окна, 12 вариантов, 16) медиана первых 16 шагов (для hindcast; у Chronos-2 из вызова H16);
    mae{H}, crps{H}, cov{H}, width{H} — (окна, 12) метрики окна в единицах ряда, как в src/cascade.py;
    fallback     — (окна,) scale заменён на level из-за малого MAD.
Порядок 12 вариантов — VARIANTS. Прерванный прогон продолжается: готовые ячейки пропускаются.
Сверки (при провале — ошибка): Original тем же путём = кеш L512 (≤ 1e-4 std контекста); одиночный вызов = батч.
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

from src import cascade as cas
from src import signals
from src.adaptation import alignment as al
from src.cache import load_dataset_meta, load_forecasts, load_series, load_windows, model_dir
from src.models import QUANTILE_LEVELS, load_forecaster

ALPHAS = (1.0, 0.25, 0.5, 0.75)  # α = 1 первыми: их квантили хранятся полностью
VARIANTS = [(k, a) for a in ALPHAS for k in al.KINDS]
SUBDIR = "aligned_v1"
L = 512
TOL = 1e-4  # в std контекста, как в cache_forecasts


def variant_name(kind: str, alpha: float) -> str:
    return kind if alpha == 1 else f"{kind}_{alpha:g}"


def build_contexts(x: np.ndarray, series: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Контексты всех вариантов: (окна, 12, L) и флаг fallback scale (окна,)."""
    ctx = x[series[:, None], t[:, None] - L + 1 + np.arange(L)]
    scale = signals.naive_scale(ctx)
    out = np.empty((len(ctx), len(VARIANTS), L))
    fb = np.zeros(len(ctx), bool)
    for j, (kind, alpha) in enumerate(VARIANTS):
        out[:, j], f = al.aligned_context(ctx, kind, alpha, scale)
        fb |= f
    return out, fb


def forecast(model, ctxs: np.ndarray, horizon: int, batch: int) -> np.ndarray:
    n, v, _ = ctxs.shape
    q = model.predict_quantiles(list(ctxs.reshape(n * v, L)), horizon=horizon, batch_size=batch)
    return q.reshape(n, v, *q.shape[1:])


def window_metrics(q: np.ndarray, fut: np.ndarray) -> dict[str, np.ndarray]:
    """Метрики (окна, варианты) по квантилям (окна, варианты, 9, H)."""
    n, v = q.shape[:2]
    m = cas.forecast_metrics(q.reshape(n * v, *q.shape[2:]), np.repeat(fut, v, axis=0), QUANTILE_LEVELS)
    return {k: val.reshape(n, v).astype(np.float32) for k, val in m.items()}


def check_original(model, root: Path, model_id: str, x, w, horizon: int, batch: int, n: int = 16) -> float:
    """Original через тот же путь (контекст L512, тот же вызов) против кеша; одиночный = батч."""
    rows = np.random.default_rng(0).choice(len(w), n, replace=False)
    s, t = w.series_idx.to_numpy()[rows], w.t.to_numpy()[rows]
    ctx = x[s[:, None], t[:, None] - L + 1 + np.arange(L)]
    q = model.predict_quantiles(list(ctx), horizon=horizon, batch_size=batch)
    ref = load_forecasts(root, model_id, horizon, variants=[0])[rows, 0]
    std = ctx.std(axis=1)[:, None, None]
    worst = float(np.max(np.abs(q - ref) / std))
    single = model.predict_quantiles([ctx[0]], horizon=horizon)
    worst_single = float(np.max(np.abs(single[0] - q[0]) / std[0]))
    if worst > TOL or worst_single > TOL:
        raise AssertionError(f"сверка Original: кеш {worst:.2e}, одиночный {worst_single:.2e} std контекста")
    return max(worst, worst_single)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="amazon/chronos-bolt-small")
    parser.add_argument("--cache", type=Path, default=Path("results/cache/synth_v1"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--cells", nargs="*", help="только эти ячейки (пробный прогон)")
    parser.add_argument("--limit", type=int, help="только первые N окон ячейки (пробный прогон)")
    parser.add_argument("--out", type=Path, help="другая папка вывода (пробный прогон)")
    args = parser.parse_args()
    t_start = time.perf_counter()
    torch.manual_seed(0)
    ds = load_dataset_meta(args.cache)
    w = load_windows(args.cache)
    x, _ = load_series(args.cache)
    out_dir = args.out or (model_dir(args.cache, args.model) / SUBDIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = load_forecaster(args.model, device=args.device)
    horizons = [max(ds["horizons"])] if model.short_horizon_is_prefix else sorted(ds["horizons"], reverse=True)
    check = {h: check_original(model, args.cache, args.model, x, w, h, args.batch_size) for h in horizons}
    print(f"сверка Original пройдена: {check}", flush=True)

    cells = args.cells or sorted(w.cell.unique())
    n_fc, sec_fc, fallback_total = 0, 0.0, 0
    for cell in cells:
        path = out_dir / f"{cell}.npz"
        if path.exists() and not args.limit:
            continue
        wc = w[w.cell == cell]
        if args.limit:
            wc = wc.iloc[: args.limit]
        s, t = wc.series_idx.to_numpy(), wc.t.to_numpy()
        ctxs, fb = build_contexts(x, s, t)
        res = {"row": wc.row.to_numpy(), "fallback": fb}
        t0 = time.perf_counter()
        for h in horizons:
            q = forecast(model, ctxs, h, args.batch_size)
            fut = x[s[:, None], t[:, None] + 1 + np.arange(h)]
            res[f"q{h}_full"] = q[:, :len(al.KINDS)].astype(np.float32)
            for k, v in window_metrics(q, fut).items():
                res[f"{'cov' if k == 'coverage' else k}{h}"] = v
            if h == 16 or model.short_horizon_is_prefix:
                res["med16"] = q[:, :, QUANTILE_LEVELS.index(0.5), :16].astype(np.float32)
            if model.short_horizon_is_prefix:  # у Bolt H16 = начало H64: метрики H16 из того же прогноза
                for k, v in window_metrics(q[..., :16], fut[:, :16]).items():
                    res[f"{'cov' if k == 'coverage' else k}16"] = v
            n_fc += q.shape[0] * q.shape[1]
        sec_fc += time.perf_counter() - t0
        fallback_total += int(fb.sum())
        np.savez_compressed(path, **res)
        print(f"  {cell}: {len(wc)} окон, {time.perf_counter() - t0:.0f} с", flush=True)

    meta = {"date": datetime.now().strftime("%Y-%m-%d %H:%M"), "model": args.model, "cache": str(args.cache),
            "dataset_hash": ds["hash"], "protocol": "docs/protocol_v3_aligned.md v1",
            "variants": [variant_name(k, a) for k, a in VARIANTS], "w_recent": al.W_RECENT,
            "mad_eps": al.MAD_EPS, "k_clip": [al.K_MIN, al.K_MAX], "horizons": horizons,
            "device": args.device, "batch_size": args.batch_size, "checks_original_max_std": check,
            "n_forecasts_this_run": n_fc, "seconds_forecast_this_run": round(sec_fc, 1),
            "ms_per_forecast": round(1000 * sec_fc / max(n_fc, 1), 2), "fallback_windows_this_run": fallback_total,
            "seconds_total": round(time.perf_counter() - t_start, 1), "limit": args.limit,
            "env": (f"python {platform.python_version()}, numpy {np.__version__}, torch {torch.__version__}, "
                    f"transformers {transformers.__version__}, chronos-forecasting {chronos.__version__}")}
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(json.dumps({k: meta[k] for k in ("n_forecasts_this_run", "ms_per_forecast", "seconds_total")}))


if __name__ == "__main__":
    main()
