"""Сверка двух кешей прогнозов: одни ли данные и одни ли прогнозы на общих окнах.

Нужна, когда кеш посчитан в другом месте (Colab, GPU) и его надо положить рядом с локальным:
    python -m scripts.compare_caches results/cache/synth_v1 ~/Downloads/synth_v1 --model amazon/chronos-2
    python -m scripts.compare_caches results/cache/synth_v1_n1 ~/Downloads/synth_v1 --model amazon/chronos-2

1. Ряды: у общих пар (ячейка, номер ряда) ряды должны совпадать точно. Номер ряда внутри ячейки
   задаёт seed генератора, поэтому первый ряд ячейки в пробном кеше (--n-per-cell 1) — тот же, что
   в полном. Если ряды не совпали, второй кеш построен по другим данным и смешивать их нельзя.
2. Прогнозы: на общих окнах (ячейка, ряд, t) — разница квантилей всех вариантов контекста в единицах
   std основного контекста. Пробный кеш synth_v1_n1 посчитан на CPU, поэтому сверка с ним показывает,
   совпадает ли прогноз на GPU с прогнозом на CPU. Порог 1e-3 std: разница из-за порядка операций
   в float32 на порядки меньше, а реальное расхождение (другая модель, bf16) — больше.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.cache import load_series, load_windows, model_dir

TOL = 1e-3  # допустимая разница прогнозов, std контекста


def cell_forecasts(root: Path, model_id: str, cell: str) -> dict[str, np.ndarray] | None:
    """{"row": …, "q64": …, "q16": …} одной ячейки; q16 достраивается из q64, если он его начало."""
    f = model_dir(root, model_id) / f"{cell}.npz"
    if not f.exists():
        return None
    meta = json.loads((model_dir(root, model_id) / "meta.json").read_text())
    with np.load(f) as z:
        out = {k: z[k] for k in z.files}
    if meta["short_horizon_is_prefix"]:
        h_max = max(int(k[1:]) for k in out if k.startswith("q"))
        for h in yaml.safe_load((root / "config.yaml").read_text())["windows"]["horizons"]:
            out.setdefault(f"q{h}", out[f"q{h_max}"][..., :h])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("a", type=Path, help="первый кеш (например, локальный)")
    parser.add_argument("b", type=Path, help="второй кеш (например, скачанный из Colab)")
    parser.add_argument("--model", default="amazon/chronos-2")
    args = parser.parse_args()

    ds = [json.loads((r / "dataset.json").read_text()) for r in (args.a, args.b)]
    print(f"хеш конфига: {ds[0]['hash']} и {ds[1]['hash']}; варианты совпадают: {ds[0]['variants'] == ds[1]['variants']}")
    if ds[0]["variants"] != ds[1]["variants"]:
        raise SystemExit("разные варианты контекста — прогнозы несравнимы")
    L = yaml.safe_load((args.a / "config.yaml").read_text())["windows"]["context"]

    # 1. Ряды.
    (xa, sa), (xb, sb) = load_series(args.a), load_series(args.b)
    pairs = sa.merge(sb, on=["cell", "series"], suffixes=("_a", "_b"))
    diff = max(float(np.abs(xa[i] - xb[j]).max()) for i, j in zip(pairs.series_idx_a, pairs.series_idx_b))
    print(f"ряды: общих {len(pairs)}, макс. разница {diff:.1e}")
    if diff != 0:
        raise SystemExit("ряды различаются: кеши построены по разным данным")

    # 2. Прогнозы на общих окнах.
    wa, wb = load_windows(args.a), load_windows(args.b)
    worst, n_win, rows = 0.0, 0, []
    for cell in sorted(set(wa.cell) & set(wb.cell)):
        fa, fb = cell_forecasts(args.a, args.model, cell), cell_forecasts(args.b, args.model, cell)
        if fa is None or fb is None:
            continue
        ka = wa.set_index("row").loc[fa["row"], ["series", "t", "series_idx"]].reset_index().reset_index(names="ia")
        kb = wb.set_index("row").loc[fb["row"], ["series", "t"]].reset_index().reset_index(names="ib")
        m = ka.merge(kb, on=["series", "t"])
        if m.empty:
            continue
        t, s = m.t.to_numpy(), m.series_idx.to_numpy()
        scale = xa[s[:, None], t[:, None] - L + 1 + np.arange(L)].std(axis=1)
        for key in sorted(k for k in fa if k.startswith("q") and k in fb):
            d = np.abs(fa[key][m.ia.to_numpy()] - fb[key][m.ib.to_numpy()]).max(axis=(1, 2, 3)) / scale
            rows.append({"cell": cell, "key": key, "windows": len(m), "max": d.max(), "median": np.median(d)})
            worst = max(worst, float(d.max()))
        n_win += len(m)
    if not rows:
        print(f"прогнозов {args.model}, посчитанных в обоих кешах, нет — сверены только ряды")
        return
    print(pd.DataFrame(rows).to_string(index=False, float_format="%.1e"))
    status = "OK" if worst < TOL else "FAIL"
    print(f"[{status}] прогнозы на {n_win} общих окнах: макс. разница {worst:.1e} std контекста (порог {TOL:g})")
    if status == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
