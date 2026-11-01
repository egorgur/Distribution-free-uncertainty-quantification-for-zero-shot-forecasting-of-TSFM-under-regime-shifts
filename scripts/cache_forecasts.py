"""Кеш прогнозов на синтетике: ряды → окна → квантили всех вариантов контекста.

Запуск из корня репозитория:
    python -m scripts.cache_forecasts                                      # все модели из конфига
    python -m scripts.cache_forecasts --model amazon/chronos-bolt-small
    python -m scripts.cache_forecasts --model amazon/chronos-2 --device cuda   # Colab с GPU
    python -m scripts.cache_forecasts --n-per-cell 1                       # пробный прогон, отдельная папка
    python -m scripts.cache_forecasts --seed 1 --model amazon/chronos-bolt-small  # та же сетка на новых рядах

Что делает:
1. По конфигу генерирует ряды (src/synth.py) и таблицу окон (src/windows.py).
   Данные от модели не зависят и строятся один раз.
2. Для каждой модели и ячейки сетки считает квантильные прогнозы всех вариантов
   контекста окна (основной, S4, S7) и сохраняет в отдельный файл ячейки.
3. Посчитанные ячейки пропускает, поэтому прерванный прогон можно продолжить той же командой.
4. В конце пересчитывает несколько случайных окон заново и сверяет с кешем.

Результат: results/cache/<name>/ (хранится в git, логи — нет; чтение — src/cache.py).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from datetime import datetime
from pathlib import Path

import chronos
import numpy as np
import pandas as pd
import torch
import transformers
import yaml

from src.cache import model_dir
from src.models import load_forecaster
from src.synth import grid, layout, generate
from src.windows import noise_rng, variant_names, window_contexts, window_ends


def dataset_hash(cfg: dict) -> str:
    """Хеш всего, от чего зависят ряды и окна (список моделей не входит)."""
    data = {k: v for k, v in cfg.items() if k != "models"}
    return hashlib.sha1(json.dumps(data, sort_keys=True).encode()).hexdigest()[:12]


def build_dataset(cfg: dict, root: Path) -> None:
    tau, n = layout(cfg)
    t_all, d_all = window_ends(tau, cfg)
    series, s_rows, w_rows = [], [], []
    for cell in grid(cfg):
        for i in range(cfg["n_per_cell"]):
            x, params = generate(cell, i, cfg)
            s_idx = len(series)
            series.append(x)
            s_rows.append({"series_idx": s_idx, "cell": cell.cell_id, "series": i, **params})
            for t, d in zip(t_all, d_all):
                w_rows.append({
                    "row": len(w_rows), "cell": cell.cell_id, "cls": cell.cls, "kind": cell.kind,
                    "level": cell.level, "value": cell.value, "series": i, "series_idx": s_idx,
                    "t": int(t), "pos": int(d),  # pos = t − τ, есть и у рядов без смены
                    "d": np.nan if cell.cls == "none" else int(d),
                })
    root.mkdir(parents=True, exist_ok=True)
    np.savez(root / "series.npz", x=np.stack(series))
    pd.DataFrame(s_rows).to_csv(root / "series.csv", index=False)
    pd.DataFrame(w_rows).to_csv(root / "windows.csv", index=False)
    (root / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    meta = {
        "hash": dataset_hash(cfg), "name": cfg["name"], "tau": tau, "length": n,
        "horizons": cfg["windows"]["horizons"], "variants": variant_names(cfg),
        "n_cells": len(grid(cfg)), "n_series": len(series), "n_windows": len(w_rows),
        "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    (root / "dataset.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"Данные: {len(series)} рядов длины {n}, {len(w_rows)} окон, τ = {tau} → {root}/")


def contexts_for(rows: pd.DataFrame, x_all: np.ndarray, cfg: dict) -> list[np.ndarray]:
    """Контексты всех вариантов для строк таблицы окон, подряд: окно 0 (все варианты), окно 1, …"""
    out = []
    for r in rows.itertuples():
        rng = noise_rng(cfg["seed"], r.cell, r.series, r.t)
        out += window_contexts(x_all[r.series_idx], r.t, cfg, rng)
    return out


def predict_rows(model, rows: pd.DataFrame, x_all: np.ndarray, cfg: dict, batch_size: int) -> dict[str, np.ndarray]:
    """{"q64": (окна, варианты, 9, 64), "q16": …}; q16 не хранится, если он равен началу q64."""
    n_var = len(variant_names(cfg))
    ctxs = contexts_for(rows, x_all, cfg)
    horizons = sorted(cfg["windows"]["horizons"], reverse=True)
    out = {}
    for h in horizons:
        if h != horizons[0] and model.short_horizon_is_prefix:
            continue
        q = model.predict_quantiles(ctxs, horizon=h, batch_size=batch_size)
        out[f"q{h}"] = q.reshape(len(rows), n_var, *q.shape[1:]).astype(np.float32)
    return out


def calls_per_window(model, cfg: dict) -> int:
    return 1 if model.short_horizon_is_prefix else len(cfg["windows"]["horizons"])


def verify(model, mdir: Path, windows: pd.DataFrame, x_all: np.ndarray, cfg: dict, n: int, seed: int) -> None:
    """Пересчитывает n случайных окон из уже посчитанных ячеек и сверяет с кешем."""
    done = {f.stem for f in mdir.glob("*.npz")}
    pool = windows[windows.cell.isin(done)]
    if pool.empty or n == 0:
        return
    sample = pool.sample(min(n, len(pool)), random_state=seed)
    fresh = predict_rows(model, sample, x_all, cfg, batch_size=256)
    worst = 0.0
    for j, r in enumerate(sample.itertuples()):
        with np.load(mdir / f"{r.cell}.npz") as z:
            k = int(np.flatnonzero(z["row"] == r.row)[0])
            scale = np.std(x_all[r.series_idx][r.t - cfg["windows"]["context"] + 1 : r.t + 1])
            for key, q in fresh.items():
                worst = max(worst, float(np.abs(z[key][k] - q[j]).max() / scale))
    status = "OK" if worst < 1e-4 else "FAIL"
    print(f"  [{status}] пересчёт {len(sample)} случайных окон совпадает с кешем: макс. разница {worst:.1e} std контекста")
    if status == "FAIL":
        raise RuntimeError("кеш не совпадает с пересчётом")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/synth_v1.yaml"))
    parser.add_argument("--model", nargs="+", help="модели (по умолчанию — список из конфига)")
    parser.add_argument("--n-per-cell", type=int, help="переопределить n_per_cell (пробный прогон)")
    parser.add_argument("--seed", type=int, help="переопределить seed: та же сетка на новых рядах, отдельная папка")
    parser.add_argument("--cells", nargs="+", help="считать прогнозы только для этих ячеек")
    parser.add_argument("--out", type=Path, help="папка кеша (по умолчанию results/cache/<name>)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--verify", type=int, default=8, help="сколько окон пересчитать для сверки")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    default_name = cfg["name"]
    if args.n_per_cell is not None:
        cfg["n_per_cell"] = args.n_per_cell
        default_name += f"_n{args.n_per_cell}"
    if args.seed is not None:
        cfg["seed"] = args.seed  # меняет ряды и шум S7; хеш данных тоже меняется
        default_name += f"_seed{args.seed}"
    root = args.out or Path("results/cache") / default_name
    torch.manual_seed(cfg["seed"])

    ds_file = root / "dataset.json"
    if ds_file.exists():
        old = json.loads(ds_file.read_text())["hash"]
        if old != dataset_hash(cfg):
            raise SystemExit(
                f"{root}/ построен по другому конфигу (хеш {old}, сейчас {dataset_hash(cfg)}). "
                "Удалите папку или укажите другую через --out."
            )
        print(f"Данные уже есть: {root}/")
    else:
        build_dataset(cfg, root)

    windows = pd.read_csv(root / "windows.csv")
    x_all = np.load(root / "series.npz")["x"]
    n_var = len(variant_names(cfg))
    all_cells = [c.cell_id for c in grid(cfg)]
    cells = args.cells or all_cells
    unknown = set(cells) - set(all_cells)
    if unknown:
        raise SystemExit(f"неизвестные ячейки: {sorted(unknown)}")

    for model_id in args.model or cfg["models"]:
        print(f"\n{model_id}")
        model = load_forecaster(model_id, device=args.device)
        mdir = model_dir(root, model_id)
        mdir.mkdir(exist_ok=True)
        for f in mdir.glob("*.tmp.npz"):  # остатки прерванного прогона
            f.unlink()
        meta_file = mdir / "meta.json"
        meta = json.loads(meta_file.read_text()) if meta_file.exists() else {"cells": {}}
        meta.update({
            "model": model_id, "short_horizon_is_prefix": model.short_horizon_is_prefix,
            "device": args.device, "torch_threads": torch.get_num_threads(),
            "env": (f"python {platform.python_version()}, torch {torch.__version__}, "
                    f"transformers {transformers.__version__}, chronos-forecasting {chronos.__version__}"),
        })

        todo = [c for c in cells if not (mdir / f"{c}.npz").exists()]
        per_window = n_var * calls_per_window(model, cfg)
        total = int(windows.cell.isin(todo).sum()) * per_window
        print(f"  ячеек к расчёту: {len(todo)} из {len(cells)}; прогнозов: {total}")
        spent, made = 0.0, 0
        for c in todo:
            rows = windows[windows.cell == c]
            t0 = time.perf_counter()
            q = predict_rows(model, rows, x_all, cfg, args.batch_size)
            sec = time.perf_counter() - t0
            if not all(np.isfinite(a).all() for a in q.values()):
                raise RuntimeError(f"{c}: в прогнозе есть NaN")
            tmp = mdir / f"{c}.tmp.npz"
            np.savez(tmp, row=rows.row.to_numpy(), **q)
            os.replace(tmp, mdir / f"{c}.npz")  # файл ячейки появляется только целиком
            n_fc = len(rows) * per_window
            spent, made = spent + sec, made + n_fc
            meta["cells"][c] = {"seconds": round(sec, 1), "forecasts": n_fc}
            meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
            eta = spent / made * (total - made)
            print(f"  {c}: {sec:.0f} с, {sec / n_fc * 1e3:.1f} мс на прогноз; осталось ~{eta / 60:.0f} мин")

        verify(model, mdir, windows, x_all, cfg, args.verify, cfg["seed"])

        # Оценка полной сетки по измеренной скорости этой модели.
        timed = meta["cells"].values()
        ms = sum(c["seconds"] for c in timed) / max(1, sum(c["forecasts"] for c in timed)) * 1e3
        n_windows_full = len(all_cells) * yaml.safe_load(args.config.read_text())["n_per_cell"] * len(
            window_ends(0, cfg)[0])
        full = n_windows_full * per_window
        meta["ms_per_forecast"] = round(ms, 2)
        meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        print(f"  скорость {ms:.1f} мс на прогноз ({args.device}); "
              f"полная сетка конфига: {full} прогнозов ≈ {full * ms / 1e3 / 3600:.1f} ч")
        del model


if __name__ == "__main__":
    main()
