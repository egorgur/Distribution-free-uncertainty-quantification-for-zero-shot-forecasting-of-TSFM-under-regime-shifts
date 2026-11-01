"""S5: сдвиг эмбеддингов энкодера в конце контекста — расчёт для всех окон кеша.

Запуск из корня репозитория (нужны данные кеша, см. scripts/cache_forecasts.py):
    python -m scripts.cache_s5                                          # synth_v1, chronos-bolt-small
    python -m scripts.cache_s5 --cache results/cache/synth_v1_seed1

Для каждого окна берётся основной контекст (L = 512, 32 патча по 16 точек), модель считает
эмбеддинги патчей, по ним — S5 (src/signals.py, embedding_shift). Варианты выбраны до расчёта
(журнал решений CLAUDE.md, 2026-11-01):
    слой — выход последнего слоя энкодера (основной) и вход энкодера, до трансформера (абляция);
    m    — последние 4 патча = 64 точки (основной, как KS в S3); 2 и 8 патчей (абляции);
    мера — (а) косинусное расстояние, (б) диагональное расстояние Махаланобиса.
Ключи: s5_<слой>_m<m>_<a|b>; основной S5а = s5_final_m4_a, S5б = s5_final_m4_b.

Модель вызывается заново, потому что кеш хранит только квантили. В работе S5 берётся из того же
прохода энкодера, что и основной прогноз, то есть дополнительных прогонов не требует.

Результат: results/cache/<name>/<модель>/signals/s5.npz и s5.json (в git не попадают, как и весь
кеш). Читает scripts/stage4_signals.py.
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

from src import signals
from src.cache import load_dataset_meta, load_series, load_windows, signals_path
from src.models import load_forecaster

LAYERS = ("final", "input")  # первый — основной
M_PATCHES = (4, 2, 8)  # первый — основной
SEED = 0


def s5_keys() -> list[str]:
    return [f"s5_{layer}_m{m}_{k}" for layer in LAYERS for m in M_PATCHES for k in "ab"]


def contexts(x: np.ndarray, series_idx: np.ndarray, t: np.ndarray, L: int) -> np.ndarray:
    """Основные контексты окон: (окна, L), последняя точка — t."""
    return x[series_idx[:, None], t[:, None] - L + 1 + np.arange(L)]


def s5_values(model, ctx: np.ndarray, batch_size: int) -> dict[str, np.ndarray]:
    inputs, finals = model.encode_patches(ctx, batch_size=batch_size)
    out = {}
    for layer, tokens in (("final", finals), ("input", inputs)):
        for m in M_PATCHES:
            out[f"s5_{layer}_m{m}_a"], out[f"s5_{layer}_m{m}_b"] = signals.embedding_shift(tokens, m)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path, default=Path("results/cache/synth_v1"))
    parser.add_argument("--model", default="amazon/chronos-bolt-small")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--chunk", type=int, default=2048, help="окон за раз (память ~ chunk × 32 × 512 × 8 байт × 2)")
    parser.add_argument("--verify", type=int, default=8, help="сколько окон пересчитать поодиночке для сверки")
    args = parser.parse_args()
    t0 = time.perf_counter()

    ds = load_dataset_meta(args.cache)
    L = yaml.safe_load((args.cache / "config.yaml").read_text())["windows"]["context"]
    w = load_windows(args.cache)
    assert (w.row.to_numpy() == np.arange(len(w))).all()
    x, _ = load_series(args.cache)
    t, s = w.t.to_numpy(), w.series_idx.to_numpy()
    model = load_forecaster(args.model, device=args.device)
    torch.manual_seed(SEED)

    n = len(w)
    out = {k: np.empty(n) for k in s5_keys()}
    print(f"S5: {n} окон, L = {L}, {args.model}")
    for a in range(0, n, args.chunk):
        sl = slice(a, a + args.chunk)
        for k, v in s5_values(model, contexts(x, s[sl], t[sl], L), args.batch_size).items():
            out[k][sl] = v
        done = min(a + args.chunk, n)
        el = time.perf_counter() - t0
        print(f"  {done}/{n} окон, {el:.0f} с; осталось ~{el / done * (n - done) / 60:.1f} мин")
    bad = [k for k, v in out.items() if not np.isfinite(v).all()]
    if bad:
        raise RuntimeError(f"нечисловые значения S5: {bad}")

    # Сверка: окно, посчитанное в одиночку, даёт тот же S5, что и в общем батче.
    idx = np.random.default_rng(SEED).choice(n, min(args.verify, n), replace=False)
    worst = 0.0
    for i in idx:
        one = s5_values(model, contexts(x, s[[i]], t[[i]], L), batch_size=1)
        for k, v in one.items():
            worst = max(worst, abs(float(v[0]) - out[k][i]) / np.median(out[k]))
    status = "OK" if worst < 1e-3 else "FAIL"
    print(f"  [{status}] пересчёт {len(idx)} окон поодиночке: макс. разница {worst:.1e} медианы сигнала")
    if status == "FAIL":
        raise RuntimeError("S5 зависит от состава батча")

    path = signals_path(args.cache, args.model, "s5")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, row=w.row.to_numpy(), **out)
    meta = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"), "model": args.model, "dataset_hash": ds["hash"],
        "L": L, "layers": LAYERS, "m_patches": M_PATCHES, "keys": s5_keys(), "seconds": round(time.perf_counter() - t0),
        "device": args.device, "verify_max_rel_diff": worst,
        "env": (f"python {platform.python_version()}, torch {torch.__version__}, "
                f"transformers {transformers.__version__}, chronos-forecasting {chronos.__version__}"),
    }
    path.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"Готово за {time.perf_counter() - t0:.0f} с → {path}")


if __name__ == "__main__":
    main()
