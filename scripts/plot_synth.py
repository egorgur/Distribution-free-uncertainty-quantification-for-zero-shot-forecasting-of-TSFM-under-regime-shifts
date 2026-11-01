"""Примеры синтетических рядов: по одной панели на тип изменения.

Запуск из корня репозитория:
    python -m scripts.plot_synth                          # configs/synth_v1.yaml
    python -m scripts.plot_synth --config configs/другой.yaml

На каждой панели — один и тот же исходный ряд (одинаковый шум) при силе weak / medium / strong
и серым — тот же ряд без изменения. Значения до масштабирования, в единицах σ (std шума).
Вертикальная линия — точка смены τ; заливка — длительность дрейфа или аномалии.

Результат: results/<name>/examples.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # без окна: только сохранение в файл
import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.synth import LEVELS, Cell, grid, layout, generate, series_rng

BEFORE, AFTER = 256, 384  # сколько точек показать до и после τ
COLORS = {"weak": "tab:blue", "medium": "tab:orange", "strong": "tab:red"}
TITLES = {
    "mean": "уровень процесса",
    "variance": "дисперсия шума",
    "trend": "появление тренда",
    "ar": "автокорреляция (AR)",
    "seasonality": "амплитуда сезонности",
    "frequency": "частота сезонности",
    "sensor_bias": "смещение датчика",
    "spike": "выброс",
    "freeze": "залипание датчика",
    "none": "без изменений",
}
CLASS_TITLES = {"change": "смена", "drift": "дрейф", "anomaly": "аномалия", "none": "контроль"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/synth_v1.yaml"))
    parser.add_argument("--series", type=int, default=0, help="номер исходного ряда")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    tau, _ = layout(cfg)
    cells = grid(cfg)
    panels = list(dict.fromkeys((c.cls, c.kind) for c in cells))  # уникальные (класс, тип) по порядку
    by_panel = {p: {c.level: c for c in cells if (c.cls, c.kind) == p} for p in panels}

    def series(cell: Cell) -> np.ndarray:
        # Один и тот же rng для всех сил → одинаковые параметры и шум, отличается только смена.
        x, p = generate(cell, args.series, cfg, rng=series_rng(cfg["seed"], "examples", args.series))
        return ((x - p["offset"]) / p["scale"])[tau - BEFORE : tau + AFTER]

    ncols = 3
    nrows = -(-len(panels) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.1 * nrows), sharex=True)
    t = np.arange(-BEFORE, AFTER)
    baseline = series(Cell("none", "none", "-", float("nan")))
    for ax, (cls, kind) in zip(axes.flat, panels):
        if cls != "none":
            ax.plot(t, baseline, color="0.75", lw=0.8, label="без изменения")
        for level in LEVELS if cls != "none" else ["-"]:
            cell = by_panel[(cls, kind)][level]
            label = f"{level}: {cell.value:g}" if cls != "none" else None
            ax.plot(t, series(cell), color=COLORS.get(level, "k"), lw=0.8, label=label)
        ax.axvline(0, color="k", lw=0.8, ls="--")
        if cls == "drift":
            ax.axvspan(0, cfg["drift"]["duration"], color="0.9", zorder=0)
        if cls == "anomaly" and kind == "freeze":
            ax.axvspan(0, max(cfg["anomaly"]["freeze"]), color="0.9", zorder=0)
        ax.set_title(f"{CLASS_TITLES[cls]}: {TITLES[kind]}", fontsize=10)
        if cls != "none":
            ax.legend(fontsize=7, loc="upper left", ncol=2)
    for ax in axes.flat[len(panels) :]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("t − τ")
    fig.suptitle(
        f"{cfg['name']}: примеры рядов (ряд {args.series}, в единицах σ; пунктир — точка смены τ)", fontsize=12
    )
    fig.tight_layout()

    out = Path("results") / cfg["name"] / "examples.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"Сохранено: {out}")


if __name__ == "__main__":
    main()
