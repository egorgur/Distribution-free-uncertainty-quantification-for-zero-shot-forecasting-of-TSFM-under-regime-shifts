"""Сравнение моделей Chronos на синтетических рядах без точки смены.

Запуск из корня репозитория:
    python -m scripts.compare_models                       # все модели из MODELS
    python -m scripts.compare_models --models amazon/chronos-bolt-small amazon/chronos-2
    python -m scripts.compare_models --n-per-family 5      # быстрый пробный прогон

Что сравнивается. Все модели получают одни и те же окна, поэтому сравнение парное.
- Точность: MASE медианы и нормированный CRPS при H = 16 и 64, L = 512.
- Калибровка: покрытие интервала [q0.1, q0.9] (номинал 0.8) и PCE по 9 квантилям,
  в среднем по всем окнам (глобальная калибровка, как у Adler et al., ICLR 2026).
- Скорость и размер: мс на прогноз на CPU, число параметров и слоёв.
- Фон S4/S7: насколько меняется медиана прогноза при другой длине контекста и
  малых возмущениях на рядах БЕЗ смены. Это не критерий выбора, а уровень шума,
  над которым S4/S7 должны подниматься при смене режима.

Ряды со сменой режима здесь сознательно не используются: выбирать модель по тому,
как на ней выглядят гипотезы H1–H3, означало бы подгонку под результат.

Результаты: results/model_comparison/
    windows.csv     — метрики по каждому окну (модель × окно × горизонт);
    background.csv  — фон S4/S7 по каждому окну;
    summary.md      — сводные таблицы с 95% бутстреп-интервалами;
    meta.json       — версии, параметры запуска, сведения о моделях.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import time
from datetime import datetime
from pathlib import Path

import chronos
import numpy as np
import pandas as pd
import torch
import transformers

from src.metrics import interval_coverage, mase, pce, quantile_hits, scaled_crps, seasonal_naive_scale
from src.models import QUANTILE_LEVELS, load_forecaster

MODELS = [
    "amazon/chronos-bolt-tiny",
    "amazon/chronos-bolt-mini",
    "amazon/chronos-bolt-small",
    "amazon/chronos-bolt-base",
    "autogluon/chronos-2-small",
    "amazon/chronos-2",
]
REFERENCE = "amazon/chronos-bolt-small"  # текущая модель по умолчанию; с ней считаются парные разности

SEED = 0
L = 512  # основной контекст
HORIZONS = (16, 64)
H_MAX = max(HORIZONS)
FAMILIES = ("sine", "ar1", "two_seasons", "trend_sine", "random_walk")
S4_LENGTHS = (256, 128, 64)  # плюс основной L = 512
S7_TAIL_K = (3, 9)  # длина L − k: начало окна сдвигается, конец (момент t) на месте
S7_NOISE_ALPHA = 0.1  # шум N(0, (α · std разностей контекста)²)
S7_NOISE_DRAWS = 2
N_BOOT = 2000
OUT_DIR = Path("results/model_comparison")

I_LO, I_MED, I_HI = (QUANTILE_LEVELS.index(q) for q in (0.1, 0.5, 0.9))
N_VARIANTS = 1 + len(S4_LENGTHS) + len(S7_TAIL_K) + S7_NOISE_DRAWS


# ---------- данные ----------


def make_series(family: str, rng: np.random.Generator) -> np.ndarray:
    """Один ряд длины L + H_MAX без точки смены; масштаб и уровень случайные."""
    n = L + H_MAX
    t = np.arange(n)
    if family == "sine":  # одна сезонность + шум
        period = rng.choice([12, 24, 48])
        x = np.sin(2 * np.pi * t / period + rng.uniform(0, 2 * np.pi)) + rng.normal(0, rng.uniform(0.1, 0.5), n)
    elif family == "ar1":  # AR(1) с разной «памятью»
        phi, burn = rng.uniform(0.5, 0.95), 200
        e = rng.normal(0, 1, n + burn)
        x = np.zeros(n + burn)
        for i in range(1, n + burn):
            x[i] = phi * x[i - 1] + e[i]
        x = x[burn:]
    elif family == "two_seasons":  # периоды 24 и 168: короткий контекст видит только первый
        x = (
            np.sin(2 * np.pi * t / 24 + rng.uniform(0, 2 * np.pi))
            + 0.5 * np.sin(2 * np.pi * t / 168 + rng.uniform(0, 2 * np.pi))
            + rng.normal(0, 0.3, n)
        )
    elif family == "trend_sine":  # линейный тренд + сезонность
        slope = rng.uniform(-5, 5) / n
        x = slope * t + np.sin(2 * np.pi * t / 24 + rng.uniform(0, 2 * np.pi)) + rng.normal(0, 0.3, n)
    elif family == "random_walk":  # наивный прогноз здесь оптимален: модель может лишь сравняться с ним
        x = np.cumsum(rng.normal(0, 1, n))
    else:
        raise ValueError(family)
    scale = 10 ** rng.uniform(-2, 2)  # модели должны быть нечувствительны к масштабу
    offset = scale * rng.uniform(-20, 20)  # и к уровню
    return offset + scale * x


def build_windows(n_per_family: int, rng: np.random.Generator) -> list[dict]:
    """Окна «контекст L + будущее H_MAX» и заранее сгенерированный шум для S7 (один на все модели)."""
    windows = []
    for family in FAMILIES:
        for i in range(n_per_family):
            x = make_series(family, rng)
            ctx, fut = x[:L], x[L : L + H_MAX]
            sigma = np.std(np.diff(ctx))
            noisy = [ctx + rng.normal(0, S7_NOISE_ALPHA * sigma, L) for _ in range(S7_NOISE_DRAWS)]
            windows.append(
                {"family": family, "series": i, "context": ctx, "future": fut, "noisy": noisy,
                 "scale": seasonal_naive_scale(ctx, m=1)}
            )
    return windows


def variants(w: dict) -> list[np.ndarray]:
    """Варианты контекста одного окна: [L=512, S4-длины, S7-хвосты, S7-шум]. Конец у всех — момент t."""
    ctx = w["context"]
    return [ctx] + [ctx[-n:] for n in S4_LENGTHS] + [ctx[k:] for k in S7_TAIL_K] + w["noisy"]


def naive_mase(windows: list[dict]) -> dict[int, np.ndarray]:
    """MASE наивного прогноза «последнее значение» — точка отсчёта для моделей."""
    return {
        h: np.array([mase(w["future"][:h], np.full(h, w["context"][-1]), w["scale"]) for w in windows])
        for h in HORIZONS
    }


# ---------- прогон одной модели ----------


def evaluate(model, windows: list[dict], naive: dict[int, np.ndarray]) -> tuple[list[dict], list[dict], float]:
    """Метрики по окнам и фон S4/S7. Возвращает (строки метрик, строки фона, секунды)."""
    t0 = time.perf_counter()
    all_ctx = [c for w in windows for c in variants(w)]
    q_max = model.predict_quantiles(all_ctx, horizon=H_MAX)
    q_max = q_max.reshape(len(windows), N_VARIANTS, len(QUANTILE_LEVELS), H_MAX)
    q_by_h = {H_MAX: q_max[:, 0]}
    for h in HORIZONS:
        if h != H_MAX:  # у Chronos-2 прогноз на 16 шагов — отдельный проход, не начало прогноза на 64
            q_by_h[h] = model.predict_quantiles([w["context"] for w in windows], horizon=h)
    seconds = time.perf_counter() - t0

    rows, background = [], []
    n_s4 = 1 + len(S4_LENGTHS)
    for j, w in enumerate(windows):
        key = {"model": model.model_id, "window": j, "family": w["family"], "series": w["series"]}
        for h in HORIZONS:
            q, fut = q_by_h[h][j], w["future"][:h]
            hits = quantile_hits(fut, q)
            rows.append(
                {**key, "horizon": h,
                 "mase": mase(fut, q[I_MED], w["scale"]),
                 "crps": scaled_crps(fut, q, QUANTILE_LEVELS, w["scale"]),
                 "naive_mase": naive[h][j],
                 "coverage": interval_coverage(fut, q[I_LO], q[I_HI]),
                 **{f"hit_{lv}": v for lv, v in zip(QUANTILE_LEVELS, hits)}}
            )
        # Медианы всех вариантов в единицах масштаба наивного прогноза (как MASE).
        med = q_max[j, :, I_MED, :] / w["scale"]
        background.append(
            {**key,
             "s4_spread": float(med[:n_s4].std(axis=0).mean()),
             "s7_tail": float(np.abs(med[n_s4 : n_s4 + len(S7_TAIL_K)] - med[0]).mean()),
             "s7_noise": float(np.abs(med[-S7_NOISE_DRAWS:] - med[0]).mean())}
        )
    return rows, background, seconds


def ms_per_forecast(model, windows: list[dict], n: int = 64, repeats: int = 3) -> float:
    """Медиана по повторам: батч из n окон, L = 512, H = 64 (батч всегда из n, даже если окон меньше)."""
    ctxs = [windows[i % len(windows)]["context"] for i in range(n)]
    model.predict_quantiles(ctxs[:8], horizon=H_MAX)  # прогрев
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        model.predict_quantiles(ctxs, horizon=H_MAX)
        times.append(time.perf_counter() - t0)
    return float(np.median(times)) / len(ctxs) * 1e3


def model_info(model) -> dict:
    net = model.pipeline.model
    return {
        "model": model.model_id,
        "params_m": sum(p.numel() for p in net.parameters()) / 1e6,
        "weights_mb": sum(p.numel() * p.element_size() for p in net.parameters()) / 2**20,
        "enc_layers": net.config.num_layers,
        "dec_layers": getattr(net.config, "num_decoder_layers", 0) or 0,
        "d_model": net.config.d_model,
        "max_horizon": model.max_horizon,
    }


# ---------- сводка ----------


def boot_mean(values: np.ndarray, idx: np.ndarray) -> tuple[float, float, float]:
    """Среднее и 95% бутстреп-интервал; idx — общие для всех моделей индексы (парный бутстреп)."""
    lo, hi = np.percentile(values[idx].mean(axis=1), [2.5, 97.5])
    return float(values.mean()), float(lo), float(hi)


def boot_gmean(ratios: np.ndarray, idx: np.ndarray) -> tuple[float, float, float]:
    """Геометрическое среднее отношений и 95% бутстреп-интервал.

    Для отношений ошибок (модель / эталон) это стандартный способ усреднения:
    0.5 и 2.0 гасят друг друга, а большие ошибки одного семейства не доминируют.
    """
    logs = np.log(ratios)
    lo, hi = np.exp(np.percentile(logs[idx].mean(axis=1), [2.5, 97.5]))
    return float(np.exp(logs.mean())), float(lo), float(hi)


def boot_median(values: np.ndarray, idx: np.ndarray) -> tuple[float, float, float]:
    lo, hi = np.percentile(np.median(values[idx], axis=1), [2.5, 97.5])
    return float(np.median(values)), float(lo), float(hi)


def fmt(mean: float, lo: float, hi: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} [{lo:.{digits}f}; {hi:.{digits}f}]"


def md_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows])


def short(model_id: str) -> str:
    return model_id.split("/")[-1]


def summarize(df: pd.DataFrame, bg: pd.DataFrame, infos: list[dict], meta: dict) -> str:
    models = [i["model"] for i in infos]
    n = meta["n_windows"]
    idx = np.random.default_rng(SEED + 1).integers(0, n, size=(N_BOOT, n))
    levels = np.array(QUANTILE_LEVELS)
    hit_cols = [f"hit_{lv}" for lv in QUANTILE_LEVELS]
    has_ref = REFERENCE in models

    out = [
        "# Сравнение моделей Chronos на рядах без точки смены",
        "",
        f"Дата: {meta['date']}. Скрипт: `python -m scripts.compare_models` (seed {SEED}).",
        f"Окружение: {meta['env']}. CPU: {meta['cpu']}, потоков torch: {meta['torch_threads']}.",
        f"Окна: {n} = {len(FAMILIES)} семейств × {meta['n_per_family']} рядов; контекст L = {L}; "
        f"горизонты {', '.join(map(str, HORIZONS))}. Все модели видят одни и те же окна.",
        "В квадратных скобках — 95% бутстреп-интервал по окнам "
        f"({N_BOOT} повторов, одинаковые выборки окон для всех моделей).",
        "",
        "## Модели",
        "",
    ]
    out.append(md_table(
        ["Модель", "Параметры, M", "Слои энкодер / декодер", "d_model", "Веса, МБ",
         "мс на прогноз (CPU, батч 64, L 512, H 64)", "Время всего прогона, с", "Макс. горизонт за проход"],
        [[short(i["model"]), f"{i['params_m']:.1f}", f"{i['enc_layers']} / {i['dec_layers']}", str(i["d_model"]),
          f"{i['weights_mb']:.0f}", f"{i['ms_per_forecast']:.1f}", f"{i['eval_seconds']:.0f}", str(i["max_horizon"])]
         for i in infos],
    ))

    out += [
        "",
        "Точность — относительно наивного прогноза «последнее значение», геометрическое среднее по окнам: "
        "0.30 = ошибка в среднем в 0.3 раза от наивной; 1 = не лучше наивного. Отношение к "
        f"{short(REFERENCE)} — так же, но в паре с ней на тех же окнах: 0.95 = в среднем на 5% точнее. "
        "Простое среднее MASE не используется: его определяет random_walk, где MASE при H = 64 порядка 8.",
    ]
    for h in HORIZONS:
        d = df[df.horizon == h]

        def col(m: str, c: str) -> np.ndarray:
            return d[d.model == m].sort_values("window")[c].to_numpy()

        rows = []
        for m in models:
            mase_m, naive_m = col(m, "mase"), col(m, "naive_mase")
            hits = d[d.model == m].sort_values("window")[hit_cols].to_numpy()
            pce_boot = np.abs(hits[idx].mean(axis=1) - levels).mean(axis=1)
            row = [short(m), fmt(*boot_gmean(mase_m / naive_m, idx)), fmt(*boot_gmean(col(m, "crps") / naive_m, idx))]
            if has_ref:
                ratio = mase_m / col(REFERENCE, "mase")
                row += ["—", "—"] if m == REFERENCE else [fmt(*boot_gmean(ratio, idx)), f"{(ratio < 1).mean():.0%}"]
            row += [fmt(*boot_mean(col(m, "coverage"), idx), digits=2),
                    fmt(pce(hits.mean(axis=0), QUANTILE_LEVELS), *np.percentile(pce_boot, [2.5, 97.5]))]
            rows.append(row)

        header = ["Модель", "MASE / наивный ↓", "CRPS / наивный ↓"]
        if has_ref:
            header += [f"MASE / {short(REFERENCE)} ↓", f"Окон точнее {short(REFERENCE)}"]
        header += ["Покрытие 80%-интервала", "PCE ↓"]
        out += ["", f"## Точность и калибровка, H = {h}", "", md_table(header, rows)]

    h = H_MAX
    d = df[df.horizon == h].assign(rel=lambda x: np.log(x.mase / x.naive_mase))
    by_fam = np.exp(d.groupby(["model", "family"]).rel.mean())
    naive_fam = d[d.model == models[0]].groupby("family").naive_mase.mean()
    rows = [[short(m)] + [f"{by_fam[(m, f)]:.3f}" for f in FAMILIES] for m in models]
    rows.append(["MASE самого наивного прогноза"] + [f"{naive_fam[f]:.2f}" for f in FAMILIES])
    out += ["", f"## MASE / наивный по семействам рядов, H = {h}", "",
            "Геометрическое среднее по окнам семейства; последняя строка — абсолютный MASE наивного прогноза.", "",
            md_table(["Модель", *FAMILIES], rows)]

    bg_cols = ("s4_spread", "s7_tail", "s7_noise")
    rows = []
    for m in models:
        b = bg[bg.model == m].sort_values("window")
        rows.append([short(m)] + [fmt(*boot_median(b[c].to_numpy(), idx)) for c in bg_cols])
    s4_fam = bg.groupby(["model", "family"]).s4_spread.median()
    rows_fam = [[short(m)] + [f"{s4_fam[(m, f)]:.3f}" for f in FAMILIES] for m in models]
    out += [
        "",
        f"## Фон S4/S7 на рядах без смены, H = {h}",
        "",
        "Насколько меняется медиана прогноза, в единицах масштаба наивного прогноза (как MASE); "
        "медиана по окнам. "
        f"S4 — std медиан по L ∈ {{{', '.join(map(str, sorted(S4_LENGTHS + (L,))))}}}; "
        f"S7-хвост — средняя |Δ медианы| при L − k, k ∈ {{{', '.join(map(str, S7_TAIL_K))}}}; "
        f"S7-шум — то же при шуме α = {S7_NOISE_ALPHA} std разностей ({S7_NOISE_DRAWS} реализации). "
        "Это уровень, над которым сигналы должны подниматься при смене режима; критерием выбора модели он не является.",
        "",
        md_table(["Модель", "S4: разброс по длинам", "S7: изменение хвоста", "S7: слабый шум"], rows),
        "",
        "S4 по семействам (медиана по окнам семейства):",
        "",
        md_table(["Модель", *FAMILIES], rows_fam),
    ]

    out += [
        "",
        "## Ограничения",
        "",
        "- Только синтетические ряды без точки смены. В предобучении Chronos использовались синтетические ряды "
        "(KernelSynth, arXiv:2403.07815), поэтому простые синусоиды и AR близки к обучающим данным. "
        "О промышленных данных и о поведении при смене режима эта таблица ничего не говорит; смена режима "
        "исключена сознательно (см. начало скрипта).",
        "- MASE считается с наивным масштабом (m = 1) по контексту. При H > 1 даже лучший прогноз "
        "случайного блуждания даёт MASE > 1, поэтому модели сравниваются с наивным прогнозом на тех же окнах.",
        "- CRPS приближён по 9 квантилям 0.1…0.9 без хвостов. У Chronos-2 квантилей больше, "
        "для сравнимости берутся те же 9.",
        "- Покрытие и PCE усреднены по всем окнам (глобальная калибровка).",
        "- Время измерено на CPU этой машины; на GPU в Colab соотношение моделей может быть другим.",
    ]
    return "\n".join(out) + "\n"


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "?"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--n-per-family", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    windows = build_windows(args.n_per_family, np.random.default_rng(SEED))
    naive = naive_mase(windows)
    print(f"Окон: {len(windows)}; вариантов контекста на окно: {N_VARIANTS}")

    all_rows, all_bg, infos = [], [], []
    for model_id in args.models:
        print(f"\n{model_id}")
        model = load_forecaster(model_id, device=args.device)
        info = model_info(model)
        info["ms_per_forecast"] = ms_per_forecast(model, windows)
        rows, bg, seconds = evaluate(model, windows, naive)
        info["eval_seconds"] = seconds
        infos.append(info)
        all_rows += rows
        all_bg += bg
        d = pd.DataFrame(rows)
        print(f"  {info['ms_per_forecast']:.1f} мс на прогноз, прогон {seconds:.0f} с; "
              f"MASE H=64 {d[d.horizon == H_MAX].mase.mean():.3f}")
        del model
        gc.collect()  # модели загружаются по одной: памяти на машине мало

    df, bg = pd.DataFrame(all_rows), pd.DataFrame(all_bg)
    meta = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "env": (f"python {platform.python_version()}, torch {torch.__version__}, "
                f"transformers {transformers.__version__}, chronos-forecasting {chronos.__version__}, "
                f"numpy {np.__version__}, pandas {pd.__version__}"),
        "cpu": cpu_name(),
        "torch_threads": torch.get_num_threads(),
        "device": args.device,
        "seed": SEED,
        "n_per_family": args.n_per_family,
        "n_windows": len(windows),
        "config": {"L": L, "horizons": HORIZONS, "families": FAMILIES, "s4_lengths": S4_LENGTHS,
                   "s7_tail_k": S7_TAIL_K, "s7_noise_alpha": S7_NOISE_ALPHA, "s7_noise_draws": S7_NOISE_DRAWS,
                   "n_boot": N_BOOT, "reference": REFERENCE},
        "models": infos,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "windows.csv", index=False)
    bg.to_csv(args.out / "background.csv", index=False)
    (args.out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    summary = summarize(df, bg, infos, meta)
    (args.out / "summary.md").write_text(summary)
    print("\n" + summary)
    print(f"Сохранено в {args.out}/")


if __name__ == "__main__":
    main()
