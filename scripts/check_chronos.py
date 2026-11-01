"""Проверка этапа 0: Chronos-Bolt загружается и ведёт себя так, как мы рассчитываем.

Запуск из корня репозитория:
    python -m scripts.check_chronos
    python -m scripts.check_chronos --model amazon/chronos-bolt-base
    python -m scripts.check_chronos --model amazon/chronos-2

Что проверяется и зачем:
1. Формы выхода и отсутствие NaN.
2. Детерминизм: два одинаковых вызова дают одинаковый прогноз. Нужен, чтобы
   расхождение прогнозов в S4/S7 было реакцией модели, а не случайностью.
3. Батч из контекстов разной длины даёт то же, что вызовы поодиночке.
   Тогда S4 и S7 для одного окна считаются одним вызовом модели.
4. Здравый смысл: на синусоиде с шумом модель точнее наивного прогноза.
5. Горизонт больше максимального запрещён (без авторегрессии).
6. Время прогноза — для оценки объёма вычислений по сетке синтетики.
"""

from __future__ import annotations

import argparse
import platform
import sys
import time

import chronos
import numpy as np
import torch
import transformers

from src.models import load_forecaster

SEED = 0
PERIOD = 24  # период синусоиды в тестовом ряде
L, H = 512, 64  # основной контекст и максимальный горизонт из CLAUDE.md

failures: list[str] = []


def check(ok: bool, what: str, detail: str = "") -> None:
    """Печатает результат проверки и запоминает провалы (скрипт идёт до конца)."""
    print(f"  [{'OK' if ok else 'FAIL'}] {what}" + (f": {detail}" if detail else ""))
    if not ok:
        failures.append(what)


def make_series(n: int, rng: np.random.Generator) -> np.ndarray:
    """Стационарный тестовый ряд без смены режима: 10 + 3·sin(2πt/24) + шум N(0, 0.5²)."""
    t = np.arange(n)
    return 10 + 3 * np.sin(2 * np.pi * t / PERIOD) + rng.normal(0, 0.5, n)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="amazon/chronos-bolt-small")
    parser.add_argument("--device", default="cpu", help="cpu или cuda (в Colab с GPU)")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    print("Окружение:")
    print(
        f"  python {platform.python_version()}, torch {torch.__version__}, "
        f"transformers {transformers.__version__}, chronos-forecasting {chronos.__version__}, "
        f"numpy {np.__version__}"
    )
    print(f"  потоков CPU для torch: {torch.get_num_threads()}")

    t0 = time.perf_counter()
    model = load_forecaster(args.model, device=args.device)
    net = model.pipeline.model
    n_params = sum(p.numel() for p in net.parameters())
    q_levels = model.quantile_levels
    i_lo, i_med, i_hi = q_levels.index(0.1), q_levels.index(0.5), q_levels.index(0.9)
    print(f"\nМодель {args.model} загружена за {time.perf_counter() - t0:.1f} с")
    print(f"  параметров {n_params / 1e6:.1f} M, dtype {net.dtype}, устройство {net.device}")
    print(
        f"  слоёв: энкодер {net.config.num_layers}, декодер {getattr(net.config, 'num_decoder_layers', 0) or 0}; "
        f"d_model {net.config.d_model}"
    )
    print(f"  патч {model.patch_size}, макс. горизонт {model.max_horizon}, макс. контекст {model.max_context}")
    print(f"  квантили {q_levels}")

    # Тестовое окно: контекст x[:end], будущее x[end:end+H].
    x = make_series(L + H, rng)
    end = L
    scale = x[:end].std()

    print("\n1. Формы выхода")
    lengths = [64, 128, 256, 512]
    for h in (16, 64):
        q = model.predict_quantiles([x[end - n : end] for n in lengths], horizon=h)
        expected = (len(lengths), len(q_levels), h)
        check(q.shape == expected and np.isfinite(q).all(), f"H={h}", f"форма {q.shape}, NaN нет")
    crossing = (np.diff(q, axis=1) < 0).mean()
    print(f"  доля пересечений соседних квантилей: {crossing:.4f} (монотонность модель не гарантирует)")

    print("\n2. Детерминизм")
    q1 = model.predict_quantiles([x[:end]], horizon=H)
    q2 = model.predict_quantiles([x[:end]], horizon=H)
    check(np.array_equal(q1, q2), "повторный вызов даёт тот же прогноз", f"макс. разница {np.abs(q1 - q2).max():.1e}")

    print("\n3. Батч из контекстов разной длины = вызовы поодиночке")
    # Все контексты заканчиваются в одной точке, как в S4 (длины в разы) и S7 (L - k, k мало).
    lengths = [512, 256, 128, 64, 509, 503]
    ctxs = [x[end - n : end] for n in lengths]
    batched = model.predict_quantiles(ctxs, horizon=H)
    single = np.concatenate([model.predict_quantiles([c], horizon=H) for c in ctxs])
    noise_floor = np.abs(batched - single).max() / scale
    check(noise_floor < 1e-4, "батч совпадает с поодиночными вызовами", f"макс. разница {noise_floor:.1e} std ряда")
    print("  для справки, ряд без смены: среднее |Δ медианы| относительно L=512, в std ряда:")
    for i, n in enumerate(lengths[1:], start=1):
        diff = np.abs(batched[i, i_med] - batched[0, i_med]).mean() / scale
        print(f"    L={n}: {diff:.4f}")

    print(f"\n4. Здравый смысл: ряд без смены, L={L}, H={H}")
    q = model.predict_quantiles([x[:end]], horizon=H)[0]
    future = x[end : end + H]
    mae_model = np.abs(q[i_med] - future).mean()
    mae_naive = np.abs(x[end - 1] - future).mean()
    snaive = x[end - PERIOD : end][np.arange(H) % PERIOD]  # повтор последнего периода
    mae_snaive = np.abs(snaive - future).mean()
    check(
        mae_model < mae_naive,
        "модель точнее наивного прогноза",
        f"MAE модели {mae_model:.3f}, наивного {mae_naive:.3f}, сезонного наивного {mae_snaive:.3f} "
        f"(шум ряда даёт MAE ≈ 0.40)",
    )
    coverage = ((future >= q[i_lo]) & (future <= q[i_hi])).mean()
    print(f"  покрытие [q0.1, q0.9] на этом окне: {coverage:.2f} (номинал 0.80; по одному окну оценка грубая)")

    print("\n5. Горизонт больше максимального запрещён")
    try:
        model.predict_quantiles([x[:end]], horizon=model.max_horizon + 1)
        check(False, "ожидалась ошибка ValueError")
    except ValueError:
        check(True, f"horizon={model.max_horizon + 1} даёт ValueError")

    print(f"\n6. Время прогноза (L={L}, H={H})")
    long_x = make_series(L + 200, rng)
    ctxs = [long_x[i : i + L] for i in range(64)]  # 64 разных окна
    model.predict_quantiles(ctxs[:4], horizon=H)  # прогрев
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        model.predict_quantiles(ctxs, horizon=H)
        times.append(time.perf_counter() - t0)
    t_batch = float(np.median(times))
    print(f"  батч из {len(ctxs)} окон: {t_batch:.2f} с, {t_batch / len(ctxs) * 1e3:.1f} мс на прогноз")

    print()
    if failures:
        print(f"Провалены проверки: {failures}")
        sys.exit(1)
    print("Все проверки пройдены.")


if __name__ == "__main__":
    main()
