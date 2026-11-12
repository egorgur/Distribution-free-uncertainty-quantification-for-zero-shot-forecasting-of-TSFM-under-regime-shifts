"""Описательный экран кандидатов эталона, не доказательство стационарности."""

import numpy as np
import pandas as pd

from src.gas_audit import SENSORS

TAIL_SECONDS = 120
GUARD_SECONDS = 30
BLOCK_SECONDS = (5, 10)
BOOTSTRAPS = 1000
SEED = 0
PAIRS = ((0, 1), (0, 2), (1, 2))


def select_tail(seconds, segment):
    """Возвращает только заранее заданный хвост либо причину исключения."""
    end = int(np.floor(segment.boundary_s))
    start = end - TAIL_SECONDS
    if segment.left_censored or segment.right_censored:
        return None, "censored", start, end
    if start - segment.start_s < GUARD_SECONDS:
        return None, "insufficient_duration", start, end
    tail = seconds.loc[(seconds.second >= start) & (seconds.second < end)]
    if not np.array_equal(tail.second.to_numpy(), np.arange(start, end)):
        return None, "missing_or_duplicate_bins", start, end
    same = (~tail.mixed_setpoint
            & (tail.gas_ppm_min == segment.gas_ppm)
            & (tail.gas_ppm_max == segment.gas_ppm)
            & (tail.ethylene_ppm_min == segment.ethylene_ppm)
            & (tail.ethylene_ppm_max == segment.ethylene_ppm))
    if not same.all():
        return None, "mixed_or_wrong_setpoint", start, end
    values = tail[SENSORS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        return None, "nonfinite_sensor", start, end
    return values, "eligible", start, end


def moving_indices(length, block, repeats, rng):
    """Обычные moving blocks без циклического соединения концов."""
    if length < 1 or block < 1 or length % block or repeats < 1:
        raise ValueError("positive length divisible by block and repeats required")
    starts = rng.integers(0, length - block + 1,
                          size=(repeats, length // block))
    return (starts[..., None] + np.arange(block)).reshape(repeats, length)


def passes(mean_upper, log_variance_upper, mean_margin=.25, variance_ratio=1.5):
    """Решение без epsilon, nan/inf никогда не проходят."""
    return bool(np.isfinite(mean_upper) and np.isfinite(log_variance_upper)
                and mean_upper <= mean_margin
                and log_variance_upper <= np.log(variance_ratio))


def lag_correlation(x, lag):
    a, b = x[:-lag], x[lag:]
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def screen_tail(values, seed, repeats=BOOTSTRAPS):
    """Синхронный bootstrap каналов в трёх последовательных блоках по40с.

    Границы приближённые; длинная память и неоднородность могут нарушать
    предпосылки bootstrap. Ошибки TSFM сюда не поступают.
    """
    x = np.asarray(values, dtype=float)
    if x.shape != (TAIL_SECONDS, len(SENSORS)) or not np.isfinite(x).all():
        raise ValueError("expected finite (120, 16) diagnostic means")
    # Центрирование снижает потерю точности при больших исходных уровнях.
    x = x - x.mean(axis=0)
    blocks = x.reshape(3, 40, len(SENSORS))
    scale = x.std(axis=0)
    means, variances = blocks.mean(axis=1), blocks.var(axis=1)
    regular = (scale > 0) & (variances > 0).all(axis=0)
    mean_pairs = np.full((3, len(SENSORS)), np.nan)
    log_pairs = mean_pairs.copy()
    for k, (a, b) in enumerate(PAIRS):
        mean_pairs[k, regular] = (means[a, regular] - means[b, regular]) / scale[regular]
        log_pairs[k, regular] = np.log(variances[a, regular]) - np.log(variances[b, regular])
    rows = []
    for block in BLOCK_SECONDS:
        rng = np.random.default_rng(np.random.SeedSequence([*seed, block]))
        indices = [moving_indices(40, block, repeats, rng) for _ in range(3)]
        # Один набор индексов для всех16каналов, а не независимая их перестановка.
        draws = np.stack([blocks[k][indices[k]] for k in range(3)], axis=1)
        boot_means, boot_vars = draws.mean(axis=2), draws.var(axis=2)
        for j, sensor in enumerate(SENSORS):
            row = {"sensor": sensor, "block_s": block, "status": "ok",
                   "tail_std_1s": scale[j], "mean_spread": np.nan,
                   "log_variance_spread": np.nan, "mean_upper": np.nan,
                   "log_variance_upper": np.nan,
                   "lag1": lag_correlation(x[:, j], 1),
                   "lag10": lag_correlation(x[:, j], 10)}
            if not regular[j]:
                row["status"] = "zero_observed_variance"
            else:
                row["mean_spread"] = float(np.max(np.abs(mean_pairs[:, j])))
                row["log_variance_spread"] = float(np.max(np.abs(log_pairs[:, j])))
                if not (boot_vars[:, :, j] > 0).all():
                    row["status"] = "zero_bootstrap_variance"
                else:
                    mean_errors, variance_errors = [], []
                    for k, (a, b) in enumerate(PAIRS):
                        mean_errors.append((boot_means[:, a, j] - boot_means[:, b, j])
                                           / scale[j] - mean_pairs[k, j])
                        variance_errors.append(np.log(boot_vars[:, a, j])
                                               - np.log(boot_vars[:, b, j]) - log_pairs[k, j])
                    for name, errors, observed in (
                        ("mean_upper", mean_errors, row["mean_spread"]),
                        ("log_variance_upper", variance_errors, row["log_variance_spread"]),
                    ):
                        maximum = np.max(np.abs(np.asarray(errors)), axis=0)
                        row[name] = observed + float(np.quantile(maximum, .975, method="linear"))
            row["primary"] = passes(row["mean_upper"], row["log_variance_upper"])
            row["sensitivity"] = passes(row["mean_upper"], row["log_variance_upper"], .5, 2.)
            rows.append(row)
    return pd.DataFrame(rows)


def adjacent_candidates(flags, segment_ids):
    """Событие имеет кандидатов с двух сторон только у одного и того же канала."""
    lookup = {(int(row.segment), row.sensor): row for row in flags.itertuples()}
    rows = []
    for segment in segment_ids:
        if segment == 0:
            continue
        for sensor in SENSORS:
            old, new = lookup.get((segment - 1, sensor)), lookup.get((segment, sensor))
            rows.append({"new_segment": segment, "event": segment - 1, "sensor": sensor,
                         "primary": bool(old is not None and new is not None
                                         and old.primary and new.primary),
                         "sensitivity": bool(old is not None and new is not None
                                             and old.sensitivity and new.sensitivity)})
    return pd.DataFrame(rows)


def collapse_candidates(diagnostics):
    """Не разрешать неполную/дублированную пару bootstrap5/10с."""
    keys = ["recording", "segment", "sensor"]
    for _, group in diagnostics.groupby(keys):
        if sorted(group.block_s.tolist()) != list(BLOCK_SECONDS):
            raise ValueError("each segment/channel must have exactly block lengths 5 and 10")
    return diagnostics.groupby(keys, as_index=False)[["primary", "sensitivity"]].all()