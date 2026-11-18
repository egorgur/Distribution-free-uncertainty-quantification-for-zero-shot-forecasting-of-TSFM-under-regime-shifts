"""Regime-Component: статистики компонент и правило (docs/protocol_v3_component.md)."""

import unittest

import numpy as np

from src.adaptation import components as cp


def series(n=4, L=512, w=64, seed=0, change=None):
    """AR(1)+сезонность; change меняет последние w точек."""
    rng = np.random.default_rng(seed)
    i = np.arange(L)
    out = np.empty((n, L))
    for r in range(n):
        e = rng.normal(size=L)
        x = np.zeros(L)
        for t in range(1, L):
            x[t] = 0.7 * x[t - 1] + e[t]
        x += 2 * np.sin(2 * np.pi * i / 24)
        if change == "level":
            x[L - w:] += 8
        elif change == "scale":
            x[L - w:] = x[L - w:].mean() + 4 * (x[L - w:] - x[L - w:].mean())
        elif change == "trend":
            x[L - w:] += 0.3 * np.arange(w)
        elif change == "freq":
            x[L - w:] = x[L - w:] - 2 * np.sin(2 * np.pi * i[L - w:] / 24) + 2 * np.sin(2 * np.pi * i[L - w:] / 6)
        out[r] = x
    return out


def scale_of(x):
    return np.abs(np.diff(x, axis=1)).mean(axis=1)


class ComponentTests(unittest.TestCase):
    def test_each_change_raises_its_statistic(self):
        base = series()
        z0 = cp.statistics(base, scale_of(base))
        for kind in cp.STATS:
            x = series(change=kind)
            z = cp.statistics(x, scale_of(x))
            self.assertTrue(np.all(z[kind] > z0[kind]), kind)

    def test_decide_no_adaptation_and_largest_ratio(self):
        z = {"level": np.array([0.5, 3.0, 3.0]), "scale": np.array([0.1, 0.1, 0.9]),
             "trend": np.array([0.2, 0.2, 0.2]), "freq": np.array([0.1, 0.1, 0.1])}
        thr = {"level": 1.0, "scale": 0.2, "trend": 1.0, "freq": 1.0}
        np.testing.assert_array_equal(cp.decide(z, thr), [0, 1, 2])  # 4.5 > 3 у scale в третьем окне

    def test_flat_context_gives_no_adaptation(self):
        x = np.full((1, 512), 5.0)
        z = cp.statistics(x, scale_of(x))
        self.assertTrue(all(z[k][0] == 0 for k in cp.STATS))
        x = series(n=1)
        x[0, -64:] = x[0, -65]  # залипание недавнего участка
        z = cp.statistics(x, scale_of(x))
        self.assertEqual(z["scale"][0], 0.0)
        self.assertEqual(z["freq"][0], 0.0)

    def test_thresholds_quantile_and_true_mapping(self):
        z = {k: np.arange(100, dtype=float) for k in cp.STATS}
        thr = cp.thresholds(z, np.ones(100, bool))
        self.assertAlmostEqual(thr["level"], 94.05)
        comp = cp.true_component(np.array(["change", "change", "none", "drift"]),
                                 np.array(["sensor_bias", "ar", "none", "frequency"]))
        np.testing.assert_array_equal(comp, [1, 0, 0, 4])


if __name__ == "__main__":
    unittest.main()
