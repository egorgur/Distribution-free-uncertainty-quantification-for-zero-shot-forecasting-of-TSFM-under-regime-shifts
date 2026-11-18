"""Multi-Context (протокол docs/protocol_v3_multicontext.md): признаки, выбор, веса, смесь, сводки."""

import unittest

import numpy as np

from src.adaptation import evaluate as ev
from src.adaptation import multicontext as mc

LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def brute_mixture_quantile(dists: np.ndarray, w: np.ndarray, tau: float) -> float:
    """Медленная проверка: CDF смеси на очень мелкой сетке y, первая точка с G ≥ τ."""
    taus = np.r_[0.0, LEVELS, 1.0]
    knots = []
    for d in np.sort(dists, axis=1):
        knots.append(np.r_[d[0] - (d[1] - d[0]), d, d[-1] + (d[-1] - d[-2])])
    y = np.linspace(min(k[0] for k in knots) - 1, max(k[-1] for k in knots) + 1, 400_001)
    g = sum(wk * np.interp(y, k, taus) for wk, k in zip(w, knots))
    return float(y[np.argmax(g >= tau)])


def candidates(n=3, K=4, H=2, seed=0):
    rng = np.random.default_rng(seed)
    centre = rng.normal(size=(n, K, 1, H)) * 3
    spread = np.sort(rng.gamma(2.0, size=(n, K, 9, H)), axis=2).cumsum(axis=2)
    return centre + spread - spread[:, :, 4:5]


class ScoreTests(unittest.TestCase):
    def test_previous_rows_same_series_only(self):
        series = np.array([0, 0, 0, 1, 1])
        pos = np.array([-128, -112, -96, -128, -112])
        rows = mc.previous_rows(series, pos, np.arange(5), 16)
        np.testing.assert_array_equal(rows, [-1, 0, 1, -1, 3])

    def test_hindcast_scale_and_shape(self):
        med = np.zeros((1, 4, 16))
        med[0, 2] = 1.0
        actual = np.ones((1, 16))
        s = mc.hindcast_scores(med, actual, np.array([2.0]))
        np.testing.assert_allclose(s, [[0.5, 0.5, 0.0, 0.5]])

    def test_consistency_zero_for_identical(self):
        q = np.repeat(candidates(K=1), 4, axis=1)
        np.testing.assert_allclose(mc.consistency_scores(q, np.ones(3)), 0.0)


class SelectTests(unittest.TestCase):
    def test_argmin_tie_goes_longer_and_nan_to_full(self):
        s = np.array([[0.3, 0.1, 0.1 + 1e-9, 0.5],
                      [0.1, 0.2, 0.3, 0.4],
                      [np.nan, 0.1, 0.1, 0.1]])
        np.testing.assert_array_equal(mc.select_index(s), [2, 0, mc.FULL])

    def test_softmax_limits(self):
        s = np.array([[0.0, 1.0, 2.0, 3.0]])
        np.testing.assert_allclose(mc.softmax_weights(s, 1e-6), [[1, 0, 0, 0]], atol=1e-12)
        np.testing.assert_allclose(mc.softmax_weights(s, 1e6), [[0.25] * 4], atol=1e-5)
        np.testing.assert_allclose(mc.softmax_weights(np.full((1, 4), np.nan), 1.0), [[0, 0, 0, 1]])
        with self.assertRaises(ValueError):
            mc.softmax_weights(s, 0.0)


class MixtureTests(unittest.TestCase):
    def test_one_hot_returns_candidate(self):
        q = candidates()
        w = mc.one_hot(np.array([0, 3, 1]))
        out, crossed = mc.mixture_quantiles(q, w, LEVELS, LEVELS)
        np.testing.assert_allclose(out, mc.pick(q, np.array([0, 3, 1])), atol=1e-9)
        self.assertEqual(crossed, 0)

    def test_identical_candidates_give_same_distribution(self):
        q = np.repeat(candidates(K=1), 4, axis=1)
        out, _ = mc.mixture_quantiles(q, np.full((3, 4), 0.25), LEVELS, LEVELS)
        np.testing.assert_allclose(out, q[:, 0], atol=1e-9)

    def test_against_brute_force(self):
        q = candidates(n=2, H=1, seed=3)
        w = np.array([[0.1, 0.2, 0.3, 0.4], [0.7, 0.0, 0.0, 0.3]])
        out, _ = mc.mixture_quantiles(q, w, LEVELS, [0.05, 0.25, 0.5, 0.95])
        for i in range(2):
            for j, tau in enumerate([0.05, 0.25, 0.5, 0.95]):
                ref = brute_mixture_quantile(q[i, :, :, 0], w[i], tau)
                self.assertAlmostEqual(out[i, j, 0], ref, delta=2e-3)

    def test_mixture_differs_from_vincent(self):
        # Две далёкие узкие компоненты: медиана смеси между ними, а среднее квантилей — посередине
        # тоже, но 0.1-квантиль смеси лежит в левой компоненте, а у среднего квантилей — нет.
        base = np.linspace(-0.4, 0.4, 9)
        q = np.zeros((1, 4, 9, 1))
        q[0, 0, :, 0], q[0, 1, :, 0] = base - 10, base + 10
        q[0, 2:, :, 0] = base + 10
        w = np.array([[0.5, 0.5, 0.0, 0.0]])
        mix, _ = mc.mixture_quantiles(q, w, LEVELS, [0.1])
        vin = mc.vincent(q, w)[:, 0]
        self.assertLess(mix[0, 0, 0], -9)
        self.assertAlmostEqual(vin[0, 0], -0.4, places=9)

    def test_crossing_and_ties_are_handled(self):
        q = candidates(n=1, H=1)
        q[0, 0, 3, 0], q[0, 0, 4, 0] = q[0, 0, 4, 0], q[0, 0, 3, 0]  # нарушение порядка
        q[0, 1, :, 0] = 5.0  # все квантили совпадают: точечная масса
        out, crossed = mc.mixture_quantiles(q, np.full((1, 4), 0.25), LEVELS, LEVELS)
        self.assertEqual(crossed, 1)
        self.assertTrue(np.all(np.diff(out[0, :, 0]) >= 0))
        self.assertTrue(np.isfinite(out).all())


class SummaryTests(unittest.TestCase):
    def test_series_weighted_equally(self):
        values = np.array([0.0, 0.0, 0.0, 1.0])
        series = np.array([0, 0, 0, 1])
        ids, m = ev.series_means(values, series, np.ones(4, bool))
        np.testing.assert_array_equal(ids, [0, 1])
        draws = ev.bootstrap_draws(2, 50, 0)
        point, boot = ev.group_mean(values, series, np.ones(4, bool), draws)
        self.assertAlmostEqual(point, 0.5)
        self.assertEqual(boot.shape, (50,))

    def test_gain_and_harm_signs(self):
        tg, lo, hi = ev.gain(np.log(0.9), np.log(np.array([0.85, 0.9, 0.95])))
        self.assertAlmostEqual(tg, 0.1)
        self.assertLessEqual(lo, tg)
        self.assertLessEqual(tg, hi)
        sh, lo, hi = ev.harm(np.log(1.02), np.log(np.array([1.0, 1.02, 1.04])))
        self.assertAlmostEqual(sh, 0.02)
        self.assertLessEqual(lo, sh)

    def test_method_equal_to_full_has_zero_gain(self):
        self.assertEqual(ev.log_ratio(np.array([2.0]), np.array([2.0]))[0], 0.0)


if __name__ == "__main__":
    unittest.main()
