"""Regime-Aligned Context: преобразования старой части контекста (docs/protocol_v3_aligned.md)."""

import unittest

import numpy as np

from src.adaptation import alignment as al


def ctx_with_shift(level_old=10.0, level_new=20.0, amp_old=1.0, amp_new=1.0, L=512, w=64):
    """Сезонный ряд, у которого в недавних w точках другой уровень и амплитуда."""
    i = np.arange(L)
    x = np.sin(2 * np.pi * i / 16)
    x[: L - w] = level_old + amp_old * x[: L - w]
    x[L - w:] = level_new + amp_new * x[L - w:]
    return x[None, :]


class AlignmentTests(unittest.TestCase):
    def test_instruction_example_level(self):
        # Пример инструкции: колебания 10, 11, 9 при новом уровне 20 → 20, 21, 19.
        ctx = np.array([[10.0, 11.0, 9.0, 20.0, 20.0]])
        out, _ = al.aligned_context(ctx, "level", 1.0, np.ones(1), w=2)
        np.testing.assert_allclose(out[0], [20, 21, 19, 20, 20])

    def test_recent_never_changes_and_alpha_zero_is_identity(self):
        ctx = np.random.default_rng(0).normal(size=(3, 512)).cumsum(axis=1)
        for kind in al.KINDS:
            out, _ = al.aligned_context(ctx, kind, 0.0, np.ones(3))
            np.testing.assert_array_equal(out, ctx)
            out, _ = al.aligned_context(ctx, kind, 0.7, np.ones(3))
            np.testing.assert_array_equal(out[:, -al.W_RECENT:], ctx[:, -al.W_RECENT:])

    def test_scale_matches_recent_median_and_mad(self):
        ctx = ctx_with_shift(amp_old=1.0, amp_new=3.0)
        O2, fb = al.align_old(ctx, "scale", np.ones(1))
        O, R = al.split(ctx)
        self.assertFalse(fb[0])
        self.assertAlmostEqual(float(np.median(O2)), float(np.median(R)), places=9)
        self.assertAlmostEqual(float(al.mad(O2)[0]), float(al.mad(R)[0]), places=9)

    def test_scale_falls_back_to_level_on_zero_mad_and_clips(self):
        ctx = ctx_with_shift()
        ctx[0, -al.W_RECENT:] = 5.0  # залипание: MAD(R) = 0
        O2, fb = al.align_old(ctx, "scale", np.ones(1))
        lev, _ = al.align_old(ctx, "level", np.ones(1))
        self.assertTrue(fb[0])
        np.testing.assert_allclose(O2, lev)
        ctx = ctx_with_shift(amp_old=1.0, amp_new=1000.0)
        O2, _ = al.align_old(ctx, "scale", np.ones(1))
        O, _ = al.split(ctx)
        self.assertAlmostEqual(float(al.mad(O2)[0] / al.mad(O)[0]), al.K_MAX, places=6)

    def test_trend_replaces_slope_keeps_residuals(self):
        L, w = 512, 64
        i = np.arange(L, dtype=float)
        wave = np.sin(2 * np.pi * i / 16)
        x = 0.01 * i + wave
        x[L - w:] = x[L - w - 1] + 0.5 * (i[L - w:] - (L - w - 1)) + wave[L - w:]
        O2, _ = al.align_old(x[None, :], "trend", np.ones(1))
        a, b = al.ols_line(O2, np.arange(L - w, dtype=float))
        _, b_r = al.ols_line(x[None, L - w:], np.arange(L - w, L, dtype=float))
        self.assertAlmostEqual(float(b[0]), float(b_r[0]), places=9)
        detr = O2[0] - (a[0] + b[0] * np.arange(L - w))
        np.testing.assert_allclose(detr - detr.mean(), wave[:L - w] - wave[:L - w].mean(), atol=0.05)

    def test_old_part_depends_only_on_recent_statistics(self):
        # Перестановка точек внутри R не меняет медиану/MAD, поэтому level и scale дают ту же O′.
        ctx = ctx_with_shift(amp_new=2.0)
        perm = ctx.copy()
        perm[0, -al.W_RECENT:] = perm[0, -al.W_RECENT:][::-1]
        for kind in ("level", "scale"):
            np.testing.assert_allclose(al.align_old(ctx, kind, np.ones(1))[0],
                                       al.align_old(perm, kind, np.ones(1))[0])

    def test_choose_tie_prefers_least_intervention_and_nan_to_original(self):
        e = np.array([[0.2, 0.1, 0.1 + 1e-9, 0.3],
                      [0.5, 0.4, 0.3, 0.1],
                      [np.nan, 0.1, 0.1, 0.1]])
        np.testing.assert_array_equal(al.choose(e, np.ones(3)), [1, 3, 0])
        np.testing.assert_array_equal(al.choose(e[:1], np.zeros(1)), [0])


if __name__ == "__main__":
    unittest.main()
