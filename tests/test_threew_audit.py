"""Аудит 3W: разбор меток, достаточность длин, пригодность переменных (docs/protocol_3w_audit.md)."""

import unittest

import numpy as np
import pandas as pd

from scripts.download_3w import git_blob_sha1
from src import threew_audit as ta


def labels(*parts):
    """labels((np.nan, 3), (0, 5), (102, 4), (2, 6)) → массив меток."""
    return np.concatenate([np.full(n, v, dtype=float) for v, n in parts])


class LabelTests(unittest.TestCase):
    def test_periods_with_nan(self):
        p = ta.periods(labels((np.nan, 3), (0, 5), (106, 4), (6, 2)))
        self.assertEqual(p, [(-1, 0, 3), (0, 3, 5), (106, 8, 4), (6, 12, 2)])

    def test_full_event(self):
        ev = ta.event_of(labels((np.nan, 3), (0, 5), (106, 4), (6, 2)))
        self.assertEqual((ev.structure, ev.tau, ev.old_len, ev.trans_len, ev.new_len, ev.new_start,
                          ev.total_from_tau), ("0>T>S", 8, 5, 4, 2, 12, 6))

    def test_event_without_steady_and_without_normal(self):
        self.assertEqual(ta.event_of(labels((0, 5), (102, 4))).structure, "0>T")
        self.assertEqual(ta.event_of(labels((np.nan, 2), (105, 4), (5, 3))).structure, "T>S")  # NaN перед τ ≠ норма

    def test_no_transient(self):
        self.assertEqual(ta.event_of(labels((np.nan, 2), (0, 5))).structure, "0 only")
        self.assertEqual(ta.event_of(labels((0, 2), (4, 5))).structure, "S only")


class SufficiencyTests(unittest.TestCase):
    def test_thresholds_follow_protocol(self):
        ev = ta.Event("0>T>S", 0, old_len=512 + 128, trans_len=10, new_len=576, new_start=10,
                      total_from_tau=704)
        self.assertEqual(ta.sufficient(ev, 1, 512, 64), {"old": True, "span": True, "new": True, "all": True})
        ev2 = ta.Event("0>T>S", 0, 639, 10, 575, 10, 703)
        self.assertEqual(ta.sufficient(ev2, 1, 512, 64), {"old": False, "span": False, "new": False, "all": False})
        self.assertFalse(ta.sufficient(ev, 10, 512, 64)["all"])  # при k = 10 блоков в 10 раз меньше


class VariableTests(unittest.TestCase):
    def test_usable_frozen_interp(self):
        self.assertFalse(ta.usable(np.array([1.0, np.nan, 2.0])))
        self.assertFalse(ta.usable(np.ones(5)))
        self.assertTrue(ta.usable(np.array([1.0, 2.0])))
        v = np.r_[np.arange(10.0), np.full(70, 3.0), np.arange(20.0)]
        self.assertAlmostEqual(ta.frozen_share(v), 70 / 100)
        self.assertAlmostEqual(ta.interp_share(np.arange(50.0)), 1.0)
        self.assertLess(ta.interp_share(np.random.default_rng(0).normal(size=200)), 0.05)

    def test_time_check(self):
        idx = pd.date_range("2020-01-01", periods=5, freq="1s")
        r = ta.time_check(idx)
        self.assertEqual((r["median_step_s"], r["share_step_1s"], r["duplicates"]), (1.0, 1.0, 0))

    def test_git_blob_sha1_matches_git(self):
        # `printf 'hello\\n' | git hash-object --stdin`
        self.assertEqual(git_blob_sha1(b"hello\n"), "ce013625030ba8dba906f756967f9e9ca394464a")


if __name__ == "__main__":
    unittest.main()
