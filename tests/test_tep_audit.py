import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.download_tep_skab import safe_path, store
from src.tep_audit import compare_runs, load_dat, rolling_geometry


class TepAuditTests(unittest.TestCase):
    def test_boundary_single_window(self):
        self.assertEqual(rolling_geometry(576, 512, 64, 160)["row_order"], 1)
        self.assertEqual(rolling_geometry(575, 512, 64, 160)["row_order"], 0)

    def test_reference_geometry(self):
        r = rolling_geometry(960, 512, 64, 160)
        self.assertEqual((r["row_order"], r["first_d"]), (49, 351))
        self.assertEqual((r["before"], r["change_in_horizon"]), (0, 0))
        self.assertEqual(sum(r[k] for k in ("before", "change_in_horizon", "mixed_context", "fully_post")), 49)
        self.assertEqual(rolling_geometry(960, 128, 16, 160)["before"], 3)

    def test_no_onset_is_not_invented(self):
        self.assertNotIn("before", rolling_geometry(480, 128, 16, None))

    def test_dat_orientation(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            x = np.arange(52 * 5).reshape(52, 5)
            np.savetxt(p / "d00.dat", x)
            np.savetxt(p / "d01.dat", x.T)
            np.testing.assert_equal(load_dat(p / "d00.dat")[0], x.T)
            np.testing.assert_equal(load_dat(p / "d01.dat")[0], x.T)

    def test_ragged_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "d01.dat"
            p.write_text("1 2\n3\n")
            with self.assertRaises(ValueError):
                load_dat(p)

    def test_exact_prefix(self):
        x = np.ones((4, 52))
        y = x.copy()
        y[2, 1] = 3
        r = compare_runs({"a": x, "b": y}).iloc[0]
        self.assertEqual(r.common_prefix_rows, 2)
        self.assertEqual(r.identical_rows_same_index, 3)

    def test_unsafe_paths(self):
        for name in ("../x", "/x", "a\\b", "C:x"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                safe_path(name)

    def test_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x"
            store(p, b"original")
            store(p, b"original")
            with self.assertRaises(ValueError):
                store(p, b"changed")
            self.assertEqual(p.read_bytes(), b"original")


if __name__ == "__main__":
    unittest.main()