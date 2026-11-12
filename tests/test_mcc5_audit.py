import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.mcc5_audit import aggregate, analyse, key_phase_stats, load_recording, plateau_stats


class Mcc5AuditTests(unittest.TestCase):
    def load(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selected.csv"
            path.write_text(text)
            return load_recording(path)

    def test_no_header_first_sample_preserved(self):
        data, schema = self.load("0,1,2,3,4,5,6,7,8\n.000078125,1,2,3,4,5,6,7,8\n")
        self.assertEqual(len(data), 2)
        self.assertIsNone(schema["header"])
        self.assertEqual(data[0, 0], 0)

    def test_header_recognised_only_first_row(self):
        data, schema = self.load("a,b,c,d,e,f,g,h\n1,2,3,4,5,6,7,8\n")
        self.assertEqual(schema["header"][0], "a")
        self.assertEqual(data.shape, (1, 8))

    def test_blank_and_short_rows_rejected(self):
        for suffix in ("\n", "1,2\n", "bad,2,3,4,5,6,7,8\n"):
            with self.subTest(suffix=suffix), self.assertRaises(ValueError):
                self.load("1,2,3,4,5,6,7,8\n" + suffix)

    def test_nonfinite_rejected(self):
        for value in ("nan", "inf", "-inf"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.load(f"1,2,3,4,5,6,7,{value}\n")

    def test_corrupt_first_numeric_row_not_mistaken_for_header(self):
        with self.assertRaises(ValueError):
            self.load("bad,2,3,4,5,6,7,8\n1,2,3,4,5,6,7,8\n")

    def test_rows_preserved_across_parse_blocks(self):
        data, schema = self.load("1,2,3,4,5,6,7,8\n" * 8193)
        self.assertEqual(data.shape, (8193, 8))
        self.assertEqual(schema["data_rows"], 8193)

    def test_invalid_width_and_empty_rejected(self):
        for text in ("", "1,2,3\n", "a,b,c,d,e,f,g,h\n"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.load(text)

    def test_time_grid_and_duplicates(self):
        signals = np.ones((8, 8))
        data = np.column_stack((np.arange(8) / 4, signals))
        result = analyse(data, fs=4)
        self.assertEqual(result["time"]["max_abs_deviation_from_nominal_s"], 0)
        self.assertEqual(result["time"]["dt_zero"], 0)
        self.assertEqual(len(result["exact_duplicate_channels"]), 28)
        self.assertFalse(result["stability_established"])

    def test_time_defects_reported_not_fixed(self):
        data = np.column_stack(([0, .25, .25, .1], np.ones((4, 8))))
        result = analyse(data, fs=4)
        self.assertEqual(result["time"]["dt_zero"], 1)
        self.assertEqual(result["time"]["dt_negative"], 1)
        self.assertFalse(result["time"]["hardware_continuity_verified"])

    def test_missing_time_not_silently_measured(self):
        result = analyse(np.ones((4, 8)), fs=4)
        self.assertFalse(result["time"]["timestamp_present"])
        self.assertEqual(result["time"]["count_divided_by_fs_s"], 1)

    def test_bins_include_partial_final_bin(self):
        data = np.zeros((5, 8))
        data[:, 1] = np.arange(5)
        rows = aggregate(data, fs=4, seconds=1, columns=[1])
        self.assertEqual([r["count"] for r in rows], [4, 1])
        self.assertEqual(rows[-1]["mean"], 4)

    def test_fractional_second_bins(self):
        rows = aggregate(np.ones((8, 8)), fs=4, seconds=.25, columns=[1])
        self.assertEqual(len(rows), 8)
        self.assertTrue(all(row["count"] == 1 for row in rows))

    def test_fixed_plateaus_and_slope(self):
        data = np.zeros((90 * 4, 8))
        data[:, 1] = 2 * np.arange(len(data)) / 4 + 3
        rows = plateau_stats(data, fs=4)
        self.assertEqual(len(rows), 15)
        self.assertEqual((rows[0]["start_s"], rows[0]["stop_exclusive_s"]), (10, 30))
        self.assertAlmostEqual(rows[0]["slope_original_units_per_s"], 2)
        self.assertFalse(rows[0]["stability_established"])

    def test_phase_edges_do_not_imply_rpm(self):
        result = key_phase_stats(np.array([0, 0, 2, 2, 0, 0, 2, 2]), fs=4)
        self.assertEqual(result["rising_edges"], 2)
        self.assertEqual(result["inter_edge_seconds_quantiles"], [1] * 5)
        self.assertFalse(result["rpm_derived"])

    def test_json_roundtrip(self):
        result = analyse(np.ones((4, 8)), fs=4)
        self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), result)

    def test_analysis_requires_finite_nonempty_array(self):
        for data in (np.ones((1, 8)), np.ones((2, 7)), np.full((2, 8), np.nan)):
            with self.subTest(shape=data.shape), self.assertRaises(ValueError):
                analyse(data)


if __name__ == "__main__":
    unittest.main()