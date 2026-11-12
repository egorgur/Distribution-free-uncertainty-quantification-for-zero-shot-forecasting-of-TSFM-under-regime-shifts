import csv
import io
import unittest
from decimal import Decimal

from src.pronto_test11 import parse_test11, window_geometry


def fixture(n=20, bad_index=None, missing=False):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerows([["Start Time", "6/26/2018 11:00", "End Time", "6/26/2018 12:00", "Interval", "1s"],
                     ["", "P", "P", "fault", "", ""],
                     ["TIMESTAMP", "pressure", "duplicate", "error", "", ""]])
    start = Decimal(43277) + Decimal(11) / 24
    for i in range(n):
        stamp = start + Decimal(i + (2 if bad_index == i else 0)) / 86400
        row = [f"{stamp:.5f}", str(i), str(i), "Tag does not exist.\r\n", "", ""]
        writer.writerow(row[:-1] if missing and i == 0 else row)
    return stream.getvalue().encode()


class Test11Tests(unittest.TestCase):
    def test_rounding_and_multiline_strings(self):
        summary, columns = parse_test11(fixture())
        self.assertEqual(summary["logical_data_rows"], 20)
        self.assertTrue(summary["nominal_grid_consistent_with_5_decimal_rounding"])
        self.assertEqual(summary["exact_one_second_steps"], 0)
        self.assertEqual(columns[2]["nonnumeric"], 20)
        self.assertEqual(summary["duplicate_column_pairs"], [{"left_index": 1, "right_index": 2, "tag": "P"}])

    def test_missing_logical_field_rejected(self):
        with self.assertRaisesRegex(ValueError, "width"):
            parse_test11(fixture(missing=True))

    def test_missing_second_not_silently_repaired(self):
        summary, _ = parse_test11(fixture(bad_index=5))
        self.assertFalse(summary["nominal_grid_consistent_with_5_decimal_rounding"])
        self.assertGreater(summary["nonpositive_time_steps"], 0)

    def test_geometry_against_direct_slices(self):
        n = 3601
        result = window_geometry(n, 2220, 3120)
        accepted = [t for t in range(511, n - 64, 8)
                    if all(2220 <= i < 3120 for i in range(t - 511, t + 65))]
        self.assertEqual(result["rolling_windows_global_phase"], len(accepted))
        self.assertEqual(result["max_disjoint_target_blocks"], 6)

    def test_no_complete_window(self):
        result = window_geometry(1000, 0, 575)
        self.assertEqual(result["rolling_windows_global_phase"], 0)


if __name__ == "__main__":
    unittest.main()