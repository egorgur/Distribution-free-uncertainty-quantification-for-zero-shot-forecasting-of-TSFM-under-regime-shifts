"""Граничные случаи аудита на маленьких файлах, без сети и моделей."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.gas_audit import SENSORS, audit_file, late_tail_diagnostics, run_summary


class GasAuditTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "gas.txt"

    def fixture(self, times, gas=None, ethylene=None, sensors=None):
        n = len(times)
        array = np.column_stack([times, np.zeros(n) if gas is None else gas,
            np.zeros(n) if ethylene is None else ethylene,
            np.ones((n, 16)) if sensors is None else sensors])
        np.savetxt(self.path, array, header="Time (seconds), gas ppm, ethylene ppm, sensors", comments="")

    def test_chunk_boundary_simultaneous_event_and_final_segment(self):
        self.fixture(np.arange(6) / 100, [0, 0, 0, 2, 2, 2], [0, 0, 0, 3, 3, 3])
        summary, tables = audit_file(self.path, chunksize=3)
        self.assertEqual(summary["setpoint_events"], 1)
        event = tables["events"].iloc[0]
        self.assertEqual(event.row, 3)
        self.assertTrue(event.gas_changed and event.ethylene_changed)
        segments = tables["segments"]
        self.assertEqual(segments.rows.tolist(), [3, 3])
        self.assertEqual(segments.end_row_exclusive.tolist(), [3, 6])
        self.assertEqual(segments.right_censored.tolist(), [False, True])
        self.assertAlmostEqual(segments.iloc[-1].duration_s, .03)
        self.assertTrue(tables["seconds"].iloc[0].mixed_setpoint)

    def test_chunksize_invariance(self):
        sensors = np.tile(np.arange(8)[:, None] % 3, (1, 16)).astype(float)
        sensors[4, 0] = np.nan
        self.fixture([0, .01, .01, .03, .02, 1, 1.01, 2], [0, 0, 1, 1, 1, 2, 2, 2], sensors=sensors)
        expected, expected_tables = audit_file(self.path, chunksize=100)
        for chunk in (1, 2, 3, 4):
            summary, tables = audit_file(self.path, chunksize=chunk)
            self.assertEqual(summary, expected)
            for name in tables:
                pd.testing.assert_frame_equal(tables[name], expected_tables[name], check_exact=False)

    def test_duplicate_backward_gap(self):
        self.fixture([0, .01, .01, .04, .02])
        summary, tables = audit_file(self.path, chunksize=2)
        self.assertEqual(summary["adjacent_duplicate_times"], 1)
        self.assertEqual(summary["duplicate_time_labels_beyond_first"], 1)
        self.assertEqual(summary["backward_steps"], 1)
        self.assertEqual(summary["gaps_over_nominal"], 1)
        self.assertEqual(tables["time_anomalies"].row.tolist(), [2, 3, 4])

    def test_constant_channel_and_nonfinite(self):
        sensors = np.ones((7, 16))
        sensors[2, 1] = np.inf
        sensors[3, 1] = np.nan
        self.fixture(np.arange(7) / 100, sensors=sensors)
        summary, tables = audit_file(self.path, chunksize=2)
        self.assertEqual(summary["nonfinite_values"], 2)
        columns = tables["columns"].set_index("column")
        self.assertEqual(columns.loc["s01", "longest_equal_run_rows"], 7)
        self.assertEqual(columns.loc["s01", "equal_adjacent_pairs"], 6)
        self.assertEqual(columns.loc["s02", "longest_equal_run_rows"], 3)
        self.assertEqual(columns.loc["s02", "finite"], 5)
        self.assertEqual(columns.loc["s02", "at_min"], 5)
        self.assertEqual(tables["seconds"].iloc[0].s02, 1)

    def test_no_change_single_row(self):
        self.fixture([0])
        summary, tables = audit_file(self.path, chunksize=1)
        self.assertEqual(summary["setpoint_events"], 0)
        self.assertEqual(summary["non_nominal_steps"], 0)
        self.assertEqual(len(tables["segments"]), 1)
        self.assertTrue(tables["segments"].iloc[0].left_censored)

    def test_empty_and_wrong_columns(self):
        for payload in ("header\n", "header\n1 2 3\n", "header\n" + " ".join(["0"] * 20)):
            self.path.write_text(payload)
            with self.assertRaises(ValueError):
                audit_file(self.path)

    def test_nonfinite_time_or_setpoint_rejected(self):
        self.fixture([0, np.nan])
        with self.assertRaisesRegex(ValueError, "nonfinite time/setpoint"):
            audit_file(self.path)

    def test_short_row_after_valid_row_not_padded_or_skipped(self):
        valid = " ".join(["0"] * 19)
        for invalid in (" ".join(["0"] * 18), "", " ".join(["0"] * 20)):
            self.path.write_text(f"header\n{valid}\n{invalid}\n{valid}\n")
            with self.assertRaisesRegex(ValueError, "data row 1"):
                audit_file(self.path, chunksize=1)

    def test_run_summary_boundary(self):
        self.assertEqual(run_summary(np.ones(3), 1, 4), (3, 7, 7))
        self.assertEqual(run_summary(np.array([np.nan, 2, 2])), (1, 2, 2))

    def test_late_tail_excludes_mixed_and_partial_bins(self):
        seconds = pd.DataFrame({"second": np.arange(60), "mixed_setpoint": False,
                                "gas_ppm_min": 0, "ethylene_ppm_min": 0})
        for sensor in SENSORS:
            seconds[sensor] = seconds.second.astype(float)
        seconds.loc[55, "mixed_setpoint"] = True
        seconds.loc[55, SENSORS] = 10000
        segments = pd.DataFrame({"segment": [0], "start_s": [.01], "boundary_s": [59.01],
            "gas_ppm": [0], "ethylene_ppm": [0], "right_censored": [False]})
        result = late_tail_diagnostics(seconds, segments)
        self.assertEqual(result.iloc[0].first_bins, 9)
        self.assertEqual(result.iloc[0].last_bins, 8)
        self.assertLess(result.iloc[0].delta, 30)


if __name__ == "__main__":
    unittest.main()