"""SKAB: граничные случаи без сети, моделей и доступа к реальным CSV."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.skab_audit import (COMMIT, SECOND_NS, SENSORS, audit_frame, bounds_from_cuts,
    brute_window_counts, continuous, cross_file_overlaps, event_geometry, load_csv, longest_equal_run,
    rolling_windows, tail_diagnostics, time_ns, verify_originals, window_counts)


def fixture(n=20, *, cps=(), anomaly=None, elapsed=None, labels=True):
    seconds = np.arange(n) if elapsed is None else np.asarray(elapsed)
    frame = pd.DataFrame({"datetime": pd.Timestamp("2020-01-01") + pd.to_timedelta(seconds, unit="s")})
    for j, sensor in enumerate(SENSORS):
        frame[sensor] = np.arange(n, dtype=float) + j
    if labels:
        frame["anomaly"] = np.zeros(n, dtype=int) if anomaly is None else np.asarray(anomaly)
        frame["changepoint"] = np.isin(np.arange(n), cps).astype(int)
    return frame


class SkabWindowsTests(unittest.TestCase):
    def test_exact_L_plus_H_one_window_and_target_bounds(self):
        windows = rolling_windows(fixture(8), 5, 3)
        self.assertEqual(windows.t.tolist(), [4])
        self.assertEqual(windows.start.tolist(), [0])
        self.assertEqual(windows.stop.tolist(), [8])
        self.assertTrue(windows.strict_1s.all())

    def test_too_short_and_invalid_lengths(self):
        self.assertTrue(rolling_windows(fixture(7), 5, 3).empty)
        for config in ((0, 3, 8), (5, -1, 8), (5, 3, 0)):
            with self.assertRaises(ValueError):
                rolling_windows(fixture(), *config)

    def test_anchor_never_restarts_at_cp_or_gap(self):
        frame = fixture(28, cps=(5, 18), elapsed=np.r_[np.arange(12), np.arange(12, 28) + 3])
        self.assertEqual(rolling_windows(frame, 4, 2).t.tolist(), [3, 11, 19])

    def test_gap_context_target_join_and_last_target_interval(self):
        for bad_interval in (1, 4, 6):
            elapsed = np.arange(8)
            elapsed[bad_interval + 1:] += 1
            self.assertFalse(rolling_windows(fixture(8, elapsed=elapsed), 5, 3).strict_1s.iloc[0])

    def test_bad_interval_outside_window_not_counted(self):
        elapsed = np.arange(24)
        elapsed[8:] += 5
        windows = rolling_windows(fixture(24, elapsed=elapsed), 5, 3)
        self.assertEqual(windows.strict_1s.tolist(), [True, True, True])

    def test_duplicate_backward_and_subsecond_are_not_1s(self):
        for times in ([0, 1, 1, 2], [0, 1, 0, 1], [0, 1, 2.5, 3.5]):
            self.assertFalse(rolling_windows(fixture(4, elapsed=times), 2, 2).strict_1s.iloc[0])

    def test_cp_first_point_allowed_last_point_forbidden(self):
        self.assertTrue(rolling_windows(fixture(8, cps=(0,)), 5, 3).same_cp_segment.iloc[0])
        for mark in (1, 4, 5, 7):
            self.assertFalse(rolling_windows(fixture(8, cps=(mark,)), 5, 3).same_cp_segment.iloc[0])

    def test_no_cp_does_not_imply_anomaly0(self):
        frame = fixture(8, anomaly=np.ones(8))
        window = rolling_windows(frame, 5, 3).iloc[0]
        self.assertTrue(window.same_cp_segment)
        self.assertFalse(window.all_anomaly0)

    def test_anomaly0_does_not_imply_no_cp(self):
        window = rolling_windows(fixture(8, cps=(3,)), 5, 3).iloc[0]
        self.assertFalse(window.same_cp_segment)
        self.assertTrue(window.all_anomaly0)

    def test_normal_without_labels_is_not_filled_with_zeros(self):
        windows = rolling_windows(fixture(8, labels=False), 5, 3)
        self.assertTrue(windows.all_anomaly0.isna().all())
        self.assertTrue(windows.same_cp_segment.isna().all())
        counts = window_counts(windows)
        self.assertEqual((counts["row_order"], counts["strict_1s"], counts["anomaly0"]), (1, 1, 0))

    def test_randomized_prefix_counts_equal_direct_slicing(self):
        rng = np.random.default_rng(126)
        for n in (1, 20, 200, 600):
            frame = fixture(n, cps=np.flatnonzero(rng.random(n) < .03),
                            anomaly=rng.integers(0, 2, n), elapsed=np.cumsum(rng.choice([1, 1, 1, 2], n)))
            for length, horizon in ((1, 1), (12, 3), (128, 16), (512, 64)):
                self.assertEqual(window_counts(rolling_windows(frame, length, horizon)),
                                 brute_window_counts(frame, length, horizon))


class SkabGeometryTests(unittest.TestCase):
    def test_adjacent_boundaries_exact_lengths(self):
        geometry = event_geometry(fixture(32, cps=(8, 20)), 8, 4)
        self.assertEqual(geometry.before_rows.tolist(), [8, 12])
        self.assertEqual(geometry.after_rows.tolist(), [12, 12])
        self.assertTrue(geometry.L_before_LH_after.all())
        self.assertTrue(geometry.L_before_LH_after_strict.all())
        self.assertEqual(geometry.left_file_censored.tolist(), [True, False])
        self.assertEqual(geometry.right_file_censored.tolist(), [False, True])

    def test_short_previous_segment_and_next_cp_exclusion(self):
        geometry = event_geometry(fixture(20, cps=(5, 8, 19)), 5, 4)
        self.assertEqual(geometry.L_before_H_after.tolist(), [False, False, False])
        self.assertEqual(geometry.before_rows.tolist(), [5, 3, 11])
        self.assertEqual(geometry.after_rows.tolist(), [3, 11, 1])

    def test_gap_at_cp_blocks_strict_event(self):
        elapsed = np.r_[np.arange(8), np.arange(8, 24) + 4]
        geometry = event_geometry(fixture(24, cps=(8,), elapsed=elapsed), 8, 4)
        self.assertTrue(geometry.L_before_LH_after.iloc[0])
        self.assertFalse(geometry.L_before_H_after_strict.iloc[0])

    def test_cp_at_zero_no_empty_segment(self):
        self.assertEqual(bounds_from_cuts(10, np.array([0, 5])), [(0, 5), (5, 10)])
        geometry = event_geometry(fixture(10, cps=(0, 5)), 5, 2)
        self.assertFalse(geometry.L_before.iloc[0])
        self.assertEqual(geometry.before_rows.tolist(), [0, 5])


class SkabSummaryTests(unittest.TestCase):
    def test_cross_file_shared_samples_and_label_conflicts(self):
        left = fixture(20)
        right = left.iloc[10:].copy()
        right.loc[10, "anomaly"] = 1
        right.loc[11, SENSORS[0]] += 1
        table = cross_file_overlaps({"left": left, "right": right, "unrelated": fixture(3, elapsed=[100, 101, 102])})
        self.assertEqual(len(table), 1)
        self.assertEqual(table.shared_timestamp_pairs.iloc[0], 10)
        self.assertEqual(table.identical_8sensor_timestamp_pairs.iloc[0], 9)
        self.assertEqual(table.anomaly_conflicts_on_identical_rows.iloc[0], 1)

    def test_cross_file_missing_labels_not_claimed_comparable(self):
        table = cross_file_overlaps({"left": fixture(4), "normal": fixture(4, labels=False)})
        self.assertFalse(table.anomaly_labels_comparable.iloc[0])
        self.assertIsNone(table.anomaly_conflicts_on_identical_rows.iloc[0])

    def test_empty_normal_event_tables_preserve_integer_row_indices(self):
        from scripts.audit_skab import combine_tables

        _, normal = audit_frame(fixture(20, labels=False), "normal.csv")
        _, labeled = audit_frame(fixture(20, cps=(6, 12)), "labeled.csv")
        combined = combine_tables([normal["event_geometry"], labeled["event_geometry"]])
        self.assertTrue(pd.api.types.is_integer_dtype(combined.row))
        self.assertTrue(pd.api.types.is_integer_dtype(combined.L))
        self.assertTrue(pd.api.types.is_bool_dtype(combined.L_before_H_after))

    def test_cp_marks_not_independent_anomaly_onsets(self):
        frame = fixture(20, cps=(4, 6, 10, 13), anomaly=[0] * 6 + [1] * 7 + [0] * 7)
        summary, tables = audit_frame(frame, "data/other/1.csv")
        self.assertEqual((summary["cp_marks"], summary["anomaly_onsets"], summary["anomaly_offsets"]), (4, 1, 1))
        self.assertEqual(tables["cp_marks"].at_onset.tolist(), [False, True, False, False])
        self.assertEqual(tables["cp_marks"].at_offset.tolist(), [False, False, False, True])
        self.assertEqual(tables["anomaly_runs"].rows.tolist(), [6, 7, 7])
        self.assertEqual(summary["prehistory_rows"], 6)

    def test_contiguous_cp_marks_count_as_marks_not_new_physical_events(self):
        summary, _ = audit_frame(fixture(20, cps=(4, 5, 6)), "test.csv")
        self.assertEqual(summary["cp_marks"], 3)
        self.assertEqual(summary["cp_0_to_1_runs"], 1)

    def test_initial_anomaly_is_not_observed_onset(self):
        summary, _ = audit_frame(fixture(20, anomaly=np.ones(20)), "test.csv")
        self.assertEqual(summary["anomaly_onsets"], 0)
        self.assertEqual(summary["prehistory_rows"], 0)

    def test_missing_labels_stay_missing(self):
        summary, tables = audit_frame(fixture(20, labels=False), "normal.csv")
        self.assertIsNone(summary["anomaly0_rows"])
        self.assertIsNone(summary["first_anomaly_row"])
        self.assertTrue(tables["anomaly_runs"].empty)
        self.assertEqual(tables["cp_segments"].rows.sum(), 20)

    def test_nonfinite_constant_and_equal_run(self):
        frame = fixture(20)
        frame[SENSORS[0]] = 4.
        frame.loc[2, SENSORS[1]] = np.nan
        frame.loc[3, SENSORS[1]] = np.inf
        frame.loc[4, SENSORS[1]] = -np.inf
        summary, tables = audit_frame(frame, "test.csv")
        self.assertEqual(summary["nonfinite_sensor_values"], 3)
        self.assertEqual(summary["constant_sensors"], 1)
        self.assertEqual(tables["columns"].longest_equal_run_rows.iloc[0], 20)
        self.assertEqual(longest_equal_run(np.array([1, 1, np.nan, 1, 1, 1])), 3)

    def test_gap_duplicate_reverse_histogram(self):
        summary, tables = audit_frame(fixture(6, elapsed=[0, 1, 1, 5, 4, 5]), "test.csv")
        self.assertEqual(summary["non1s_steps"], 3)
        self.assertEqual(summary["duplicate_times"], 2)
        self.assertEqual(summary["backward_steps"], 1)
        self.assertEqual(summary["max_continuous_1s_rows"], 2)
        self.assertEqual(tables["continuous_runs"].rows.sum(), 6)
        self.assertEqual(tables["time_gaps"].row.tolist(), [2, 3, 4])
        self.assertEqual(tables["dt_histogram"]["count"].sum(), 5)
        self.assertFalse(continuous(time_ns(fixture(6, elapsed=[0, 1, 1, 5, 4, 5])), 0, 6))

    def test_tail_is_fixed_128_and_constant_ratio_undefined(self):
        frame = fixture(150)
        frame[SENSORS[1]] = 4.
        table = tail_diagnostics(frame, [(0, 150)]).set_index("sensor")
        self.assertEqual(table.loc[SENSORS[0], "start"], 22)
        self.assertAlmostEqual(table.loc[SENSORS[0], "delta"], 64.)
        self.assertAlmostEqual(table.loc[SENSORS[0], "abs_mean_difference_over_std"],
                               64 / np.std(np.arange(128)))
        self.assertTrue(table.loc[SENSORS[1], "constant_tail"])
        self.assertTrue(np.isnan(table.loc[SENSORS[1], "abs_mean_difference_over_std"]))
        self.assertTrue(tail_diagnostics(frame, [(0, 127)]).empty)


class SkabInputTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        (self.root / "data").mkdir()
        self.path = self.root / "data/test.csv"

    def test_schema_missing_labels_and_flow_name(self):
        fixture(3, labels=False).to_csv(self.path, sep=";", index=False)
        self.assertEqual(len(load_csv(self.path)), 3)
        self.assertIn("Volume Flow RateRMS", load_csv(self.path))
        frame = fixture(3).rename(columns={SENSORS[-1]: "RateRMS"})
        frame.to_csv(self.path, sep=";", index=False)
        with self.assertRaisesRegex(ValueError, "schema"):
            load_csv(self.path)

    def test_invalid_labels_and_time_rejected(self):
        for col, value in (("anomaly", 2), ("changepoint", np.nan), ("datetime", pd.NaT)):
            frame = fixture(3)
            frame.loc[1, col] = value
            frame.to_csv(self.path, sep=";", index=False)
            with self.assertRaises(ValueError):
                load_csv(self.path)

    def test_malformed_physical_rows_not_silently_skipped(self):
        header = ";".join(["datetime", *SENSORS])
        for invalid in ("", "2020-01-01 00:00:00;1", ";".join(["0"] * 12)):
            self.path.write_text(header + "\n" + invalid + "\n")
            with self.assertRaises(ValueError):
                load_csv(self.path)

    def test_manifest_hashes_and_tampering(self):
        fixture(3).to_csv(self.path, sep=";", index=False)
        raw = self.path.read_bytes()
        record = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                      git_blob_sha1=hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest())
        manifest = dict(commit=COMMIT, files={"data/test.csv": record})
        (self.root / "manifest.json").write_text(json.dumps(manifest))
        self.assertTrue(verify_originals(self.root, pinned=False).verified.all())
        self.path.write_bytes(raw + b"\n")
        with self.assertRaisesRegex(ValueError, "integrity"):
            verify_originals(self.root, pinned=False)

    def test_inventory_extra_csv_rejected(self):
        (self.root / "manifest.json").write_text(json.dumps(dict(commit=COMMIT, files={})))
        self.path.write_text("extra")
        with self.assertRaisesRegex(ValueError, "inventory"):
            verify_originals(self.root, pinned=False)


if __name__ == "__main__":
    unittest.main()