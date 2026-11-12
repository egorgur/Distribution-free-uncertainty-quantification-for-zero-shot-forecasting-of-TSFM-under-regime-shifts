import unittest
import json

from src.uci447_audit import cycle_geometry, parse_profile, parse_ps1


class TestUci447Audit(unittest.TestCase):
    def test_profile_runs_and_flags(self):
        content = b"3\t100\t0\t130\t1\n3\t100\t0\t130\t1\n100\t100\t0\t90\t0\n"
        result = parse_profile(content)
        self.assertEqual(result["cycles"], 3)
        self.assertEqual(result["stable_flag_counts"], {"1": 2, "0": 1})
        self.assertEqual(len(result["profile_runs"]), 2)
        self.assertFalse(result["stationarity_established"])

    def test_profile_width_rejected(self):
        with self.assertRaises(ValueError):
            parse_profile(b"1\t2\t3\n")

    def test_profile_flag_rejected(self):
        with self.assertRaises(ValueError):
            parse_profile(b"1\t2\t3\t4\t2\n")

    def test_ps1_shape_and_finiteness(self):
        row = "\t".join(str(i) for i in range(4))
        result = parse_ps1(((row + "\n") * 2).encode(), expected_cycles=2, expected_samples=4)
        self.assertEqual(result["finite_values"], 8)
        self.assertEqual(result["cycle_stats"][0]["min"], 0.0)
        self.assertEqual(result["cycle_stats"][0]["max"], 3.0)
        self.assertFalse(result["continuity_between_cycles_established"])

    def test_json_roundtrip(self):
        result = parse_profile(b"3\t100\t0\t130\t1\n100\t100\t0\t90\t0\n")
        self.assertEqual(json.loads(json.dumps(result)), result)

    def test_empty_profile_rejected(self):
        with self.assertRaises(ValueError):
            parse_profile(b"")

    def test_blank_profile_row_rejected(self):
        with self.assertRaises(ValueError):
            parse_profile(b"3\t100\t0\t130\t1\n\n")

    def test_blank_signal_row_rejected(self):
        with self.assertRaises(ValueError):
            parse_ps1(b"1\t2\t3\t4\n\n", expected_cycles=1, expected_samples=4)

    def test_nonfinite_signal_rejected(self):
        for value in (b"nan", b"inf", b"-inf"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_ps1(b"1\t2\t3\t" + value, expected_cycles=1, expected_samples=4)

    def test_ps1_width_rejected(self):
        with self.assertRaises(ValueError):
            parse_ps1(b"1\t2\n", expected_cycles=1, expected_samples=4)

    def test_ps1_nonnumeric_rejected(self):
        with self.assertRaises(ValueError):
            parse_ps1(b"1\tnope\t3\t4\n", expected_cycles=1, expected_samples=4)

    def test_ps1_cycle_count_rejected(self):
        with self.assertRaises(ValueError):
            parse_ps1(b"1\t2\t3\t4\n", expected_cycles=2, expected_samples=4)

    def test_geometry_does_not_join_cycles(self):
        result = cycle_geometry()
        self.assertEqual(result[0]["full_windows_per_cycle_stride1"], 0)
        self.assertEqual(result[1]["full_windows_per_cycle_stride1"], 25)
        self.assertEqual(result[2]["full_windows_per_cycle_stride1"], 5425)

    def test_geometry_seconds(self):
        result = cycle_geometry(sample_rates=(100,), length=512, horizon=64)[0]
        self.assertAlmostEqual(result["full_window_seconds"], 5.76)
        self.assertAlmostEqual(result["length_seconds"], 5.12)


if __name__ == "__main__":
    unittest.main()