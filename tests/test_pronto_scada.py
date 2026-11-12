import csv
import io
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from src.pronto_scada import header_time, timestamp, seconds, parse_scada, journal_rows
from scripts import download_pronto_scada as download


def fixture(times, values=None):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerows([["Start Time", "09/11/2017 16:20", "End Time", "9/11/2017  16:21:00 PM", "Interval", "1s"],
                     ["", "sensor", "copy", "bad", "const", ""],
                     ["TIMESTAMP", "A", "B", "err", "C", ""]])
    for i, time in enumerate(times):
        writer.writerow([time, str(i), str(i), 'Tag error\nsecond line', "0", ""] if values is None else [time, *values])
    return stream.getvalue().encode()


class TestProntoScada(unittest.TestCase):
    def test_redundant_header_suffix(self):
        dt, kind = header_time("9/11/2017  17:00:00 PM")
        self.assertEqual(dt.hour, 17)
        self.assertIn("redundant", kind)

    def test_ambiguous_suffix_rejected(self):
        with self.assertRaises(ValueError):
            header_time("9/11/2017 05:00 PM")

    def test_bad_header_rejected(self):
        with self.assertRaises(ValueError):
            header_time("not a date")

    def test_minute_encoding(self):
        value, kind = timestamp("09/11/2017 16:20")
        self.assertEqual(kind, "text_minute")
        self.assertEqual(value, seconds(datetime(2017, 9, 11, 16, 20)))

    def test_unknown_time_not_repaired(self):
        for raw in ("nan", "inf", "bad", ""):
            self.assertEqual(timestamp(raw), (None, "unparsed"))

    def test_minute_grid(self):
        s, cols, runs = parse_scada(fixture(["09/11/2017 16:20"] * 60 + ["09/11/2017 16:21"]))
        self.assertTrue(s["nominal_grid_consistent"])
        self.assertTrue(s["nominal_endpoint_matches_header"])
        self.assertEqual(s["duplicate_raw_timestamps"], 59)
        self.assertEqual(s["backwards_parsed_steps"], 0)
        self.assertEqual(runs[0]["rows"], 61)
        self.assertEqual(cols[2]["nonnumeric"], 61)
        self.assertEqual(cols[4]["blank"], 61)
        self.assertEqual(s["duplicate_column_pairs"][0]["right_index"], 2)

    def test_coarse_label_cannot_cover_missing_minute(self):
        s, _, _ = parse_scada(fixture(["09/11/2017 16:20"] * 61))
        self.assertEqual(s["first_inconsistent_index"], 60)

    def test_mixed_formats_consistent(self):
        s, _, runs = parse_scada(fixture(["09/11/2017 16:20", "42989.68057"]))
        self.assertEqual(len(runs), 2)
        self.assertTrue(s["nominal_grid_consistent"])
        self.assertEqual(s["unparsed_timestamps"], 0)

    def test_missing_second_detected_in_serial(self):
        s, _, _ = parse_scada(fixture(["42989.68056", "42989.68058"]))
        self.assertFalse(s["nominal_grid_consistent"])

    def test_invalid_time_retained(self):
        s, _, runs = parse_scada(fixture(["broken"]))
        self.assertEqual(s["unparsed_timestamps"], 1)
        self.assertEqual(runs[0]["first_raw"], "broken")

    def test_nonnumeric_nonfinite_separate(self):
        s, cols, _ = parse_scada(fixture(["broken"], ["nan", "inf", "x", "1", ""]))
        self.assertEqual(cols[0]["nonfinite"], 1)
        self.assertEqual(cols[2]["nonnumeric"], 1)
        self.assertEqual(s["finite_numeric_columns"], 1)

    def test_unequal_widths_rejected(self):
        with self.assertRaises(ValueError):
            parse_scada(fixture(["broken"]) + b"x,2\n")

    def test_journal_does_not_fill_missing_time(self):
        cells = [{"sheet": "Operation Log 0912 ", "cell": cell, "display": value}
                 for cell, value in [("L30", "11:33:00"), ("M30", "Increase water flow rate"),
                                     ("M31", "Reduce air flow rate")]]
        rows, timeline = journal_rows(cells)
        self.assertEqual(len(rows), 2)
        self.assertEqual(timeline[1]["clock"], "")
        self.assertEqual(timeline[1]["time_cell"], "L31")

    def test_action_and_measurement_columns_separate(self):
        cells = [{"sheet": "Operation Log 0907", "cell": cell, "display": value}
                 for cell, value in [("C11", "12:57:00"), ("D11", "Start Labview measurements"),
                                     ("J11", "Waterflow rate:"), ("K11", "13:01:00")]]
        _, timeline = journal_rows(cells)
        self.assertEqual([r["clock"] for r in timeline], ["13:01:00", "12:57:00"])

    def test_header_endpoint_mismatch_not_silent(self):
        s, _, _ = parse_scada(fixture(["09/11/2017 16:20"]))
        self.assertFalse(s["nominal_endpoint_matches_header"])

    def test_serial_tolerance_boundary(self):
        for serial, ok in [("42989.68056", True), ("42989.680561", False)]:
            s, _, _ = parse_scada(fixture([serial]))
            self.assertEqual(s["nominal_grid_consistent"], ok)


class TestOriginalVerification(unittest.TestCase):
    def setUp(self):
        import zlib
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = self.root / "raw"
        self.raw.mkdir()
        data = fixture(["09/11/2017 16:20"])
        (self.raw / "day.csv").write_bytes(data)
        entry = {"member": "pronto/Process Data/day.csv", "local_file": "day.csv",
                 "size": len(data), "sha256": download.digest(data),
                 "crc32": f"{zlib.crc32(data):08x}"}
        self.manifest = {"files": [entry]}
        self.write_manifest()
        inventory = self.root / "results/pronto_documents_v2/archive_inventory.csv"
        inventory.parent.mkdir(parents=True)
        with inventory.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["member", "size", "crc32"])
            writer.writeheader()
            writer.writerow({k: entry[k] for k in writer.fieldnames})
        self.addCleanup(patch.stopall)
        patch.object(download, "ROOT", self.root).start()
        patch.object(download, "RAW", self.raw).start()

    def write_manifest(self):
        (self.raw / "manifest.json").write_text(json.dumps(self.manifest))

    def test_verified_fixture(self):
        self.assertTrue(download.verify()["passed"])

    def test_unmanifested_csv_rejected(self):
        (self.raw / "extra.csv").write_bytes((self.raw / "day.csv").read_bytes())
        with self.assertRaisesRegex(ValueError, "Unmanifested"):
            download.verify()

    def test_modified_original_rejected(self):
        (self.raw / "day.csv").write_bytes(b"corruption")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            download.verify()

    def test_modified_manifest_crc_rejected(self):
        self.manifest["files"][0]["crc32"] = "00000000"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "CRC mismatch"):
            download.verify()

    def test_missing_member_rejected(self):
        self.manifest["files"] = []
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "Member set"):
            download.verify()


if __name__ == "__main__":
    unittest.main()