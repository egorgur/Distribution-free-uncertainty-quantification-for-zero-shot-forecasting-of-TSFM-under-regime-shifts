"""Tiny local ZIPs and mocked HTTP responses; no network or external packages."""

import hashlib
import io
import json
import tempfile
import unittest
import warnings
import zipfile
import zlib
from datetime import datetime
from http.client import IncompleteRead
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError

from scripts import download_gas_data as gas


PAYLOADS = {
    "ethylene_CO.txt": b"Time CO ethylene\r\n0 1 2\r\n",
    "ethylene_methane.txt": b"Time methane ethylene\n0 3 4\n",
}


def tiny_zip(entries=None):
    buffer = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # Deliberate duplicate fixture.
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
            for name, payload in (PAYLOADS.items() if entries is None else entries):
                archive.writestr(name, payload)
    return buffer.getvalue()


class Response(io.BytesIO):
    def __init__(self, payload, *, status=200, headers=None):
        super().__init__(payload)
        self.status = status
        self.headers = {"Content-Length": str(len(payload)), **(headers or {})}

    def getcode(self):
        return self.status


class GasDownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        self.archive = self.out / gas.ARCHIVE_NAME

    def write_archive(self, payload=None):
        self.archive.write_bytes(tiny_zip() if payload is None else payload)

    def test_streamed_checksum(self):
        payload = b"abc" * 100
        path = self.out / "sample"
        path.write_bytes(payload)
        with patch.object(gas, "CHUNK_BYTES", 7):
            self.assertEqual(gas.sha256_file(path), hashlib.sha256(payload).hexdigest())

    def test_expected_members_and_original_bytes(self):
        self.write_archive()
        self.assertEqual([m.filename for m in gas.validate_archive(self.archive)],
                         list(gas.EXPECTED_FILES))
        records = gas.extract_archive(self.archive, self.out)
        for record in records:
            payload = PAYLOADS[record["name"]]
            self.assertEqual((self.out / record["name"]).read_bytes(), payload)
            self.assertEqual(record["bytes"], len(payload))
            self.assertEqual(record["sha256"], hashlib.sha256(payload).hexdigest())
            self.assertEqual(record["zip_crc32"], f"{zlib.crc32(payload):08x}")
        self.assertEqual(list(self.out.glob("*.part")), [])

    def test_safe_folders_are_flattened_and_directory_entries_allowed(self):
        entries = [("folder/", b"")]
        entries += [(f"folder/{name}", data) for name, data in PAYLOADS.items()]
        self.write_archive(tiny_zip(entries))
        gas.extract_archive(self.archive, self.out)
        self.assertFalse((self.out / "folder").exists())
        for name in PAYLOADS:
            self.assertTrue((self.out / name).is_file())

    def test_broken_zip_and_bad_crc(self):
        corrupt = bytearray(tiny_zip())
        corrupt[corrupt.index(PAYLOADS["ethylene_CO.txt"])] ^= 1
        for payload in (b"not a ZIP", bytes(corrupt)):
            with self.subTest(payload=payload[:20]):
                self.write_archive(payload)
                with self.assertRaises(zipfile.BadZipFile):
                    gas.extract_archive(self.archive, self.out)
                self.assertFalse((self.out / "ethylene_CO.txt").exists())

    def test_missing_unexpected_duplicate_and_unsafe_members_rejected(self):
        valid = list(PAYLOADS.items())
        cases = [valid[:1], valid + [("readme.txt", b"extra")],
                 valid + [valid[0]], valid + [("folder/ethylene_CO.txt", b"duplicate")]]
        for name in ("../ethylene_CO.txt", "/ethylene_CO.txt",
                     "folder/../../ethylene_CO.txt", "..\\ethylene_CO.txt",
                     "C:/ethylene_CO.txt", "../"):
            cases.append(valid + [(name, b"unsafe")])
        for entries in cases:
            with self.subTest(names=[name for name, _ in entries]):
                self.write_archive(tiny_zip(entries))
                with self.assertRaises(ValueError):
                    gas.extract_archive(self.archive, self.out)
                self.assertFalse((self.out / "ethylene_CO.txt").exists())
        self.assertFalse((self.out.parent / "ethylene_CO.txt").exists())

    def test_repeated_run_reuses_archive_and_matching_files(self):
        self.write_archive()
        with patch.object(gas, "urlopen", side_effect=AssertionError("No network")):
            first = gas.download_data(self.out)
            paths = [self.archive, *(self.out / name for name in PAYLOADS)]
            stats = [(p.stat().st_ino, p.stat().st_mtime_ns) for p in paths]
            second = gas.download_data(self.out)
        self.assertEqual(first["archive"], second["archive"])
        self.assertEqual(first["files"], second["files"])
        self.assertEqual(stats, [(p.stat().st_ino, p.stat().st_mtime_ns) for p in paths])

    def test_differing_existing_file_is_never_overwritten(self):
        self.write_archive()
        target = self.out / "ethylene_CO.txt"
        target.write_bytes(b"different local measurements")
        with self.assertRaisesRegex(ValueError, "differs"):
            gas.download_data(self.out)
        self.assertEqual(target.read_bytes(), b"different local measurements")
        self.assertFalse((self.out / "manifest.json").exists())
        self.assertEqual(list(self.out.glob("*.part")), [])

    def test_download_and_manifest_fields(self):
        payload = tiny_zip()
        response = Response(payload, headers={"ETag": '"test"', "Set-Cookie": "secret"})
        with patch.object(gas, "urlopen", return_value=response) as request:
            manifest = gas.download_data(self.out, timeout=12)
        self.assertEqual(request.call_args.kwargs["timeout"], 12)
        self.assertFalse(request.call_args.args[0].has_header("Range"))
        self.assertEqual(self.archive.read_bytes(), payload)
        self.assertEqual(set(manifest), {"url", "doi", "verified_at_utc", "archive",
                                        "files", "source_headers", "hash_note"})
        self.assertEqual(manifest["url"], gas.ARCHIVE_URL)
        self.assertEqual(manifest["doi"], "10.24432/C5WP4C")
        self.assertEqual(datetime.fromisoformat(manifest["verified_at_utc"]).utcoffset().total_seconds(), 0)
        self.assertEqual(manifest["archive"], {
            "name": gas.ARCHIVE_NAME, "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        })
        self.assertEqual(manifest["source_headers"], {
            "Content-Length": str(len(payload)), "ETag": '"test"',
        })
        self.assertIn("not trusted upstream", manifest["hash_note"])
        saved = json.loads((self.out / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(saved, manifest)
        self.assertEqual(list(self.out.glob("*.part")), [])

    def test_ignored_range_stale_partial_is_restarted_not_appended(self):
        payload = tiny_zip()
        part = self.archive.with_name(self.archive.name + ".part")
        part.write_bytes(b"stale incomplete content")
        with patch.object(gas, "urlopen", return_value=Response(payload)) as request:
            gas.download_archive(self.archive)
        self.assertFalse(request.call_args.args[0].has_header("Range"))
        self.assertEqual(self.archive.read_bytes(), payload)
        self.assertFalse(part.exists())

    def test_retry_after_truncation_restarts(self):
        payload = tiny_zip()
        responses = [Response(payload[:20], headers={"Content-Length": str(len(payload))}),
                     Response(payload)]
        with patch.object(gas, "urlopen", side_effect=responses) as request:
            gas.download_archive(self.archive, attempts=2)
        self.assertEqual(request.call_count, 2)
        self.assertEqual(self.archive.read_bytes(), payload)
        self.assertEqual(list(self.out.glob("*.part")), [])

    def test_interrupted_http_body_is_retried(self):
        payload = tiny_zip()
        interrupted = Response(payload)
        with patch.object(interrupted, "read", side_effect=IncompleteRead(b"partial")):
            with patch.object(gas, "urlopen", side_effect=[interrupted, Response(payload)]) as request:
                gas.download_archive(self.archive, attempts=2)
        self.assertEqual(request.call_count, 2)
        self.assertEqual(self.archive.read_bytes(), payload)
        self.assertEqual(list(self.out.glob("*.part")), [])

    def test_failures_are_bounded_and_leave_no_archive_or_part(self):
        for failure in (URLError("offline"), Response(b"broken ZIP"),
                        Response(tiny_zip(), status=206)):
            with self.subTest(failure=failure):
                # A fresh response per attempt (context managers close responses).
                if isinstance(failure, URLError):
                    effect = failure
                else:
                    content, status = failure.getvalue(), failure.status
                    effect = lambda *args, **kwargs: Response(content, status=status)
                with patch.object(gas, "urlopen", side_effect=effect) as request:
                    with self.assertRaises((URLError, ValueError, zipfile.BadZipFile)):
                        gas.download_archive(self.archive, attempts=2)
                self.assertEqual(request.call_count, 2)
                self.assertFalse(self.archive.exists())
                self.assertEqual(list(self.out.glob("*.part")), [])

    def test_invalid_existing_archive_is_not_replaced(self):
        self.write_archive(b"broken archive")
        with patch.object(gas, "urlopen", side_effect=AssertionError("No network")):
            with self.assertRaises(zipfile.BadZipFile):
                gas.download_archive(self.archive)
        self.assertEqual(self.archive.read_bytes(), b"broken archive")


if __name__ == "__main__":
    unittest.main()