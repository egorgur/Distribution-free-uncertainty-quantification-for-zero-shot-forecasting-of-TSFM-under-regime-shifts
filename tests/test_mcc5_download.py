import io
import unittest

from src.mcc5_download import BudgetRangeReader, MAX_RAW, safe_member, select_member


class Response(io.BytesIO):
    def __init__(self, content, status, content_range):
        super().__init__(content)
        self.status = status
        self.headers = {"Content-Range": content_range}


class Mcc5DownloadTests(unittest.TestCase):
    def test_total_budget_and_cache(self):
        calls = []
        def opener(request, timeout):
            a, b = map(int, request.get_header("Range")[6:].split("-"))
            calls.append((a, b))
            return Response(b"abcdefghij"[a:b + 1], 206, f"bytes {a}-{b}/10")
        reader = BudgetRangeReader("https://example.test/file", 10, opener, budget=5)
        self.assertEqual(reader.read(3), b"abc")
        reader.seek(0)
        self.assertEqual(reader.read(2), b"ab")
        reader.seek(3)
        with self.assertRaisesRegex(ValueError, "budget"):
            reader.read(3)
        self.assertEqual(calls, [(0, 2)])

    def test_200_rejected_before_body(self):
        class MustNotRead(Response):
            def read(self, *args):
                raise AssertionError("Body of full archive must not be consumed")
        reader = BudgetRangeReader("https://example.test/file", 100,
                                   lambda *a, **k: MustNotRead(b"x", 200, ""))
        with self.assertRaisesRegex(ValueError, "honor"):
            reader.read(1)

    def test_selection_is_metadata_only_and_sorted(self):
        records = [{"member": name, "size": 10, "compressed_size": 5} for name in (
            "z/Healthy_torque_circulation_20Nm_1000rpm.csv",
            "a/._Healthy_torque_circulation_20Nm_1000rpm.csv",
            "a/Healthy_torque_circulation_20Nm_1000rpm.csv",
            "a/Healthy_speed_circulation_20Nm_1000rpm.csv")]
        self.assertEqual(select_member(records)["member"], records[2]["member"])

    def test_no_candidate(self):
        with self.assertRaises(ValueError):
            select_member([])

    def test_oversize_member(self):
        with self.assertRaisesRegex(ValueError, "size limit"):
            select_member([{"member": "Healthy_torque_20Nm_1000rpm.csv",
                            "size": MAX_RAW + 1, "compressed_size": 2}])

    def test_unsafe_paths(self):
        for path in ("../a.csv", "/a.csv", "x\\a.csv"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                safe_member(path)


if __name__ == "__main__":
    unittest.main()