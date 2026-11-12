import io
import unittest
from zipfile import ZIP_DEFLATED, ZipFile

from src.pronto_documents import RangeReader, excel_display, workbook_cells


class Response(io.BytesIO):
    def __init__(self, content, status, content_range):
        super().__init__(content)
        self.status = status
        self.headers = {"Content-Range": content_range}


class RangeTests(unittest.TestCase):
    def make_reader(self, content):
        def opener(request, timeout):
            self.assertEqual(timeout, 60)
            a, b = map(int, request.get_header("Range").removeprefix("bytes=").split("-"))
            return Response(content[a:b + 1], 206, f"bytes {a}-{b}/{len(content)}")
        return RangeReader("https://example.test/a.zip", len(content), opener)

    def test_seek_cache_eof(self):
        reader = self.make_reader(b"abcdefgh")
        reader.seek(-4, 2)
        self.assertEqual(reader.read(8), b"efgh")
        reader.seek(-3, 1)
        self.assertEqual(reader.read(2), b"fg")
        self.assertEqual(reader.transferred, 4)
        reader.seek(20)
        self.assertEqual(reader.read(), b"")
        with self.assertRaises(ValueError):
            reader.seek(-1)

    def test_full_response_rejected_before_reading(self):
        reader = RangeReader("https://example.test/a.zip", 20, lambda *a, **k: Response(b"x" * 20, 200, ""))
        with self.assertRaisesRegex(ValueError, "honor"):
            reader.read(3)
        self.assertEqual(reader.transferred, 0)

    def test_wrong_range_rejected(self):
        reader = RangeReader("https://example.test/a.zip", 20, lambda *a, **k: Response(b"abc", 206, "bytes 1-3/20"))
        with self.assertRaisesRegex(ValueError, "honor"):
            reader.read(3)

    def test_truncated_rejected(self):
        reader = RangeReader("https://example.test/a.zip", 20, lambda *a, **k: Response(b"ab", 206, "bytes 0-2/20"))
        with self.assertRaisesRegex(ValueError, "Truncated"):
            reader.read(3)

    def test_limit(self):
        reader = self.make_reader(b"abcdefgh")
        reader.limit = 3
        with self.assertRaises(ValueError):
            reader.read()

    def test_real_zip_over_ranges(self):
        buffer = io.BytesIO()
        with ZipFile(buffer, "w", ZIP_DEFLATED) as zipped:
            zipped.writestr("Operation log.xlsx", b"test workbook")
        reader = self.make_reader(buffer.getvalue())
        with ZipFile(reader) as zipped:
            self.assertEqual(zipped.read("Operation log.xlsx"), b"test workbook")


class ExcelTests(unittest.TestCase):
    def test_display_time_date_duration(self):
        self.assertEqual(excel_display("0.5", "n", "h:mm"), "12:00:00")
        self.assertEqual(excel_display("1.5", "n", "[h]:mm:ss"), "36:00:00")
        self.assertEqual(excel_display("2", "n", "mm-dd-yy"), "1900-01-02 00:00:00")
        self.assertEqual(excel_display("1", "n", "mm-dd-yy", True), "1904-01-02 00:00:00")

    def test_percent_and_fictitious_date(self):
        self.assertEqual(excel_display("0.2", "n", "0%"), "20%")
        self.assertEqual(excel_display("0.201", "n", "0.00%"), "20.10%")
        with self.assertRaisesRegex(ValueError, "fictitious"):
            excel_display("60", "n", "mm-dd-yy")

    def test_plain_values_not_dates(self):
        self.assertEqual(excel_display("120", "n", "0.00"), "120")
        self.assertEqual(excel_display("12:00", "str", "h:mm"), "12:00")

    def test_all_cells_hidden_sheet_and_formulas(self):
        buffer = io.BytesIO()
        ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
        with ZipFile(buffer, "w") as zipped:
            zipped.writestr("xl/workbook.xml", f'<workbook xmlns="{ns}" '
                            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
                            '<sheets><sheet name="Test11" sheetId="1" r:id="r1" state="hidden"/></sheets></workbook>')
            zipped.writestr("xl/_rels/workbook.xml.rels", '<Relationships><Relationship Id="r1" Target="worksheets/sheet1.xml"/></Relationships>')
            zipped.writestr("xl/sharedStrings.xml", f'<sst xmlns="{ns}"><si><t>normal</t></si></sst>')
            zipped.writestr("xl/styles.xml", f'<styleSheet xmlns="{ns}"><cellXfs><xf numFmtId="0"/><xf numFmtId="20"/></cellXfs></styleSheet>')
            zipped.writestr("xl/worksheets/sheet1.xml", f'<worksheet xmlns="{ns}"><sheetData><row r="1">'
                            '<c r="A1" t="s"><v>0</v></c><c r="B1" s="1"><v>0.5</v></c>'
                            '<c r="C1"><f>B1*2</f><v>1</v></c><c r="D1" t="inlineStr"><is><t>valve</t></is></c>'
                            '</row></sheetData><mergeCells><mergeCell ref="A1:A2"/></mergeCells></worksheet>')
        cells, sheets = workbook_cells(buffer.getvalue())
        self.assertEqual(len(cells), 4)
        self.assertEqual(cells[0]["display"], "normal")
        self.assertEqual(cells[1]["raw"], "0.5")
        self.assertEqual(cells[1]["display"], "12:00:00")
        self.assertEqual(cells[2]["formula"], "B1*2")
        self.assertEqual(cells[3]["display"], "valve")
        self.assertEqual(sheets[0]["state"], "hidden")
        self.assertEqual(sheets[0]["merged_cells"], ["A1:A2"])


if __name__ == "__main__":
    unittest.main()