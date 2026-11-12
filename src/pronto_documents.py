"""Чтение малых документов из большого ZIP, без загрузки сигналов PRONTO."""

import io
import posixpath
from datetime import datetime, timedelta
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET
from zipfile import ZipFile


class RangeReader(io.RawIOBase):
    """Seekable HTTP-файл: каждый ответ обязан быть точным диапазоном 206."""

    def __init__(self, url, size, opener=urlopen, limit=16 * 1024 * 1024):
        self.url, self.size, self.opener, self.limit = url, size, opener, limit
        self.position = 0
        self.transferred = 0
        self.requests = []
        self.cache = []

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        base = {0: 0, 1: self.position, 2: self.size}[whence]
        position = base + offset
        if position < 0:
            raise ValueError("Negative seek")
        self.position = position
        return position

    def read(self, size=-1):
        start = self.position
        stop = self.size if size < 0 else min(self.size, start + size)
        if stop <= start:
            return b""
        if stop - start > self.limit:
            raise ValueError("Refusing large range: full archive is not requested")
        for cached_start, content in self.cache:
            if cached_start <= start and stop <= cached_start + len(content):
                self.position = stop
                return content[start - cached_start:stop - cached_start]
        request = Request(self.url, headers={
            "Range": f"bytes={start}-{stop - 1}", "Accept-Encoding": "identity",
            "User-Agent": "TSFM-research-document-audit/1.0",
        })
        with self.opener(request, timeout=60) as response:
            expected = f"bytes {start}-{stop - 1}/{self.size}"
            if response.status != 206 or response.headers.get("Content-Range") != expected:
                raise ValueError("Server did not honor exact HTTP Range; download stopped")
            content = response.read(stop - start + 1)
        if len(content) != stop - start:
            raise ValueError("Truncated or overlong HTTP range")
        self.requests.append({"start": start, "end_inclusive": stop - 1})
        self.transferred += len(content)
        self.cache.append((start, content))
        self.position = stop
        return content


NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def excel_display(value, kind, format_code, date1904=False):
    """Даты/время отображаются отдельно от исходного значения Excel, без TZ."""
    if kind in ("s", "inlineStr", "str", "b", "e") or not value:
        return value
    try:
        number = float(value)
    except ValueError:
        return value
    code = format_code.lower()
    if code in ("0%", "0.00%"):
        return f"{number * 100:.{2 if '.00' in code else 0}f}%"
    # Только известные форматы времени/даты, а не любой числовой стиль.
    if not any(c in code for c in ("yy", "dd", "hh", "ss", "h:", "m:", "[h]")):
        return value
    if "[h]" in code:
        seconds = round(number * 86400)
        return f"{seconds // 3600}:{seconds // 60 % 60:02}:{seconds % 60:02}"
    # Excel содержит фиктивное 29.02.1900 (serial60). Не переводим его молча.
    if not date1904 and 60 <= number < 61:
        raise ValueError("Excel fictitious date 1900-02-29")
    base = datetime(1904, 1, 1) if date1904 else datetime(1899, 12, 30)
    if not date1904 and 1 <= number < 60:
        base += timedelta(days=1)
    stamp = base + timedelta(seconds=round(number * 86400))
    return stamp.isoformat(sep=" ") if number >= 1 else stamp.strftime("%H:%M:%S")


def workbook_cells(content):
    """Извлекает все непустые ячейки XLSX с адресом, raw, формулой и стилем.

    Формулы не исполняются; cached value явно отделено от формулы. Объединённые
    ячейки не распространяются автоматически на другие строки.
    """
    builtin = {0: "General", 1: "0", 2: "0.00", 3: "#,##0", 4: "#,##0.00",
               9: "0%", 10: "0.00%", 11: "0.00E+00", 12: "# ?/?", 13: "# ??/??",
               14: "mm-dd-yy", 15: "d-mmm-yy", 16: "d-mmm", 17: "mmm-yy",
               18: "h:mm AM/PM", 19: "h:mm:ss AM/PM", 20: "h:mm", 21: "h:mm:ss",
               22: "m/d/yy h:mm", 45: "mm:ss", 46: "[h]:mm:ss", 47: "mmss.0"}
    cells, sheets = [], []
    with ZipFile(io.BytesIO(content)) as book:
        names = set(book.namelist())
        if sum(info.file_size for info in book.infolist()) > 64 * 1024 * 1024:
            raise ValueError("Workbook XML too large")
        shared = []
        if "xl/sharedStrings.xml" in names:
            shared = ["".join(item.itertext()) for item in
                      ET.fromstring(book.read("xl/sharedStrings.xml")).findall("s:si", NS)]
        formats, styles = builtin.copy(), [0]
        if "xl/styles.xml" in names:
            root = ET.fromstring(book.read("xl/styles.xml"))
            formats.update({int(n.attrib["numFmtId"]): n.attrib["formatCode"]
                            for n in root.findall("s:numFmts/s:numFmt", NS)})
            styles = [int(n.attrib.get("numFmtId", 0))
                      for n in root.findall("s:cellXfs/s:xf", NS)]
        relations = {node.attrib["Id"]: node.attrib["Target"] for node in
                     ET.fromstring(book.read("xl/_rels/workbook.xml.rels"))
                     if node.attrib.get("TargetMode") != "External"}
        root = ET.fromstring(book.read("xl/workbook.xml"))
        properties = root.find("s:workbookPr", NS)
        date1904 = properties is not None and properties.attrib.get("date1904") in ("1", "true")
        for sheet in root.findall("s:sheets/s:sheet", NS):
            name = sheet.attrib["name"]
            target = relations[sheet.attrib[f"{{{REL}}}id"]]
            path = target.lstrip("/") if target.startswith("/") else posixpath.normpath("xl/" + target)
            if not path.startswith("xl/") or path not in names:
                raise ValueError("Unsafe or missing worksheet target")
            data = ET.fromstring(book.read(path))
            merged = [item.attrib["ref"] for item in data.findall("s:mergeCells/s:mergeCell", NS)]
            sheets.append({"sheet": name, "state": sheet.attrib.get("state", "visible"),
                           "merged_cells": merged, "date1904": date1904})
            for cell in data.findall("s:sheetData/s:row/s:c", NS):
                kind = cell.attrib.get("t", "n")
                raw = cell.findtext("s:v", default="", namespaces=NS)
                value = raw
                if kind == "s":
                    value = shared[int(raw)]
                elif kind == "inlineStr":
                    value = "".join(cell.find("s:is", NS).itertext())
                style = int(cell.attrib.get("s", 0))
                fmt = formats.get(styles[style], f"unresolved-builtin-{styles[style]}")
                formula = cell.findtext("s:f", default="", namespaces=NS)
                if value or formula:
                    cells.append({"sheet": name, "cell": cell.attrib["r"], "type": kind,
                                  "raw": raw, "value": value, "format": fmt,
                                  "display": excel_display(value, kind, fmt, date1904),
                                  "formula": formula})
    return cells, sheets