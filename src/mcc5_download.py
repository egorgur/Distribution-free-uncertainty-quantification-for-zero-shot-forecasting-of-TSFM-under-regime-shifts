"""Bounded single-member access to the pinned MCC5 torque ZIP."""

import hashlib
import json
from pathlib import PurePosixPath
from urllib.request import urlopen

from src.pronto_documents import RangeReader

COMMIT = "897bc5d2504980418249f98e733bfd7f61056186"
BASE = "https://huggingface.co/datasets/Samlzy/MCC5-THU-Motor"
TREE_URL = f"https://huggingface.co/api/datasets/Samlzy/MCC5-THU-Motor/tree/{COMMIT}?limit=100"
ARCHIVE_NAME = "MCC5-THU Motor_torque_circulation.zip"
ARCHIVE_URL = f"{BASE}/resolve/{COMMIT}/MCC5-THU%20Motor_torque_circulation.zip"
README_URL = f"{BASE}/raw/{COMMIT}/README.md"
MAX_NETWORK = 128 * 1024**2
MAX_COMPRESSED = 96 * 1024**2
MAX_RAW = 256 * 1024**2


class BudgetRangeReader(RangeReader):
    """Keep exact-206 checks and bound cumulative network transfer as well."""

    def __init__(self, url, size, opener=urlopen, budget=MAX_NETWORK):
        super().__init__(url, size, opener=opener)
        self.budget = budget

    def read(self, size=-1):
        start = self.position
        stop = self.size if size < 0 else min(self.size, start + size)
        cached = any(a <= start and stop <= a + len(b) for a, b in self.cache)
        if not cached and self.transferred + max(0, stop - start) > self.budget:
            raise ValueError("Cumulative range budget exceeded")
        return super().read(size)


def small_document(url, opener=urlopen):
    with opener(url, timeout=60) as response:
        content = response.read(2 * 1024**2 + 1)
    if len(content) > 2 * 1024**2:
        raise ValueError("Document exceeds 2 MiB")
    return content


def sha256(content):
    return hashlib.sha256(content).hexdigest()


def safe_member(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError("Unsafe archive path")
    return path


def select_member(inventory):
    """Selection uses only filename metadata, never the signal values."""
    candidates = []
    for item in inventory:
        path = safe_member(item["member"])
        lower = path.name.lower()
        if (lower.endswith(".csv") and not lower.startswith("._")
                and any(word in lower for word in ("healthy", "health", "normal"))
                and "torque" in lower and "20nm" in lower and "1000rpm" in lower):
            candidates.append(item)
    if not candidates:
        raise ValueError("No unambiguous healthy20Nm1000rpm torque file; inspect inventory")
    chosen = sorted(candidates, key=lambda item: item["member"])[0]
    if chosen["size"] > MAX_RAW or chosen["compressed_size"] > MAX_COMPRESSED:
        raise ValueError("Selected member exceeds the pre-recorded size limit")
    return chosen


def archive_metadata(tree_content):
    tree = json.loads(tree_content)
    matches = [item for item in tree if item.get("path") == ARCHIVE_NAME]
    if len(matches) != 1 or matches[0]["size"] != 6513549892:
        raise ValueError("Pinned archive metadata differs from source audit")
    return matches[0]