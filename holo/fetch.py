"""Download HoloAssist open annotations (labels + label2idx + splits). No video, no login."""
from __future__ import annotations

import urllib.request

from . import config as C


def _get(url, out):
    if out.exists() and out.stat().st_size > 0:
        print("skip (exists)", out.name)
        return
    print("download", url)
    urllib.request.urlretrieve(url, out)
    print("  ->", out.name, out.stat().st_size, "bytes")


def main() -> None:
    _get(C.LABELS_URL, C.LABELS_JSON)
    _get(C.LABEL2IDX_URL, C.LABEL2IDX_JSON)
    for s, url in C.SPLIT_URLS.items():
        _get(url, C.RAW / f"{s}.txt")


if __name__ == "__main__":
    main()
