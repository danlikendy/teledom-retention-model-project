"""Download the TeleDom SQLite dump into data/."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

DATA_URL = "https://code.s3.yandex.net/data-scientist/ds-plus-final.db"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download TeleDom SQLite dump")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "ds-plus-final.db",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists() and not args.force:
        print(f"already present: {args.out}")
        return
    print("downloading…")
    urllib.request.urlretrieve(DATA_URL, args.out)
    print(f"ok: {args.out}")


if __name__ == "__main__":
    main()
