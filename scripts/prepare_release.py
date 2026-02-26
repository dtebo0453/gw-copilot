#!/usr/bin/env python3
"""Copy the Vite production build into gw/_frontend/ so it ships with the wheel.

Usage
-----
    # 1. Build the frontend first:
    cd ui && npm run build

    # 2. Then run this script from the project root:
    python scripts/prepare_release.py

The script will:
  - Verify ui/dist/ exists (tells you to build if not)
  - Remove any stale gw/_frontend/ directory
  - Copy ui/dist/* → gw/_frontend/
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "ui" / "dist"
DST = ROOT / "gw" / "_frontend"


def main() -> None:
    if not SRC.exists():
        print(f"ERROR: {SRC} does not exist.")
        print("       Run 'cd ui && npm run build' first, then re-run this script.")
        sys.exit(1)

    # Wipe stale copy
    if DST.exists():
        print(f"Removing old {DST} ...")
        shutil.rmtree(DST)

    print(f"Copying {SRC}  →  {DST} ...")
    shutil.copytree(SRC, DST)

    # Count files for a quick sanity check
    count = sum(1 for _ in DST.rglob("*") if _.is_file())
    print(f"Done — {count} files copied.")


if __name__ == "__main__":
    main()
