#!/usr/bin/env python3
"""Regenerate the committed graph golden files.

Run this deliberately, review the diff, and commit it. A graph change that
nobody looked at is exactly what the golden files exist to catch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.workflows.cases import GOLDEN_DIR, cases


def main() -> int:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    for name, graph in cases():
        path = GOLDEN_DIR / f"{name}.json"
        path.write_text(json.dumps(graph, indent=2, sort_keys=True) + "\n")
        print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
