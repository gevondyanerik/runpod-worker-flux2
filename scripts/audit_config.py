#!/usr/bin/env python3
"""Fail if anything outside ``app/config.py`` reads the environment.

The worker's whole configuration story rests on one claim: there is exactly one
place where a deployment can influence behaviour. That claim is worth nothing
if it is only documented, so CI checks it. A stray ``os.getenv`` in a feature
branch fails the build rather than quietly growing the configuration surface.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDITED = ("app", "bootstrap")

# ``comfy_client`` copies the environment into the ComfyUI subprocess and never
# reads a worker setting from it; ``models``/``preflight`` use ``os`` for paths
# and sysconf. Each exemption is by file, and each one is deliberate.
ALLOWED = {
    "app/config.py",  # the one place that may
    "app/comfy_client.py",  # dict(os.environ) for the subprocess
}

MARKERS = ("os.environ", "os.getenv")


def main() -> int:
    offenders: list[str] = []
    for package in AUDITED:
        for path in sorted((ROOT / package).rglob("*.py")):
            relative = path.relative_to(ROOT).as_posix()
            if relative in ALLOWED:
                continue
            text = path.read_text()
            for number, line in enumerate(text.splitlines(), start=1):
                if any(marker in line for marker in MARKERS):
                    offenders.append(f"{relative}:{number}: {line.strip()}")

    if offenders:
        print("environment access outside app/config.py:", file=sys.stderr)
        for offender in offenders:
            print(f"  {offender}", file=sys.stderr)
        return 1

    print(f"config audit ok ({', '.join(AUDITED)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
