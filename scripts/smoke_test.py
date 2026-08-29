#!/usr/bin/env python3
"""Exercise a deployed endpoint end to end.

    python scripts/smoke_test.py <endpoint-id> --api-key $RUNPOD_API_KEY

Runs the three requests that between them cover everything a caller can do:
capabilities, text-to-image, and an edit with a reference image. Images are
written to ./smoke-output so the result can be looked at, because "returned
200" is not the same as "produced the right picture".
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "reference_bag.png"
OUTPUT = Path("smoke-output")

BASE = "https://api.runpod.ai/v2/{endpoint}/runsync"


def call(endpoint: str, api_key: str, payload: dict, timeout: int) -> dict:
    request = urllib.request.Request(
        BASE.format(endpoint=endpoint),
        data=json.dumps({"input": payload}).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        return {"status": "HTTP_ERROR", "error": f"{exc.code} {exc.read()[:400]!r}"}


def save(result: dict, name: str) -> None:
    images = (result.get("output") or {}).get("images") or []
    OUTPUT.mkdir(exist_ok=True)
    for image in images:
        if "b64" in image:
            suffix = image["mime_type"].split("/")[-1]
            path = OUTPUT / f"{name}-{image['index']}.{suffix}"
            path.write_bytes(base64.b64decode(image["b64"]))
            print(f"    wrote {path}")
        elif "url" in image:
            print(f"    uploaded to {image['url']}")


def report(name: str, result: dict, elapsed: float) -> bool:
    output = result.get("output") or {}
    if result.get("status") == "COMPLETED" and "error" not in output:
        print(f"  PASS  {name}  ({elapsed:.1f}s)")
        if "timings_ms" in output:
            print(f"    timings: {output['timings_ms']}")
        save(result, name)
        return True

    error = output.get("error") or result.get("error") or result
    print(f"  FAIL  {name}  ({elapsed:.1f}s)")
    print(f"    {json.dumps(error)[:500]}")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("endpoint", help="Runpod endpoint id")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("RUNPOD_API_KEY", ""),
        help="defaults to $RUNPOD_API_KEY",
    )
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    if not args.api_key:
        print("no API key: pass --api-key or set RUNPOD_API_KEY", file=sys.stderr)
        return 2

    reference = (
        "data:image/png;base64," + base64.b64encode(FIXTURE.read_bytes()).decode()
    )

    cases = [
        ("capabilities", {"op": "capabilities"}),
        (
            "text-to-image",
            {
                "prompt": "a red bicycle leaning against a white wall, soft daylight",
                "width": 768,
                "height": 768,
                "seed": 42,
            },
        ),
        (
            "image-edit",
            {
                "prompt": "change the bag colour to deep blue, keep everything else",
                "images": [reference],
                "seed": 7,
            },
        ),
    ]

    passed = 0
    for name, payload in cases:
        print(f"\n{name}...")
        started = time.monotonic()
        result = call(args.endpoint, args.api_key, payload, args.timeout)
        if report(name, result, time.monotonic() - started):
            passed += 1

    print(f"\n{passed}/{len(cases)} passed")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
