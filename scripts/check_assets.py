#!/usr/bin/env python3
"""Check every pinned model file still exists, with the size and digest we recorded.

Run weekly in CI. A repository being renamed, a file being replaced, or a
revision disappearing all fail a deployment at cold start, in front of a paying
request. Finding out on a schedule is much cheaper.

Needs network access; no token, because every profile is public.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.variants import VARIANTS, Asset

API = "https://huggingface.co/api/models/{repo}?blobs=true"


def fetch(repo: str, attempts: int = 3) -> dict:
    """Read a repository's metadata, retrying transient network failures.

    Without this the weekly run reports "the model is gone" every time a TLS
    handshake times out, and a check that cries wolf stops being read.
    """
    request = urllib.request.Request(
        API.format(repo=repo), headers={"User-Agent": "runpod-worker-flux2"}
    )
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except Exception:
            if attempt == attempts:
                raise
            time.sleep(2 * attempt)
    raise AssertionError("unreachable")


def check(asset: Asset, metadata: dict) -> list[str]:
    problems: list[str] = []
    siblings = {s["rfilename"]: s for s in metadata.get("siblings", [])}

    entry = siblings.get(asset.path)
    if entry is None:
        return [f"{asset.repo}: {asset.path} no longer exists"]

    lfs = entry.get("lfs") or {}
    if lfs.get("size") and lfs["size"] != asset.size_bytes:
        problems.append(
            f"{asset.repo}/{asset.path}: size is {lfs['size']}, "
            f"registry says {asset.size_bytes}"
        )
    if asset.sha256 and lfs.get("sha256") and lfs["sha256"] != asset.sha256:
        problems.append(
            f"{asset.repo}/{asset.path}: digest is {lfs['sha256']}, "
            f"registry says {asset.sha256}"
        )

    licence = (metadata.get("cardData") or {}).get("license")
    if licence != asset.license_id:
        problems.append(
            f"{asset.repo}: licence is {licence!r}, registry says {asset.license_id!r}"
        )
    if metadata.get("gated"):
        problems.append(f"{asset.repo}: repository is now gated")

    return problems


def main() -> int:
    seen: dict[str, dict] = {}
    problems: list[str] = []
    checked = 0

    for variant in VARIANTS.values():
        for asset in variant.assets:
            if asset.repo not in seen:
                try:
                    seen[asset.repo] = fetch(asset.repo)
                except Exception as exc:
                    problems.append(f"{asset.repo}: cannot be read ({exc})")
                    seen[asset.repo] = {}
                    continue
            if not seen[asset.repo]:
                continue  # already reported as unreadable
            problems.extend(check(asset, seen[asset.repo]))
            checked += 1

    if problems:
        print("pinned assets have drifted:", file=sys.stderr)
        for problem in sorted(set(problems)):
            print(f"  {problem}", file=sys.stderr)
        return 1

    print(f"asset audit ok ({checked} references across {len(seen)} repositories)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
