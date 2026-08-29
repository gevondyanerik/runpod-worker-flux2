#!/usr/bin/env python3
"""Keep the three descriptions of the configuration surface in agreement.

``app/config.py`` is the truth. ``.env.example`` is what a self-hoster reads,
``.runpod/hub.json`` is what a Hub deployer sees. Documentation that drifts
from the code is worse than no documentation, because it is trusted — so this
runs in CI rather than living as a convention.

``FLUX2_BAKED_VARIANT`` is exempt: the Dockerfile writes it into the image, and
it is not an operator setting.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Written by the image, not by an operator: it belongs in neither document.
IMAGE_ONLY = {"FLUX2_BAKED_VARIANT"}


def from_config() -> set[str]:
    source = (ROOT / "app" / "config.py").read_text()
    names = set(re.findall(r'_env[a-z_]*\(\s*"([A-Z0-9_]+)"', source))
    # The three asset overrides are built from a prefix, so they never appear
    # as literals.
    for prefix in ("DIFFUSION_MODEL", "TEXT_ENCODER", "VAE"):
        names.discard(f"{prefix}_REPO")
        names.discard(f"{prefix}_FILE")
        names |= {f"{prefix}_REPO", f"{prefix}_FILE"}
    return names - IMAGE_ONLY


def from_env_example() -> set[str]:
    source = (ROOT / ".env.example").read_text()
    return set(re.findall(r"^#?\s*([A-Z0-9_]+)=", source, flags=re.MULTILINE))


def from_hub() -> set[str]:
    config = json.loads((ROOT / ".runpod" / "hub.json").read_text())
    return {entry["key"] for entry in config["config"]["env"]}


def report(label: str, missing: set[str], extra: set[str]) -> bool:
    ok = True
    if missing:
        print(f"{label}: missing {', '.join(sorted(missing))}", file=sys.stderr)
        ok = False
    if extra:
        print(f"{label}: documents unknown {', '.join(sorted(extra))}", file=sys.stderr)
        ok = False
    return ok


def main() -> int:
    config = from_config()
    example = from_env_example()
    hub = from_hub()

    ok = True
    ok &= report(".env.example", config - example, example - config)
    ok &= report(".runpod/hub.json", config - hub, hub - config)

    if not ok:
        print(
            "\napp/config.py is the source of truth; update the documents to match it.",
            file=sys.stderr,
        )
        return 1

    print(f"env audit ok ({len(config)} variables, all optional)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
