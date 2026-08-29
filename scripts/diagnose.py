#!/usr/bin/env python3
"""Explain what this worker would do, from inside the container.

Run it over SSH on a pod, or as a one-off container command, when a deployment
misbehaves. It prints the resolved configuration, the hardware, and whether the
model files are actually where ComfyUI will look for them — the three things
that explain almost every startup failure.

It never starts ComfyUI and never downloads anything, so it is safe to run on a
worker that is already unhappy.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import config as config_module
from app.errors import WorkerError
from bootstrap import models, preflight


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> int:
    section("configuration")
    try:
        config = config_module.load()
    except WorkerError as error:
        print(f"FAILS AT STARTUP: [{error.code}] {error.message}")
        return 1

    print(json.dumps(config_module.describe(config), indent=2))
    variant = config.variant
    print(
        f"sampler defaults: steps={variant.sampling.steps} "
        f"guidance={variant.sampling.guidance} sampler={variant.sampling.sampler}"
    )
    print(f"reference limit:  {config.max_reference_images}")
    print(f"total weights:    {variant.total_download_bytes / 1e9:.2f} GB")

    section("hardware")
    gpu = preflight._gpu()
    if gpu is None:
        print("no CUDA device visible")
    else:
        name, capability, vram = gpu
        print(f"gpu:              {name}")
        print(f"compute:          {capability[0]}.{capability[1]}")
        print(
            f"vram:             {vram:.1f} GB "
            f"(recommended {variant.recommended_vram_gb} GB)"
        )
    ram = preflight._system_ram_gb()
    if ram is not None:
        print(
            f"system ram:       {ram:.1f} GB "
            f"(needs {variant.system_ram_min_gb} GB to stage weights)"
        )
    try:
        print(f"free disk:        {models.free_space_gb():.1f} GB")
    except OSError as exc:
        print(f"free disk:        unknown ({exc})")

    try:
        preflight.check(config)
        print("preflight:        ok")
    except WorkerError as error:
        print(f"preflight:        FAILS [{error.code}] {error.message}")
        return 1

    section("model files")
    missing = 0
    for asset in variant.assets:
        target = models.MODELS_ROOT / asset.dest_dir / asset.filename
        volume = models.VOLUME_ROOT / asset.dest_dir / asset.filename
        if target.is_file():
            size = target.stat().st_size / 1e9
            link = " (symlink)" if target.is_symlink() else ""
            print(f"  [image ] {asset.filename}  {size:.2f} GB{link}")
        elif volume.is_file():
            print(f"  [volume] {asset.filename}  not yet linked into the image")
        else:
            missing += 1
            print(
                f"  [MISSING] {asset.filename}  would download "
                f"{asset.size_gb:.2f} GB from {asset.repo}"
            )

    if missing and config.model_source != "auto":
        print(
            f"\nMODEL_SOURCE={config.model_source} will not fetch the missing "
            "files; startup would fail."
        )
        return 1

    print(
        "\nnothing above is fatal"
        if not missing
        else "\nmissing files would be downloaded on first boot"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
