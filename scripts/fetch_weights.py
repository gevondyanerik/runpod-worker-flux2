#!/usr/bin/env python3
"""Download a profile's weights into the image at build time.

Run during ``docker build`` so the published image starts serving without
touching the network. Files land as real files, not symlinks into a cache, so
the image is self-contained and the cache layer can be discarded.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import constants
from app.variants import TEXT_ENCODERS, VARIANTS, Asset, get_variant
from bootstrap.models import sha256_of

MODELS_ROOT = Path(constants.COMFY_MODELS_ROOT)


def fetch(asset: Asset, cache: Path) -> None:
    from huggingface_hub import hf_hub_download

    target = MODELS_ROOT / asset.dest_dir / asset.filename
    if target.is_file() and target.stat().st_size == asset.size_bytes:
        print(f"  already present: {asset.filename}")
        return

    print(f"  downloading {asset.filename} ({asset.size_gb:.2f} GB) from {asset.repo}")
    downloaded = Path(
        hf_hub_download(
            repo_id=asset.repo,
            filename=asset.path,
            revision=asset.revision,
            cache_dir=str(cache),
        )
    )

    if asset.sha256:
        actual = sha256_of(downloaded)
        if actual != asset.sha256:
            raise SystemExit(
                f"checksum mismatch for {asset.filename}: "
                f"expected {asset.sha256}, got {actual}"
            )
        print("  checksum ok")

    target.parent.mkdir(parents=True, exist_ok=True)
    # move, not copy: the cache and the target are on the same layer, and a
    # copy would double the image size for the duration of the build.
    shutil.move(str(downloaded.resolve()), target)
    print(f"  installed {target}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "variant",
        nargs="?",
        default="klein-4b",
        choices=sorted(VARIANTS),
        help="profile whose weights to bake into the image",
    )
    parser.add_argument(
        "--text-encoder",
        default="bf16",
        choices=sorted(TEXT_ENCODERS),
        help="text-encoder precision to bake",
    )
    parser.add_argument(
        "--cache",
        default="/tmp/hf-cache",
        help="scratch download cache, removed afterwards",
    )
    args = parser.parse_args()

    variant = get_variant(args.variant).with_text_encoder(args.text_encoder)
    cache = Path(args.cache)

    total_gb = variant.total_download_bytes / 1e9
    print(f"baking profile {variant.name} ({total_gb:.1f} GB)")
    try:
        for asset in variant.assets:
            fetch(asset, cache)
    finally:
        shutil.rmtree(cache, ignore_errors=True)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
