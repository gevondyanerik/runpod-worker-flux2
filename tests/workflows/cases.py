"""The graphs pinned by golden files.

Kept next to the test rather than inside it so ``scripts/update_goldens.py``
regenerates exactly what the test compares against.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from app import workflow
from app.variants import get_variant

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"


def cases() -> Iterator[tuple[str, dict[str, Any]]]:
    yield (
        "text_to_image",
        workflow.build(
            variant=get_variant("klein-4b"),
            prompt="a red bicycle leaning on a wall",
            width=1024,
            height=1024,
            seed=42,
            steps=4,
            guidance=1.0,
        ),
    )
    yield (
        "text_to_image_base",
        workflow.build(
            variant=get_variant("klein-4b-base"),
            prompt="a red bicycle leaning on a wall",
            width=1024,
            height=1024,
            seed=42,
            steps=28,
            guidance=4.0,
        ),
    )
    yield (
        "edit_one_reference",
        workflow.build(
            variant=get_variant("klein-4b"),
            prompt="change the bag colour to blue",
            width=1024,
            height=1024,
            seed=42,
            steps=4,
            guidance=1.0,
            reference_filenames=("job-ref0.png",),
            match_reference_size=True,
        ),
    )
    yield (
        "edit_two_references_fixed_size",
        workflow.build(
            variant=get_variant("klein-4b"),
            prompt="put the logo from image 2 onto the bag in image 1",
            width=768,
            height=1024,
            seed=42,
            steps=4,
            guidance=1.0,
            batch_size=2,
            reference_filenames=("job-ref0.png", "job-ref1.png"),
        ),
    )
