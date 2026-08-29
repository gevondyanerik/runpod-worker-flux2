"""Shared fixtures.

Every test in this suite runs on a laptop with no GPU and no network. That is
deliberate: a test that needs a 24 GB card is a test nobody runs.
"""

from __future__ import annotations

import io
import os
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import config as config_module  # noqa: E402
from app.config import Config  # noqa: E402

# Every variable the worker reads. Cleared before each test so a developer's
# shell cannot change what the suite asserts.
_WORKER_ENV = (
    "FLUX2_VARIANT",
    "FLUX2_BAKED_VARIANT",
    "FLUX2_TEXT_ENCODER",
    "MODEL_SOURCE",
    "HF_TOKEN",
    "DEFAULT_WIDTH",
    "DEFAULT_HEIGHT",
    "MAX_PIXELS",
    "MAX_IMAGES_PER_REQUEST",
    "MAX_INPUT_IMAGES",
    "REF_MAX_PIXELS",
    "DEFAULT_OUTPUT_FORMAT",
    "DEFAULT_OUTPUT_QUALITY",
    "DIFFUSION_MODEL_REPO",
    "DIFFUSION_MODEL_FILE",
    "TEXT_ENCODER_REPO",
    "TEXT_ENCODER_FILE",
    "VAE_REPO",
    "VAE_FILE",
    "BUCKET_ENDPOINT_URL",
    "BUCKET_ACCESS_KEY_ID",
    "BUCKET_SECRET_ACCESS_KEY",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name in _WORKER_ENV:
        monkeypatch.delenv(name, raising=False)
    yield


@pytest.fixture
def config() -> Config:
    """The Tier 0 configuration: nothing set at all."""
    return config_module.load()


@pytest.fixture
def png_bytes() -> bytes:
    """A small valid PNG, for reference-image tests."""
    buffer = io.BytesIO()
    Image.new("RGB", (64, 48), (200, 120, 60)).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def large_png_bytes() -> bytes:
    """A PNG well above the reference pixel budget."""
    buffer = io.BytesIO()
    Image.new("RGB", (2400, 1600), (30, 90, 200)).save(buffer, format="PNG")
    return buffer.getvalue()


def pytest_configure() -> None:
    os.environ.setdefault("PYTHONHASHSEED", "0")
