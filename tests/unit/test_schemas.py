"""Request validation: one required field, and a hard line on what is exposed."""

from __future__ import annotations

from dataclasses import replace

import pytest

from app import constants, schemas
from app.config import Config
from app.errors import ErrorCode, WorkerError


def test_prompt_alone_is_a_complete_request(config: Config) -> None:
    request = schemas.parse({"prompt": "a red bicycle"}, config)
    assert request.width == config.default_width
    assert request.height == config.default_height
    assert request.steps == config.variant.sampling.steps
    assert request.guidance == config.variant.sampling.guidance
    assert request.n == 1
    assert request.size_explicit is False


def test_missing_prompt_has_its_own_code(config: Config) -> None:
    with pytest.raises(WorkerError) as excinfo:
        schemas.parse({}, config)
    assert excinfo.value.code is ErrorCode.MISSING_PROMPT


def test_blank_prompt_is_missing_not_invalid(config: Config) -> None:
    with pytest.raises(WorkerError) as excinfo:
        schemas.parse({"prompt": "   "}, config)
    assert excinfo.value.code is ErrorCode.MISSING_PROMPT


@pytest.mark.parametrize(
    "field",
    ["sampler", "scheduler", "model", "workflow", "unet_name", "cfg"],
)
def test_implementation_details_are_refused(config: Config, field: str) -> None:
    # The line this schema holds: generation parameters are exposed,
    # implementation is not. An unknown key is an error, never ignored.
    with pytest.raises(WorkerError) as excinfo:
        schemas.parse({"prompt": "x", field: "euler"}, config)
    assert excinfo.value.code is ErrorCode.INVALID_INPUT


def test_dimensions_are_aligned_downwards(config: Config) -> None:
    request = schemas.parse({"prompt": "x", "width": 1023, "height": 777}, config)
    assert request.width == 1008
    assert request.height == 768
    assert request.adjusted is True
    assert request.size_explicit is True


def test_aspect_ratio_is_never_stretched(config: Config) -> None:
    # Rounding down both dimensions keeps the requested shape; stretching one
    # to fit would silently change the image the caller asked for.
    request = schemas.parse({"prompt": "x", "width": 900, "height": 600}, config)
    assert request.width % 16 == 0 and request.height % 16 == 0
    assert request.width <= 900 and request.height <= 600


def test_resolution_below_the_minimum_is_rejected(config: Config) -> None:
    with pytest.raises(WorkerError) as excinfo:
        schemas.parse({"prompt": "x", "width": 64, "height": 64}, config)
    assert excinfo.value.code is ErrorCode.INVALID_RESOLUTION


def test_pixel_budget_is_enforced(config: Config) -> None:
    with pytest.raises(WorkerError) as excinfo:
        schemas.parse({"prompt": "x", "width": 4096, "height": 4096}, config)
    assert excinfo.value.code is ErrorCode.INVALID_RESOLUTION
    assert excinfo.value.details["max_pixels"] == config.max_pixels


def test_too_many_references_names_the_limit(config: Config) -> None:
    urls = ["https://example.invalid/a.png"] * 20
    with pytest.raises(WorkerError) as excinfo:
        schemas.parse({"prompt": "x", "images": urls}, config)
    assert excinfo.value.code is ErrorCode.TOO_MANY_IMAGES
    assert excinfo.value.details["max_reference_images"] == config.max_reference_images


def test_steps_above_the_ceiling_are_rejected(config: Config) -> None:
    with pytest.raises(WorkerError):
        schemas.parse({"prompt": "x", "steps": constants.MAX_STEPS + 1}, config)


def test_overlong_prompt_is_rejected(config: Config) -> None:
    with pytest.raises(WorkerError):
        schemas.parse({"prompt": "a" * (constants.MAX_PROMPT_CHARS + 1)}, config)


def test_seed_is_random_but_stable_once_resolved(config: Config) -> None:
    first = schemas.parse({"prompt": "x"}, config)
    second = schemas.parse({"prompt": "x"}, config)
    assert first.seed != second.seed  # astronomically unlikely to collide

    fixed = schemas.parse({"prompt": "x", "seed": 42}, config)
    assert fixed.seed == 42
    assert fixed.seeds == [42]


def test_a_batch_reports_one_seed_per_image(config: Config) -> None:
    # One seed covers the whole batch; every image carries it, because image i
    # of a batch is not reproducible from that seed alone.
    request = schemas.parse({"prompt": "x", "seed": 7, "n": 3}, config)
    assert request.seeds == [7, 7, 7]


def test_capabilities_describes_the_endpoint(config: Config) -> None:
    described = schemas.capabilities(config, "NVIDIA RTX PRO 4500 Blackwell")
    assert described["variant"] == config.variant.name
    assert described["license"] == "apache-2.0"
    assert described["api_version"] == constants.API_VERSION
    assert described["max_reference_images"] == config.max_reference_images
    assert described["gpu"] == "NVIDIA RTX PRO 4500 Blackwell"


def test_default_steps_reaches_a_request_that_omits_them(config: Config) -> None:
    endpoint = replace(config, default_steps=8)
    assert schemas.parse({"prompt": "a red bicycle"}, endpoint).steps == 8


def test_a_request_still_outranks_the_endpoint_default(config: Config) -> None:
    # The endpoint sets a house default; a caller who names a number gets it.
    endpoint = replace(config, default_steps=8)
    parsed = schemas.parse({"prompt": "a red bicycle", "steps": 4}, endpoint)
    assert parsed.steps == 4


def test_capabilities_reports_the_effective_step_count(config: Config) -> None:
    # An integrator reads this to learn the endpoint's behaviour, so it must
    # report what will actually happen, not what the profile would do alone.
    assert schemas.capabilities(config)["default_steps"] == (
        config.variant.sampling.steps
    )
    endpoint = replace(config, default_steps=8)
    assert schemas.capabilities(endpoint)["default_steps"] == 8
