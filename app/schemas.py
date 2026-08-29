"""Public request and response shapes.

Only ``prompt`` is required. Everything else is defaulted from the active
profile, so a caller can send one field and get an image.

The line this schema holds: **generation parameters are exposed, implementation
is not.** ``steps`` and ``guidance`` are things a caller reasonably varies per
image. ``sampler``, ``scheduler``, model filenames, node ids and workflow JSON
are properties of the profile and never appear in a request.
"""

from __future__ import annotations

import secrets
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app import constants
from app.config import Config
from app.errors import ErrorCode, WorkerError

SEED_MAX = 2**63 - 1


class GenerationRequest(BaseModel):
    """A validated, profile-aware request.

    Built through :func:`parse` rather than instantiated directly, because the
    defaults depend on the active profile.
    """

    model_config = ConfigDict(extra="forbid")

    prompt: Annotated[str, Field(min_length=1, max_length=constants.MAX_PROMPT_CHARS)]
    images: list[str] = Field(default_factory=list)
    width: int | None = None
    height: int | None = None
    n: Annotated[int, Field(ge=1)] = 1
    steps: Annotated[int, Field(ge=1, le=constants.MAX_STEPS)] | None = None
    guidance: Annotated[float, Field(ge=0.0, le=20.0)] | None = None
    seed: Annotated[int, Field(ge=0, le=SEED_MAX)] | None = None
    output_format: Literal["webp", "png", "jpeg"] | None = None
    quality: Annotated[int, Field(ge=1, le=100)] | None = None

    @field_validator("prompt")
    @classmethod
    def _strip_prompt(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("prompt must not be blank")
        return stripped


class ResolvedRequest(BaseModel):
    """A request with every default filled in from the active profile."""

    model_config = ConfigDict(extra="forbid")

    prompt: str
    images: list[str]
    width: int
    height: int
    n: int
    steps: int
    guidance: float
    seed: int
    output_format: str
    quality: int
    adjusted: bool = False
    size_explicit: bool = False

    @property
    def seeds(self) -> list[int]:
        """The seed reported for each returned image.

        One seed covers the whole batch, and every image carries it. ComfyUI
        draws a batch of noise from a single generator, so image *i* of a batch
        of *n* is not the image you get from that seed with ``n=1`` — the
        reproducible unit is ``(seed, n, index)``, not ``seed`` alone.
        Reporting ``seed + i`` here would look tidier and be a lie.
        """
        return [self.seed] * self.n


def parse(payload: dict[str, Any], config: Config) -> ResolvedRequest:
    """Validate a job payload and resolve it against the active profile."""
    from app.workflow import align

    if not isinstance(payload, dict):
        raise WorkerError(ErrorCode.INVALID_INPUT, "input must be an object")

    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise WorkerError(
            ErrorCode.MISSING_PROMPT,
            "input.prompt is required and must be a non-empty string",
        )

    try:
        request = GenerationRequest.model_validate(payload)
    except Exception as exc:  # pydantic ValidationError
        raise WorkerError(ErrorCode.INVALID_INPUT, _first_error(exc)) from None

    cap = config.max_reference_images
    if len(request.images) > cap:
        raise WorkerError(
            ErrorCode.TOO_MANY_IMAGES,
            f"Profile {config.variant.name} accepts at most {cap} reference "
            f"image{'s' if cap != 1 else ''}; {len(request.images)} provided.",
            details={"max_reference_images": cap, "provided": len(request.images)},
        )

    if request.n > config.max_images_per_request:
        raise WorkerError(
            ErrorCode.INVALID_INPUT,
            f"n must be at most {config.max_images_per_request}, got {request.n}",
        )

    width = request.width or config.default_width
    height = request.height or config.default_height
    for name, value in (("width", width), ("height", height)):
        if not constants.MIN_DIMENSION <= value <= constants.MAX_DIMENSION:
            raise WorkerError(
                ErrorCode.INVALID_RESOLUTION,
                f"{name} must be between {constants.MIN_DIMENSION} and "
                f"{constants.MAX_DIMENSION}, got {value}",
            )

    if width * height > config.max_pixels:
        raise WorkerError(
            ErrorCode.INVALID_RESOLUTION,
            f"{width}x{height} is {width * height} pixels, above the "
            f"{config.max_pixels} limit.",
            details={"max_pixels": config.max_pixels},
        )

    # Round down to a valid latent size. Never stretch: the aspect ratio a
    # caller asked for is not ours to change.
    aligned_w, aligned_h = align(width), align(height)

    return ResolvedRequest(
        prompt=request.prompt,
        images=request.images,
        width=aligned_w,
        height=aligned_h,
        n=request.n,
        steps=request.steps or config.effective_steps,
        guidance=(
            request.guidance
            if request.guidance is not None
            else (config.variant.sampling.guidance or 1.0)
        ),
        seed=request.seed if request.seed is not None else secrets.randbelow(SEED_MAX),
        output_format=request.output_format or config.output_format,
        quality=request.quality or config.output_quality,
        adjusted=(aligned_w, aligned_h) != (width, height),
        size_explicit=request.width is not None or request.height is not None,
    )


def _first_error(exc: Exception) -> str:
    errors = getattr(exc, "errors", None)
    if not callable(errors):
        return str(exc)
    try:
        first = errors()[0]
    except (IndexError, TypeError):
        return str(exc)
    location = ".".join(str(part) for part in first.get("loc", ())) or "input"
    return f"{location}: {first.get('msg', 'invalid value')}"


def capabilities(config: Config, gpu_name: str | None = None) -> dict[str, Any]:
    """What this endpoint can do.

    Because an operator configures almost nothing, a caller cannot infer the
    limits from a deployment manifest. This is how an integrator discovers them
    without a failed request.
    """
    variant = config.variant
    return {
        "api_version": constants.API_VERSION,
        "variant": variant.name,
        "description": variant.description,
        "license": variant.diffusion.license_id,
        "distilled": variant.distilled,
        "precision": variant.precision,
        "text_encoder": "fp4" if "fp4" in variant.text_encoder.filename else "bf16",
        "default_steps": config.effective_steps,
        "default_guidance": variant.sampling.guidance,
        "default_width": config.default_width,
        "default_height": config.default_height,
        "max_reference_images": config.max_reference_images,
        "ref_max_pixels": config.ref_max_pixels,
        "max_output_pixels": config.max_pixels,
        "max_images_per_request": config.max_images_per_request,
        "max_prompt_chars": constants.MAX_PROMPT_CHARS,
        "output": "s3" if config.uses_s3 else "base64",
        "gpu": gpu_name,
        "measured_limits": [
            {
                "width": p.width,
                "height": p.height,
                "refs": p.refs,
                "peak_vram_gb": p.peak_reserved_gb,
                "gpu": p.gpu_name,
            }
            for p in variant.vram_probes
        ],
    }
