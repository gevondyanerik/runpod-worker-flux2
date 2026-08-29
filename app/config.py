"""Deployment configuration.

This is the **only** module that reads ``os.environ``. CI enforces that (see
``.github/workflows/ci.yml``), so a stray ``getenv`` in a feature branch fails
the build rather than quietly growing the configuration surface back.

Three tiers:

* **Tier 0 — nothing set.** The worker boots on the default profile and serves
  requests. An endpoint created with an empty environment must produce an
  image; that is a supported production configuration, not a shortcut.
* **Tier 1 — ``FLUX2_VARIANT``.** Selects a complete internal profile. Unset
  falls back to whatever the image baked (``FLUX2_BAKED_VARIANT``, written by
  the Dockerfile) and says so in the log. An *unknown* value fails startup: a
  typo must never silently deploy a different model.
* **Tier 2/3 — optional refinements and expert overrides.** All defaulted, all
  ``advanced`` in the deployment UI.

No variable is required, and no credential is required on the default path.
``HF_TOKEN`` is read only when a Tier 3 override names a gated repository —
no built-in profile can need it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Final

from app import constants
from app.errors import ErrorCode, WorkerError
from app.variants import (
    DEFAULT_VARIANT,
    TEXT_ENCODERS,
    Asset,
    UnknownVariantError,
    VariantConfig,
    get_variant,
)

_MODEL_SOURCES: Final = ("auto", "baked", "volume", "download")
_OUTPUT_FORMATS: Final = ("webp", "png", "jpeg")


def _env(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = _env(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise WorkerError(
            ErrorCode.INVALID_INPUT, f"{name} must be an integer, got {raw!r}"
        ) from None
    if not minimum <= value <= maximum:
        raise WorkerError(
            ErrorCode.INVALID_INPUT,
            f"{name} must be between {minimum} and {maximum}, got {value}",
        )
    return value


def _env_int_or_none(name: str, *, minimum: int, maximum: int) -> int | None:
    """An integer that, when unset, means "whatever the profile says".

    Distinct from ``_env_int``: there is no single sensible default here,
    because the right value depends on the profile. Unset must therefore stay
    unset rather than collapse to a number.

    Zero counts as unset. The Hub renders a number input as a stepper that
    cannot be left blank: it starts at 0 and submits it, so without this a
    deploy that changed nothing would be refused over a value nobody chose.
    """
    if _env(name) in (None, "0"):
        return None
    return _env_int(name, 0, minimum=minimum, maximum=maximum)


def _env_choice(name: str, default: str, allowed: tuple[str, ...]) -> str:
    raw = _env(name)
    if raw is None:
        return default
    value = raw.lower()
    if value not in allowed:
        raise WorkerError(
            ErrorCode.INVALID_INPUT,
            f"{name} must be one of {', '.join(allowed)}, got {raw!r}",
        )
    return value


@dataclass(frozen=True, slots=True)
class S3Config:
    """Opt-in object storage.

    When these are absent the worker returns base64 and never mentions S3 —
    a feature you have not enabled should be invisible, not a reproach.
    """

    endpoint_url: str
    access_key_id: str
    secret_access_key: str


@dataclass(frozen=True, slots=True)
class Config:
    variant: VariantConfig
    variant_was_defaulted: bool

    model_source: str
    hf_token: str | None

    default_width: int
    default_height: int
    default_steps: int | None
    max_pixels: int
    max_images_per_request: int

    max_input_images: int
    ref_max_pixels: int

    output_format: str
    output_quality: int

    s3: S3Config | None
    overrides: dict[str, str] = field(default_factory=dict)

    @property
    def uses_s3(self) -> bool:
        return self.s3 is not None

    @property
    def effective_steps(self) -> int:
        """Steps for a request that does not specify them.

        ``DEFAULT_STEPS`` outranks the profile, and a request outranks both.
        The fallback of 4 is unreachable in practice — every shipped profile
        has confirmed sampling defaults — but the type says ``int | None`` and
        this keeps the resolution total.
        """
        return self.default_steps or self.variant.sampling.steps or 4

    @property
    def max_reference_images(self) -> int:
        """The effective cap.

        ``MAX_INPUT_IMAGES`` may lower a profile's cap but never raise it:
        accepting more references than the active profile can actually serve
        would be promising something the hardware cannot honour.
        """
        return min(self.max_input_images, self.variant.max_reference_images)


def _resolve_variant() -> tuple[VariantConfig, bool, dict[str, str]]:
    name = _env("FLUX2_VARIANT")
    defaulted = name is None

    # FLUX2_BAKED_VARIANT is written into the image by the Dockerfile, not by
    # an operator. It exists so a privately rebuilt image that bakes a
    # different profile still starts correctly with an empty environment.
    fallback = _env("FLUX2_BAKED_VARIANT") or DEFAULT_VARIANT

    try:
        variant = get_variant(name or fallback)
    except UnknownVariantError as exc:
        raise WorkerError(ErrorCode.UNSUPPORTED_VARIANT, str(exc)) from None

    overrides: dict[str, str] = {}

    encoder = _env("FLUX2_TEXT_ENCODER")
    if encoder is not None:
        if encoder not in TEXT_ENCODERS:
            raise WorkerError(
                ErrorCode.INVALID_INPUT,
                f"FLUX2_TEXT_ENCODER must be one of {', '.join(TEXT_ENCODERS)}, "
                f"got {encoder!r}",
            )
        variant = variant.with_text_encoder(encoder)
        overrides["FLUX2_TEXT_ENCODER"] = encoder

    variant, asset_overrides = _apply_asset_overrides(variant)
    overrides.update(asset_overrides)
    return variant, defaulted, overrides


def _apply_asset_overrides(
    variant: VariantConfig,
) -> tuple[VariantConfig, dict[str, str]]:
    """Tier 3: point the worker at a different checkpoint without forking.

    Deliberately explicit rather than a loop over ``setattr``: three named
    branches are longer but they type-check, and an override that silently
    landed on the wrong field would be very hard to notice.
    """
    applied: dict[str, str] = {}

    diffusion = _override_asset("DIFFUSION_MODEL", "diffusion_models", applied)
    encoder = _override_asset("TEXT_ENCODER", "text_encoders", applied)
    vae = _override_asset("VAE", "vae", applied)

    if diffusion is not None:
        variant = replace(variant, diffusion=diffusion)
    if encoder is not None:
        variant = replace(variant, text_encoder=encoder)
    if vae is not None:
        variant = replace(variant, vae=vae)

    return variant, applied


def _override_asset(
    prefix: str, dest_dir: str, applied: dict[str, str]
) -> Asset | None:
    """Read one ``*_REPO``/``*_FILE`` pair, or return None if unset."""
    repo = _env(f"{prefix}_REPO")
    path = _env(f"{prefix}_FILE")

    if repo is None and path is None:
        return None
    if repo is None or path is None:
        raise WorkerError(
            ErrorCode.INVALID_INPUT,
            f"{prefix}_REPO and {prefix}_FILE must be set together",
        )
    if path.endswith(".gguf"):
        raise WorkerError(
            ErrorCode.INVALID_INPUT,
            f"{prefix}_FILE points at a GGUF file, which this worker cannot "
            "load: no GGUF loader is installed. Use a .safetensors file.",
        )

    applied[f"{prefix}_REPO"] = repo
    applied[f"{prefix}_FILE"] = path
    # No size and no digest: an override is the operator's own file, so the
    # worker can only check that it arrives, not that it is the right one.
    return Asset(
        repo=repo,
        path=path,
        dest_dir=dest_dir,
        size_bytes=0,
        license_id="unknown",
    )


def _resolve_s3() -> S3Config | None:
    endpoint = _env("BUCKET_ENDPOINT_URL")
    key_id = _env("BUCKET_ACCESS_KEY_ID")
    secret = _env("BUCKET_SECRET_ACCESS_KEY")
    if not any((endpoint, key_id, secret)):
        return None
    if not all((endpoint, key_id, secret)):
        raise WorkerError(
            ErrorCode.INVALID_INPUT,
            "S3 output needs BUCKET_ENDPOINT_URL, BUCKET_ACCESS_KEY_ID and "
            "BUCKET_SECRET_ACCESS_KEY together, or none of them.",
        )
    assert endpoint and key_id and secret  # narrowed by the check above
    return S3Config(endpoint, key_id, secret)


def load() -> Config:
    variant, defaulted, overrides = _resolve_variant()

    return Config(
        variant=variant,
        variant_was_defaulted=defaulted,
        model_source=_env_choice("MODEL_SOURCE", "auto", _MODEL_SOURCES),
        hf_token=_env("HF_TOKEN"),
        default_width=_env_int(
            "DEFAULT_WIDTH",
            1024,
            minimum=constants.MIN_DIMENSION,
            maximum=constants.MAX_DIMENSION,
        ),
        default_height=_env_int(
            "DEFAULT_HEIGHT",
            1024,
            minimum=constants.MIN_DIMENSION,
            maximum=constants.MAX_DIMENSION,
        ),
        default_steps=_env_int_or_none(
            "DEFAULT_STEPS", minimum=1, maximum=constants.MAX_STEPS
        ),
        max_pixels=_env_int(
            "MAX_PIXELS",
            variant.max_output_pixels,
            minimum=256 * 256,
            maximum=variant.max_output_pixels,
        ),
        max_images_per_request=_env_int(
            "MAX_IMAGES_PER_REQUEST", 4, minimum=1, maximum=8
        ),
        max_input_images=_env_int(
            "MAX_INPUT_IMAGES", variant.max_reference_images, minimum=0, maximum=10
        ),
        ref_max_pixels=_env_int(
            "REF_MAX_PIXELS",
            variant.ref_max_pixels,
            minimum=256 * 1024,
            maximum=4_194_304,
        ),
        output_format=_env_choice("DEFAULT_OUTPUT_FORMAT", "webp", _OUTPUT_FORMATS),
        output_quality=_env_int("DEFAULT_OUTPUT_QUALITY", 95, minimum=1, maximum=100),
        s3=_resolve_s3(),
        overrides=overrides,
    )


def step_warning(config: Config) -> str | None:
    """Whether ``DEFAULT_STEPS`` is set to something that will not help.

    Measured on a distilled profile (RTX PRO 4500, 2026-08-29): raising 4 steps
    to 20 changed the composition and cost 2.8x the time without improving
    detail or text rendering. A distilled model is trained to converge in its
    native step count, so extra steps buy a different image, not a better one.

    A warning rather than a clamp: the operator asked for this, and an
    endpoint that silently ignored its own configuration would be worse than
    one that does what it was told and says what it thinks.
    """
    steps = config.default_steps
    if steps is None:
        return None
    native = config.variant.sampling.steps
    if native is None or not config.variant.distilled or steps <= native:
        return None
    return (
        f"DEFAULT_STEPS={steps} on distilled profile {config.variant.name}, "
        f"which is trained for {native}. Extra steps change the composition "
        f"rather than improve it, and cost proportionally more. For higher "
        f"quality prefer FLUX2_VARIANT=klein-4b-base."
    )


def describe(config: Config) -> dict[str, object]:
    """Non-default settings, for the startup log.

    With so little configurable, this handful of lines fully explains a
    worker's behaviour — which is what makes support tractable.
    """
    described: dict[str, object] = {
        "variant": config.variant.name,
        "variant_source": "default" if config.variant_was_defaulted else "environment",
        "text_encoder": config.variant.text_encoder.filename,
        "model_source": config.model_source,
        "output": "s3" if config.uses_s3 else "base64",
    }
    if config.default_steps is not None:
        described["default_steps"] = config.default_steps
    if config.overrides:
        described["overrides"] = config.overrides
    return described
