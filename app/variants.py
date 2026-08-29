"""The profile registry — the single source of truth for model configuration.

``FLUX2_VARIANT`` selects one entry here, and that entry determines the exact
model files, the workflow template, the loader, the precision, the text
encoder, the VAE, the sampling defaults and the reference limits. An operator
sets one string; everything downstream follows. No model path, sampler name or
step count may appear anywhere else in the codebase.

Every model referenced here is Apache-2.0. FLUX.2-dev and the klein 9B family
are ``flux-non-commercial-license`` and are deliberately absent — see
``docs/design.md``.

Sizes, SHA-256 digests and repository revisions were read from the Hugging
Face API on 2026-08-29 and are pinned: a profile resolves to exactly the
weights it was tested against, whatever upstream does later.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

APACHE_2_0 = "apache-2.0"

# The Comfy-Org repository that repackages the klein 4B assets for ComfyUI.
# The typo in the repo name ("encorder") is real upstream — do not "fix" it.
COMFY_ORG_KLEIN_4B = "Comfy-Org/vae-text-encorder-for-flux-klein-4b"
COMFY_ORG_REVISION = "5f526678002e43af5551dadb73ce2e8c91b43afe"


@dataclass(frozen=True, slots=True)
class Asset:
    """One downloadable model file."""

    repo: str
    path: str
    dest_dir: str  # diffusion_models | text_encoders | vae
    size_bytes: int
    license_id: str = APACHE_2_0
    revision: str = "main"
    sha256: str | None = None

    @property
    def filename(self) -> str:
        return self.path.rsplit("/", 1)[-1]

    @property
    def size_gb(self) -> float:
        return self.size_bytes / 1e9


@dataclass(frozen=True, slots=True)
class VramProbe:
    """One measured VRAM data point.

    Written only by ``scripts/benchmark.py`` running on real hardware and
    committed with the GPU name and date. Never hand-authored: a fabricated
    probe is worse than no probe, because it will be trusted.
    """

    width: int
    height: int
    refs: int
    text_encoder: str  # "bf16" | "fp4"
    peak_allocated_gb: float
    peak_reserved_gb: float
    gpu_name: str
    measured_at: str  # ISO date


@dataclass(frozen=True, slots=True)
class SamplingDefaults:
    """Inference defaults for a profile.

    Taken verbatim from the official ComfyUI workflow templates shipped in
    ``comfyui_workflow_templates`` (``image_flux2_klein_*``), not invented. A
    profile whose defaults are unconfirmed does not ship (see ``is_ready``):
    guessed numbers produce a worker that runs and is quietly wrong.
    """

    steps: int | None = None
    guidance: float | None = None
    sampler: str | None = None

    @property
    def confirmed(self) -> bool:
        return None not in (self.steps, self.guidance, self.sampler)


# Distilled: taken verbatim from the official ComfyUI templates, confirmed
# 2026-08-29 (Flux2Scheduler steps=4, CFGGuider cfg=1, KSamplerSelect euler).
# The distilled models are trained to converge at 4 steps, so this is the
# model's number, not a preference — raising it changes the image rather than
# improving it.
DISTILLED_SAMPLING = SamplingDefaults(steps=4, guidance=1.0, sampler="euler")

# Base: the official template ships 20 steps at cfg 5.0. This worker serves 28
# at cfg 4.0, chosen by a grid search rather than by taste — steps in
# {12, 20, 28, 36, 50} against guidance in {3, 4, 5, 6}, three seeds per cell,
# scored on whether a sign prompt spelled its two words correctly (see
# docs/samples/grid-base.webp).
#
# 28 steps was the peak of that grid for the fp8 profile: 7 of 12 cells
# legible, against 4 of 12 at both 12 and 50. More steps than that made
# spelling worse, not better. The bf16 profile does not fall off the same way
# — the collapse at 50 is the quantisation, not the model — but its gap
# between 28 and 50 is one cell in twelve on three seeds, which is noise, so
# both base profiles share this constant rather than getting one each.
#
# An identical product-shot grid showed no visible gain past 20 for either.
# Guidance barely moved the result anywhere in 3.0-5.0, so 4.0 sits in the
# middle of a flat region rather than on a peak. Sampler unchanged.
#
# This is the one shipped default that departs from the template, so it is
# explained here rather than left to be discovered.
BASE_SAMPLING = SamplingDefaults(steps=28, guidance=4.0, sampler="euler")


# --------------------------------------------------------------------------
# Shared assets — identical for every profile
# --------------------------------------------------------------------------

TEXT_ENCODER_BF16 = Asset(
    repo=COMFY_ORG_KLEIN_4B,
    path="split_files/text_encoders/qwen_3_4b.safetensors",
    dest_dir="text_encoders",
    size_bytes=8_044_982_048,
    revision=COMFY_ORG_REVISION,
    sha256="6c671498573ac2f7a5501502ccce8d2b08ea6ca2f661c458e708f36b36edfc5a",
)

TEXT_ENCODER_FP4 = Asset(
    repo=COMFY_ORG_KLEIN_4B,
    path="split_files/text_encoders/qwen_3_4b_fp4_flux2.safetensors",
    dest_dir="text_encoders",
    size_bytes=3_848_213_998,
    revision=COMFY_ORG_REVISION,
    sha256="3eab03a77adb0ee5304a4e677d5c10ac22f9049c1d7c894adca4f8bb39206ca8",
)

TEXT_ENCODERS = {"bf16": TEXT_ENCODER_BF16, "fp4": TEXT_ENCODER_FP4}

VAE = Asset(
    repo=COMFY_ORG_KLEIN_4B,
    path="split_files/vae/flux2-vae.safetensors",
    dest_dir="vae",
    size_bytes=336_211_292,
    revision=COMFY_ORG_REVISION,
    sha256="868fe7b343cc8f3a19dbcfcafbc3d5f888802be3f89bd81b65b3621a066ce8f3",
)


@dataclass(frozen=True, slots=True)
class VariantConfig:
    """A complete, self-contained deployment profile."""

    name: str
    description: str
    distilled: bool
    precision: str  # "bf16" | "fp8" | "nvfp4"

    diffusion: Asset
    text_encoder: Asset = TEXT_ENCODER_BF16
    vae: Asset = VAE

    sampling: SamplingDefaults = field(default_factory=SamplingDefaults)

    # nvfp4 needs Blackwell (compute capability >= 12.0). On older
    # architectures it may load without delivering the memory win, so the
    # worker refuses to start rather than running slowly and silently.
    #
    # Note that nvfp4 is not the fast option. Measured on an RTX PRO 4500
    # Blackwell it took 3.5 s to fp8's 2.0 s at 1024x1024 — the kernels are not
    # the win here; the 2.5 GB download is. Pick it to shrink cold starts, not
    # to shorten inference.
    requires_compute_capability: tuple[int, int] | None = None

    max_reference_images: int = 6
    ref_max_pixels: int = 1_048_576
    max_output_pixels: int = 4_194_304

    recommended_vram_gb: int = 16
    hard_min_vram_gb: int = 12
    system_ram_min_gb: int = 20

    vram_probes: tuple[VramProbe, ...] = ()

    @property
    def workflow_name(self) -> str:
        return self.name

    @property
    def assets(self) -> tuple[Asset, ...]:
        return (self.diffusion, self.text_encoder, self.vae)

    @property
    def total_download_bytes(self) -> int:
        return sum(a.size_bytes for a in self.assets)

    @property
    def is_ready(self) -> bool:
        """Whether this profile may be served.

        False while its sampling defaults are unconfirmed. The startup path
        refuses to serve an unready profile instead of guessing.
        """
        return self.sampling.confirmed

    def with_text_encoder(self, precision: str) -> VariantConfig:
        return replace(self, text_encoder=TEXT_ENCODERS[precision])


# --------------------------------------------------------------------------
# Profiles
#
#   distilled  ~4 steps, fast          base  full step count, better adherence
#   bf16 / fp8 / nvfp4                 precision, i.e. download size and speed
#
# The peak-VRAM floor is set by the text encoder (8.0 GB in bf16), not by the
# diffusion model, so it is the same for every profile: choosing fp8 over bf16
# buys download size and load time, not a smaller GPU.
# --------------------------------------------------------------------------

_PROFILES: tuple[VariantConfig, ...] = (
    VariantConfig(
        name="klein-4b",
        description="Distilled 4B, fp8 — the default. Best speed/size balance.",
        distilled=True,
        precision="fp8",
        sampling=DISTILLED_SAMPLING,
        diffusion=Asset(
            repo="black-forest-labs/FLUX.2-klein-4b-fp8",
            path="flux-2-klein-4b-fp8.safetensors",
            dest_dir="diffusion_models",
            size_bytes=4_070_624_520,
            revision="5b4408e59397a4a37ccb46afe426d8ed86379441",
            sha256="97ed34fe0567e436200f2faee3939b88f2b5d99f8af2a4dc16532c4245c0ccb6",
        ),
    ),
    VariantConfig(
        name="klein-4b-bf16",
        description="Distilled 4B, bf16 — full precision reference for the fast path.",
        distilled=True,
        precision="bf16",
        sampling=DISTILLED_SAMPLING,
        diffusion=Asset(
            repo=COMFY_ORG_KLEIN_4B,
            path="split_files/diffusion_models/flux-2-klein-4b.safetensors",
            dest_dir="diffusion_models",
            size_bytes=7_751_105_712,
            revision=COMFY_ORG_REVISION,
            sha256="ec3d4e733a771f61c052fb4856c48b336c55eaf2c65487c2a1faeb9bbda7a343",
        ),
    ),
    VariantConfig(
        name="klein-4b-nvfp4",
        description="Distilled 4B, NVFP4 — smallest download. Blackwell GPUs only.",
        distilled=True,
        precision="nvfp4",
        sampling=DISTILLED_SAMPLING,
        diffusion=Asset(
            repo="black-forest-labs/FLUX.2-klein-4b-nvfp4",
            path="flux-2-klein-4b-nvfp4.safetensors",
            dest_dir="diffusion_models",
            size_bytes=2_460_413_488,
            revision="1db2b2f776c24b76f1122e5f69ab1949fc620068",
            sha256="d8c5007b6a3bbbdfd38538bbcef5101a55dfde81894f58d2e3c8701cdef3542b",
        ),
        requires_compute_capability=(12, 0),
    ),
    VariantConfig(
        name="klein-4b-base",
        description="Base (undistilled) 4B, fp8 — slower, stronger prompt adherence.",
        distilled=False,
        precision="fp8",
        sampling=BASE_SAMPLING,
        diffusion=Asset(
            repo="black-forest-labs/FLUX.2-klein-base-4b-fp8",
            path="flux-2-klein-base-4b-fp8.safetensors",
            dest_dir="diffusion_models",
            size_bytes=4_089_498_488,
            revision="103db268c10d4d3921101b46057671f9ac460da6",
            sha256="44bab3a86fe98b85d21dd2a4729ebdc3ae51fb8a39f76e457e18c724219e6840",
        ),
    ),
    VariantConfig(
        name="klein-4b-base-bf16",
        description="Base 4B, bf16 — the quality ceiling of this family.",
        distilled=False,
        precision="bf16",
        sampling=BASE_SAMPLING,
        diffusion=Asset(
            repo=COMFY_ORG_KLEIN_4B,
            path="split_files/diffusion_models/flux-2-klein-base-4b.safetensors",
            dest_dir="diffusion_models",
            size_bytes=7_751_105_712,
            revision=COMFY_ORG_REVISION,
            sha256="9c5fed22b76baea749d88fc2abe3ad53245e7b21a0d353a762665eea00043b92",
        ),
    ),
)

VARIANTS: dict[str, VariantConfig] = {p.name: p for p in _PROFILES}

DEFAULT_VARIANT = "klein-4b"

# Baked into the published image, so a deployment with an empty environment
# starts without downloading anything. Other profiles fetch only their
# diffusion model, because the encoder and VAE are already present.
BAKED_ASSETS: tuple[Asset, ...] = (
    VARIANTS[DEFAULT_VARIANT].diffusion,
    TEXT_ENCODER_BF16,
    VAE,
)


class UnknownVariantError(ValueError):
    """Raised for an unrecognised FLUX2_VARIANT.

    Unset defaults; wrong never does. A typo in a model name must not silently
    deploy a different model.
    """

    def __init__(self, name: str) -> None:
        valid = ", ".join(sorted(VARIANTS))
        super().__init__(f"Unknown FLUX2_VARIANT {name!r}. Valid values: {valid}")
        self.name = name


def get_variant(name: str) -> VariantConfig:
    try:
        return VARIANTS[name]
    except KeyError:
        raise UnknownVariantError(name) from None
