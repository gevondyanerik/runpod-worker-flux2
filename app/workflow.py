"""Builds the ComfyUI API-format graph for a request.

The workflow is owned by the server. Callers send a prompt and optional
reference images; they never see node identifiers, loader names or sampler
settings. That is what keeps the public API stable when the internal graph
changes.

The graph shape mirrors the official ComfyUI templates
(``image_flux2_klein_text_to_image`` and ``image_flux2_klein_image_edit_4b_*``),
flattened out of their subgraphs:

    UNETLoader ─────────────────────────────► CFGGuider.model
    CLIPLoader ──► CLIPTextEncode ──────────► (positive chain)
               └─► negative branch ─────────► (negative chain)
    VAELoader ──┬─────────────────────────► VAEDecode.vae
                └─────────────────────────► VAEEncode.vae

    for each reference image i:
        LoadImage ─► ImageScaleToTotalPixels ─► VAEEncode ─┬─► ReferenceLatent(pos)
                                                           └─► ReferenceLatent(neg)
        …chained, so reference i+1 conditions on the output of reference i.

    RandomNoise ┐
    CFGGuider   ├─► SamplerCustomAdvanced ─► VAEDecode ─► SaveImage
    KSamplerSelect, Flux2Scheduler, EmptyFlux2LatentImage ┘

The negative branch differs by profile, and it is not cosmetic. The distilled
templates zero the positive conditioning out (``ConditioningZeroOut``), which
is free and harmless because they sample at cfg 1.0, where the negative is
never evaluated. The base templates encode an actual empty prompt instead,
because at cfg 5.0 the negative *is* evaluated — and a zeroed tensor is not the
same thing as the encoding of an empty string. Using the zeroed tensor there
produces wildly oversaturated images that ignore the prompt.

Why this is built in code rather than loaded from a static JSON template: the
reference chain has a variable length. ComfyUI's API format has no notion of a
bypassed node — a node is either present or absent — so a static template would
need N pre-made slots plus deletion logic, and its node ids would churn every
time someone re-saved it from the ComfyUI UI. Building the graph from named
constants keeps a single source of truth, and ``tests/workflows`` pins the
output against committed golden files so an accidental change to the graph
fails CI.
"""

from __future__ import annotations

from typing import Any

from app.variants import VariantConfig

# ComfyUI node ids. Strings, because that is what the /prompt API expects.
# These are internal and never leave the process.
N_UNET = "unet"
N_CLIP = "clip"
N_VAE = "vae"
N_PROMPT = "prompt"
N_NEGATIVE = "negative"
N_LATENT = "latent"
N_SIGMAS = "sigmas"
N_SAMPLER = "sampler"
N_NOISE = "noise"
N_GUIDER = "guider"
N_SAMPLE = "sample"
N_DECODE = "decode"
N_SAVE = "save"
N_REF_SIZE = "ref_size"

OUTPUT_NODE = N_SAVE
FILENAME_PREFIX = "flux2"

# ``CLIPLoader`` needs to know which text-encoder family it is loading.
CLIP_TYPE = "flux2"

# Latent dimensions must be a multiple of this. FLUX.2 packs a 16x-downscaled
# latent, so both dimensions are rounded down to a multiple of 16 rather than
# being stretched — the aspect ratio a caller asks for is never silently
# changed.
LATENT_ALIGNMENT = 16

# ``ImageScaleToTotalPixels`` multiplies its ``megapixels`` input by 1024*1024,
# not by 1e6. Using the decimal million here would quietly push every reference
# image about 5% over the intended pixel budget.
MEBIPIXEL = 1024 * 1024


def align(value: int) -> int:
    """Round down to a valid latent dimension, never below one block."""
    return max(LATENT_ALIGNMENT, (value // LATENT_ALIGNMENT) * LATENT_ALIGNMENT)


def _megapixels(pixels: int) -> float:
    """Convert a pixel budget into the node's mebipixel input, within range."""
    return round(min(16.0, max(0.01, pixels / MEBIPIXEL)), 2)


def _node(class_type: str, inputs: dict[str, Any], title: str) -> dict[str, Any]:
    return {"class_type": class_type, "inputs": inputs, "_meta": {"title": title}}


def build(
    *,
    variant: VariantConfig,
    prompt: str,
    width: int,
    height: int,
    seed: int,
    steps: int,
    guidance: float,
    batch_size: int = 1,
    reference_filenames: tuple[str, ...] = (),
    match_reference_size: bool = False,
) -> dict[str, Any]:
    """Return a ComfyUI API-format graph.

    ``reference_filenames`` are names already present in ComfyUI's input
    directory, in the caller's original order — order is semantic, because a
    prompt may say "the product in image 1".

    ``match_reference_size`` takes the output geometry from the first scaled
    reference instead of ``width``/``height``, which is what the official edit
    template does. The handler sets it only when references are present and the
    caller did not pin a resolution: editing a 3:4 photo should return a 3:4
    image, and a caller who did name a size must still get exactly that size.
    """
    width, height = align(width), align(height)

    graph: dict[str, Any] = {
        N_UNET: _node(
            "UNETLoader",
            {"unet_name": variant.diffusion.filename, "weight_dtype": "default"},
            "FLUX2_DIFFUSION_LOADER",
        ),
        N_CLIP: _node(
            "CLIPLoader",
            {
                "clip_name": variant.text_encoder.filename,
                "type": CLIP_TYPE,
                "device": "default",
            },
            "FLUX2_TE_LOADER",
        ),
        N_VAE: _node(
            "VAELoader",
            {"vae_name": variant.vae.filename},
            "FLUX2_VAE_LOADER",
        ),
        N_PROMPT: _node(
            "CLIPTextEncode",
            {"clip": [N_CLIP, 0], "text": prompt},
            "FLUX2_PROMPT",
        ),
        N_NEGATIVE: (
            _node(
                "ConditioningZeroOut",
                {"conditioning": [N_PROMPT, 0]},
                "FLUX2_NEGATIVE",
            )
            if variant.distilled
            else _node(
                "CLIPTextEncode",
                {"clip": [N_CLIP, 0], "text": ""},
                "FLUX2_NEGATIVE",
            )
        ),
        N_SAMPLER: _node(
            "KSamplerSelect",
            {"sampler_name": variant.sampling.sampler},
            "FLUX2_SAMPLER",
        ),
        N_NOISE: _node(
            "RandomNoise",
            {"noise_seed": seed},
            "FLUX2_NOISE",
        ),
    }

    # Reference chain. Each image conditions both the positive and the negative
    # branch, exactly as the official edit template does, and each one chains
    # onto the previous so the model sees all of them.
    ref_megapixels = _megapixels(variant.ref_max_pixels)
    positive: Any = [N_PROMPT, 0]
    negative: Any = [N_NEGATIVE, 0]
    first_scale: str | None = None

    for index, filename in enumerate(reference_filenames):
        load = f"ref{index}_load"
        scale = f"ref{index}_scale"
        encode = f"ref{index}_encode"
        pos = f"ref{index}_pos"
        neg = f"ref{index}_neg"
        first_scale = first_scale or scale

        graph[load] = _node(
            "LoadImage",
            {"image": filename},
            f"FLUX2_REF_{index + 1}",
        )
        graph[scale] = _node(
            "ImageScaleToTotalPixels",
            {
                "image": [load, 0],
                "upscale_method": "nearest-exact",
                "megapixels": ref_megapixels,
                "resolution_steps": 1,
            },
            f"FLUX2_REF_{index + 1}_SCALE",
        )
        graph[encode] = _node(
            "VAEEncode",
            {"pixels": [scale, 0], "vae": [N_VAE, 0]},
            f"FLUX2_REF_{index + 1}_ENCODE",
        )
        graph[pos] = _node(
            "ReferenceLatent",
            {"conditioning": positive, "latent": [encode, 0]},
            f"FLUX2_REF_{index + 1}_POS",
        )
        graph[neg] = _node(
            "ReferenceLatent",
            {"conditioning": negative, "latent": [encode, 0]},
            f"FLUX2_REF_{index + 1}_NEG",
        )
        positive = [pos, 0]
        negative = [neg, 0]

    # Output geometry. Literal dimensions unless the request is an edit with no
    # explicit size, in which case the first reference decides.
    if match_reference_size and first_scale is not None:
        graph[N_REF_SIZE] = _node(
            "GetImageSize", {"image": [first_scale, 0]}, "FLUX2_REF_SIZE"
        )
        out_width: Any = [N_REF_SIZE, 0]
        out_height: Any = [N_REF_SIZE, 1]
    else:
        out_width, out_height = width, height

    graph[N_LATENT] = _node(
        "EmptyFlux2LatentImage",
        {"width": out_width, "height": out_height, "batch_size": batch_size},
        "FLUX2_LATENT",
    )
    graph[N_SIGMAS] = _node(
        "Flux2Scheduler",
        {"steps": steps, "width": out_width, "height": out_height},
        "FLUX2_SCHEDULER",
    )
    graph[N_GUIDER] = _node(
        "CFGGuider",
        {
            "model": [N_UNET, 0],
            "positive": positive,
            "negative": negative,
            "cfg": guidance,
        },
        "FLUX2_GUIDER",
    )
    graph[N_SAMPLE] = _node(
        "SamplerCustomAdvanced",
        {
            "noise": [N_NOISE, 0],
            "guider": [N_GUIDER, 0],
            "sampler": [N_SAMPLER, 0],
            "sigmas": [N_SIGMAS, 0],
            "latent_image": [N_LATENT, 0],
        },
        "FLUX2_SAMPLE",
    )
    graph[N_DECODE] = _node(
        "VAEDecode",
        {"samples": [N_SAMPLE, 0], "vae": [N_VAE, 0]},
        "FLUX2_DECODE",
    )
    graph[N_SAVE] = _node(
        "SaveImage",
        {"filename_prefix": FILENAME_PREFIX, "images": [N_DECODE, 0]},
        "FLUX2_OUTPUT",
    )

    return graph


class WorkflowInvalidError(RuntimeError):
    """A generated graph failed its structural checks."""


def validate(graph: dict[str, Any], *, expected_references: int) -> None:
    """Assert the graph is internally consistent before it is submitted.

    A broken graph must fail loudly here rather than deep inside ComfyUI, where
    it surfaces as an opaque prompt-validation error.
    """
    if OUTPUT_NODE not in graph:
        raise WorkflowInvalidError("graph has no output node")

    references = sum(1 for key in graph if key.endswith("_pos"))
    if references != expected_references:
        raise WorkflowInvalidError(
            f"graph wires {references} reference images, expected {expected_references}"
        )

    for node_id, node in graph.items():
        for key, value in node["inputs"].items():
            if not (isinstance(value, list) and len(value) == 2):
                continue
            target = value[0]
            if not isinstance(target, str) or target not in graph:
                raise WorkflowInvalidError(
                    f"node {node_id!r} input {key!r} points at missing node {target!r}"
                )

    # Every reference must actually reach the guider, otherwise we would be
    # accepting images and silently ignoring them.
    if expected_references:
        last = f"ref{expected_references - 1}_pos"
        if graph[N_GUIDER]["inputs"]["positive"] != [last, 0]:
            raise WorkflowInvalidError("reference chain is not connected to the guider")
