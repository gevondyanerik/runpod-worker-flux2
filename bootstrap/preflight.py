"""Hardware checks that run once, before ComfyUI is started.

Serverless bills for the whole cold start, so a worker that boots on the wrong
hardware and only discovers it on the first paying request has already cost the
operator money and produced a confusing failure. Every check here has a clear
remedy in its message; a check with no remedy would just be a slower crash.

Only the architecture check is fatal. Memory checks warn, because a VRAM
estimate that refuses to start would make this worker impossible to run on
hardware it may well handle — the operator is better placed to judge than a
table of numbers is.
"""

from __future__ import annotations

import logging

from app.config import Config
from app.errors import ErrorCode, WorkerError

log = logging.getLogger(__name__)


def _gpu() -> tuple[str, tuple[int, int], float] | None:
    """(name, compute capability, total VRAM in GB), or None without CUDA."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    properties = torch.cuda.get_device_properties(0)
    return (
        str(properties.name),
        (int(properties.major), int(properties.minor)),
        float(properties.total_memory) / 1e9,
    )


def _system_ram_gb() -> float | None:
    try:
        import os

        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError, AttributeError):
        return None


def check(config: Config) -> None:
    """Validate hardware against the active profile."""
    variant = config.variant

    gpu = _gpu()
    if gpu is None:
        # Local development and CI: there is nothing to check, and refusing to
        # start would make the worker untestable off a GPU host.
        log.warning("no CUDA device visible; skipping hardware checks")
        return

    name, capability, vram_gb = gpu
    log.info(
        "gpu detected",
        extra={
            "gpu": name,
            "compute_capability": f"{capability[0]}.{capability[1]}",
            "vram_gb": round(vram_gb, 1),
        },
    )

    required = variant.requires_compute_capability
    if required is not None and capability < required:
        raise WorkerError(
            ErrorCode.UNSUPPORTED_GPU_ARCH,
            f"profile {variant.name} needs compute capability "
            f"{required[0]}.{required[1]} or newer (Blackwell: RTX 50xx, RTX PRO, "
            f"B200), but this worker has {name} at "
            f"{capability[0]}.{capability[1]}. Use FLUX2_VARIANT=klein-4b, or "
            "restrict the endpoint to Blackwell GPUs.",
        )

    if vram_gb < variant.hard_min_vram_gb:
        log.warning(
            "gpu has %.1f GB of VRAM, below the %d GB minimum for profile %s; "
            "generation is likely to fail with an out-of-memory error",
            vram_gb,
            variant.hard_min_vram_gb,
            variant.name,
        )
    elif vram_gb < variant.recommended_vram_gb:
        log.warning(
            "gpu has %.1f GB of VRAM, below the %d GB recommended for profile "
            "%s; large resolutions or several reference images may not fit",
            vram_gb,
            variant.recommended_vram_gb,
            variant.name,
        )

    ram_gb = _system_ram_gb()
    if ram_gb is not None and ram_gb < variant.system_ram_min_gb:
        log.warning(
            "host has %.1f GB of RAM, below the %d GB needed to stage this "
            "profile's weights; model loading may be killed by the OOM killer",
            ram_gb,
            variant.system_ram_min_gb,
        )
