"""The Runpod Serverless handler.

Thin on purpose: validate, fetch references, build the graph, run it, encode
the result. Everything substantial lives in a module that can be tested without
a GPU.

Model selection happens once, at startup — never per request. One profile is
loaded per worker process, which is what makes the memory behaviour
predictable.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Any

from app import config as config_module
from app import constants, image_loader, logging_setup, output, schemas, workflow
from app.comfy_client import ComfyClient
from app.errors import ErrorCode, WorkerError

log = logging.getLogger(__name__)

# One generation at a time per GPU. ComfyUI's own queue would otherwise accept
# a second prompt and both would race for VRAM. Horizontal worker scaling is
# the right lever for throughput, not concurrency inside one worker.
_LOCK = threading.Lock()

_CONFIG: config_module.Config | None = None
_COMFY: ComfyClient | None = None
_GPU_NAME: str | None = None


def init(comfy: ComfyClient | None = None) -> None:
    """Resolve configuration and attach a ComfyUI client.

    Separated from module import so tests can inject a fake ComfyUI and run the
    whole request path on a laptop.
    """
    global _CONFIG, _COMFY, _GPU_NAME

    logging_setup.configure()
    _CONFIG = config_module.load()
    _COMFY = comfy or ComfyClient()

    if not _CONFIG.variant.is_ready:
        raise WorkerError(
            ErrorCode.PROFILE_NOT_READY,
            f"profile {_CONFIG.variant.name} has unconfirmed sampling defaults "
            "and must not be served",
        )

    _GPU_NAME = _detect_gpu()
    log.info("worker configured", extra=config_module.describe(_CONFIG))


def _detect_gpu() -> str | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return str(torch.cuda.get_device_name(0))


def handler(job: dict[str, Any]) -> dict[str, Any]:
    """Runpod entry point. Never raises: every failure becomes a coded error."""
    if _CONFIG is None or _COMFY is None:
        return WorkerError(
            ErrorCode.INFERENCE_FAILED, "worker is not initialised"
        ).to_response()

    job_id = str(job.get("id") or uuid.uuid4())
    payload = job.get("input") or {}

    if isinstance(payload, dict) and payload.get("op") == "capabilities":
        return schemas.capabilities(_CONFIG, _GPU_NAME)

    started = time.monotonic()
    try:
        with _LOCK:
            return _generate(payload, job_id, started)
    except WorkerError as error:
        log.warning(
            "job failed",
            extra={"job_id": job_id, "code": str(error.code)},
        )
        _COMFY.handle_failure(error)
        return error.to_response()
    except Exception as exc:  # pragma: no cover - defensive
        log.exception("unexpected failure", extra={"job_id": job_id})
        return WorkerError(
            ErrorCode.INFERENCE_FAILED,
            f"unexpected failure ({type(exc).__name__})",
        ).to_response()


def _generate(payload: dict[str, Any], job_id: str, started: float) -> dict[str, Any]:
    assert _CONFIG is not None and _COMFY is not None
    config, comfy = _CONFIG, _COMFY

    request = schemas.parse(payload, config)

    t0 = time.monotonic()
    references = image_loader.load_references(
        request.images, max_pixels=config.ref_max_pixels
    )
    download_ms = int((time.monotonic() - t0) * 1000)

    comfy.clear_inputs()
    uploaded: list[str] = []
    for reference in references:
        name = comfy.upload_image(f"{job_id}-ref{reference.index}.png", reference.data)
        uploaded.append(name)

    graph = workflow.build(
        variant=config.variant,
        prompt=request.prompt,
        width=request.width,
        height=request.height,
        seed=request.seed,
        steps=request.steps,
        guidance=request.guidance,
        batch_size=request.n,
        reference_filenames=tuple(uploaded),
        match_reference_size=bool(uploaded) and not request.size_explicit,
    )
    try:
        workflow.validate(graph, expected_references=len(uploaded))
    except workflow.WorkflowInvalidError as exc:
        raise WorkerError(ErrorCode.WORKFLOW_INVALID, str(exc)) from exc

    t1 = time.monotonic()
    prompt_id = comfy.submit(graph)
    generated = comfy.wait_for_result(prompt_id)
    inference_ms = int((time.monotonic() - t1) * 1000)

    t2 = time.monotonic()
    encoded = output.encode(
        [image.data for image in generated],
        request.seeds,
        fmt=request.output_format,
        quality=request.quality,
    )
    images = output.build_payload(encoded, config, job_id)
    upload_ms = int((time.monotonic() - t2) * 1000)

    # Reported from the decoded image, not from the request: with references
    # and no explicit size the graph takes its geometry from the first
    # reference, so the requested numbers would be wrong.
    out_width = encoded[0].width
    out_height = encoded[0].height

    total_ms = int((time.monotonic() - started) * 1000)
    log.info(
        "job completed",
        extra={
            "job_id": job_id,
            "variant": config.variant.name,
            "references": len(references),
            "width": out_width,
            "height": out_height,
            "steps": request.steps,
            "n": request.n,
            "inference_ms": inference_ms,
            "total_ms": total_ms,
            "gpu": _GPU_NAME,
        },
    )

    response: dict[str, Any] = {
        "images": images,
        "variant": config.variant.name,
        "width": out_width,
        "height": out_height,
        "steps": request.steps,
        "guidance": request.guidance,
        "reference_count": len(references),
        "timings_ms": {
            "download": download_ms,
            "inference": inference_ms,
            "upload": upload_ms,
            "total": total_ms,
        },
        "api_version": constants.API_VERSION,
    }
    if references:
        response["references"] = [ref.to_report() for ref in references]
    if request.adjusted or (out_width, out_height) != (request.width, request.height):
        response["adjusted"] = True
    return response


def main() -> None:
    """Boot the worker: check hardware, provision models, start ComfyUI, serve.

    The order is deliberate. Hardware first, because an unsupported GPU should
    fail before a multi-gigabyte download. Models next, because ComfyUI indexes
    its model directories at startup and will not see a file that arrives
    later.
    """
    import runpod

    from bootstrap import models as model_bootstrap
    from bootstrap import preflight

    logging_setup.configure()
    config = config_module.load()
    log.info("worker starting", extra=config_module.describe(config))

    preflight.check(config)
    model_bootstrap.ensure_assets(config)

    comfy = ComfyClient()
    comfy.start()
    model_bootstrap.verify_visible(comfy, config)

    init(comfy)
    log.info("worker ready")
    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    main()
