"""The whole request path, against a ComfyUI stand-in.

This is the test that matters most: it exercises validation, reference
handling, graph construction, execution and encoding together, on a laptop,
with no GPU and no network.
"""

from __future__ import annotations

import base64
import io
import threading
from typing import Any

import pytest
from PIL import Image

from app import config
from app import handler as handler_module
from app.errors import ErrorCode, WorkerError
from tests.fake_comfy import FakeComfy


@pytest.fixture
def comfy(monkeypatch: pytest.MonkeyPatch) -> FakeComfy:
    fake = FakeComfy()
    handler_module.init(fake)  # type: ignore[arg-type]
    yield fake
    handler_module._CONFIG = None
    handler_module._COMFY = None


def data_uri(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode()


def run(payload: dict[str, Any], job_id: str = "job-1") -> dict[str, Any]:
    return handler_module.handler({"id": job_id, "input": payload})


# ------------------------------------------------------------------ happy paths


def test_prompt_only_produces_an_image(comfy: FakeComfy) -> None:
    response = run({"prompt": "a red bicycle"})
    assert "error" not in response
    assert len(response["images"]) == 1
    assert response["variant"] == "klein-4b"
    assert response["reference_count"] == 0
    assert response["api_version"] == "1"
    assert set(response["timings_ms"]) == {"download", "inference", "upload", "total"}

    decoded = base64.b64decode(response["images"][0]["b64"])
    assert Image.open(io.BytesIO(decoded)).size == (1024, 1024)


def test_defaults_come_from_the_profile(comfy: FakeComfy) -> None:
    run({"prompt": "x"})
    graph = comfy.submitted[0]
    assert graph["sigmas"]["inputs"]["steps"] == 4
    assert graph["guider"]["inputs"]["cfg"] == 1.0
    assert graph["sampler"]["inputs"]["sampler_name"] == "euler"


def test_a_batch_returns_that_many_images(comfy: FakeComfy) -> None:
    response = run({"prompt": "x", "n": 3, "seed": 11})
    assert len(response["images"]) == 3
    assert [image["index"] for image in response["images"]] == [0, 1, 2]
    assert {image["seed"] for image in response["images"]} == {11}


def test_references_are_uploaded_and_wired(comfy: FakeComfy, png_bytes: bytes) -> None:
    response = run({"prompt": "edit this", "images": [data_uri(png_bytes)] * 2})
    assert response["reference_count"] == 2
    assert len(comfy.uploaded) == 2
    assert comfy.cleared == 1  # inputs wiped before the job, not left to pile up

    graph = comfy.submitted[0]
    assert graph["guider"]["inputs"]["positive"] == ["ref1_pos", 0]
    assert response["references"][0]["index"] == 0


def test_an_edit_without_a_size_follows_the_reference(
    comfy: FakeComfy, png_bytes: bytes
) -> None:
    comfy.output_size = (896, 672)
    response = run({"prompt": "edit", "images": [data_uri(png_bytes)]})
    graph = comfy.submitted[0]
    assert graph["latent"]["inputs"]["width"] == ["ref_size", 0]
    # The reported size is the image that came back, not the one we assumed.
    assert (response["width"], response["height"]) == (896, 672)
    assert response["adjusted"] is True


def test_an_explicit_size_is_honoured(comfy: FakeComfy, png_bytes: bytes) -> None:
    comfy.output_size = (768, 512)
    response = run(
        {"prompt": "edit", "images": [data_uri(png_bytes)], "width": 768, "height": 512}
    )
    graph = comfy.submitted[0]
    assert graph["latent"]["inputs"]["width"] == 768
    assert (response["width"], response["height"]) == (768, 512)
    assert "adjusted" not in response


def test_capabilities_needs_no_gpu(comfy: FakeComfy) -> None:
    response = handler_module.handler({"id": "job", "input": {"op": "capabilities"}})
    assert response["variant"] == "klein-4b"
    assert response["license"] == "apache-2.0"
    assert comfy.submitted == []


# ---------------------------------------------------------------------- failures


def test_a_validation_error_never_reaches_comfyui(comfy: FakeComfy) -> None:
    response = run({})
    assert response["error"]["code"] == "MISSING_PROMPT"
    assert response["error"]["retryable"] is False
    assert comfy.submitted == []


def test_an_unknown_field_is_refused(comfy: FakeComfy) -> None:
    response = run({"prompt": "x", "sampler": "dpmpp_2m"})
    assert response["error"]["code"] == "INVALID_INPUT"


def test_out_of_memory_is_reported_and_recovered(comfy: FakeComfy) -> None:
    comfy.raise_on_wait = WorkerError(
        ErrorCode.CUDA_OUT_OF_MEMORY, "the GPU ran out of memory"
    )
    response = run({"prompt": "x"})
    assert response["error"]["code"] == "CUDA_OUT_OF_MEMORY"
    assert response["error"]["retryable"] is True
    # A worker whose GPU is in an unknown state must not just carry on.
    assert comfy.restarts == 1
    assert comfy.freed == 1


def test_a_rejected_graph_surfaces_as_workflow_invalid(comfy: FakeComfy) -> None:
    comfy.raise_on_submit = WorkerError(ErrorCode.WORKFLOW_INVALID, "bad node")
    response = run({"prompt": "x"})
    assert response["error"]["code"] == "WORKFLOW_INVALID"


def test_an_unexpected_exception_becomes_a_coded_error(
    comfy: FakeComfy, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The handler is the last line of defence: Runpod must never see a traceback
    # where a response belongs.
    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("something exploded")

    monkeypatch.setattr(handler_module.schemas, "parse", boom)
    response = run({"prompt": "x"})
    assert response["error"]["code"] == "INFERENCE_FAILED"
    assert "traceback" not in str(response).lower()


def test_a_bad_reference_url_fails_before_the_gpu(comfy: FakeComfy) -> None:
    response = run({"prompt": "x", "images": ["file:///etc/passwd"]})
    assert response["error"]["code"] == "INVALID_IMAGE_URL"
    assert comfy.submitted == []


def test_an_uninitialised_worker_answers_instead_of_crashing() -> None:
    handler_module._CONFIG = None
    handler_module._COMFY = None
    response = handler_module.handler({"id": "job", "input": {"prompt": "x"}})
    assert response["error"]["code"] == "INFERENCE_FAILED"


# ------------------------------------------------------------------- boot order


@pytest.fixture
def booting(monkeypatch: pytest.MonkeyPatch) -> threading.Event:
    """A worker mid-boot: registered with Runpod, ComfyUI not up yet."""
    event = threading.Event()
    monkeypatch.setattr(handler_module, "_BOOT", event)
    monkeypatch.setattr(handler_module, "_BOOT_ERROR", None)
    monkeypatch.setattr(handler_module, "_CONFIG", config.load())
    monkeypatch.setattr(handler_module, "_COMFY", None)
    return event


def test_capabilities_answers_while_the_worker_is_still_booting(
    booting: threading.Event,
) -> None:
    # The reason config is published before the boot thread starts: this must
    # not block on ComfyUI, and Runpod's own health probing uses it.
    response = handler_module.handler({"id": "job", "input": {"op": "capabilities"}})
    assert response["variant"] == "klein-4b"
    assert not booting.is_set()


def test_a_job_waits_for_the_boot_instead_of_failing(
    booting: threading.Event, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeComfy()

    def finish() -> None:
        handler_module.init(fake)  # type: ignore[arg-type]
        booting.set()

    threading.Timer(0.05, finish).start()
    response = handler_module.handler({"id": "job", "input": {"prompt": "x"}})
    assert "error" not in response
    assert len(response["images"]) == 1


def test_a_boot_failure_becomes_the_job_error(
    booting: threading.Event, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A worker that cannot start must say why on every job, not crash on the
    # way up where Runpod only sees an unresponsive runtime.
    monkeypatch.setattr(
        handler_module,
        "_BOOT_ERROR",
        WorkerError(ErrorCode.UNSUPPORTED_GPU_ARCH, "needs Blackwell"),
    )
    booting.set()
    response = handler_module.handler({"id": "job", "input": {"prompt": "x"}})
    assert response["error"]["code"] == "UNSUPPORTED_GPU_ARCH"


def test_a_job_gives_up_on_a_boot_that_never_finishes(
    booting: threading.Event, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(handler_module.constants, "WORKER_BOOT_TIMEOUT_S", 0.05)
    response = handler_module.handler({"id": "job", "input": {"prompt": "x"}})
    assert response["error"]["code"] == "COMFYUI_START_FAILED"


def test_boot_records_a_failure_rather_than_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # _boot runs on a thread nobody joins, so it must never let an exception
    # escape: the event has to be set either way or every job hangs.
    def boom(_: object) -> None:
        raise RuntimeError("no GPU here")

    monkeypatch.setattr(handler_module, "_BOOT_ERROR", None)
    monkeypatch.setattr("bootstrap.preflight.check", boom)
    ready = threading.Event()
    handler_module._boot(config.load(), ready)
    assert ready.is_set()
    assert handler_module._BOOT_ERROR is not None
    assert handler_module._BOOT_ERROR.code is ErrorCode.COMFYUI_START_FAILED
