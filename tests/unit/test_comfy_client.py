"""The ComfyUI client's parsing and recovery logic, without a subprocess.

The recovery paths matter more than the happy path here: they are what never
gets exercised against real hardware, and getting them wrong means a worker
that fails every job it is handed instead of dying and being replaced.
"""

from __future__ import annotations

import json

import pytest

from app import constants
from app.comfy_client import ComfyClient
from app.errors import ErrorCode, WorkerError


@pytest.fixture
def client() -> ComfyClient:
    return ComfyClient()


def patch(monkeypatch: pytest.MonkeyPatch, name: str, replacement) -> None:
    """Replace a method on the class.

    ``ComfyClient`` is a slots dataclass, so per-instance patching is not
    possible — and the slots are worth keeping for a type that exists once per
    worker and is touched on every job.
    """
    monkeypatch.setattr(ComfyClient, name, replacement)


# ------------------------------------------------------------ error reporting


def test_prompt_error_extracts_the_node_message(client: ComfyClient) -> None:
    body = json.dumps(
        {
            "error": {"message": "Prompt outputs failed validation"},
            "node_errors": {
                "unet": {
                    "errors": [{"message": "value not in list: model.safetensors"}]
                }
            },
        }
    ).encode()
    detail = client._prompt_error(body)
    assert "failed validation" in detail
    assert "model.safetensors" in detail


def test_prompt_error_survives_a_non_json_body(client: ComfyClient) -> None:
    # ComfyUI can return an HTML error page; that must not mask the failure
    # with a JSONDecodeError.
    assert "Internal Server Error" in client._prompt_error(
        b"<html>Internal Server Error</html>"
    )


def test_out_of_memory_gets_its_own_code(client: ComfyClient) -> None:
    status = {
        "messages": [
            [
                "execution_error",
                {
                    "exception_type": "torch.OutOfMemoryError",
                    "exception_message": "CUDA out of memory. Tried to allocate 2 GiB",
                },
            ]
        ]
    }
    with pytest.raises(WorkerError) as excinfo:
        client._raise_execution_error(status)
    assert excinfo.value.code is ErrorCode.CUDA_OUT_OF_MEMORY
    # The message must suggest something the caller can actually do.
    assert "fewer reference images" in excinfo.value.message


def test_other_failures_stay_generic(client: ComfyClient) -> None:
    status = {
        "messages": [
            ["execution_error", {"exception_message": "size mismatch for weight"}]
        ]
    }
    with pytest.raises(WorkerError) as excinfo:
        client._raise_execution_error(status)
    assert excinfo.value.code is ErrorCode.INFERENCE_FAILED
    assert "size mismatch" in excinfo.value.message


def test_an_empty_status_still_fails_cleanly(client: ComfyClient) -> None:
    with pytest.raises(WorkerError) as excinfo:
        client._raise_execution_error({})
    assert excinfo.value.code is ErrorCode.INFERENCE_FAILED


# ------------------------------------------------------------ model discovery


def test_known_models_reads_the_loader_options(
    client: ComfyClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch(
        monkeypatch,
        "object_info",
        lambda self, class_type: {
            "UNETLoader": {
                "input": {
                    "required": {
                        "unet_name": [["a.safetensors", "b.safetensors"]],
                        "weight_dtype": [["default"]],
                    }
                }
            }
        },
    )
    assert client.known_models("UNETLoader", "unet_name") == [
        "a.safetensors",
        "b.safetensors",
    ]


def test_known_models_is_empty_when_comfyui_says_nothing(
    client: ComfyClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch(monkeypatch, "object_info", lambda self, class_type: {})
    assert client.known_models("UNETLoader", "unet_name") == []


# ----------------------------------------------------------------- collection


def test_collect_ignores_temp_previews(
    client: ComfyClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    viewed: list[str] = []

    def fake_view(self, filename: str, subfolder: str, type_: str) -> bytes:
        viewed.append(filename)
        return b"PNGDATA"

    patch(monkeypatch, "_view", fake_view)
    outputs = {
        "save": {
            "images": [
                {"filename": "final.png", "subfolder": "", "type": "output"},
                {"filename": "preview.png", "subfolder": "", "type": "temp"},
            ]
        }
    }
    images = client._collect(outputs)
    assert [image.filename for image in images] == ["final.png"]
    assert viewed == ["final.png"]


def test_collect_fails_when_nothing_was_produced(client: ComfyClient) -> None:
    with pytest.raises(WorkerError) as excinfo:
        client._collect({"save": {"images": []}})
    assert excinfo.value.code is ErrorCode.INFERENCE_FAILED


# ------------------------------------------------------------------- recovery


def test_a_validation_failure_leaves_the_worker_alone(
    client: ComfyClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Restarting ComfyUI because a caller sent a bad width would turn a cheap
    # 400 into a cold start.
    restarted: list[int] = []
    patch(monkeypatch, "restart", lambda self: restarted.append(1))
    client.handle_failure(WorkerError(ErrorCode.INVALID_RESOLUTION, "bad size"))
    assert restarted == []


def test_the_first_out_of_memory_restarts(
    client: ComfyClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    patch(monkeypatch, "restart", lambda self: events.append("restart"))
    patch(monkeypatch, "free_memory", lambda self: events.append("free"))
    patch(monkeypatch, "_exit", lambda self: events.append("exit"))

    client.handle_failure(WorkerError(ErrorCode.CUDA_OUT_OF_MEMORY, "oom"))
    assert events == ["free", "restart"]


def test_a_repeated_out_of_memory_exits(
    client: ComfyClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A worker whose GPU is in a bad state keeps pulling jobs off the queue and
    # failing them. The platform replaces a dead worker; it keeps feeding a
    # sick one.
    events: list[str] = []
    patch(monkeypatch, "restart", lambda self: events.append("restart"))
    patch(monkeypatch, "free_memory", lambda self: events.append("free"))
    patch(monkeypatch, "_exit", lambda self: events.append("exit"))

    for _ in range(constants.OOM_STRIKES_BEFORE_EXIT):
        client.handle_failure(WorkerError(ErrorCode.CUDA_OUT_OF_MEMORY, "oom"))

    assert "exit" in events


def test_old_strikes_expire(
    client: ComfyClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # One oversized request an hour apart is not a broken worker.
    events: list[str] = []
    patch(monkeypatch, "restart", lambda self: events.append("restart"))
    patch(monkeypatch, "free_memory", lambda self: None)
    patch(monkeypatch, "_exit", lambda self: events.append("exit"))

    clock = [1000.0]
    monkeypatch.setattr("app.comfy_client.time.monotonic", lambda: clock[0])

    client.handle_failure(WorkerError(ErrorCode.CUDA_OUT_OF_MEMORY, "oom"))
    clock[0] += constants.OOM_STRIKE_WINDOW_S + 1
    client.handle_failure(WorkerError(ErrorCode.CUDA_OUT_OF_MEMORY, "oom"))

    assert "exit" not in events
    assert events.count("restart") == 2


def test_exiting_when_a_restart_fails(
    client: ComfyClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []

    def failing_restart(self) -> None:
        raise WorkerError(ErrorCode.COMFYUI_START_FAILED, "did not come back")

    patch(monkeypatch, "restart", failing_restart)
    patch(monkeypatch, "free_memory", lambda self: None)
    patch(monkeypatch, "_exit", lambda self: events.append("exit"))

    client.handle_failure(WorkerError(ErrorCode.INFERENCE_TIMEOUT, "too slow"))
    assert events == ["exit"]
