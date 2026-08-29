"""Owning the ComfyUI process and talking to its HTTP API.

ComfyUI is an implementation detail: it listens on loopback, nothing outside
the container can reach it, and its configuration is not exposed to operators.

The part that matters most here is failure handling. On serverless, a worker
whose GPU is in a bad state keeps pulling jobs off the queue and failing them,
which is worse than a worker that dies: the platform replaces a dead worker but
happily keeps feeding a sick one. So an out-of-memory or a dead ComfyUI first
triggers a restart, and if that does not help — or it happens again inside the
strike window — the process exits and lets Runpod schedule a fresh worker.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import urllib3

from app import constants
from app.errors import ErrorCode, WorkerError

log = logging.getLogger(__name__)

_OOM_MARKERS = (
    "out of memory",
    "cuda error: out of memory",
    "torch.outofmemoryerror",
    "hip out of memory",
)


@dataclass(slots=True)
class GeneratedImage:
    filename: str
    data: bytes


@dataclass(slots=True)
class ComfyClient:
    base_url: str = constants.COMFY_BASE_URL
    client_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    _http: urllib3.PoolManager = field(
        default_factory=lambda: urllib3.PoolManager(retries=False), repr=False
    )
    _process: subprocess.Popen[bytes] | None = field(default=None, repr=False)
    _oom_times: list[float] = field(default_factory=list, repr=False)

    # ------------------------------------------------------------------
    # Process lifecycle
    # ------------------------------------------------------------------
    def start(self, timeout_s: int = constants.COMFY_START_TIMEOUT_S) -> None:
        if self.is_healthy():
            log.info("comfyui already running")
            return

        env = dict(os.environ)
        env["PYTORCH_CUDA_ALLOC_CONF"] = constants.TORCH_ALLOC_CONF

        log.info("starting comfyui")
        # sys.executable, not "python3": the same interpreter that imported this
        # module, so ComfyUI cannot end up on a different Python than the one
        # the worker's dependencies were installed into.
        self._process = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
            [
                sys.executable,
                "main.py",
                "--listen",
                constants.COMFY_HOST,
                "--port",
                str(constants.COMFY_PORT),
                "--disable-auto-launch",
                "--disable-metadata",
            ],
            cwd=constants.COMFY_ROOT,
            env=env,
        )
        self.wait_until_healthy(timeout_s)

    def wait_until_healthy(self, timeout_s: int) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.is_healthy():
                log.info("comfyui is ready")
                return
            if self._process is not None and self._process.poll() is not None:
                raise WorkerError(
                    ErrorCode.COMFYUI_START_FAILED,
                    "ComfyUI exited during startup with code "
                    f"{self._process.returncode}",
                )
            time.sleep(constants.COMFY_HEALTH_POLL_INTERVAL_S)
        raise WorkerError(
            ErrorCode.COMFYUI_START_FAILED,
            f"ComfyUI did not become healthy within {timeout_s}s",
        )

    def is_healthy(self) -> bool:
        try:
            response = self._http.request(
                "GET", f"{self.base_url}/system_stats", timeout=2.0
            )
            return response.status == 200
        except Exception:
            return False

    def stop(self) -> None:
        process = self._process
        if process is None or process.poll() is not None:
            return
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)

    def restart(self) -> None:
        log.warning("restarting comfyui")
        self.stop()
        self._process = None
        self.start(constants.COMFY_RESTART_TIMEOUT_S)

    # ------------------------------------------------------------------
    # Model visibility
    # ------------------------------------------------------------------
    def object_info(self, class_type: str) -> dict[str, Any]:
        response = self._http.request(
            "GET", f"{self.base_url}/object_info/{class_type}", timeout=30.0
        )
        if response.status != 200:
            return {}
        payload: dict[str, Any] = json.loads(response.data)
        return payload

    def known_models(self, class_type: str, input_name: str) -> list[str]:
        """The filenames ComfyUI can actually see for a loader input.

        Used at startup so a missing checkpoint fails with a clear message
        instead of an opaque prompt-validation error on the first request.
        """
        info = self.object_info(class_type).get(class_type, {})
        required = info.get("input", {}).get("required", {})
        entry = required.get(input_name)
        if isinstance(entry, list) and entry and isinstance(entry[0], list):
            return [str(name) for name in entry[0]]
        return []

    # ------------------------------------------------------------------
    # Job execution
    # ------------------------------------------------------------------
    def upload_image(self, filename: str, data: bytes) -> str:
        response = self._http.request(
            "POST",
            f"{self.base_url}/upload/image",
            fields={
                "image": (filename, data, "image/png"),
                "overwrite": "true",
                "type": "input",
            },
            timeout=60.0,
        )
        if response.status != 200:
            raise WorkerError(
                ErrorCode.INFERENCE_FAILED,
                f"ComfyUI rejected reference upload (HTTP {response.status})",
            )
        payload = json.loads(response.data)
        name = str(payload.get("name", filename))
        subfolder = payload.get("subfolder") or ""
        return f"{subfolder}/{name}" if subfolder else name

    def submit(self, graph: dict[str, Any]) -> str:
        body = json.dumps({"prompt": graph, "client_id": self.client_id}).encode()
        response = self._http.request(
            "POST",
            f"{self.base_url}/prompt",
            body=body,
            headers={"Content-Type": "application/json"},
            timeout=60.0,
        )
        if response.status != 200:
            detail = self._prompt_error(response.data)
            raise WorkerError(
                ErrorCode.WORKFLOW_INVALID,
                f"ComfyUI rejected the workflow: {detail}",
            )
        return str(json.loads(response.data)["prompt_id"])

    @staticmethod
    def _prompt_error(data: bytes) -> str:
        try:
            payload = json.loads(data)
        except (json.JSONDecodeError, ValueError):
            return data.decode("utf-8", "replace")[:400]
        error = payload.get("error") or {}
        message = error.get("message") or "unknown error"
        node_errors = payload.get("node_errors") or {}
        if node_errors:
            first = next(iter(node_errors.values()))
            errors = first.get("errors") or []
            if errors:
                message = f"{message}: {errors[0].get('message', '')}"
        return str(message)[:400]

    def wait_for_result(
        self, prompt_id: str, timeout_s: int = constants.INFERENCE_TIMEOUT_S
    ) -> list[GeneratedImage]:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            history = self._history(prompt_id)
            if history:
                status = history.get("status", {})
                if status.get("status_str") == "error":
                    self._raise_execution_error(status)

                outputs = history.get("outputs") or {}
                if outputs:
                    return self._collect(outputs)

                # Finished, but with nothing to show. Only conclusive once the
                # entry says it completed: an entry that is merely present may
                # still be executing, and treating that as a failure would kill
                # every slow job.
                if status.get("completed") is True:
                    raise WorkerError(
                        ErrorCode.INFERENCE_FAILED,
                        "the workflow completed without producing an image",
                    )
            if self._process is not None and self._process.poll() is not None:
                raise WorkerError(
                    ErrorCode.INFERENCE_FAILED, "ComfyUI died while executing the job"
                )
            time.sleep(0.5)

        raise WorkerError(
            ErrorCode.INFERENCE_TIMEOUT,
            f"generation exceeded the {timeout_s}s limit",
        )

    def _history(self, prompt_id: str) -> dict[str, Any]:
        try:
            response = self._http.request(
                "GET", f"{self.base_url}/history/{prompt_id}", timeout=30.0
            )
        except Exception:
            return {}
        if response.status != 200:
            return {}
        payload = json.loads(response.data)
        entry: dict[str, Any] = payload.get(prompt_id) or {}
        return entry

    def _raise_execution_error(self, status: dict[str, Any]) -> None:
        messages: list[str] = []
        for item in status.get("messages") or []:
            if isinstance(item, list) and len(item) >= 2 and isinstance(item[1], dict):
                for key in ("exception_message", "exception_type"):
                    value = item[1].get(key)
                    if value:
                        messages.append(str(value))
        detail = " | ".join(messages) or "execution failed"
        if any(marker in detail.lower() for marker in _OOM_MARKERS):
            raise WorkerError(
                ErrorCode.CUDA_OUT_OF_MEMORY,
                "the GPU ran out of memory for this request; try fewer reference "
                "images or a smaller resolution",
            )
        raise WorkerError(ErrorCode.INFERENCE_FAILED, detail[:400])

    def _collect(self, outputs: dict[str, Any]) -> list[GeneratedImage]:
        images: list[GeneratedImage] = []
        for node_output in outputs.values():
            for image in node_output.get("images", []):
                if image.get("type") != "output":
                    continue
                images.append(
                    GeneratedImage(
                        filename=image["filename"],
                        data=self._view(
                            image["filename"],
                            image.get("subfolder", ""),
                            image.get("type", "output"),
                        ),
                    )
                )
        if not images:
            raise WorkerError(
                ErrorCode.INFERENCE_FAILED, "the workflow produced no output image"
            )
        return images

    def _view(self, filename: str, subfolder: str, type_: str) -> bytes:
        response = self._http.request(
            "GET",
            f"{self.base_url}/view",
            fields={"filename": filename, "subfolder": subfolder, "type": type_},
            timeout=120.0,
        )
        if response.status != 200:
            raise WorkerError(
                ErrorCode.INFERENCE_FAILED,
                f"could not read the generated image (HTTP {response.status})",
            )
        return bytes(response.data)

    def free_memory(self) -> None:
        """Ask ComfyUI to unload models and free VRAM."""
        try:
            self._http.request(
                "POST",
                f"{self.base_url}/free",
                body=json.dumps({"unload_models": True, "free_memory": True}).encode(),
                headers={"Content-Type": "application/json"},
                timeout=60.0,
            )
        except Exception:  # best effort; the caller is already in a bad state
            log.warning("free_memory call failed", exc_info=True)

    # ------------------------------------------------------------------
    # Recovery
    # ------------------------------------------------------------------
    def handle_failure(self, error: WorkerError) -> None:
        """Recover after a failed job, or exit so Runpod replaces this worker.

        Called once per failed job. Anything that leaves the GPU in an unknown
        state — OOM, a dead process — must not be met with "carry on".
        """
        fatal = error.code in (
            ErrorCode.CUDA_OUT_OF_MEMORY,
            ErrorCode.INFERENCE_TIMEOUT,
        )
        dead = self._process is not None and self._process.poll() is not None

        if not fatal and not dead:
            return

        if error.code is ErrorCode.CUDA_OUT_OF_MEMORY:
            now = time.monotonic()
            self._oom_times = [
                t for t in self._oom_times if now - t < constants.OOM_STRIKE_WINDOW_S
            ]
            self._oom_times.append(now)
            if len(self._oom_times) >= constants.OOM_STRIKES_BEFORE_EXIT:
                log.error(
                    "%d out-of-memory failures within %ds; exiting so the platform "
                    "replaces this worker",
                    len(self._oom_times),
                    constants.OOM_STRIKE_WINDOW_S,
                )
                self._exit()

        self.free_memory()
        try:
            self.restart()
        except WorkerError:
            log.exception("comfyui could not be restarted; exiting")
            self._exit()

    def _exit(self) -> None:
        self.stop()
        os._exit(1)

    # ------------------------------------------------------------------
    def clear_inputs(self) -> None:
        """Remove uploaded reference images between jobs."""
        inputs = os.path.join(constants.COMFY_ROOT, "input")
        if not os.path.isdir(inputs):
            return
        for entry in os.scandir(inputs):
            if entry.is_file():
                with contextlib.suppress(OSError):
                    os.unlink(entry.path)
            elif entry.is_dir():
                shutil.rmtree(entry.path, ignore_errors=True)
