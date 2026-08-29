"""A ComfyUI stand-in with failure injection.

The real client owns a subprocess and speaks HTTP. This one implements the same
surface in memory so the whole request path — validation, reference upload,
graph submission, result collection, error mapping — can be exercised without a
GPU. Failure injection matters more than the happy path: recovery behaviour is
exactly what never gets tested against real hardware.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from app.comfy_client import GeneratedImage
from app.errors import ErrorCode, WorkerError


def _png(width: int, height: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (10, 140, 90)).save(buffer, format="PNG")
    return buffer.getvalue()


@dataclass
class FakeComfy:
    """Records what it was asked to do and returns plausible results."""

    uploaded: list[tuple[str, bytes]] = field(default_factory=list)
    submitted: list[dict[str, Any]] = field(default_factory=list)
    cleared: int = 0
    restarts: int = 0
    freed: int = 0
    failures: list[WorkerError] = field(default_factory=list)

    # Injection points
    raise_on_submit: WorkerError | None = None
    raise_on_wait: WorkerError | None = None
    output_size: tuple[int, int] = (1024, 1024)
    available_models: dict[str, list[str]] | None = None

    # ------------------------------------------------------------------
    def known_models(self, class_type: str, input_name: str) -> list[str]:
        if self.available_models is None:
            return []
        return self.available_models.get(class_type, [])

    def clear_inputs(self) -> None:
        self.cleared += 1

    def upload_image(self, filename: str, data: bytes) -> str:
        self.uploaded.append((filename, data))
        return filename

    def submit(self, graph: dict[str, Any]) -> str:
        if self.raise_on_submit is not None:
            raise self.raise_on_submit
        self.submitted.append(graph)
        return f"prompt-{len(self.submitted)}"

    def wait_for_result(self, prompt_id: str) -> list[GeneratedImage]:
        if self.raise_on_wait is not None:
            raise self.raise_on_wait
        graph = self.submitted[-1]
        batch = graph["latent"]["inputs"]["batch_size"]
        width, height = self.output_size
        return [
            GeneratedImage(filename=f"{prompt_id}-{i}.png", data=_png(width, height))
            for i in range(batch)
        ]

    def free_memory(self) -> None:
        self.freed += 1

    def restart(self) -> None:
        self.restarts += 1

    def handle_failure(self, error: WorkerError) -> None:
        self.failures.append(error)
        if error.code in (ErrorCode.CUDA_OUT_OF_MEMORY, ErrorCode.INFERENCE_TIMEOUT):
            self.free_memory()
            self.restart()
