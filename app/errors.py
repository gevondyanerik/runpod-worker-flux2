"""Stable, machine-readable error codes.

Clients get a code, a human-readable message and a ``retryable`` flag. They
never get a stack trace — those are logged server-side.

``retryable`` exists because it is not inferable from the code alone: an
integrator should retry ``IMAGE_DOWNLOAD_FAILED`` and must not retry
``INVALID_RESOLUTION``. Encoding that here saves every caller from guessing.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class ErrorCode(StrEnum):
    # Request validation
    INVALID_INPUT = "INVALID_INPUT"
    MISSING_PROMPT = "MISSING_PROMPT"
    INVALID_RESOLUTION = "INVALID_RESOLUTION"
    TOO_MANY_IMAGES = "TOO_MANY_IMAGES"

    # Reference images
    INVALID_IMAGE_URL = "INVALID_IMAGE_URL"
    IMAGE_DOWNLOAD_FAILED = "IMAGE_DOWNLOAD_FAILED"
    IMAGE_TOO_LARGE = "IMAGE_TOO_LARGE"
    TOTAL_INPUT_TOO_LARGE = "TOTAL_INPUT_TOO_LARGE"
    INVALID_IMAGE = "INVALID_IMAGE"

    # Deployment / startup
    UNSUPPORTED_VARIANT = "UNSUPPORTED_VARIANT"
    UNSUPPORTED_GPU_ARCH = "UNSUPPORTED_GPU_ARCH"
    INSUFFICIENT_VRAM = "INSUFFICIENT_VRAM"
    INSUFFICIENT_SYSTEM_RAM = "INSUFFICIENT_SYSTEM_RAM"
    INSUFFICIENT_VRAM_FOR_REFERENCES = "INSUFFICIENT_VRAM_FOR_REFERENCES"
    MODEL_ASSET_MISSING = "MODEL_ASSET_MISSING"
    MODEL_CHECKSUM_MISMATCH = "MODEL_CHECKSUM_MISMATCH"
    MODEL_AUTH_REQUIRED = "MODEL_AUTH_REQUIRED"
    INSUFFICIENT_DISK = "INSUFFICIENT_DISK"
    PROFILE_NOT_READY = "PROFILE_NOT_READY"
    COMFYUI_START_FAILED = "COMFYUI_START_FAILED"
    WORKFLOW_INVALID = "WORKFLOW_INVALID"

    # Inference
    CUDA_OUT_OF_MEMORY = "CUDA_OUT_OF_MEMORY"
    INFERENCE_FAILED = "INFERENCE_FAILED"
    INFERENCE_TIMEOUT = "INFERENCE_TIMEOUT"

    # Output
    OUTPUT_TOO_LARGE = "OUTPUT_TOO_LARGE"
    OUTPUT_UPLOAD_FAILED = "OUTPUT_UPLOAD_FAILED"


RETRYABLE: frozenset[ErrorCode] = frozenset(
    {
        ErrorCode.IMAGE_DOWNLOAD_FAILED,
        ErrorCode.CUDA_OUT_OF_MEMORY,
        ErrorCode.INFERENCE_FAILED,
        ErrorCode.INFERENCE_TIMEOUT,
        ErrorCode.OUTPUT_UPLOAD_FAILED,
        ErrorCode.COMFYUI_START_FAILED,
    }
)


class WorkerError(Exception):
    """An error with a stable code, safe to return to a client."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}

    @property
    def retryable(self) -> bool:
        return self.code in RETRYABLE

    def to_response(self) -> dict[str, Any]:
        """The wire form of a failure.

        The structure sits at the top level rather than nested under ``error``,
        and that is forced by the platform. Runpod's SDK pops a top-level
        ``error`` out of whatever the handler returns (``rp_job.py``), moves it
        to the job's own error field, and then drops ``output`` altogether when
        the pop left it empty. The platform keeps that field only if it is a
        string, so a structured error returned under ``error`` alone arrived as
        a job with no output *and* no error: every code defined here was
        invisible in production while passing every test locally.

        ``error`` therefore carries a plain string, for Runpod's own reporting,
        and everything a caller needs sits beside it where nothing removes it.
        """
        payload: dict[str, Any] = {
            "error": f"{self.code}: {self.message}",
            "code": str(self.code),
            "message": self.message,
            "retryable": self.retryable,
        }
        if self.details:
            payload["details"] = self.details
        return payload
