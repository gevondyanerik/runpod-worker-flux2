"""Turning generated images into a response.

base64 is the primary path, not a degraded fallback: a deployment with no
storage configured is a fully supported production configuration. S3 is opt-in,
and when it is not configured the worker never mentions it — a feature you have
not enabled should be invisible, not a reproach.
"""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass
from typing import Any

from PIL import Image

from app import constants
from app.config import Config
from app.errors import ErrorCode, WorkerError

log = logging.getLogger(__name__)

_MIME = {"webp": "image/webp", "png": "image/png", "jpeg": "image/jpeg"}


@dataclass(slots=True)
class EncodedImage:
    index: int
    seed: int
    data: bytes
    mime_type: str
    width: int
    height: int


def encode(
    images: list[bytes], seeds: list[int], *, fmt: str, quality: int
) -> list[EncodedImage]:
    """Re-encode ComfyUI's PNG output into the requested format."""
    encoded: list[EncodedImage] = []
    for index, raw in enumerate(images):
        image = Image.open(io.BytesIO(raw))
        image.load()
        buffer = io.BytesIO()
        if fmt == "png":
            image.save(buffer, format="PNG", optimize=True)
        elif fmt == "jpeg":
            image.convert("RGB").save(
                buffer, format="JPEG", quality=quality, optimize=True
            )
        else:
            image.save(buffer, format="WEBP", quality=quality, method=4)
        encoded.append(
            EncodedImage(
                index=index,
                seed=seeds[index] if index < len(seeds) else seeds[-1],
                data=buffer.getvalue(),
                mime_type=_MIME[fmt],
                width=image.width,
                height=image.height,
            )
        )
    return encoded


def build_payload(
    encoded: list[EncodedImage], config: Config, job_id: str
) -> list[dict[str, Any]]:
    """Return the ``images`` array, uploading to S3 when it is configured."""
    if config.uses_s3:
        return _upload(encoded, job_id)

    total = sum(len(item.data) for item in encoded)
    if total > constants.MAX_INLINE_RESPONSE_BYTES:
        raise WorkerError(
            ErrorCode.OUTPUT_TOO_LARGE,
            f"the response would be {total} bytes, above the "
            f"{constants.MAX_INLINE_RESPONSE_BYTES} byte inline limit. Request "
            "fewer images or a smaller resolution, or configure S3 output.",
            details={
                "bytes": total,
                "limit": constants.MAX_INLINE_RESPONSE_BYTES,
            },
        )

    return [
        {
            "b64": base64.b64encode(item.data).decode("ascii"),
            "mime_type": item.mime_type,
            "seed": item.seed,
            "index": item.index,
            "width": item.width,
            "height": item.height,
        }
        for item in encoded
    ]


def _upload(encoded: list[EncodedImage], job_id: str) -> list[dict[str, Any]]:
    """Upload through the Runpod SDK helper.

    The bucket variables are read by the SDK itself, which is why this worker
    reuses upstream's exact names rather than inventing a parallel set.
    """
    try:
        from runpod.serverless.utils import rp_upload
    except ImportError as exc:  # pragma: no cover - only in a broken image
        raise WorkerError(
            ErrorCode.OUTPUT_UPLOAD_FAILED,
            "S3 output is configured but the Runpod SDK upload helper is unavailable",
        ) from exc

    results: list[dict[str, Any]] = []
    for item in encoded:
        suffix = item.mime_type.split("/")[-1]
        try:
            url = rp_upload.upload_in_memory_object(
                job_id, item.data, f"{item.index}.{suffix}"
            )
        except Exception as exc:
            raise WorkerError(
                ErrorCode.OUTPUT_UPLOAD_FAILED,
                f"upload failed ({type(exc).__name__})",
            ) from exc
        results.append(
            {
                "url": url,
                "mime_type": item.mime_type,
                "seed": item.seed,
                "index": item.index,
                "width": item.width,
                "height": item.height,
            }
        )
    return results
