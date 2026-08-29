"""Response encoding. base64 is the primary path, not a fallback."""

from __future__ import annotations

import base64
import io
from dataclasses import replace

import pytest
from PIL import Image

from app import constants, output
from app.config import Config, S3Config
from app.errors import ErrorCode, WorkerError


def raw_png(width: int = 64, height: int = 64) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (12, 200, 90)).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("fmt", "mime"),
    [("webp", "image/webp"), ("png", "image/png"), ("jpeg", "image/jpeg")],
)
def test_every_format_encodes(fmt: str, mime: str) -> None:
    encoded = output.encode([raw_png()], [7], fmt=fmt, quality=90)
    assert len(encoded) == 1
    assert encoded[0].mime_type == mime
    assert Image.open(io.BytesIO(encoded[0].data)).size == (64, 64)


def test_dimensions_come_from_the_image_not_the_request() -> None:
    # With references and no explicit size the graph picks the geometry, so the
    # request's numbers would be wrong.
    encoded = output.encode([raw_png(768, 512)], [1], fmt="png", quality=90)
    assert (encoded[0].width, encoded[0].height) == (768, 512)


def test_each_image_carries_its_seed_and_index() -> None:
    encoded = output.encode([raw_png(), raw_png()], [5, 5], fmt="webp", quality=80)
    assert [(e.index, e.seed) for e in encoded] == [(0, 5), (1, 5)]


def test_base64_payload_is_self_describing(config: Config) -> None:
    encoded = output.encode([raw_png()], [3], fmt="webp", quality=80)
    payload = output.build_payload(encoded, config, "job-1")
    assert len(payload) == 1
    item = payload[0]
    assert set(item) == {"b64", "mime_type", "seed", "index", "width", "height"}
    assert base64.b64decode(item["b64"]) == encoded[0].data


def test_oversized_response_fails_with_a_usable_message(config: Config) -> None:
    # Better a clear error than a response the platform silently rejects.
    huge = output.EncodedImage(
        index=0,
        seed=1,
        data=b"\x00" * (constants.MAX_INLINE_RESPONSE_BYTES + 1),
        mime_type="image/png",
        width=4096,
        height=4096,
    )
    with pytest.raises(WorkerError) as excinfo:
        output.build_payload([huge], config, "job-1")
    assert excinfo.value.code is ErrorCode.OUTPUT_TOO_LARGE
    assert "S3" in excinfo.value.message


def test_s3_path_is_used_when_configured(
    config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    uploaded: list[tuple[str, str]] = []

    class FakeUpload:
        @staticmethod
        def upload_in_memory_object(job_id: str, data: bytes, name: str) -> str:
            uploaded.append((job_id, name))
            return f"https://bucket.invalid/{job_id}/{name}"

    import sys
    import types

    module = types.ModuleType("runpod.serverless.utils.rp_upload")
    module.upload_in_memory_object = FakeUpload.upload_in_memory_object  # type: ignore[attr-defined]
    utils = types.ModuleType("runpod.serverless.utils")
    utils.rp_upload = module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "runpod.serverless.utils", utils)
    monkeypatch.setitem(sys.modules, "runpod.serverless.utils.rp_upload", module)

    with_s3 = replace(config, s3=S3Config("https://s3.invalid", "key", "secret"))
    encoded = output.encode([raw_png()], [9], fmt="png", quality=90)
    payload = output.build_payload(encoded, with_s3, "job-42")

    assert uploaded == [("job-42", "0.png")]
    assert payload[0]["url"].endswith("job-42/0.png")
    assert "b64" not in payload[0]


def test_upload_failure_is_reported_as_retryable(
    config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys
    import types

    def boom(job_id: str, data: bytes, name: str) -> str:
        raise ConnectionError("bucket unreachable")

    module = types.ModuleType("runpod.serverless.utils.rp_upload")
    module.upload_in_memory_object = boom  # type: ignore[attr-defined]
    utils = types.ModuleType("runpod.serverless.utils")
    utils.rp_upload = module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "runpod.serverless.utils", utils)
    monkeypatch.setitem(sys.modules, "runpod.serverless.utils.rp_upload", module)

    with_s3 = replace(config, s3=S3Config("https://s3.invalid", "key", "secret"))
    encoded = output.encode([raw_png()], [9], fmt="png", quality=90)
    with pytest.raises(WorkerError) as excinfo:
        output.build_payload(encoded, with_s3, "job-42")
    assert excinfo.value.code is ErrorCode.OUTPUT_UPLOAD_FAILED
    assert excinfo.value.retryable is True
