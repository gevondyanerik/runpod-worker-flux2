"""Error contract: stable codes, no stack traces, an honest retryable flag."""

from __future__ import annotations

import pytest

from app.errors import RETRYABLE, ErrorCode, WorkerError


def test_response_shape_is_stable() -> None:
    response = WorkerError(ErrorCode.INVALID_INPUT, "bad width").to_response()
    assert response == {
        "error": {
            "code": "INVALID_INPUT",
            "message": "bad width",
            "retryable": False,
        }
    }


def test_details_are_included_when_present() -> None:
    error = WorkerError(ErrorCode.TOO_MANY_IMAGES, "too many", details={"max": 6})
    assert error.to_response()["error"]["details"] == {"max": 6}


@pytest.mark.parametrize(
    "code",
    [
        ErrorCode.IMAGE_DOWNLOAD_FAILED,
        ErrorCode.CUDA_OUT_OF_MEMORY,
        ErrorCode.INFERENCE_TIMEOUT,
        ErrorCode.OUTPUT_UPLOAD_FAILED,
    ],
)
def test_transient_failures_are_retryable(code: ErrorCode) -> None:
    assert WorkerError(code, "x").retryable is True


@pytest.mark.parametrize(
    "code",
    [
        ErrorCode.INVALID_INPUT,
        ErrorCode.INVALID_RESOLUTION,
        ErrorCode.MISSING_PROMPT,
        ErrorCode.UNSUPPORTED_VARIANT,
        ErrorCode.UNSUPPORTED_GPU_ARCH,
        ErrorCode.OUTPUT_TOO_LARGE,
    ],
)
def test_caller_errors_are_not_retryable(code: ErrorCode) -> None:
    # Retrying these wastes GPU time and produces the same failure.
    assert WorkerError(code, "x").retryable is False


def test_every_retryable_code_is_a_real_code() -> None:
    for code in RETRYABLE:
        assert code in set(ErrorCode)


def test_codes_are_their_own_strings() -> None:
    # Clients switch on these; the enum name and the wire value must not drift.
    for code in ErrorCode:
        assert str(code) == code.name
