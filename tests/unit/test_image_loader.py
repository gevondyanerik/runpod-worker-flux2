"""Reference loading — the worker's main attack surface.

An operator deploys this inside their own network and then lets arbitrary
callers hand it URLs. Every test here is about a URL that must not be fetched.
"""

from __future__ import annotations

import base64
import io
import socket

import pytest
from PIL import Image

from app import image_loader
from app.errors import ErrorCode, WorkerError


def data_uri(payload: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64,{base64.b64encode(payload).decode()}"


# ---------------------------------------------------------------- address rules


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",  # loopback
        "10.0.0.5",  # RFC1918
        "192.168.1.1",
        "172.16.0.1",
        "169.254.169.254",  # cloud metadata: the prize
        "100.64.0.1",  # carrier-grade NAT
        "0.0.0.0",
        "::1",
        "fe80::1",
        "fc00::1",
        "::ffff:127.0.0.1",  # loopback wearing an IPv6 hat
        "::ffff:169.254.169.254",
        "224.0.0.1",
        "not-an-address",
    ],
)
def test_non_public_addresses_are_refused(address: str) -> None:
    assert image_loader._is_public(address) is False


@pytest.mark.parametrize(
    "address", ["1.1.1.1", "8.8.8.8", "93.184.216.34", "2606:4700::1"]
)
def test_public_addresses_are_allowed(address: str) -> None:
    assert image_loader._is_public(address) is True


def test_a_host_resolving_to_a_private_address_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_getaddrinfo(host: str, port: int, **kwargs: object) -> list:
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", port))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    with pytest.raises(WorkerError) as excinfo:
        image_loader._resolve("metadata.example", 443, 0)
    assert excinfo.value.code is ErrorCode.INVALID_IMAGE_URL
    assert "non-public" in excinfo.value.message


def test_a_host_with_one_bad_address_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A host that resolves to both a public and a private address is a
    # rebinding attempt, not a valid target. Every address must pass.
    def fake_getaddrinfo(host: str, port: int, **kwargs: object) -> list:
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("1.1.1.1", port)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port)),
        ]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    with pytest.raises(WorkerError):
        image_loader._resolve("rebind.example", 443, 0)


def test_unresolvable_host_fails_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_getaddrinfo(host: str, port: int, **kwargs: object) -> list:
        raise socket.gaierror("nope")

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    with pytest.raises(WorkerError) as excinfo:
        image_loader._resolve("nowhere.invalid", 443, 0)
    assert excinfo.value.code is ErrorCode.INVALID_IMAGE_URL


# ---------------------------------------------------------------------- schemes


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "gopher://example.com/",
        "ftp://example.com/x.png",
        "//example.com/x.png",
    ],
)
def test_only_http_and_https_are_fetched(url: str, png_bytes: bytes) -> None:
    with pytest.raises(WorkerError) as excinfo:
        image_loader.load_references([url], max_pixels=1_048_576)
    assert excinfo.value.code is ErrorCode.INVALID_IMAGE_URL


def test_empty_url_is_rejected() -> None:
    with pytest.raises(WorkerError) as excinfo:
        image_loader.load_references(["   "], max_pixels=1_048_576)
    assert excinfo.value.code is ErrorCode.INVALID_IMAGE_URL


# -------------------------------------------------------------------- data URIs


def test_data_uri_round_trips(png_bytes: bytes) -> None:
    references = image_loader.load_references(
        [data_uri(png_bytes)], max_pixels=1_048_576
    )
    assert len(references) == 1
    assert references[0].source_size == (64, 48)
    assert references[0].downscaled is False


def test_non_base64_data_uri_is_rejected() -> None:
    with pytest.raises(WorkerError) as excinfo:
        image_loader.load_references(["data:image/png,hello"], max_pixels=1_048_576)
    assert excinfo.value.code is ErrorCode.INVALID_IMAGE_URL


def test_malformed_base64_is_rejected() -> None:
    with pytest.raises(WorkerError) as excinfo:
        image_loader.load_references(
            ["data:image/png;base64,!!!not base64!!!"], max_pixels=1_048_576
        )
    assert excinfo.value.code is ErrorCode.INVALID_IMAGE


def test_undecodable_payload_is_rejected() -> None:
    with pytest.raises(WorkerError) as excinfo:
        image_loader.load_references(
            [data_uri(b"this is not an image")], max_pixels=1_048_576
        )
    assert excinfo.value.code is ErrorCode.INVALID_IMAGE


def test_oversized_inline_image_is_rejected() -> None:
    with pytest.raises(WorkerError) as excinfo:
        image_loader.load_references(
            [data_uri(b"\x00" * (20 * 1024 * 1024 + 1))], max_pixels=1_048_576
        )
    assert excinfo.value.code is ErrorCode.IMAGE_TOO_LARGE


# ------------------------------------------------------------------ preparation


def test_large_references_are_downscaled_and_reported(large_png_bytes: bytes) -> None:
    # A silent resize is as dishonest as a silent drop, so the response says so.
    reference = image_loader._prepare(large_png_bytes, 0, 1_048_576)
    assert reference.source_size == (2400, 1600)
    assert reference.used_size[0] * reference.used_size[1] <= 1_048_576
    assert reference.downscaled is True
    assert reference.to_report()["downscaled"] is True


def test_aspect_ratio_survives_downscaling(large_png_bytes: bytes) -> None:
    reference = image_loader._prepare(large_png_bytes, 0, 262_144)
    source_ratio = 2400 / 1600
    used_ratio = reference.used_size[0] / reference.used_size[1]
    assert used_ratio == pytest.approx(source_ratio, rel=0.01)


def test_small_references_are_left_alone(png_bytes: bytes) -> None:
    reference = image_loader._prepare(png_bytes, 0, 1_048_576)
    assert reference.used_size == (64, 48)
    assert reference.downscaled is False


def test_output_is_always_png_rgb() -> None:
    buffer = io.BytesIO()
    Image.new("P", (32, 32)).save(buffer, format="PNG")
    reference = image_loader._prepare(buffer.getvalue(), 0, 1_048_576)
    decoded = Image.open(io.BytesIO(reference.data))
    assert decoded.format == "PNG"
    assert decoded.mode == "RGB"


def test_order_is_preserved(png_bytes: bytes, large_png_bytes: bytes) -> None:
    # images[0] is "image 1" in the prompt; reordering would change the result.
    references = image_loader.load_references(
        [data_uri(png_bytes), data_uri(large_png_bytes)], max_pixels=1_048_576
    )
    assert [r.index for r in references] == [0, 1]
    assert references[0].source_size == (64, 48)
    assert references[1].source_size == (2400, 1600)


def test_empty_list_is_fine() -> None:
    assert image_loader.load_references([], max_pixels=1_048_576) == []
