"""Fetching and preparing reference images.

This code runs in front of other people's infrastructure: an operator deploys
the worker inside their own network and then lets arbitrary callers hand it
URLs. Everything here is written on the assumption that the URL is hostile.

The protections, and why each one is necessary rather than decorative:

* **Scheme allowlist.** ``file://``, ``gopher://`` and friends are not fetched.
* **Resolve-then-pin.** The hostname is resolved once, *every* returned address
  is validated, and the connection is made to a validated IP with the original
  ``Host`` header and SNI. Validating a hostname and then letting the HTTP
  client resolve it again is a DNS-rebinding hole: the second lookup can return
  a private address.
* **Per-hop redirect validation.** Redirects are followed manually and each hop
  is fully re-validated, including its own DNS resolution. A public URL that
  302s to ``169.254.169.254`` is the classic cloud-metadata exfiltration path.
* **Streamed byte ceiling.** The limit is enforced while reading, not from
  ``Content-Length``, which the server controls and can lie about.
* **Total budget.** Ten references of just-under-the-limit each would otherwise
  add up to a memory problem.
* **IPv4-mapped IPv6 unwrapping.** ``::ffff:127.0.0.1`` is loopback wearing a
  different hat.
"""

from __future__ import annotations

import base64
import binascii
import io
import ipaddress
import socket
import time
from dataclasses import dataclass
from urllib.parse import urlsplit

import urllib3
from PIL import Image, UnidentifiedImageError

from app import constants
from app.errors import ErrorCode, WorkerError

_DATA_PREFIX = "data:"
_BLOCKED_V4 = tuple(
    ipaddress.ip_network(net) for net in constants.BLOCKED_IPV4_NETWORKS
)
_BLOCKED_V6 = tuple(
    ipaddress.ip_network(net) for net in constants.BLOCKED_IPV6_NETWORKS
)


@dataclass(frozen=True, slots=True)
class Reference:
    """One prepared reference image."""

    index: int
    data: bytes
    source_size: tuple[int, int]
    used_size: tuple[int, int]
    downscaled: bool

    def to_report(self) -> dict[str, object]:
        return {
            "index": self.index,
            "source_px": list(self.source_size),
            "used_px": list(self.used_size),
            "downscaled": self.downscaled,
        }


def _is_public(address: str) -> bool:
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False

    # ::ffff:127.0.0.1 must be judged as the IPv4 address it wraps.
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped

    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
        return False
    if ip.is_multicast or ip.is_unspecified:
        return False

    blocked = _BLOCKED_V4 if ip.version == 4 else _BLOCKED_V6
    return not any(ip in network for network in blocked)


def _resolve(host: str, port: int, index: int) -> str:
    """Resolve a hostname and reject it unless every address is public.

    Every address, not just the one we would use: a host that resolves to both
    a public and a private address is a rebinding attempt, not a valid target.
    """
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise WorkerError(
            ErrorCode.INVALID_IMAGE_URL,
            f"images[{index}]: cannot resolve host {host!r}",
        ) from exc

    addresses = [str(info[4][0]) for info in infos]
    if not addresses:
        raise WorkerError(
            ErrorCode.INVALID_IMAGE_URL,
            f"images[{index}]: host {host!r} resolved to no addresses",
        )
    for address in addresses:
        if not _is_public(address):
            raise WorkerError(
                ErrorCode.INVALID_IMAGE_URL,
                f"images[{index}]: host {host!r} resolves to a non-public "
                "address, which is not allowed",
            )
    return addresses[0]


def _fetch(url: str, index: int, deadline: float, budget: int) -> bytes:
    """Fetch a URL with per-hop validation and a streamed size ceiling."""
    seen = 0
    current = url

    while True:
        parts = urlsplit(current)
        if parts.scheme not in ("http", "https"):
            raise WorkerError(
                ErrorCode.INVALID_IMAGE_URL,
                f"images[{index}]: only http and https URLs are supported",
            )
        host = parts.hostname
        if not host:
            raise WorkerError(
                ErrorCode.INVALID_IMAGE_URL, f"images[{index}]: URL has no host"
            )
        port = parts.port or (443 if parts.scheme == "https" else 80)
        ip = _resolve(host, port, index)

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise WorkerError(
                ErrorCode.IMAGE_DOWNLOAD_FAILED,
                f"images[{index}]: download budget exhausted",
            )

        timeout = urllib3.Timeout(
            connect=min(constants.IMAGE_DOWNLOAD_TIMEOUT_S, remaining),
            read=min(constants.IMAGE_DOWNLOAD_TIMEOUT_S, remaining),
        )
        # Connect to the validated IP; keep the real hostname for the Host
        # header and for TLS SNI/certificate verification.
        if parts.scheme == "https":
            pool: urllib3.HTTPConnectionPool = urllib3.HTTPSConnectionPool(
                host=ip,
                port=port,
                timeout=timeout,
                retries=False,
                server_hostname=host,
                assert_hostname=host,
                cert_reqs="CERT_REQUIRED",
            )
        else:
            pool = urllib3.HTTPConnectionPool(
                host=ip, port=port, timeout=timeout, retries=False
            )

        target = parts.path or "/"
        if parts.query:
            target = f"{target}?{parts.query}"

        try:
            with pool:
                response = pool.request(
                    "GET",
                    target,
                    headers={"Host": host, "Accept": "image/*"},
                    redirect=False,
                    preload_content=False,
                )
                if response.status in (301, 302, 303, 307, 308):
                    location = response.headers.get("Location")
                    response.release_conn()
                    seen += 1
                    if seen > constants.MAX_REDIRECTS:
                        raise WorkerError(
                            ErrorCode.INVALID_IMAGE_URL,
                            f"images[{index}]: too many redirects",
                        )
                    if not location:
                        raise WorkerError(
                            ErrorCode.IMAGE_DOWNLOAD_FAILED,
                            f"images[{index}]: redirect without a Location header",
                        )
                    current = urllib3.util.parse_url(location).url
                    if not urlsplit(current).scheme:
                        current = f"{parts.scheme}://{host}:{port}{location}"
                    continue

                if response.status != 200:
                    raise WorkerError(
                        ErrorCode.IMAGE_DOWNLOAD_FAILED,
                        f"images[{index}]: server returned HTTP {response.status}",
                    )

                raw_type = response.headers.get("Content-Type", "")
                content_type = raw_type.split(";")[0].strip().lower()
                if content_type and content_type not in constants.ALLOWED_IMAGE_MIME:
                    raise WorkerError(
                        ErrorCode.INVALID_IMAGE,
                        f"images[{index}]: unsupported content type "
                        f"{content_type!r}; allowed: "
                        f"{', '.join(sorted(constants.ALLOWED_IMAGE_MIME))}",
                    )

                limit = min(constants.MAX_IMAGE_BYTES, budget)
                chunks: list[bytes] = []
                total = 0
                for chunk in response.stream(64 * 1024):
                    total += len(chunk)
                    if total > limit:
                        raise WorkerError(
                            ErrorCode.IMAGE_TOO_LARGE
                            if limit == constants.MAX_IMAGE_BYTES
                            else ErrorCode.TOTAL_INPUT_TOO_LARGE,
                            f"images[{index}]: exceeds the {limit} byte limit",
                        )
                    chunks.append(chunk)
                return b"".join(chunks)
        except WorkerError:
            raise
        except Exception as exc:
            raise WorkerError(
                ErrorCode.IMAGE_DOWNLOAD_FAILED,
                f"images[{index}]: download failed ({type(exc).__name__})",
            ) from exc


def _decode_data_uri(url: str, index: int, budget: int) -> bytes:
    header, _, payload = url.partition(",")
    if "base64" not in header:
        raise WorkerError(
            ErrorCode.INVALID_IMAGE_URL,
            f"images[{index}]: only base64 data URIs are supported",
        )
    try:
        data = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise WorkerError(
            ErrorCode.INVALID_IMAGE, f"images[{index}]: malformed base64 payload"
        ) from exc
    if len(data) > min(constants.MAX_IMAGE_BYTES, budget):
        raise WorkerError(
            ErrorCode.IMAGE_TOO_LARGE, f"images[{index}]: inline image is too large"
        )
    return data


def _prepare(raw: bytes, index: int, max_pixels: int) -> Reference:
    """Decode, normalise and downscale one image.

    References enter the transformer context, so their resolution directly
    drives VRAM and latency. A 4000x3000 product photo would multiply the
    context length roughly twelvefold for no benefit at banner resolution, so
    every reference is capped — and the response reports it, because a silent
    resize is as dishonest as a silent drop.
    """
    try:
        opened = Image.open(io.BytesIO(raw))
        opened.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise WorkerError(
            ErrorCode.INVALID_IMAGE, f"images[{index}]: not a decodable image"
        ) from exc

    image: Image.Image = opened
    source_size = image.size
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGB")

    pixels = image.width * image.height
    downscaled = pixels > max_pixels
    if downscaled:
        scale = (max_pixels / pixels) ** 0.5
        target = (
            max(1, int(image.width * scale)),
            max(1, int(image.height * scale)),
        )
        image = image.resize(target, Image.Resampling.LANCZOS)

    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG", optimize=False)
    return Reference(
        index=index,
        data=buffer.getvalue(),
        source_size=source_size,
        used_size=image.size,
        downscaled=downscaled,
    )


def load_references(urls: list[str], *, max_pixels: int) -> list[Reference]:
    """Fetch and prepare every reference, preserving caller order.

    Order is semantic: ``images[0]`` is "image 1" in the prompt. Downloads are
    sequential so a shared byte budget can be enforced honestly; at these sizes
    the wall-clock cost is small next to inference.
    """
    deadline = time.monotonic() + constants.IMAGE_TOTAL_DOWNLOAD_TIMEOUT_S
    budget = constants.MAX_TOTAL_INPUT_BYTES
    references: list[Reference] = []

    for index, url in enumerate(urls):
        if not isinstance(url, str) or not url.strip():
            raise WorkerError(
                ErrorCode.INVALID_IMAGE_URL,
                f"images[{index}]: must be a non-empty string",
            )
        url = url.strip()

        if url.startswith(_DATA_PREFIX):
            raw = _decode_data_uri(url, index, budget)
        else:
            raw = _fetch(url, index, deadline, budget)

        budget -= len(raw)
        if budget < 0:
            raise WorkerError(
                ErrorCode.TOTAL_INPUT_TOO_LARGE,
                "reference images exceed the total input budget of "
                f"{constants.MAX_TOTAL_INPUT_BYTES} bytes",
            )
        references.append(_prepare(raw, index, max_pixels))

    return references
