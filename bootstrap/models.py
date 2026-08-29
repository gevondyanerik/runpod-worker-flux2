"""Making the active profile's model files available to ComfyUI.

``MODEL_SOURCE=auto`` is what makes a zero-configuration deployment real: the
default profile is baked into the image so a bare deploy starts immediately,
and a non-default profile downloads once and still works with nothing set.

This is the one place where an ordered fallback is deliberate rather than
sloppy. The chain runs fastest-to-slowest, logs which step it took, and always
ends in something that works. Choosing an explicit ``baked``/``volume``/
``download`` restores strictness, so a production deployment can assert its
assumption instead of discovering a silent multi-gigabyte download in its bill.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import time
from pathlib import Path

from app import constants
from app.config import Config
from app.errors import ErrorCode, WorkerError
from app.variants import Asset

log = logging.getLogger(__name__)

MODELS_ROOT = Path(constants.COMFY_MODELS_ROOT)
VOLUME_ROOT = Path("/runpod-volume/models")

# A file smaller than this fraction of its expected size is a truncated
# download wearing the right filename. Size is checked on every boot because it
# is free; the SHA-256 is checked only right after a download, because hashing
# 8 GB on every cold start would cost more than it protects against.
_MIN_SIZE_RATIO = 0.98

_HASH_CHUNK = 8 * 1024 * 1024


def _target(asset: Asset) -> Path:
    return MODELS_ROOT / asset.dest_dir / asset.filename


def _plausible(path: Path, asset: Asset) -> bool:
    if not path.is_file():
        return False
    if not asset.size_bytes:  # expert override: size unknown, trust presence
        return True
    return path.stat().st_size >= asset.size_bytes * _MIN_SIZE_RATIO


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(path: Path, asset: Asset) -> None:
    """Check a freshly downloaded file against its pinned digest."""
    if not asset.sha256:
        return  # expert override: nothing to compare against
    started = time.monotonic()
    actual = sha256_of(path)
    if actual != asset.sha256:
        path.unlink(missing_ok=True)
        raise WorkerError(
            ErrorCode.MODEL_CHECKSUM_MISMATCH,
            f"{asset.filename} does not match its pinned SHA-256 "
            f"(expected {asset.sha256[:12]}…, got {actual[:12]}…). The file has "
            "been removed; retry, and if it recurs the upstream repository has "
            "changed and this worker needs an update.",
        )
    log.info(
        "checksum verified",
        extra={"file": asset.filename, "seconds": round(time.monotonic() - started, 1)},
    )


def _from_volume(asset: Asset, target: Path) -> bool:
    source = VOLUME_ROOT / asset.dest_dir / asset.filename
    if not _plausible(source, asset):
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    # Symlink rather than copy: the volume is the storage, and copying 8 GB
    # into the container adds cold-start time for nothing.
    target.symlink_to(source)
    log.info("model from volume", extra={"file": asset.filename})
    return True


def _download(asset: Asset, target: Path, token: str | None) -> None:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    destination = VOLUME_ROOT if VOLUME_ROOT.is_dir() else MODELS_ROOT
    cache_dir = destination / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        "downloading model",
        extra={"repo": asset.repo, "file": asset.path, "gb": round(asset.size_gb, 2)},
    )
    try:
        path = hf_hub_download(
            repo_id=asset.repo,
            filename=asset.path,
            revision=asset.revision,
            cache_dir=str(cache_dir),
            token=token,
        )
    except GatedRepoError as exc:
        raise WorkerError(
            ErrorCode.MODEL_AUTH_REQUIRED,
            f"{asset.repo} is a gated repository and needs HF_TOKEN. No built-in "
            "profile requires this — it can only happen with an expert override.",
        ) from exc
    except RepositoryNotFoundError as exc:
        raise WorkerError(
            ErrorCode.MODEL_ASSET_MISSING,
            f"{asset.repo} does not exist or is not public",
        ) from exc
    except Exception as exc:
        raise WorkerError(
            ErrorCode.MODEL_ASSET_MISSING,
            f"could not download {asset.path} from {asset.repo} ({type(exc).__name__})",
        ) from exc

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(path)


def ensure_assets(config: Config) -> None:
    """Put every asset for the active profile where ComfyUI will find it."""
    source = config.model_source

    for asset in config.variant.assets:
        target = _target(asset)

        if _plausible(target, asset):
            log.info("model already present", extra={"file": asset.filename})
            continue

        if source == "baked":
            raise WorkerError(
                ErrorCode.MODEL_ASSET_MISSING,
                f"MODEL_SOURCE=baked but {asset.filename} is not in the image. "
                "Use MODEL_SOURCE=auto to download it, or deploy the image tag "
                "that bakes this profile.",
            )

        if source in ("auto", "volume") and _from_volume(asset, target):
            continue

        if source == "volume":
            raise WorkerError(
                ErrorCode.MODEL_ASSET_MISSING,
                f"MODEL_SOURCE=volume but {asset.filename} was not found under "
                f"{VOLUME_ROOT}. Populate the volume or use MODEL_SOURCE=auto.",
            )

        _download(asset, target, config.hf_token)

        if not _plausible(target, asset):
            raise WorkerError(
                ErrorCode.MODEL_CHECKSUM_MISMATCH,
                f"{asset.filename} is smaller than expected after download; "
                "the transfer was probably truncated",
            )
        _verify(target.resolve(), asset)


def verify_visible(comfy: object, config: Config) -> None:
    """Check ComfyUI's model index actually lists the profile's files.

    A file on disk that ComfyUI cannot see produces an opaque prompt-validation
    error on the first paying request. Catching it at startup is the difference
    between a clear failure and a mystery.
    """
    checks = (
        ("UNETLoader", "unet_name", config.variant.diffusion.filename),
        ("CLIPLoader", "clip_name", config.variant.text_encoder.filename),
        ("VAELoader", "vae_name", config.variant.vae.filename),
    )
    known_models = getattr(comfy, "known_models", None)
    if known_models is None:  # pragma: no cover - only with a stub client
        return

    for class_type, input_name, filename in checks:
        available = known_models(class_type, input_name)
        if available and filename not in available:
            raise WorkerError(
                ErrorCode.MODEL_ASSET_MISSING,
                f"ComfyUI does not list {filename!r} for {class_type}. "
                f"It sees: {', '.join(available[:8]) or 'nothing'}",
            )


def free_space_gb(path: Path = MODELS_ROOT) -> float:
    target = path if path.exists() else path.parent
    usage = shutil.disk_usage(os.fspath(target))
    return usage.free / 1e9
