"""Model provisioning: the ordered fallback that makes zero-config real."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from app import config as config_module
from app.config import Config
from app.errors import ErrorCode, WorkerError
from app.variants import Asset
from bootstrap import models

# The real assets are gigabytes. Nothing here is about their size, only about
# which branch of the fallback runs, so the fixtures use kilobyte stand-ins.
TINY = 4096


def _refuse(*args: object, **kwargs: object) -> None:
    raise AssertionError("downloaded a file that was already available")


@pytest.fixture
def roots(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Path]:
    image = tmp_path / "comfyui" / "models"
    volume = tmp_path / "runpod-volume" / "models"
    monkeypatch.setattr(models, "MODELS_ROOT", image)
    monkeypatch.setattr(models, "VOLUME_ROOT", volume)
    return image, volume


@pytest.fixture
def config() -> Config:
    """The default profile, with every asset shrunk to a testable size."""
    base = config_module.load()
    variant = replace(
        base.variant,
        diffusion=replace(base.variant.diffusion, size_bytes=TINY),
        text_encoder=replace(base.variant.text_encoder, size_bytes=TINY),
        vae=replace(base.variant.vae, size_bytes=TINY),
    )
    return replace(base, variant=variant)


def place(root: Path, asset: Asset, size: int = TINY) -> Path:
    path = root / asset.dest_dir / asset.filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00" * size)
    return path


def test_baked_assets_are_used_as_is(
    roots: tuple[Path, Path], config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    image, _ = roots
    for asset in config.variant.assets:
        place(image, asset)

    monkeypatch.setattr(models, "_download", _refuse)
    models.ensure_assets(config)

    for asset in config.variant.assets:
        assert (image / asset.dest_dir / asset.filename).is_file()


def test_a_volume_copy_is_symlinked_not_copied(
    roots: tuple[Path, Path], config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The volume is the storage. Copying 8 GB into the container would add cold
    # start time for nothing.
    image, volume = roots
    for asset in config.variant.assets:
        place(volume, asset)

    monkeypatch.setattr(models, "_download", _refuse)
    models.ensure_assets(config)

    for asset in config.variant.assets:
        target = image / asset.dest_dir / asset.filename
        assert target.is_symlink()
        assert target.resolve() == (volume / asset.dest_dir / asset.filename)


def test_a_truncated_file_is_not_accepted(
    roots: tuple[Path, Path], config: Config
) -> None:
    # A half-downloaded file wearing the right name is exactly the failure this
    # exists to catch.
    image, _ = roots
    for asset in config.variant.assets:
        place(image, asset, size=TINY // 2)

    with pytest.raises(WorkerError) as excinfo:
        models.ensure_assets(replace(config, model_source="baked"))
    assert excinfo.value.code is ErrorCode.MODEL_ASSET_MISSING


def test_baked_mode_never_downloads(roots: tuple[Path, Path], config: Config) -> None:
    with pytest.raises(WorkerError) as excinfo:
        models.ensure_assets(replace(config, model_source="baked"))
    assert "MODEL_SOURCE=baked" in excinfo.value.message
    assert "MODEL_SOURCE=auto" in excinfo.value.message  # says how to fix it


def test_volume_mode_never_downloads(roots: tuple[Path, Path], config: Config) -> None:
    with pytest.raises(WorkerError) as excinfo:
        models.ensure_assets(replace(config, model_source="volume"))
    assert "MODEL_SOURCE=volume" in excinfo.value.message


def test_auto_prefers_the_image_then_the_volume(
    roots: tuple[Path, Path], config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    image, volume = roots

    # Diffusion model baked, the rest only on the volume: neither may download.
    place(image, config.variant.diffusion)
    place(volume, config.variant.text_encoder)
    place(volume, config.variant.vae)

    monkeypatch.setattr(models, "_download", _refuse)
    models.ensure_assets(config)

    diffusion = (
        image / config.variant.diffusion.dest_dir / config.variant.diffusion.filename
    )
    vae = image / config.variant.vae.dest_dir / config.variant.vae.filename
    assert not diffusion.is_symlink()
    assert vae.is_symlink()


def test_auto_downloads_what_is_missing(
    roots: tuple[Path, Path], config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    image, _ = roots
    requested: list[str] = []

    def fake_download(asset: Asset, target: Path, token: str | None) -> None:
        requested.append(asset.filename)
        place(image, asset)

    monkeypatch.setattr(models, "_download", fake_download)
    monkeypatch.setattr(models, "_verify", lambda path, asset: None)
    models.ensure_assets(config)

    assert requested == [a.filename for a in config.variant.assets]


def test_checksum_mismatch_removes_the_file(tmp_path: Path) -> None:
    asset = config_module.load().variant.vae
    path = tmp_path / asset.filename
    path.write_bytes(b"wrong content")

    with pytest.raises(WorkerError) as excinfo:
        models._verify(path, asset)
    assert excinfo.value.code is ErrorCode.MODEL_CHECKSUM_MISMATCH
    assert not path.exists(), "a file that failed verification must not be left behind"


def test_checksum_is_skipped_for_expert_overrides(tmp_path: Path) -> None:
    asset = replace(config_module.load().variant.vae, sha256=None)
    path = tmp_path / asset.filename
    path.write_bytes(b"anything")
    models._verify(path, asset)  # nothing to compare against, so nothing to do


def test_verify_visible_catches_a_file_comfyui_cannot_see() -> None:
    # A file on disk that ComfyUI has not indexed produces an opaque
    # prompt-validation error on the first paying request.
    config = config_module.load()

    class Blind:
        def known_models(self, class_type: str, input_name: str) -> list[str]:
            return ["something-else.safetensors"]

    with pytest.raises(WorkerError) as excinfo:
        models.verify_visible(Blind(), config)
    assert excinfo.value.code is ErrorCode.MODEL_ASSET_MISSING
    assert config.variant.diffusion.filename in excinfo.value.message


def test_verify_visible_passes_when_everything_is_indexed() -> None:
    config = config_module.load()
    listing = {
        "UNETLoader": [config.variant.diffusion.filename],
        "CLIPLoader": [config.variant.text_encoder.filename],
        "VAELoader": [config.variant.vae.filename],
    }

    class Sighted:
        def known_models(self, class_type: str, input_name: str) -> list[str]:
            return listing[class_type]

    models.verify_visible(Sighted(), config)


def test_a_download_that_cannot_fit_fails_before_it_starts(
    roots: tuple[Path, Path], config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Running out of disk mid-download costs the whole cold start and leaves an
    # errno in the log. The size is known in advance, so the check is too.
    monkeypatch.setattr(models, "free_space_gb", lambda *_: 0.0)
    monkeypatch.setattr(models, "_download", _refuse)

    with pytest.raises(WorkerError) as excinfo:
        models.ensure_assets(config)
    assert excinfo.value.code is ErrorCode.INSUFFICIENT_DISK
    assert "container disk" in excinfo.value.message


def test_disk_is_not_checked_for_assets_already_present(
    roots: tuple[Path, Path], config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A baked image on a nearly full disk is a normal, working deployment.
    image, _ = roots
    for asset in config.variant.assets:
        place(image, asset)

    monkeypatch.setattr(models, "free_space_gb", lambda *_: 0.0)
    monkeypatch.setattr(models, "_download", _refuse)
    models.ensure_assets(config)


def test_free_space_is_measured_where_the_models_go(
    roots: tuple[Path, Path], tmp_path: Path
) -> None:
    # A default argument bound at import time would measure the production
    # path no matter what MODELS_ROOT says.
    image, _ = roots
    assert not image.exists()
    assert models.free_space_gb() > 0  # walks up to an ancestor that exists
