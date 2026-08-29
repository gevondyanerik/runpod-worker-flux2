"""The profile registry is the contract everything else depends on."""

from __future__ import annotations

import pytest

from app.variants import (
    APACHE_2_0,
    BAKED_ASSETS,
    DEFAULT_VARIANT,
    TEXT_ENCODERS,
    VARIANTS,
    UnknownVariantError,
    get_variant,
)


def test_default_variant_exists() -> None:
    assert DEFAULT_VARIANT in VARIANTS


def test_every_profile_is_ready_to_serve() -> None:
    # A profile with unconfirmed sampling defaults must never ship: the worker
    # would run and be quietly wrong.
    for name, variant in VARIANTS.items():
        assert variant.is_ready, f"{name} has unconfirmed sampling defaults"


def test_every_asset_is_apache_licensed() -> None:
    # The whole repository is publishable only because nothing here carries the
    # FLUX non-commercial licence. This test is the guard on that promise.
    for name, variant in VARIANTS.items():
        for asset in variant.assets:
            assert asset.license_id == APACHE_2_0, f"{name}/{asset.filename}"


def test_every_asset_is_pinned_and_checksummed() -> None:
    for variant in VARIANTS.values():
        for asset in variant.assets:
            assert asset.sha256, f"{asset.filename} has no pinned digest"
            assert len(asset.sha256) == 64
            assert asset.revision != "main", f"{asset.filename} is not pinned"
            assert asset.size_bytes > 0


def test_filenames_are_unique_within_a_profile() -> None:
    # Everything is flattened into ComfyUI's model directories, so a collision
    # would silently make one file shadow another.
    for name, variant in VARIANTS.items():
        filenames = [a.filename for a in variant.assets]
        assert len(filenames) == len(set(filenames)), name


def test_unknown_variant_lists_the_valid_ones() -> None:
    with pytest.raises(UnknownVariantError) as excinfo:
        get_variant("flux2-dev")
    message = str(excinfo.value)
    assert "flux2-dev" in message
    for name in VARIANTS:
        assert name in message


def test_baked_assets_cover_the_default_profile() -> None:
    baked = {a.filename for a in BAKED_ASSETS}
    for asset in VARIANTS[DEFAULT_VARIANT].assets:
        assert asset.filename in baked


def test_text_encoder_override_keeps_the_rest_of_the_profile() -> None:
    variant = get_variant("klein-4b")
    swapped = variant.with_text_encoder("fp4")
    assert swapped.text_encoder is TEXT_ENCODERS["fp4"]
    assert swapped.diffusion == variant.diffusion
    assert swapped.sampling == variant.sampling


def test_nvfp4_declares_its_architecture_requirement() -> None:
    assert get_variant("klein-4b-nvfp4").requires_compute_capability == (12, 0)
    assert get_variant("klein-4b").requires_compute_capability is None
