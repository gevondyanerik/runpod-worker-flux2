"""Configuration: nothing required, everything defaulted, no credentials."""

from __future__ import annotations

import pytest

from app import config as config_module
from app import constants
from app.errors import ErrorCode, WorkerError
from app.variants import DEFAULT_VARIANT


def test_tier_zero_boots_with_an_empty_environment() -> None:
    # The single most important test in this file: an endpoint created with no
    # environment variables at all is a supported production configuration.
    config = config_module.load()
    assert config.variant.name == DEFAULT_VARIANT
    assert config.variant_was_defaulted is True
    assert config.s3 is None
    assert config.hf_token is None


def test_variant_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLUX2_VARIANT", "klein-4b-base")
    config = config_module.load()
    assert config.variant.name == "klein-4b-base"
    assert config.variant_was_defaulted is False


def test_unknown_variant_fails_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    # A typo must never silently deploy a different model.
    monkeypatch.setenv("FLUX2_VARIANT", "klein-4b-fp8")
    with pytest.raises(WorkerError) as excinfo:
        config_module.load()
    assert excinfo.value.code is ErrorCode.UNSUPPORTED_VARIANT


def test_baked_variant_is_the_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLUX2_BAKED_VARIANT", "klein-4b-nvfp4")
    config = config_module.load()
    assert config.variant.name == "klein-4b-nvfp4"


def test_explicit_variant_beats_the_baked_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLUX2_BAKED_VARIANT", "klein-4b-nvfp4")
    monkeypatch.setenv("FLUX2_VARIANT", "klein-4b-base")
    assert config_module.load().variant.name == "klein-4b-base"


def test_text_encoder_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLUX2_TEXT_ENCODER", "fp4")
    config = config_module.load()
    assert "fp4" in config.variant.text_encoder.filename
    assert config.overrides["FLUX2_TEXT_ENCODER"] == "fp4"


def test_invalid_text_encoder_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLUX2_TEXT_ENCODER", "int4")
    with pytest.raises(WorkerError) as excinfo:
        config_module.load()
    assert excinfo.value.code is ErrorCode.INVALID_INPUT


def test_max_input_images_can_lower_but_not_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MAX_INPUT_IMAGES", "2")
    assert config_module.load().max_reference_images == 2

    # Above the profile's own limit, the profile wins: accepting references the
    # model cannot use would be promising something we cannot deliver.
    monkeypatch.setenv("MAX_INPUT_IMAGES", "10")
    config = config_module.load()
    assert config.max_reference_images == config.variant.max_reference_images


def test_out_of_range_integer_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEFAULT_WIDTH", "99999")
    with pytest.raises(WorkerError) as excinfo:
        config_module.load()
    assert excinfo.value.code is ErrorCode.INVALID_INPUT


def test_non_numeric_integer_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_IMAGES_PER_REQUEST", "lots")
    with pytest.raises(WorkerError):
        config_module.load()


def test_s3_is_all_or_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BUCKET_ENDPOINT_URL", "https://example.invalid")
    with pytest.raises(WorkerError) as excinfo:
        config_module.load()
    assert "BUCKET_ACCESS_KEY_ID" in excinfo.value.message

    monkeypatch.setenv("BUCKET_ACCESS_KEY_ID", "key")
    monkeypatch.setenv("BUCKET_SECRET_ACCESS_KEY", "secret")
    assert config_module.load().uses_s3 is True


def test_asset_override_needs_both_halves(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DIFFUSION_MODEL_REPO", "someone/else")
    with pytest.raises(WorkerError) as excinfo:
        config_module.load()
    assert "must be set together" in excinfo.value.message


def test_gguf_override_is_rejected_with_a_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # There is no GGUF loader in the image, so accepting this would fail much
    # later with a far worse message.
    monkeypatch.setenv("DIFFUSION_MODEL_REPO", "someone/else")
    monkeypatch.setenv("DIFFUSION_MODEL_FILE", "model-Q4_K_M.gguf")
    with pytest.raises(WorkerError) as excinfo:
        config_module.load()
    assert "GGUF" in excinfo.value.message


def test_asset_override_replaces_the_asset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DIFFUSION_MODEL_REPO", "someone/else")
    monkeypatch.setenv("DIFFUSION_MODEL_FILE", "custom.safetensors")
    config = config_module.load()
    assert config.variant.diffusion.repo == "someone/else"
    assert config.variant.diffusion.filename == "custom.safetensors"
    assert config.overrides["DIFFUSION_MODEL_FILE"] == "custom.safetensors"


def test_describe_reports_the_effective_setup() -> None:
    described = config_module.describe(config_module.load())
    assert described["variant"] == DEFAULT_VARIANT
    assert described["variant_source"] == "default"
    assert described["output"] == "base64"


def test_only_config_reads_the_environment() -> None:
    # The rule this whole design rests on. If it were violated, the
    # configuration surface would grow silently across the codebase. The same
    # check runs in CI, so a feature branch cannot merge past it.
    import subprocess
    import sys
    from pathlib import Path

    root = Path(config_module.__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "scripts" / "audit_config.py")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_steps_follow_the_profile_when_unset() -> None:
    # Unset must stay unset: the right step count differs per profile, so
    # collapsing it to a number here would silently mis-sample one of them.
    config = config_module.load()
    assert config.default_steps is None
    assert config.effective_steps == config.variant.sampling.steps


def test_default_steps_overrides_the_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEFAULT_STEPS", "8")
    config = config_module.load()
    assert config.default_steps == 8
    assert config.effective_steps == 8


def test_default_steps_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEFAULT_STEPS", "0")
    with pytest.raises(WorkerError) as excinfo:
        config_module.load()
    assert excinfo.value.code is ErrorCode.INVALID_INPUT

    monkeypatch.setenv("DEFAULT_STEPS", str(constants.MAX_STEPS + 1))
    with pytest.raises(WorkerError):
        config_module.load()


def test_blank_default_steps_means_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    # The Hub sends an empty string for a field the deployer left alone.
    monkeypatch.setenv("DEFAULT_STEPS", "")
    config = config_module.load()
    assert config.default_steps is None
    assert config.effective_steps == config.variant.sampling.steps


def test_raising_steps_on_a_distilled_profile_warns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Measured, not assumed: more steps on a distilled model changes the image
    # rather than improving it. The worker obeys and says so.
    monkeypatch.setenv("DEFAULT_STEPS", "20")
    warning = config_module.step_warning(config_module.load())
    assert warning is not None
    assert "klein-4b" in warning and "20" in warning


def test_no_warning_for_a_base_profile_or_a_lower_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FLUX2_VARIANT", "klein-4b-base")
    monkeypatch.setenv("DEFAULT_STEPS", "40")
    assert config_module.step_warning(config_module.load()) is None

    monkeypatch.setenv("FLUX2_VARIANT", "klein-4b")
    monkeypatch.setenv("DEFAULT_STEPS", "2")
    assert config_module.step_warning(config_module.load()) is None


def test_describe_reports_a_step_override(monkeypatch: pytest.MonkeyPatch) -> None:
    assert "default_steps" not in config_module.describe(config_module.load())
    monkeypatch.setenv("DEFAULT_STEPS", "6")
    assert config_module.describe(config_module.load())["default_steps"] == 6
