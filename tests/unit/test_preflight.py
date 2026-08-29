"""Hardware checks. Only the architecture check is fatal."""

from __future__ import annotations

import logging

import pytest

from app import config as config_module
from app.errors import ErrorCode, WorkerError
from bootstrap import preflight


def gpu(name: str, capability: tuple[int, int], vram_gb: float):
    return lambda: (name, capability, vram_gb)


def test_no_cuda_is_not_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # Refusing to start without a GPU would make the worker untestable off a
    # GPU host, which is where almost all of its development happens.
    monkeypatch.setattr(preflight, "_gpu", lambda: None)
    preflight.check(config_module.load())


def test_nvfp4_refuses_pre_blackwell(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLUX2_VARIANT", "klein-4b-nvfp4")
    monkeypatch.setattr(preflight, "_gpu", gpu("NVIDIA RTX 4090", (8, 9), 24.0))
    with pytest.raises(WorkerError) as excinfo:
        preflight.check(config_module.load())
    assert excinfo.value.code is ErrorCode.UNSUPPORTED_GPU_ARCH
    # The message must say what to do, not just what went wrong.
    assert "FLUX2_VARIANT=klein-4b" in excinfo.value.message


def test_nvfp4_accepts_blackwell(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLUX2_VARIANT", "klein-4b-nvfp4")
    monkeypatch.setattr(
        preflight, "_gpu", gpu("NVIDIA RTX PRO 4500 Blackwell", (12, 0), 32.0)
    )
    monkeypatch.setattr(preflight, "_system_ram_gb", lambda: 64.0)
    preflight.check(config_module.load())


def test_other_profiles_run_anywhere(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(preflight, "_gpu", gpu("NVIDIA A10G", (8, 6), 24.0))
    monkeypatch.setattr(preflight, "_system_ram_gb", lambda: 32.0)
    preflight.check(config_module.load())


def test_low_vram_warns_but_starts(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # An estimate that refused to start would make the worker impossible to run
    # on hardware it may well handle. The operator decides, not a table.
    monkeypatch.setattr(preflight, "_gpu", gpu("NVIDIA T4", (7, 5), 8.0))
    monkeypatch.setattr(preflight, "_system_ram_gb", lambda: 32.0)
    with caplog.at_level(logging.WARNING):
        preflight.check(config_module.load())
    assert "below the" in caplog.text


def test_low_system_ram_warns(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(preflight, "_gpu", gpu("NVIDIA L4", (8, 9), 24.0))
    monkeypatch.setattr(preflight, "_system_ram_gb", lambda: 8.0)
    with caplog.at_level(logging.WARNING):
        preflight.check(config_module.load())
    assert "RAM" in caplog.text
