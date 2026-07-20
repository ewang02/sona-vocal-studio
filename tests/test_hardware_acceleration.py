from __future__ import annotations

from types import SimpleNamespace

import pytest

from work.hardware_acceleration import onnx_provider_chain, torch_device_name
from work.separate_kim_stem import run_separator_once


def test_onnx_provider_chain_prefers_specialized_gpu_and_keeps_fallbacks() -> None:
    assert onnx_provider_chain(
        [
            "CPUExecutionProvider",
            "CoreMLExecutionProvider",
            "CUDAExecutionProvider",
            "TensorrtExecutionProvider",
        ]
    ) == [
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert onnx_provider_chain(
        ["CPUExecutionProvider", "DmlExecutionProvider"]
    ) == ["DmlExecutionProvider", "CPUExecutionProvider"]
    assert onnx_provider_chain(["CPUExecutionProvider"]) == ["CPUExecutionProvider"]


def test_onnx_provider_chain_honors_explicit_override() -> None:
    available = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    assert onnx_provider_chain(available, "cpu") == ["CPUExecutionProvider"]
    assert onnx_provider_chain(available, "coreml") == [
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ]
    with pytest.raises(RuntimeError, match="Requested CUDAExecutionProvider"):
        onnx_provider_chain(available, "cuda")
    with pytest.raises(ValueError, match="Unknown SONA_ACCELERATOR"):
        onnx_provider_chain(available, "quantum")


class Available:
    @staticmethod
    def is_available() -> bool:
        return True


class Unavailable:
    @staticmethod
    def is_available() -> bool:
        return False


def fake_torch(*, cuda=False, xpu=False, mps=False):
    return SimpleNamespace(
        cuda=Available if cuda else Unavailable,
        xpu=Available if xpu else Unavailable,
        backends=SimpleNamespace(mps=Available if mps else Unavailable),
    )


def test_torch_device_priority_and_cpu_fallback() -> None:
    assert torch_device_name(fake_torch(cuda=True, xpu=True, mps=True)) == "cuda"
    assert torch_device_name(fake_torch(xpu=True, mps=True)) == "xpu"
    assert torch_device_name(fake_torch(mps=True)) == "mps"
    assert torch_device_name(fake_torch()) == "cpu"


def test_kim_runner_treats_third_party_missing_output_as_failure(tmp_path) -> None:
    class SwallowingSeparator:
        def __init__(self, **_options):
            self.onnx_execution_provider = []

        def load_model(self, _model):
            pass

        def separate(self, _source, custom_output_names=None):
            return []

    with pytest.raises(RuntimeError, match="did not produce"):
        run_separator_once(
            SwallowingSeparator,
            {},
            ["CPUExecutionProvider"],
            "model.onnx",
            tmp_path / "source.wav",
            "Vocals",
            "vocals",
            tmp_path / "vocals.wav",
        )
