from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from work import separate_stems
from work.hardware_acceleration import (
    onnx_provider_chain,
    separator_profile,
    torch_device_name,
)
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


def test_roformer_profile_is_automatic_only_with_torch_acceleration() -> None:
    assert separator_profile(fake_torch(cuda=True), "auto") == "roformer"
    assert separator_profile(fake_torch(mps=True), "auto") == "roformer"
    # CPU-only and DirectML/XPU machines keep the smaller ONNX models, which
    # stay fast there.
    assert separator_profile(fake_torch(), "auto") == "compact"
    assert separator_profile(fake_torch(xpu=True), "auto") == "compact"
    # An explicit choice always wins.
    assert separator_profile(fake_torch(), "roformer") == "roformer"
    assert separator_profile(fake_torch(cuda=True), "compact") == "compact"
    with pytest.raises(ValueError, match="SONA_SEPARATOR"):
        separator_profile(fake_torch(), "best")


def test_stem_runner_treats_an_empty_export_as_failure(tmp_path) -> None:
    class EmptyExportSeparator:
        """Mimics audio-separator logging an export error and carrying on."""

        def __init__(self, **options):
            self.output_dir = Path(options["output_dir"])
            self.model_instance = SimpleNamespace(
                primary_stem_name="vocals", secondary_stem_name="other"
            )

        def load_model(self, _model):
            pass

        def separate(self, _source, custom_output_names=None):
            assert custom_output_names == {"vocals": "lead", "other": "backing"}
            for name in custom_output_names.values():
                (self.output_dir / f"{name}.wav").write_bytes(b"")
            return []

    with pytest.raises(RuntimeError, match="did not produce"):
        separate_stems.run_separator_once(
            EmptyExportSeparator,
            {"output_dir": str(tmp_path)},
            "cpu",
            "model.ckpt",
            tmp_path / "source.wav",
            tmp_path / "lead.wav",
            tmp_path / "backing.wav",
        )


def test_stem_pair_is_replaced_together_or_not_at_all(tmp_path) -> None:
    primary = tmp_path / "stems" / "vocals.wav"
    secondary = tmp_path / "stems" / "instrumental.wav"
    primary.parent.mkdir()
    primary.write_bytes(b"old-vocals")
    secondary.write_bytes(b"old-instrumental")

    def failing(_source, _model, _model_dir, staging, primary_name, _secondary_name):
        (staging / f"{primary_name}.wav").write_bytes(b"new-vocals")
        raise RuntimeError("accelerator ran out of memory")

    with pytest.raises(RuntimeError, match="out of memory"):
        separate_stems.separate_atomically(
            tmp_path / "song.mp3", "model.ckpt", tmp_path, primary, secondary, failing
        )
    assert primary.read_bytes() == b"old-vocals"
    assert secondary.read_bytes() == b"old-instrumental"
    assert not list(primary.parent.glob(".*.tmp"))

    def working(_source, _model, _model_dir, staging, primary_name, secondary_name):
        outputs = (staging / f"{primary_name}.wav", staging / f"{secondary_name}.wav")
        outputs[0].write_bytes(b"new-vocals")
        outputs[1].write_bytes(b"new-instrumental")
        return outputs

    separate_stems.separate_atomically(
        tmp_path / "song.mp3", "model.ckpt", tmp_path, primary, secondary, working
    )
    assert primary.read_bytes() == b"new-vocals"
    assert secondary.read_bytes() == b"new-instrumental"
    assert not list(primary.parent.glob(".*.tmp"))
