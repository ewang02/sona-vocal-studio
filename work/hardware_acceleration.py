"""Cross-platform inference-device discovery for the song pipeline.

Every stage is allowed to fall back to CPU.  Accelerators are selected only
when the installed runtime reports them as available; the order below favors
specialized/discrete GPU providers before portable platform providers.
"""

from __future__ import annotations

import os
import platform
from typing import Iterable


CPU_PROVIDER = "CPUExecutionProvider"
ONNX_ACCELERATOR_PRIORITY = (
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "MIGraphXExecutionProvider",
    "ROCMExecutionProvider",  # Older ONNX Runtime installations.
    "DmlExecutionProvider",
    "CoreMLExecutionProvider",
    "OpenVINOExecutionProvider",
    "QNNExecutionProvider",
    "WebGPUExecutionProvider",
    "XnnpackExecutionProvider",
)

ACCELERATOR_ALIASES = {
    "cpu": CPU_PROVIDER,
    "cuda": "CUDAExecutionProvider",
    "tensorrt": "TensorrtExecutionProvider",
    "rocm": "MIGraphXExecutionProvider",
    "migraphx": "MIGraphXExecutionProvider",
    "directml": "DmlExecutionProvider",
    "dml": "DmlExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
    "openvino": "OpenVINOExecutionProvider",
    "qnn": "QNNExecutionProvider",
    "webgpu": "WebGPUExecutionProvider",
    "xnnpack": "XnnpackExecutionProvider",
}


def requested_accelerator(value: str | None = None) -> str:
    """Return the normalized SONA_ACCELERATOR override (default ``auto``)."""
    requested = (value if value is not None else os.getenv("SONA_ACCELERATOR", "auto"))
    return requested.strip().lower() or "auto"


def onnx_provider_chain(
    available: Iterable[str], requested: str | None = None
) -> list[str]:
    """Choose the fastest installed ONNX provider with a CPU safety net.

    ``SONA_ACCELERATOR`` may force a provider for diagnostics.  An explicit
    unavailable provider is an error instead of silently ignoring the user's
    request.  In automatic mode an ordinary CPU install is always valid.
    """
    available_set = set(available)
    selected_request = requested_accelerator(requested)
    if selected_request != "auto":
        provider = ACCELERATOR_ALIASES.get(selected_request)
        if provider is None:
            choices = ", ".join(("auto", *ACCELERATOR_ALIASES))
            raise ValueError(
                f"Unknown SONA_ACCELERATOR={selected_request!r}; choose from {choices}"
            )
        if provider not in available_set:
            raise RuntimeError(
                f"Requested {provider}, but ONNX Runtime reports only: "
                f"{', '.join(sorted(available_set)) or 'no providers'}"
            )
    else:
        provider = next(
            (name for name in ONNX_ACCELERATOR_PRIORITY if name in available_set),
            CPU_PROVIDER if CPU_PROVIDER in available_set else None,
        )
        if provider is None:
            raise RuntimeError("ONNX Runtime reports no usable execution provider")

    chain = [provider]
    # TensorRT commonly delegates unsupported nodes to CUDA before CPU.
    if (
        provider == "TensorrtExecutionProvider"
        and "CUDAExecutionProvider" in available_set
    ):
        chain.append("CUDAExecutionProvider")
    if provider != CPU_PROVIDER and CPU_PROVIDER in available_set:
        chain.append(CPU_PROVIDER)
    return chain


def torch_device_name(torch_module=None) -> str:
    """Return the fastest Torch device supported by pipeline model code."""
    if torch_module is None:
        import torch as torch_module

    requested = requested_accelerator()
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        return "cuda" if torch_module.cuda.is_available() else "cpu"

    mps = getattr(getattr(torch_module, "backends", None), "mps", None)
    mps_available = (
        mps is not None
        and callable(getattr(mps, "is_available", None))
        and mps.is_available()
    )
    if requested == "coreml":
        return "mps" if mps_available else "cpu"
    # torchcrepe has no stable DirectML device path. The ONNX stages still use
    # DirectML, while CREPE safely uses CPU unless CUDA is available in auto.
    if requested in {"directml", "dml"}:
        return "cpu"

    if torch_module.cuda.is_available():
        return "cuda"
    xpu = getattr(torch_module, "xpu", None)
    if xpu is not None and callable(getattr(xpu, "is_available", None)):
        if xpu.is_available():
            return "xpu"
    if mps_available:
        return "mps"
    return "cpu"


def inference_hardware() -> dict[str, object]:
    """Serializable hardware provenance for stage caches and diagnostics."""
    import onnxruntime as ort

    try:
        import torch

        torch_device = torch_device_name(torch)
    except ImportError:
        torch_device = "unavailable"
    available = ort.get_available_providers()
    return {
        "platform": platform.system(),
        "machine": platform.machine(),
        "onnxAvailableProviders": available,
        "onnxSelectedProviders": onnx_provider_chain(available),
        "torchDevice": torch_device,
    }
