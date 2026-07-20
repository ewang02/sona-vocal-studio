#!/usr/bin/env python3
"""Run one Kim MDX stem on the fastest available ONNX Runtime provider."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

try:  # Support package imports and direct script execution.
    from .hardware_acceleration import CPU_PROVIDER, onnx_provider_chain
except ImportError:  # pragma: no cover - exercised by the pipeline entry point
    from hardware_acceleration import CPU_PROVIDER, onnx_provider_chain


def make_separator(separator_class, providers: list[str], **options):
    """Instantiate audio-separator and override its platform-limited choice."""
    separator = separator_class(**options)
    separator.onnx_execution_provider = providers
    return separator


def run_separator_once(
    separator_class,
    separator_options: dict,
    providers: list[str],
    model: str,
    source: Path,
    stem: str,
    output_name: str,
    output: Path,
) -> Path:
    """Run one provider chain and treat a swallowed missing output as failure."""
    separator = make_separator(separator_class, providers, **separator_options)
    separator.load_model(model)
    separator.separate(
        str(source),
        custom_output_names={stem: output_name},
    )
    if not output.exists():
        raise RuntimeError(f"separator did not produce: {output}")
    return output


def separate(
    source: Path,
    model: str,
    stem: str,
    output_name: str,
    model_dir: Path,
    output_dir: Path,
    batch_size: int = 1,
) -> Path:
    import onnxruntime as ort
    from audio_separator.separator import Separator

    available_providers = ort.get_available_providers()
    providers = onnx_provider_chain(available_providers)
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    output_dir.mkdir(parents=True, exist_ok=True)
    separator_options = dict(
        log_level=logging.WARNING,
        model_file_dir=str(model_dir),
        output_dir=str(output_dir),
        output_format="WAV",
        output_single_stem=stem,
        sample_rate=44_100,
        use_soundfile=False,
        mdx_params={
            "hop_length": 1024,
            "segment_size": 256,
            "overlap": 0.25,
            "batch_size": batch_size,
            "enable_denoise": False,
        },
    )
    # audio-separator couples its ONNX choice to Torch detection. Select ONNX
    # independently so CUDA, DirectML, CoreML, OpenVINO and CPU-only machines
    # all work even when Torch uses a different device.
    output = output_dir / f"{output_name}.wav"
    print(
        f"::hardware kim-provider={providers[0]} "
        f"fallback={providers[-1] if len(providers) > 1 else 'none'}",
        flush=True,
    )
    try:
        return run_separator_once(
            Separator,
            separator_options,
            providers,
            model,
            source,
            stem,
            output_name,
            output,
        )
    except Exception as error:
        if providers[0] == CPU_PROVIDER or CPU_PROVIDER not in available_providers:
            raise
        print(
            f"::warning {providers[0]} failed for {model}; retrying on CPU: {error}",
            flush=True,
        )
        output.unlink(missing_ok=True)
        return run_separator_once(
            Separator,
            separator_options,
            [CPU_PROVIDER],
            model,
            source,
            stem,
            output_name,
            output,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--stem", required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="MDX inference batch size; raise only after measuring memory headroom",
    )
    args = parser.parse_args()
    separate(
        args.source,
        args.model,
        args.stem,
        args.output_name,
        args.model_dir,
        args.output_dir,
        args.batch_size,
    )


if __name__ == "__main__":
    main()
