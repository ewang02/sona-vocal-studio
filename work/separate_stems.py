#!/usr/bin/env python3
"""Split one recording into two complementary stems with a Roformer model.

A single-target Roformer predicts one stem (its "primary") and the separator
derives the other by subtracting that prediction from the input, so one pass
yields both products:

* the vocal model turns a song into vocals + instrumental;
* the karaoke model turns those vocals into lead + backing.

The run uses the fastest Torch device available and retries once on CPU, so a
missing or unsupported accelerator never prevents song processing.
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

try:  # Support package imports and direct script execution.
    from .hardware_acceleration import torch_device_name
except ImportError:  # pragma: no cover - exercised by the pipeline entry point
    from hardware_acceleration import torch_device_name


def run_separator_once(
    separator_class,
    separator_options: dict,
    device: str,
    model: str,
    source: Path,
    primary: Path,
    secondary: Path,
) -> tuple[Path, Path]:
    """Run one device attempt and treat a swallowed missing output as failure."""
    import torch

    separator = separator_class(**separator_options)
    # audio-separator picks its own device at construction. Apply Sona's
    # choice (and SONA_ACCELERATOR) before the model is built on it.
    separator.torch_device = torch.device(device)
    separator.load_model(model)
    instance = separator.model_instance
    separator.separate(
        str(source),
        custom_output_names={
            instance.primary_stem_name: primary.stem,
            instance.secondary_stem_name: secondary.stem,
        },
    )
    for output in (primary, secondary):
        # audio-separator logs export errors instead of raising them, and can
        # leave an empty file behind; a bare WAV header is 44 bytes.
        if not output.is_file() or output.stat().st_size <= 44:
            raise RuntimeError(f"separator did not produce: {output}")
    return primary, secondary


def separate(
    source: Path,
    model: str,
    model_dir: Path,
    output_dir: Path,
    primary_name: str,
    secondary_name: str,
) -> tuple[Path, Path]:
    import torch
    from audio_separator.separator import Separator

    if primary_name == secondary_name:
        raise ValueError("primary and secondary outputs need different names")
    output_dir.mkdir(parents=True, exist_ok=True)
    separator_options = dict(
        log_level=logging.WARNING,
        model_file_dir=str(model_dir),
        output_dir=str(output_dir),
        output_format="WAV",
        sample_rate=44_100,
        # soundfile inherits the input's encoding and cannot write an MP3
        # source's subtype to WAV; the pydub writer always emits PCM.
        use_soundfile=False,
    )
    primary = output_dir / f"{primary_name}.wav"
    secondary = output_dir / f"{secondary_name}.wav"
    device = torch_device_name(torch)
    print(f"::hardware separator model={model} device={device}", flush=True)
    try:
        return run_separator_once(
            Separator, separator_options, device, model, source, primary, secondary
        )
    except Exception as error:
        if device == "cpu":
            raise
        print(
            f"::warning {device} failed for {model}; retrying on CPU: {error}",
            flush=True,
        )
        for output in (primary, secondary):
            output.unlink(missing_ok=True)
        accelerator = getattr(torch, device, None)
        empty_cache = getattr(accelerator, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()
        return run_separator_once(
            Separator, separator_options, "cpu", model, source, primary, secondary
        )


def separate_atomically(
    source: Path,
    model: str,
    model_dir: Path,
    primary: Path,
    secondary: Path,
    separate_function=separate,
) -> tuple[Path, Path]:
    """Write both stems to a staging directory, then move them into place.

    Either both destinations are replaced or neither is: an interrupted run
    never leaves a fresh stem beside a stale partner.
    """
    if primary.parent != secondary.parent:
        raise ValueError("both stems must be written to the same directory")
    destination = primary.parent
    destination.mkdir(parents=True, exist_ok=True)
    staging = destination / f".{primary.stem}-{secondary.stem}.tmp"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    try:
        staged_primary, staged_secondary = separate_function(
            source, model, model_dir, staging, primary.stem, secondary.stem
        )
        if not staged_primary.is_file() or not staged_secondary.is_file():
            raise RuntimeError(f"separator did not produce both stems in {staging}")
        staged_primary.replace(primary)
        staged_secondary.replace(secondary)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return primary, secondary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--primary",
        type=Path,
        required=True,
        help="destination .wav for the stem the model predicts",
    )
    parser.add_argument(
        "--secondary",
        type=Path,
        required=True,
        help="destination .wav for the remainder (input minus the primary stem)",
    )
    args = parser.parse_args()
    separate_atomically(
        args.source, args.model, args.model_dir, args.primary, args.secondary
    )


if __name__ == "__main__":
    main()
