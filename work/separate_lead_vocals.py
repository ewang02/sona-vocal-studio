#!/usr/bin/env python3
"""Split broad vocal stems into lead and backing, one song at a time.

This uses a fresh process for each song.  The karaoke model's ``Vocals`` output
is the removed lead; ``Instrumental`` is the remainder of the already-isolated
vocal stem (backing vocals and artifacts).  The VR batch size is configurable,
but defaults to the benchmarked memory-safe value of one.
"""

from __future__ import annotations

import argparse
import hashlib
import platform
import shutil
import subprocess
import sys
from pathlib import Path


MODEL = "5_HP-Karaoke-UVR.pth"
MODEL_SHA256 = "fe00891defbb61f4261500af22f7624f1a3df8dc75fa3998d1aece02e6be4537"


def torch_acceleration_status() -> tuple[str, str | None]:
    """Report the device the PyTorch-only VR separator can actually use."""
    import torch

    if torch.cuda.is_available():
        return "cuda", None
    if torch.backends.mps.is_available():
        return "mps", None
    try:
        import torch_directml

        if torch_directml.is_available():
            return "directml", None
    except ImportError:
        pass
    reason = None
    if platform.machine() == "arm64":
        built = torch.backends.mps.is_built()
        reason = (
            "MPS is built but unavailable to this process"
            if built
            else "this Torch build has no MPS support"
        )
    return "cpu", reason


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_source(root: Path, song: str, source: Path | None) -> Path:
    if source is not None:
        return source if source.is_absolute() else root / source
    candidates = (
        root / "contour_out" / "separators" / song / "vocals.wav",
        root / "contour_out" / "demucs" / "htdemucs" / song / "vocals.wav",
    )
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


def separate(
    root: Path,
    song: str,
    source: Path | None = None,
    validated_model_sha256: str | None = None,
    vr_batch_size: int = 1,
) -> None:
    source = resolve_source(root, song, source)
    model_dir = root / "contour_out" / "models" / "audio-separator"
    model_path = model_dir / MODEL
    destination = root / "contour_out" / "lead_vocals" / song
    staging = destination.with_name(f".{song}.tmp")
    if not source.exists():
        raise FileNotFoundError(f"Missing vocal stem: {source}")
    if vr_batch_size < 1:
        raise ValueError("vr_batch_size must be at least 1")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing separator model: {model_path}")
    if validated_model_sha256 is not None and validated_model_sha256 != MODEL_SHA256:
        raise RuntimeError("parent process supplied the wrong validated separator hash")
    actual_hash = validated_model_sha256 or sha256(model_path)
    if actual_hash != MODEL_SHA256:
        raise RuntimeError(
            f"Separator model hash mismatch for {MODEL}: "
            f"expected {MODEL_SHA256}, got {actual_hash}"
        )
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    executable = Path(sys.executable).with_name("audio-separator")
    command = [
        str(executable), str(source), "-m", MODEL,
        "--model_file_dir", str(model_dir),
        "--output_dir", str(staging),
        "--output_format", "WAV",
        "--vr_batch_size", str(vr_batch_size),
        "--vr_window_size", "512",
        "--use_soundfile",
    ]
    device, reason = torch_acceleration_status()
    if device == "directml":
        command.append("--use_directml")
    detail = f" reason={reason}" if reason else ""
    print(f"::hardware lead-separator device={device}{detail}", flush=True)
    print(f"[{song}] separating lead/backing with bounded settings...", flush=True)
    try:
        subprocess.run(command, check=True)
        lead_files = list(staging.glob("*(Vocals)*.wav"))
        backing_files = list(staging.glob("*(Instrumental)*.wav"))
        if len(lead_files) != 1 or len(backing_files) != 1:
            raise RuntimeError(f"Unexpected separator outputs in {staging}")
        lead_files[0].replace(staging / "lead.wav")
        backing_files[0].replace(staging / "backing.wav")
        if destination.exists():
            shutil.rmtree(destination)
        staging.replace(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"[{song}] wrote {destination / 'lead.wav'} and backing.wav", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--song",
        action="append",
        required=True,
        help="song asset name; repeat as needed",
    )
    parser.add_argument(
        "--source",
        type=Path,
        help="explicit broad vocal stem (only valid when processing one --song)",
    )
    parser.add_argument(
        "--validated-model-sha256",
        help="reuse an exact hash already verified by the parent pipeline",
    )
    parser.add_argument(
        "--vr-batch-size",
        type=int,
        default=1,
        help="VR inference batch size; raise only after measuring memory headroom",
    )
    args = parser.parse_args()
    songs = args.song
    if args.source is not None and len(songs) != 1:
        parser.error("--source requires exactly one --song")
    for song in songs:
        separate(
            args.root.resolve(),
            song,
            args.source,
            args.validated_model_sha256,
            args.vr_batch_size,
        )


if __name__ == "__main__":
    main()
