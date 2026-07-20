#!/usr/bin/env python3
"""Build final note transcriptions sequentially with bounded memory.

Every extraction runs in a fresh process and completes before the next song.
This is intentional: torchcrepe caches model state, so parallel songs can
multiply memory use dramatically.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

try:  # Support both script execution and imports from the test suite.
    from .contour_pipeline_config import PRODUCTION_CONFIG
except ImportError:  # pragma: no cover - exercised by the script entry point
    from contour_pipeline_config import PRODUCTION_CONFIG


ROOT = Path(__file__).resolve().parents[1]
def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("songs", nargs="+", help="one or more song ids")
    parser.add_argument("--model", choices=("tiny", "full"), default="tiny")
    parser.add_argument(
        "--batch-size",
        type=int,
        help="override the CREPE batch size (defaults: tiny=64, full=8)",
    )
    parser.add_argument(
        "--pyin-workers",
        type=int,
        default=0,
        help="pYIN worker processes; 0 chooses a memory-bounded automatic count",
    )
    parser.add_argument("--skip-extraction", action="store_true")
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="write the CREPE/pYIN contour and skip note decoding and preview rendering",
    )
    parser.add_argument(
        "--skip-piano",
        action="store_true",
        help="keep note decoding but skip the optional piano-guide WAV",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="copy validated final JSON into contour_out as correction evidence",
    )
    args = parser.parse_args()

    python = sys.executable
    experiment = ROOT / "experiments/transcription_final"
    raw_directory = experiment / "raw_f0"
    raw_directory.mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs").mkdir(parents=True, exist_ok=True)

    # Conservative defaults keep even the full model bounded; tiny is the
    # production default because it is substantially faster. The filtered
    # legacy full-model contour remains an audio-only pitch-center anchor.
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.pyin_workers < 0:
        parser.error("--pyin-workers must be zero or greater")
    batch_size = str(
        args.batch_size
        if args.batch_size is not None
        else (64 if args.model == "tiny" else 8)
    )
    chunk_seconds = "20" if args.model == "tiny" else "8"

    for song in args.songs:
        lead_vocals = ROOT / f"contour_out/lead_vocals/{song}/lead.wav"
        separator_vocals = ROOT / f"contour_out/separators/{song}/vocals.wav"
        legacy_vocals = ROOT / f"contour_out/demucs/htdemucs/{song}/vocals.wav"
        mixed_vocals = separator_vocals if separator_vocals.exists() else legacy_vocals
        vocals = lead_vocals if lead_vocals.exists() else mixed_vocals
        anchor = ROOT / f"contour_out/{song}_contour.csv"
        lead_contour = ROOT / f"contour_out/lead_contours/{song}_contour.csv"
        extraction_directory = (
            lead_contour.parent if vocals == lead_vocals else raw_directory
        )
        raw_contour = (
            lead_contour
            if vocals == lead_vocals
            else raw_directory / f"{song}_contour.csv"
        )
        final_notes = experiment / f"{song}_notes_final.json"
        piano = ROOT / f"outputs/{song}-auto-piano.wav"
        if not vocals.exists() or not anchor.exists():
            raise SystemExit(f"missing source assets for {song}")

        print(f"\n=== {song}: one-song-at-a-time final build ===", flush=True)
        if not args.skip_extraction:
            run(
                [
                    python,
                    "contour.py",
                    str(vocals.relative_to(ROOT)),
                    "--no-separate",
                    "--name",
                    song,
                    "--fmax",
                    "1500",
                    "--model",
                    args.model,
                    "--batch-size",
                    batch_size,
                    "--torch-threads",
                    "2",
                    "--chunk-seconds",
                    chunk_seconds,
                    "--decoder",
                    "weighted_argmax",
                    "--pitch-filter",
                    PRODUCTION_CONFIG.lead_pitch_filter,
                    "--secondary-f0",
                    "pyin",
                    "--pyin-chunk-seconds",
                    "20",
                    "--pyin-workers",
                    str(args.pyin_workers),
                    "--data-only",
                    "--outdir",
                    str(extraction_directory.relative_to(ROOT)),
                ]
            )
        if not raw_contour.exists():
            raise SystemExit(f"missing extracted contour: {raw_contour}")

        if args.extract_only:
            continue

        run(
            [
                python,
                "transcribe_notes.py",
                "--csv",
                str(raw_contour.relative_to(ROOT)),
                "--pitch-anchor",
                str(anchor.relative_to(ROOT)),
                "--vocals",
                str(vocals.relative_to(ROOT)),
                "--no-beat",
                "--out",
                str(final_notes.relative_to(ROOT)),
            ]
        )
        if not args.skip_piano:
            run(
                [
                    python,
                    "render_note_transcription.py",
                    str(final_notes.relative_to(ROOT)),
                    "--instrument",
                    "piano",
                    "--out",
                    str(piano.relative_to(ROOT)),
                ]
            )

        if args.publish:
            shutil.copy2(final_notes, ROOT / f"contour_out/{song}_notes_evidence.json")

    if args.extract_only:
        summary = "Pitch extraction complete."
    elif args.skip_piano:
        summary = "Final transcriptions complete."
    else:
        summary = "Final transcriptions and piano guides complete."
    print(f"\n{summary}", flush=True)


if __name__ == "__main__":
    main()
