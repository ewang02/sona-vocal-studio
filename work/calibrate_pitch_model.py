#!/usr/bin/env python3
"""Measure a pitch model's confidence thresholds against CREPE.

The pipeline tests confidence at the fixed levels in
``pitch_models.CONFIDENCE_LEVELS``, which were tuned with CREPE. This script
finds, for another model, the native confidence that is equally strict at
each level: the value that the same share of sung frames clears. It prints
the two rows to paste into that model's entry in ``work/pitch_models.py``.

It needs each song's separated stems, so process the songs with
``SONA_KEEP_INTERMEDIATES=1`` (or without ``--clean-intermediates``) first:

    contour_out/separators/<song>/vocals.wav      anchor row
    contour_out/lead_vocals/<song>/lead.wav       lead row

Usage:  python work/calibrate_pitch_model.py --model rmvpe SONG [SONG ...]

Matching acceptance rates makes a model as strict as CREPE; it does not say
which model is right. Judge that by ear in the editor, then adjust.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "work"))

from analyze_pitch_gate import choose_rms_floor  # noqa: E402
from contour import DEFAULT_MODEL_DIR, extract_f0  # noqa: E402
from pitch_models import CONFIDENCE_LEVELS, PITCH_MODELS, get_pitch_model  # noqa: E402

ROLES = {
    "anchor": ("separators", "vocals.wav", "full", 512),
    "lead": ("lead_vocals", "lead.wav", "tiny", 64),
}


RECOVER_LEVEL = 0.30


def matched_knots(
    reference: np.ndarray,
    native: np.ndarray,
    voicing_floor: float | None = None,
) -> tuple[float, ...]:
    """Native values cleared by the same share of frames as each level.

    When the model publishes its own voiced/unvoiced threshold, that value
    pins the "recoverable" level and scales the levels below it.
    """
    knots: list[float] = []
    previous = 0.0
    for level in CONFIDENCE_LEVELS:
        accepted = float(np.mean(reference >= level))
        value = float(np.quantile(native, 1.0 - accepted))
        if voicing_floor is not None and level <= RECOVER_LEVEL:
            value = voicing_floor * level / RECOVER_LEVEL
        # Knots must increase strictly inside (0, 1) to stay invertible.
        value = min(max(value, previous + 1e-3), 1.0 - 1e-3 * (len(CONFIDENCE_LEVELS) - len(knots)))
        knots.append(round(value, 3))
        previous = knots[-1]
    return tuple(knots)


def stem_confidences(stem: Path, role: str, model: str, model_dir: Path):
    _, _, crepe_variant, batch_size = ROLES[role]
    reference = extract_f0(stem, 10.0, crepe_variant, 1500.0, batch_size)
    candidate = extract_f0(
        stem,
        10.0,
        crepe_variant,
        1500.0,
        batch_size,
        estimator=model,
        estimator_role=role,
        model_dir=model_dir,
    )
    rms = reference[3]
    floor, _ = choose_rms_floor(rms)
    sung = rms >= floor
    reference_midi = 69 + 12 * np.log2(np.maximum(reference[6], 1e-6) / 440)
    candidate_midi = 69 + 12 * np.log2(np.maximum(candidate[6], 1e-6) / 440)
    confident = sung & (reference[7] >= 0.7)
    difference = np.abs(reference_midi - candidate_midi)[confident]
    print(
        f"  {stem.parent.name}/{stem.name}: {int(sung.sum())} sung frames; "
        f"where CREPE is confident the pitches agree within half a semitone on "
        f"{100 * float(np.mean(difference <= 0.5)):.1f}% and sit an octave apart on "
        f"{100 * float(np.mean(np.abs(difference - 12) <= 0.75)):.1f}%",
        flush=True,
    )
    return reference[7][sung], candidate[8][sung]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("songs", nargs="+", help="song ids whose stems are kept")
    parser.add_argument(
        "--model",
        required=True,
        choices=sorted(name for name in PITCH_MODELS if name != "crepe"),
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    args = parser.parse_args()
    tuning = get_pitch_model(args.model)

    rows: dict[str, tuple[float, ...]] = {}
    for role, (directory, filename, _, _) in ROLES.items():
        print(f"{tuning.label} {role} tuning:", flush=True)
        reference_parts: list[np.ndarray] = []
        native_parts: list[np.ndarray] = []
        for song in args.songs:
            stem = ROOT / "contour_out" / directory / song / filename
            if not stem.is_file():
                raise SystemExit(f"missing stem for {song}: {stem}")
            reference, native = stem_confidences(stem, role, args.model, args.model_dir)
            if len(args.songs) > 1:
                # Per-song rows show how far the measurement moves between songs.
                alone = matched_knots(reference, native, tuning.native_voicing_floor)
                print(f"    {song} alone: {alone}", flush=True)
            reference_parts.append(reference)
            native_parts.append(native)
        rows[role] = matched_knots(
            np.concatenate(reference_parts),
            np.concatenate(native_parts),
            tuning.native_voicing_floor,
        )

    print(f"\nPaste into PITCH_MODELS[{args.model!r}] in work/pitch_models.py:")
    print(f"        # levels:        {CONFIDENCE_LEVELS}")
    print(f"        anchor_confidence={rows['anchor']},")
    print(f"        lead_confidence={rows['lead']},")


if __name__ == "__main__":
    main()
