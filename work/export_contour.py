#!/usr/bin/env python3
"""Export the selected production pitch-contour configuration as app JSON.

The named configuration controls the voicing gate, conservative multi-source
pitch merge, octave correction, and evidence-aware excursion repair. The
reference MIDI never enters the voicing path; decoded notes are used only as
conservative evidence for already-recorded octave rescues.

Usage: python export_contour.py <karaoke_root> <output_dir>
"""

from __future__ import annotations

import json
import math
import csv
from pathlib import Path

import numpy as np

from analyze_pitch_gate import (
    browser_gate,
    load_csv,
    prune_unsettled_edges,
    prune_unstable_unconfirmed_segments,
    prune_unconfirmed_low_energy_voicing,
    prune_weak_edge_excursions,
)
from contour_pipeline_config import PRESETS, PRODUCTION_CONFIG, PRODUCTION_PRESET
from pyin_fallback import apply_pyin_flagged_fallback
from octave_correct_contour import (
    correct_octaves,
    merge_pitch_sources,
    recover_secondary_voicing,
    repair_isolated_excursions,
    repair_phrase_onsets_with_secondary,
)

def mask_ranges(mask: np.ndarray, times: np.ndarray, hop: float) -> list[dict]:
    edges = np.flatnonzero(np.diff(np.r_[False, np.asarray(mask, dtype=bool), False]))
    return [
        {
            "t0": round(float(times[start]), 3),
            "t1": round(float(times[end - 1] + hop), 3),
        }
        for start, end in zip(edges[::2], edges[1::2])
    ]


def finite_value_segments(
    times: np.ndarray,
    values: np.ndarray,
    hop: float,
) -> list[dict]:
    """Serialize finite pitch runs without inventing values across pYIN gaps."""

    finite = np.isfinite(values)
    edges = np.flatnonzero(np.diff(np.r_[False, finite, False]))
    return [
        {
            "t0": round(float(times[start]), 3),
            "midi": [
                round(float(value), 2)
                for value in values[start:end]
            ],
        }
        for start, end in zip(edges[::2], edges[1::2])
    ]


def export_pyin_repair_source(
    song_root: Path,
    song: str,
    out_dir: Path,
) -> Path:
    """Publish frame-aligned lead-pYIN for reversible browser contour repair."""

    evidence_path = (
        song_root / "contour_out" / "lead_contours" / f"{song}_contour.csv"
    )
    if not evidence_path.exists():
        evidence_path = (
            song_root
            / "experiments"
            / "transcription_final"
            / "raw_f0"
            / f"{song}_contour.csv"
        )
    if not evidence_path.exists():
        raise FileNotFoundError(
            f"Lead-pYIN repair source not found for {song}: {evidence_path}"
        )
    with evidence_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "midi_pyin" not in rows[0]:
        raise RuntimeError(f"Lead contour has no midi_pyin column: {evidence_path}")
    times = np.array([float(row["time_s"]) for row in rows])
    values = np.array([float(row["midi_pyin"]) for row in rows])
    hop = float(np.median(np.diff(times))) if len(times) > 1 else 0.01
    payload = {
        "hop": round(hop, 5),
        "duration": round(float(times[-1]) + hop, 3),
        "segments": finite_value_segments(times, values, hop),
    }
    out_path = out_dir / f"{song}-pyin.json"
    temporary = out_path.with_suffix(out_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")))
    temporary.replace(out_path)
    return out_path


def export(
    song_root: Path,
    song: str,
    out_dir: Path,
    config=PRODUCTION_CONFIG,
    preset: str = PRODUCTION_PRESET,
) -> None:
    lead_contour_path = song_root / "contour_out" / "lead_contours" / f"{song}_contour.csv"
    contour_path = (
        lead_contour_path
        if lead_contour_path.exists()
        else song_root / "contour_out" / f"{song}_contour.csv"
    )
    data = load_csv(contour_path)
    times = data["time_s"]
    midi = data["midi"]
    hop = float(times[1] - times[0])

    notes_path = song_root / "contour_out" / f"{song}_notes_evidence.json"
    notes = json.loads(notes_path.read_text()).get("notes", []) if notes_path.exists() else []
    evidence_path = (
        lead_contour_path
        if lead_contour_path.exists()
        else song_root / "experiments" / "transcription_final" / "raw_f0" / f"{song}_contour.csv"
    )
    evidence_times = evidence_midi = evidence_confidence = None
    if evidence_path.exists():
        with evidence_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        evidence_times = np.array([float(row["time_s"]) for row in rows])
        evidence_midi = np.array([float(row["midi_pyin"]) for row in rows])
        evidence_confidence = np.array([float(row["confidence_pyin"]) for row in rows])

    keep, rms_floor, _, gate_diagnostics = browser_gate(
        data,
        config,
        secondary_midi=evidence_midi,
        secondary_confidence=evidence_confidence,
        return_diagnostics=True,
    )

    # Lead audio is the reliable source of voicing boundaries.  Keep the
    # existing full-model contour as the pitch anchor until a bounded full
    # lead-only pass has been generated; lead-only pYIN below still controls
    # exact-octave repairs.
    pitch_source = "lead-only-tiny"
    mixed_confidence = data["confidence"]
    raw_midi = data.get("midi_raw", data["midi"])
    raw_confidence = data.get("confidence_raw", data["confidence"])
    if contour_path == lead_contour_path:
        mixed_full_path = song_root / "contour_out" / f"{song}_contour.csv"
        # Full lead-only CREPE was evaluated as an alternate anchor, but it
        # introduced more short excursions in quiet/ornamented passages. Keep
        # those contours as diagnostics and retain the steadier mixed-full
        # anchor; lead-only tiny/pYIN still controls gating and corrections.
        full_path = mixed_full_path
        if full_path.exists():
            full_data = load_csv(full_path)
            indices = np.searchsorted(full_data["time_s"], times, side="left")
            indices = np.clip(indices, 0, len(full_data["time_s"]) - 1)
            midi = full_data["midi"][indices]
            mixed_confidence = full_data["confidence"][indices]
            raw_midi = full_data.get("midi_raw", full_data["midi"])[indices]
            raw_confidence = full_data.get(
                "confidence_raw", full_data["confidence"]
            )[indices]
            pitch_source = "full-resolution-broad-vocals"
    broad_midi = midi.copy()

    if evidence_midi is not None:
        keep, secondary_recovered_frames = recover_secondary_voicing(
            times,
            keep,
            data["rms"],
            rms_floor,
            evidence_midi,
            evidence_confidence,
        )
    else:
        secondary_recovered_frames = 0

    unconfirmed_mask = np.zeros(len(times), dtype=bool)
    unconfirmed_stats = {
        "rmsPercentile": config.unconfirmed_rms_percentile,
        "removedSegments": 0,
        "removedFrames": 0,
    }
    if (
        config.unconfirmed_rms_percentile > 0
        and evidence_midi is not None
        and evidence_confidence is not None
    ):
        keep, unconfirmed_mask, unconfirmed_stats = (
            prune_unconfirmed_low_energy_voicing(
                times,
                keep,
                data["rms"],
                evidence_midi,
                evidence_confidence,
                config,
            )
        )

    merge_mask = np.zeros(len(times), dtype=bool)
    uncertainty_mask = np.zeros(len(times), dtype=bool)
    merge_stats = {"mode": "mixed_full", "leadSelectedFrames": 0, "unresolvedFrames": 0}
    if config.pitch_merge == "confidence_aware" and evidence_midi is not None:
        midi, merge_mask, uncertainty_mask, merge_stats = merge_pitch_sources(
            midi,
            mixed_confidence,
            data["midi"],
            data["confidence"],
            evidence_midi,
            evidence_confidence,
            keep,
        )
        pitch_source = "confidence-aware-full-lead-merge"
    midi, correction_mask, correction_stats = correct_octaves(
        times,
        midi,
        notes=notes,
        evidence_times=evidence_times,
        evidence_midi=evidence_midi,
        evidence_confidence=evidence_confidence,
        note_region_source_tolerance_st=(
            config.note_region_source_tolerance_st
        ),
    )
    midi, excursion_mask, excursion_stats = repair_isolated_excursions(
        times,
        midi,
        keep,
        maximum_region_frames=config.excursion_maximum_frames,
        evidence_aware=config.evidence_aware_excursions,
        evidence_midi=evidence_midi,
        evidence_confidence=evidence_confidence,
        raw_midi=raw_midi,
        raw_confidence=raw_confidence,
    )
    midi, secondary_onset_mask, secondary_onset_stats = (
        repair_phrase_onsets_with_secondary(
            times,
            midi,
            keep,
            mixed_confidence,
            evidence_midi,
            evidence_confidence,
            max_seconds=config.secondary_onset_repair_seconds,
            mode=config.secondary_onset_mode,
            min_gap_seconds=config.secondary_onset_min_gap_seconds,
            secondary_min_confidence=(
                config.secondary_onset_min_confidence
            ),
            minimum_deviation_st=(
                config.secondary_onset_min_deviation_st
            ),
            agreement_tolerance_st=(
                config.secondary_onset_agreement_st
            ),
            evidence_frames=(
                config.secondary_onset_evidence_frames
            ),
            primary_stability_frames=(
                config.secondary_onset_primary_stability_frames
            ),
            stability_tolerance_st=(
                config.secondary_onset_stability_tolerance_st
            ),
            stable_primary_confidence=(
                config.secondary_onset_primary_confidence
            ),
        )
        if evidence_midi is not None and evidence_confidence is not None
        else (
            midi,
            np.zeros(len(times), dtype=bool),
            {
                "maxSeconds": config.secondary_onset_repair_seconds,
                "repairedOnsets": 0,
                "repairedFrames": 0,
            },
        )
    )
    edge_mask = np.zeros(len(times), dtype=bool)
    edge_stats = {
        "maxSeconds": config.edge_excursion_max_seconds,
        "removedEdges": 0,
        "removedFrames": 0,
    }
    if (
        config.edge_excursion_max_seconds > 0
        and evidence_midi is not None
        and evidence_confidence is not None
    ):
        keep, edge_mask, edge_stats = prune_weak_edge_excursions(
            times,
            keep,
            midi,
            data["rms"],
            evidence_midi,
            evidence_confidence,
            config,
        )
    unsettled_mask = np.zeros(len(times), dtype=bool)
    unsettled_stats = {
        "maxSeconds": config.unsettled_edge_max_seconds,
        "trimmedEdges": 0,
        "removedFrames": 0,
    }
    if (
        config.unsettled_edge_max_seconds > 0
        and evidence_midi is not None
        and evidence_confidence is not None
    ):
        keep, unsettled_mask, unsettled_stats = prune_unsettled_edges(
            times,
            keep,
            midi,
            evidence_midi,
            evidence_confidence,
            raw_midi,
            raw_confidence,
            config,
        )
    unstable_mask = np.zeros(len(times), dtype=bool)
    unstable_stats = {
        "maxSeconds": config.unstable_segment_max_seconds,
        "removedSegments": 0,
        "removedFrames": 0,
    }
    if (
        config.unstable_segment_max_seconds > 0
        and evidence_midi is not None
        and evidence_confidence is not None
    ):
        keep, unstable_mask, unstable_stats = prune_unstable_unconfirmed_segments(
            times,
            keep,
            midi,
            data["confidence"],
            evidence_midi,
            evidence_confidence,
            config,
        )

    fallback_pitch_mask = np.zeros(len(times), dtype=bool)
    fallback_unvoiced_mask = np.zeros(len(times), dtype=bool)
    fallback_stats = {"enabled": False}
    if (
        config.pyin_fallback_enabled
        and evidence_midi is not None
        and evidence_confidence is not None
    ):
        (
            midi,
            keep,
            fallback_pitch_mask,
            fallback_unvoiced_mask,
            fallback_stats,
        ) = apply_pyin_flagged_fallback(
            times,
            midi,
            keep,
            mixed_confidence,
            data["confidence"],
            evidence_midi,
            evidence_confidence,
            config,
        )
        fallback_stats = {"enabled": True, **fallback_stats}

    reviewed_breath_mask = np.zeros(len(times), dtype=bool)
    reviewed_pitch_mask = np.zeros(len(times), dtype=bool)
    reviewed_model_stats = {
        "enabled": False,
        "modelPath": config.reviewed_model_path,
    }
    if config.reviewed_model_enabled:
        if evidence_midi is None or evidence_confidence is None:
            raise RuntimeError(
                "Reviewed contour models require frame-aligned pYIN evidence"
            )
        from reviewed_contour_classifier import apply_reviewed_contour_models

        lead_audio_path = (
            song_root / "contour_out" / "lead_vocals" / song / "lead.wav"
        )
        if not lead_audio_path.exists():
            lead_audio_path = (
                song_root / "contour_out" / "separators" / song / "vocals.wav"
            )
        model_path = song_root / config.reviewed_model_path
        base_breath_model_path = (
            song_root / "experiments" / "breath_detector" / "model.json"
        )
        (
            keep,
            midi,
            reviewed_breath_mask,
            reviewed_pitch_mask,
            reviewed_model_stats,
        ) = apply_reviewed_contour_models(
            times=times,
            keep=keep,
            midi=midi,
            broad_midi=broad_midi,
            broad_confidence=mixed_confidence,
            lead_midi=data["midi"],
            lead_confidence=data["confidence"],
            lead_rms=data["rms"],
            pyin_midi=evidence_midi,
            pyin_confidence=evidence_confidence,
            lead_audio_path=lead_audio_path,
            model_path=model_path,
            base_breath_model_path=base_breath_model_path,
            require_production_eligible=(
                config.reviewed_model_require_production_eligible
            ),
        )
        reviewed_model_stats = {
            "enabled": True,
            "modelPath": config.reviewed_model_path,
            **reviewed_model_stats,
        }

    segments: list[dict] = []
    indexes = np.flatnonzero(np.diff(np.r_[0, keep.view(np.int8), 0]))
    for start, end in zip(indexes[::2], indexes[1::2]):
        if (end - start) * hop < config.minimum_segment_seconds:
            continue
        segments.append(
            {
                "t0": round(float(times[start]), 3),
                "midi": [round(float(value), 2) for value in midi[start:end]],
            }
        )

    voiced = midi[keep]
    payload = {
        "hop": round(hop, 5),
        "duration": round(float(times[-1]) + hop, 3),
        "pipelinePreset": preset,
        "source": "lead-gated" if contour_path == lead_contour_path else "broad-vocals",
        "pitchSource": pitch_source,
        "gate": {
            **gate_diagnostics,
            "rmsFloor": round(float(rms_floor), 6),
            "maxJumpSt": config.max_jump_semitones,
        },
        "secondaryVoicingRecovery": {
            "frames": secondary_recovered_frames,
            "seconds": round(secondary_recovered_frames * hop, 3),
        },
        "weakEdgeExcursionPrune": edge_stats,
        "octaveCorrection": correction_stats,
        "excursionRepair": excursion_stats,
        "secondaryOnsetRepair": secondary_onset_stats,
        "pitchMerge": merge_stats,
        "pyinFallback": fallback_stats,
        "reviewedContourModel": reviewed_model_stats,
        "auditRanges": {
            "leadPitchSelected": mask_ranges(merge_mask, times, hop),
            "unresolvedPitchSource": mask_ranges(uncertainty_mask, times, hop),
            "weakEdgeExcursionRemoved": mask_ranges(edge_mask, times, hop),
            "octaveCorrected": mask_ranges(correction_mask, times, hop),
            "excursionRepaired": mask_ranges(excursion_mask, times, hop),
            "secondaryOnsetRepaired": mask_ranges(
                secondary_onset_mask, times, hop
            ),
            "reviewedBreathRemoved": mask_ranges(
                reviewed_breath_mask, times, hop
            ),
            "reviewedPitchRepaired": mask_ranges(
                reviewed_pitch_mask, times, hop
            ),
            "pyinFallbackPitch": mask_ranges(fallback_pitch_mask, times, hop),
            "pyinFallbackUnvoiced": mask_ranges(
                fallback_unvoiced_mask, times, hop
            ),
        },
        "range": [math.floor(float(np.min(voiced))), math.ceil(float(np.max(voiced)))],
        "segments": segments,
    }
    if config.unconfirmed_rms_percentile > 0:
        payload["unconfirmedLowEnergyPrune"] = unconfirmed_stats
        payload["auditRanges"]["unconfirmedLowEnergyRemoved"] = mask_ranges(
            unconfirmed_mask, times, hop
        )
    if config.unstable_segment_max_seconds > 0:
        payload["unstableUnconfirmedPrune"] = unstable_stats
        payload["auditRanges"]["unstableUnconfirmedRemoved"] = mask_ranges(
            unstable_mask, times, hop
        )
    if config.unsettled_edge_max_seconds > 0:
        payload["unsettledEdgePrune"] = unsettled_stats
        payload["auditRanges"]["unsettledEdgeRemoved"] = mask_ranges(
            unsettled_mask, times, hop
        )
    out_path = out_dir / f"{song}-contour.json"
    temporary_path = out_path.with_suffix(out_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, separators=(",", ":")))
    temporary_path.replace(out_path)
    export_pyin_repair_source(song_root, song, out_dir)
    total = sum(len(seg["midi"]) for seg in segments)
    print(
        f"{song}: {len(segments)} segments, {total} frames "
        f"({total * hop:.1f}s voiced of {payload['duration']:.1f}s), "
        f"range {payload['range']}, corrected {int(correction_mask.sum())} octave frames + "
        f"{int(excursion_mask.sum())} excursion frames, "
        f"{out_path.stat().st_size / 1024:.0f} KB"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("songs", nargs="+", help="one or more song ids to export")
    parser.add_argument(
        "--preset",
        default=PRODUCTION_PRESET,
        choices=sorted(PRESETS),
        help="named contour pipeline configuration (default: production)",
    )
    parser.add_argument(
        "--pyin-only",
        action="store_true",
        help="publish only the lead-pYIN browser repair source",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for song_name in args.songs:
        if args.pyin_only:
            export_pyin_repair_source(args.root, song_name, args.out)
        else:
            export(args.root, song_name, args.out, PRESETS[args.preset], args.preset)
