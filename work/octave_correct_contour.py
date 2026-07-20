#!/usr/bin/env python3
"""Conservatively repair octave errors in a continuous pitch contour.

The continuous contour stays authoritative.  We only move it by whole octaves
when an independent estimator sustains the opposite octave, or when the note
decoder has already recorded an explicit octave rescue.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class CorrectionStats:
    corrected_frames: int = 0
    corrected_seconds: float = 0.0
    note_regions: int = 0
    estimator_runs: int = 0
    gross_error_frames: int = 0


@dataclass
class ExcursionStats:
    repaired_frames: int = 0
    repaired_seconds: float = 0.0
    repaired_regions: int = 0
    evidence_protected_frames: int = 0
    evidence_protected_regions: int = 0


def repair_phrase_onsets_with_secondary(
    times: np.ndarray,
    midi: np.ndarray,
    keep: np.ndarray,
    primary_confidence: np.ndarray,
    secondary_midi: np.ndarray,
    secondary_confidence: np.ndarray,
    *,
    max_seconds: float = 0.0,
    mode: str = "prefix",
    min_gap_seconds: float = 0.05,
    secondary_min_confidence: float = 0.60,
    minimum_deviation_st: float = 2.50,
    agreement_tolerance_st: float = 1.00,
    evidence_frames: int = 3,
    primary_stability_frames: int = 3,
    stability_tolerance_st: float = 1.00,
    stable_primary_confidence: float = 0.60,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Repair a short phrase-onset pitch artifact from independent pYIN.

    This changes pitch only; it never adds or removes voicing. Only the start
    of a contiguous voiced run is eligible, and an interior pitch transition
    therefore cannot trigger the repair. Prefix mode requires stable,
    high-confidence pYIN from the first frame; available mode uses each
    high-confidence pYIN frame independently. A stable, confident primary
    pickup is protected even when it differs from pYIN, because that pattern
    may be an intentional grace note.
    """
    source = np.asarray(midi, dtype=float)
    result = source.copy()
    active = np.asarray(keep, dtype=bool) & np.isfinite(source)
    primary_score = np.asarray(primary_confidence, dtype=float)
    secondary = np.asarray(secondary_midi, dtype=float)
    secondary_score = np.asarray(secondary_confidence, dtype=float)
    repaired = np.zeros(len(source), dtype=bool)
    stats = {
        "maxSeconds": float(max_seconds),
        "minimumGapSeconds": float(min_gap_seconds),
        "repairedOnsets": 0,
        "repairedFrames": 0,
        "stablePrimaryProtectedOnsets": 0,
        "insufficientSecondaryOnsets": 0,
        "shortGapSkippedOnsets": 0,
    }
    if max_seconds <= 0 or not len(source):
        return result, repaired, stats
    if mode not in {"prefix", "available"}:
        raise ValueError(f"Unknown secondary onset repair mode: {mode}")

    time_values = np.asarray(times, dtype=float)
    hop = float(np.median(np.diff(time_values))) if len(time_values) > 1 else 0.01
    max_frames = max(1, int(round(max_seconds / hop)))
    minimum_gap_frames = max(1, int(round(min_gap_seconds / hop)))
    required_evidence_frames = max(1, int(evidence_frames))
    pickup_frames = max(1, int(primary_stability_frames))
    edges = np.flatnonzero(np.diff(np.r_[False, active, False]))

    for start, end in zip(edges[::2], edges[1::2]):
        if start > 0:
            gap_start = start
            while gap_start > 0 and not active[gap_start - 1]:
                gap_start -= 1
            if start - gap_start < minimum_gap_frames:
                stats["shortGapSkippedOnsets"] += 1
                continue

        inspect_end = min(end, start + max_frames)
        primary_end = min(inspect_end, start + pickup_frames)
        primary_prefix = source[start:primary_end]
        primary_prefix_score = primary_score[start:primary_end]
        stable_primary = (
            primary_end - start >= pickup_frames
            and np.all(np.isfinite(primary_prefix))
            and float(np.ptp(primary_prefix)) <= stability_tolerance_st
            and np.all(primary_prefix_score >= stable_primary_confidence)
        )
        if stable_primary:
            stats["stablePrimaryProtectedOnsets"] += 1
            continue

        if mode == "available":
            supported = (
                np.isfinite(secondary[start:inspect_end])
                & (
                    secondary_score[start:inspect_end]
                    >= secondary_min_confidence
                )
            )
            if not np.any(supported):
                stats["insufficientSecondaryOnsets"] += 1
                continue
            deviations = np.abs(
                source[start:inspect_end] - secondary[start:inspect_end]
            )
            effective_deviation = max(float(minimum_deviation_st), 1e-6)
            selection = supported & (deviations >= effective_deviation)
            if not np.any(selection):
                continue
            indexes = np.flatnonzero(selection) + start
            result[indexes] = secondary[indexes]
            repaired[indexes] = True
            stats["repairedOnsets"] += 1
            continue

        evidence_end = min(inspect_end, start + required_evidence_frames)
        if evidence_end - start < required_evidence_frames:
            stats["insufficientSecondaryOnsets"] += 1
            continue

        secondary_prefix = secondary[start:evidence_end]
        secondary_prefix_score = secondary_score[start:evidence_end]
        secondary_supported = (
            np.isfinite(secondary_prefix)
            & (secondary_prefix_score >= secondary_min_confidence)
        )
        if (
            not np.all(secondary_supported)
            or float(np.ptp(secondary_prefix)) > stability_tolerance_st
        ):
            stats["insufficientSecondaryOnsets"] += 1
            continue

        first_deviation = abs(float(source[start] - secondary[start]))
        if first_deviation < minimum_deviation_st:
            continue

        repaired_this_onset = False
        for index in range(start, inspect_end):
            if (
                not np.isfinite(secondary[index])
                or secondary_score[index] < secondary_min_confidence
            ):
                break
            if abs(float(source[index] - secondary[index])) <= agreement_tolerance_st:
                break
            result[index] = secondary[index]
            repaired[index] = True
            repaired_this_onset = True

        if repaired_this_onset:
            stats["repairedOnsets"] += 1

    stats["repairedFrames"] = int(np.sum(repaired))
    return result, repaired, stats


def merge_pitch_sources(
    mixed_midi: np.ndarray,
    mixed_confidence: np.ndarray,
    lead_midi: np.ndarray,
    lead_confidence: np.ndarray,
    secondary_midi: np.ndarray,
    secondary_confidence: np.ndarray,
    keep: np.ndarray,
    *,
    lead_min_confidence: float = 0.45,
    secondary_min_confidence: float = 0.60,
    lead_secondary_tolerance_st: float = 1.0,
    mixed_disagreement_st: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Prefer lead pitch only when two independent lead estimators agree.

    The mixed/full contour remains authoritative by default. A frame switches
    to lead CREPE only when lead CREPE and pYIN agree while the mixed source
    materially disagrees. Unresolved multi-source disagreements are exported
    for audit instead of silently forcing one source.
    """

    mixed = np.asarray(mixed_midi, dtype=float)
    lead = np.asarray(lead_midi, dtype=float)
    secondary = np.asarray(secondary_midi, dtype=float)
    mixed_confidence = np.asarray(mixed_confidence, dtype=float)
    lead_confidence = np.asarray(lead_confidence, dtype=float)
    secondary_confidence = np.asarray(secondary_confidence, dtype=float)
    active = np.asarray(keep, dtype=bool)
    result = mixed.copy()
    lead_secondary_agree = (
        active
        & np.isfinite(lead)
        & np.isfinite(secondary)
        & (lead_confidence >= lead_min_confidence)
        & (secondary_confidence >= secondary_min_confidence)
        & (np.abs(lead - secondary) <= lead_secondary_tolerance_st)
    )
    mixed_missing_or_disagrees = (
        ~np.isfinite(mixed) | (np.abs(mixed - lead) >= mixed_disagreement_st)
    )
    selected_lead = lead_secondary_agree & mixed_missing_or_disagrees
    result[selected_lead] = lead[selected_lead]

    mixed_supported = (
        np.isfinite(mixed)
        & (
            (np.isfinite(lead) & (np.abs(mixed - lead) <= mixed_disagreement_st))
            | (
                np.isfinite(secondary)
                & (secondary_confidence >= secondary_min_confidence)
                & (np.abs(mixed - secondary) <= mixed_disagreement_st)
            )
        )
    )
    unresolved = (
        active
        & np.isfinite(mixed)
        & np.isfinite(lead)
        & (np.abs(mixed - lead) > mixed_disagreement_st)
        & ~selected_lead
        & ~mixed_supported
    )
    return result, selected_lead, unresolved, {
        "mode": "confidence_aware",
        "leadSelectedFrames": int(np.sum(selected_lead)),
        "unresolvedFrames": int(np.sum(unresolved)),
        "mixedRetainedFrames": int(np.sum(active & ~selected_lead)),
    }


def _octave_shift(source: np.ndarray, target: np.ndarray, tolerance: float) -> np.ndarray:
    difference = target - source
    octaves = np.rint(difference / 12.0)
    shift = octaves * 12.0
    valid = (
        np.isfinite(source)
        & np.isfinite(target)
        & (octaves != 0)
        & (np.abs(difference - shift) <= tolerance)
    )
    return np.where(valid, shift, 0.0)


def correct_octaves(
    times: np.ndarray,
    midi: np.ndarray,
    *,
    notes: list[dict] | None = None,
    evidence_times: np.ndarray | None = None,
    evidence_midi: np.ndarray | None = None,
    evidence_confidence: np.ndarray | None = None,
    min_evidence_confidence: float = 0.50,
    octave_tolerance_st: float = 1.50,
    min_run_frames: int = 3,
    gross_error_radius_frames: int = 15,
    gross_error_min_confidence: float = 0.01,
    note_region_source_tolerance_st: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return corrected MIDI, a correction mask, and serializable statistics."""

    times = np.asarray(times, dtype=float)
    original = np.asarray(midi, dtype=float)
    corrected = original.copy()
    applied_shift = np.zeros(len(original), dtype=float)
    stats = CorrectionStats()

    # Propagate decoder-confirmed octave repairs across their note regions.
    for note in notes or []:
        source_note = note.get("octave_corrected_from")
        if source_note is None:
            continue
        target_note = float(note["midi"])
        expected_shift = target_note - float(source_note)
        if not np.isclose(abs(expected_shift), 12.0, atol=0.01):
            continue
        region = (times >= float(note["t0"])) & (times < float(note["t1"]))
        values = original[region & np.isfinite(original)]
        if not len(values) or abs(float(np.median(values)) - float(source_note)) > 2.0:
            continue
        # Decoder regions include smoothed attack/release frames. Shift only
        # values that are actually near the recorded wrong octave; applying
        # one whole-note offset to an already-correct edge creates a new spike.
        selection = (
            region
            & np.isfinite(original)
            & (np.abs(original - float(source_note)) <= note_region_source_tolerance_st)
            & (applied_shift == 0)
        )
        if np.any(selection):
            applied_shift[selection] = expected_shift
            stats.note_regions += 1

    # Align independent evidence to the production contour and require a
    # sustained exact-octave disagreement.  Isolated frames are ignored.
    if evidence_times is not None and evidence_midi is not None:
        evidence_times = np.asarray(evidence_times, dtype=float)
        evidence_midi = np.asarray(evidence_midi, dtype=float)
        confidence = (
            np.ones(len(evidence_midi), dtype=float)
            if evidence_confidence is None
            else np.asarray(evidence_confidence, dtype=float)
        )
        valid_evidence = np.isfinite(evidence_midi) & (confidence >= min_evidence_confidence)
        indices = np.searchsorted(evidence_times, times, side="left")
        indices = np.clip(indices, 0, len(evidence_times) - 1)
        previous = np.maximum(indices - 1, 0)
        use_previous = np.abs(evidence_times[previous] - times) < np.abs(evidence_times[indices] - times)
        indices[use_previous] = previous[use_previous]
        aligned = evidence_midi[indices]
        aligned_valid = valid_evidence[indices]
        if len(times) > 1:
            aligned_valid &= np.abs(evidence_times[indices] - times) <= 0.51 * float(np.median(np.diff(times)))
        shifts = _octave_shift(original, aligned, octave_tolerance_st)
        candidates = aligned_valid & (shifts != 0) & (applied_shift == 0)
        boundaries = np.flatnonzero(np.diff(np.r_[False, candidates, False]))
        for start, end in zip(boundaries[::2], boundaries[1::2]):
            split_points = np.flatnonzero(
                shifts[start + 1 : end] != shifts[start : end - 1]
            ) + start + 1
            for run_start, run_end in zip(
                np.r_[start, split_points], np.r_[split_points, end]
            ):
                if run_end - run_start < min_run_frames:
                    continue
                applied_shift[run_start:run_end] = shifts[run_start]
                stats.estimator_runs += 1

        # CREPE often enters/leaves an octave error through a short ramp. Its
        # edge frames are not exactly 12 st away, so exact-octave logic alone
        # leaves conspicuous spikes. Replace only gross-error edge frames near
        # an already-confirmed correction, with locally continuous secondary
        # evidence. Ordinary disagreements remain untouched.
        seeds = np.flatnonzero(applied_shift != 0)
        if len(seeds):
            for index in np.flatnonzero(applied_shift == 0):
                insertion = int(np.searchsorted(seeds, index))
                neighbors = seeds[max(0, insertion - 1) : insertion + 1]
                if not len(neighbors):
                    continue
                nearest = int(neighbors[np.argmin(np.abs(neighbors - index))])
                if abs(nearest - index) > gross_error_radius_frames:
                    continue
                if not (
                    aligned_valid[index] or confidence[indices[index]] >= gross_error_min_confidence
                ):
                    continue
                target = aligned[index]
                neighbor_target = aligned[nearest]
                difference = target - original[index]
                neighbor_direction = np.sign(applied_shift[nearest])
                if (
                    not np.isfinite(target)
                    or not np.isfinite(neighbor_target)
                    or abs(target - neighbor_target) > 2.5
                    or abs(difference) <= 4.0
                    or np.sign(difference) != neighbor_direction
                ):
                    continue
                applied_shift[index] = target - original[index]
                stats.gross_error_frames += 1

    mask = applied_shift != 0
    corrected[mask] += applied_shift[mask]
    stats.corrected_frames = int(np.sum(mask))
    hop = float(np.median(np.diff(times))) if len(times) > 1 else 0.0
    stats.corrected_seconds = round(stats.corrected_frames * hop, 3)
    return corrected, mask, asdict(stats)


def recover_secondary_voicing(
    times: np.ndarray,
    keep: np.ndarray,
    rms: np.ndarray,
    rms_floor: float,
    evidence_midi: np.ndarray,
    evidence_confidence: np.ndarray,
    *,
    min_confidence: float = 0.75,
    min_run_frames: int = 5,
) -> tuple[np.ndarray, int]:
    """Recover missing lead frames supported by a sustained second estimator."""

    recovered = np.asarray(keep, dtype=bool).copy()
    candidates = (
        np.isfinite(evidence_midi)
        & (np.asarray(evidence_confidence) >= min_confidence)
        & (np.asarray(rms) >= rms_floor)
    )
    boundaries = np.flatnonzero(np.diff(np.r_[False, candidates, False]))
    added = 0
    for start, end in zip(boundaries[::2], boundaries[1::2]):
        if end - start < min_run_frames:
            continue
        before = int(np.sum(recovered[start:end]))
        recovered[start:end] = True
        added += int(end - start - before)
    return recovered, int(added)


def repair_isolated_excursions(
    times: np.ndarray,
    midi: np.ndarray,
    keep: np.ndarray,
    *,
    median_window_frames: int = 31,
    minimum_deviation_st: float = 3.0,
    expansion_deviation_st: float = 1.0,
    maximum_region_frames: int = 30,
    return_tolerance_st: float = 2.5,
    edge_frames: int = 5,
    evidence_aware: bool = False,
    evidence_midi: np.ndarray | None = None,
    evidence_confidence: np.ndarray | None = None,
    raw_midi: np.ndarray | None = None,
    raw_confidence: np.ndarray | None = None,
    evidence_min_confidence: float = 0.60,
    raw_min_confidence: float = 0.50,
    support_tolerance_st: float = 1.5,
    minimum_supported_fraction: float = 0.50,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Remove short pitch excursions that leave and return to one local level.

    This is deliberately not a general smoother. Sustained changes, segment
    edges, vibrato, and any movement whose two sides disagree are preserved.
    """

    times = np.asarray(times, dtype=float)
    source = np.asarray(midi, dtype=float)
    active = np.asarray(keep, dtype=bool) & np.isfinite(source)
    repaired = source.copy()
    mask = np.zeros(len(source), dtype=bool)
    stats = ExcursionStats()
    evidence = (
        np.full(len(source), np.nan)
        if evidence_midi is None
        else np.asarray(evidence_midi, dtype=float)
    )
    evidence_conf = (
        np.zeros(len(source), dtype=float)
        if evidence_confidence is None
        else np.asarray(evidence_confidence, dtype=float)
    )
    raw = (
        np.full(len(source), np.nan)
        if raw_midi is None
        else np.asarray(raw_midi, dtype=float)
    )
    raw_conf = (
        np.zeros(len(source), dtype=float)
        if raw_confidence is None
        else np.asarray(raw_confidence, dtype=float)
    )
    segment_edges = np.flatnonzero(np.diff(np.r_[False, active, False]))
    window = max(3, int(median_window_frames) | 1)

    for segment_start, segment_end in zip(segment_edges[::2], segment_edges[1::2]):
        values = source[segment_start:segment_end]
        if len(values) < window:
            continue
        radius = window // 2
        padded = np.pad(values, (radius, radius), mode="edge")
        windows = np.lib.stride_tricks.sliding_window_view(padded, window)
        baseline = np.median(windows, axis=1)
        candidates = np.abs(values - baseline) >= minimum_deviation_st
        candidate_edges = np.flatnonzero(np.diff(np.r_[False, candidates, False]))

        for core_start, core_end in zip(candidate_edges[::2], candidate_edges[1::2]):
            start, end = int(core_start), int(core_end)
            while (
                start > 0
                and end - (start - 1) <= maximum_region_frames
                and abs(values[start - 1] - baseline[start - 1]) >= expansion_deviation_st
            ):
                start -= 1
            while (
                end < len(values)
                and end + 1 - start <= maximum_region_frames
                and abs(values[end] - baseline[end]) >= expansion_deviation_st
            ):
                end += 1
            if end - start > maximum_region_frames:
                continue
            left = values[max(0, start - edge_frames) : start]
            right = values[end : min(len(values), end + edge_frames)]
            if len(left) < 2 or len(right) < 2:
                continue
            left_center = float(np.median(left))
            right_center = float(np.median(right))
            if abs(left_center - right_center) > return_tolerance_st:
                continue
            replacement = np.linspace(left_center, right_center, end - start + 2)[1:-1]
            absolute_start = segment_start + start
            absolute_end = segment_start + end
            if evidence_aware:
                selection = slice(absolute_start, absolute_end)
                source_region = source[selection]
                secondary_support = (
                    np.isfinite(evidence[selection])
                    & (evidence_conf[selection] >= evidence_min_confidence)
                    & (np.abs(evidence[selection] - source_region) <= support_tolerance_st)
                )
                raw_support = (
                    np.isfinite(raw[selection])
                    & (raw_conf[selection] >= raw_min_confidence)
                    & (np.abs(raw[selection] - source_region) <= support_tolerance_st)
                )
                supported = secondary_support | raw_support
                if len(supported) and float(np.mean(supported)) >= minimum_supported_fraction:
                    stats.evidence_protected_frames += int(len(supported))
                    stats.evidence_protected_regions += 1
                    continue
            # Candidate groups can overlap after edge expansion.
            new_selection = ~mask[absolute_start:absolute_end]
            repaired[absolute_start:absolute_end] = replacement
            mask[absolute_start:absolute_end] = True
            stats.repaired_frames += int(np.sum(new_selection))
            stats.repaired_regions += 1

    hop = float(np.median(np.diff(times))) if len(times) > 1 else 0.0
    stats.repaired_seconds = round(stats.repaired_frames * hop, 3)
    return repaired, mask, asdict(stats)
