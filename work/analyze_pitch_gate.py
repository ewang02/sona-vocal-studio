#!/usr/bin/env python3
"""Compare the prototype's browser gate with its stored and manual references."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

try:
    from contour_pipeline_config import BASELINE_CONFIG, ContourPipelineConfig
except ModuleNotFoundError:
    from work.contour_pipeline_config import BASELINE_CONFIG, ContourPipelineConfig


AUTO_CONF = 0.5
AUTO_RMS_OTSU_MULT = 1.25
HYST_MAX_JUMP_ST = 0.8


def load_csv(path: Path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        key: np.array([float(row[key]) for row in rows], dtype=float)
        for key in rows[0]
    }


def otsu(values: np.ndarray, bins: int = 200) -> float:
    lo = float(np.min(values))
    hi = float(np.max(values))
    if not hi > lo:
        return lo
    hist, edges = np.histogram(values, bins=bins, range=(lo, hi))
    centers = (edges[:-1] + edges[1:]) * 0.5
    total = float(np.sum(hist))
    sum_total = float(np.sum(hist * centers))
    weight_left = 0.0
    sum_left = 0.0
    best_variance = -1.0
    best_edge = lo
    for index in range(bins - 1):
        weight_left += float(hist[index])
        sum_left += float(hist[index] * centers[index])
        weight_right = total - weight_left
        if weight_left == 0 or weight_right == 0:
            continue
        delta = sum_left / weight_left - (sum_total - sum_left) / weight_right
        variance = weight_left * weight_right * delta * delta
        if variance > best_variance:
            best_variance = variance
            best_edge = float(edges[index + 1])
    return best_edge


def otsu_diagnostics(values: np.ndarray, threshold: float) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    left = values < threshold
    right = ~left
    left_fraction = float(np.mean(left))
    right_fraction = float(np.mean(right))
    variance = float(np.var(values))
    if not np.any(left) or not np.any(right) or variance <= 0:
        separability = 0.0
    else:
        mean = float(np.mean(values))
        between = (
            left_fraction * (float(np.mean(values[left])) - mean) ** 2
            + right_fraction * (float(np.mean(values[right])) - mean) ** 2
        )
        separability = min(1.0, max(0.0, between / variance))
    hist, edges = np.histogram(values, bins=64)
    smoothed = np.convolve(
        hist.astype(float), np.array([1, 2, 3, 2, 1]) / 9.0, mode="same"
    )
    threshold_bin = int(
        np.clip(np.searchsorted(edges, threshold) - 1, 1, len(hist) - 2)
    )
    left_peak = int(np.argmax(smoothed[: threshold_bin + 1]))
    right_peak = int(np.argmax(smoothed[threshold_bin:]) + threshold_bin)
    if right_peak <= left_peak + 1:
        valley_ratio = 1.0
    else:
        valley = float(np.min(smoothed[left_peak : right_peak + 1]))
        peak_floor = min(float(smoothed[left_peak]), float(smoothed[right_peak]))
        valley_ratio = valley / peak_floor if peak_floor > 0 else 1.0
    return {
        "leftFraction": left_fraction,
        "rightFraction": right_fraction,
        "minimumClassFraction": min(left_fraction, right_fraction),
        "separability": separability,
        "valleyRatio": float(np.clip(valley_ratio, 0.0, 1.0)),
    }


def choose_rms_floor(
    rms: np.ndarray,
    config: ContourPipelineConfig = BASELINE_CONFIG,
) -> tuple[float, dict]:
    nonzero = np.asarray(rms, dtype=float)
    nonzero = nonzero[np.isfinite(nonzero) & (nonzero > 0)]
    if not len(nonzero):
        return 0.0, {
            "method": "empty",
            "otsuFloor": 0.0,
            "fallbackFloor": 0.0,
            "minimumClassFraction": 0.0,
            "separability": 0.0,
            "valleyRatio": 1.0,
        }
    log_rms = np.log10(nonzero)
    threshold = otsu(log_rms)
    diagnostics = otsu_diagnostics(log_rms, threshold)
    otsu_floor = 10 ** threshold * config.rms_otsu_multiplier
    fallback_floor = (
        float(np.percentile(nonzero, config.rms_fallback_percentile))
        * config.rms_fallback_multiplier
    )
    use_fallback = config.rms_mode == "guarded_otsu" and (
        diagnostics["minimumClassFraction"] < config.rms_min_class_fraction
        or diagnostics["separability"] < config.rms_min_separability
        or diagnostics["valleyRatio"] > config.rms_max_valley_ratio
    )
    floor = fallback_floor if use_fallback else otsu_floor
    return float(floor), {
        "method": "percentile_fallback" if use_fallback else "otsu",
        "otsuFloor": float(otsu_floor),
        "fallbackFloor": float(fallback_floor),
        "minimumClassFraction": diagnostics["minimumClassFraction"],
        "separability": diagnostics["separability"],
        "valleyRatio": diagnostics["valleyRatio"],
    }


def browser_gate(
    data: dict[str, np.ndarray],
    config: ContourPipelineConfig = BASELINE_CONFIG,
    *,
    secondary_midi: np.ndarray | None = None,
    secondary_confidence: np.ndarray | None = None,
    return_diagnostics: bool = False,
):
    midi = data["midi"]
    confidence = data["confidence"]
    rms = data["rms"]
    nonzero = rms[np.isfinite(rms) & (rms > 0)]
    p20 = float(np.percentile(nonzero, 20)) if len(nonzero) else 0.0
    rms_floor, floor_diagnostics = choose_rms_floor(rms, config)
    rms_scale = rms_floor / p20 if p20 > 0 else 0.0
    keep = np.zeros(len(midi), dtype=bool)
    segment_start = -1
    rejected_segments = 0
    rejected_frames = 0
    retained_segments = 0
    time_values = data.get("time_s")
    hop = (
        float(np.median(np.diff(time_values)))
        if time_values is not None and len(time_values) > 1
        else 0.01
    )

    def flush(end: int):
        nonlocal segment_start, rejected_segments, rejected_frames, retained_segments
        if segment_start >= 0:
            frame_count = end - segment_start
            duration_required_seeds = max(
                1,
                int(math.ceil(frame_count * hop * config.minimum_seed_rate_hz)),
            )
            required_seeds = duration_required_seeds
            seed_count = int(
                np.sum(
                    confidence[segment_start:end] >= config.seed_confidence
                )
            )
            secondary_supported = False
            if secondary_midi is not None and secondary_confidence is not None:
                secondary = (
                    np.isfinite(secondary_midi[segment_start:end])
                    & (
                        secondary_confidence[segment_start:end]
                        >= config.secondary_seed_confidence
                    )
                )
                secondary_supported = (
                    frame_count >= 5
                    and float(np.mean(secondary)) >= config.secondary_seed_fraction
                )
            if seed_count >= required_seeds or secondary_supported:
                keep[segment_start:end] = True
                retained_segments += 1
            else:
                rejected_segments += 1
                rejected_frames += frame_count
        segment_start = -1

    for index in range(len(midi)):
        growable = math.isfinite(midi[index]) and rms[index] >= rms_floor
        if not growable:
            flush(index)
            continue
        if (
            segment_start >= 0
            and abs(midi[index] - midi[index - 1]) > config.max_jump_semitones
        ):
            flush(index)
        if segment_start < 0:
            segment_start = index
    flush(len(midi))
    diagnostics = {
        **floor_diagnostics,
        "seedConfidence": config.seed_confidence,
        "retainedSegments": retained_segments,
        "rejectedSegments": rejected_segments,
        "rejectedFrames": rejected_frames,
    }
    if config.minimum_seed_rate_hz > 0:
        diagnostics["minimumSeedRateHz"] = config.minimum_seed_rate_hz
    if return_diagnostics:
        return keep, rms_floor, rms_scale, diagnostics
    return keep, rms_floor, rms_scale


def prune_unconfirmed_low_energy_voicing(
    times: np.ndarray,
    keep: np.ndarray,
    rms: np.ndarray,
    secondary_midi: np.ndarray,
    secondary_confidence: np.ndarray,
    config: ContourPipelineConfig = BASELINE_CONFIG,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | float]]:
    """Remove low-energy connected regions unsupported by pYIN.

    Separator residue can look perfectly periodic to CREPE for more than a
    second, so primary confidence and a short-duration cap do not identify it.
    The veto is intentionally conjunctive: a region must be in the quietest
    configured fraction of accepted frames *and* lack sustained pYIN support.
    """
    result = np.asarray(keep, dtype=bool).copy()
    removed = np.zeros(len(result), dtype=bool)
    percentile = config.unconfirmed_rms_percentile
    if percentile <= 0 or not len(result) or not np.any(result):
        return result, removed, {
            "rmsPercentile": percentile,
            "removedSegments": 0,
            "removedFrames": 0,
        }
    rms = np.asarray(rms, dtype=float)
    secondary_midi = np.asarray(secondary_midi, dtype=float)
    secondary_confidence = np.asarray(secondary_confidence, dtype=float)
    rms_ceiling = float(np.percentile(rms[result], percentile))
    edges = np.flatnonzero(np.diff(np.r_[False, result, False]))
    removed_segments = 0
    for start, end in zip(edges[::2], edges[1::2]):
        secondary = (
            np.isfinite(secondary_midi[start:end])
            & (
                secondary_confidence[start:end]
                >= config.unconfirmed_secondary_confidence
            )
        )
        secondary_fraction = float(np.mean(secondary))
        if (
            float(np.median(rms[start:end])) <= rms_ceiling
            and secondary_fraction <= config.unconfirmed_max_secondary_fraction
        ):
            result[start:end] = False
            removed[start:end] = True
            removed_segments += 1
    return result, removed, {
        "rmsPercentile": percentile,
        "rmsCeiling": rms_ceiling,
        "maximumSecondaryFraction": config.unconfirmed_max_secondary_fraction,
        "secondaryConfidence": config.unconfirmed_secondary_confidence,
        "removedSegments": removed_segments,
        "removedFrames": int(np.sum(removed)),
    }


def prune_weak_edge_excursions(
    times: np.ndarray,
    keep: np.ndarray,
    midi: np.ndarray,
    rms: np.ndarray,
    secondary_midi: np.ndarray,
    secondary_confidence: np.ndarray,
    config: ContourPipelineConfig = BASELINE_CONFIG,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | float]]:
    """Trim low-energy pitch dropouts attached to phrase boundaries."""
    result = np.asarray(keep, dtype=bool).copy()
    removed = np.zeros(len(result), dtype=bool)
    max_seconds = config.edge_excursion_max_seconds
    if max_seconds <= 0 or not len(result):
        return result, removed, {
            "maxSeconds": max_seconds,
            "removedEdges": 0,
            "removedFrames": 0,
        }

    times = np.asarray(times, dtype=float)
    midi = np.asarray(midi, dtype=float)
    rms = np.asarray(rms, dtype=float)
    secondary_midi = np.asarray(secondary_midi, dtype=float)
    secondary_confidence = np.asarray(secondary_confidence, dtype=float)
    hop = float(np.median(np.diff(times))) if len(times) > 1 else 0.01
    max_frames = max(1, int(round(max_seconds / hop)))
    context_frames = max(
        5, int(round(config.edge_excursion_context_seconds / hop))
    )
    minimum_frames = max(
        2, int(round(config.edge_excursion_min_seconds / hop))
    )

    def find_suffix_start(
        pitch: np.ndarray,
        energy: np.ndarray,
        secondary_pitch: np.ndarray,
        secondary_score: np.ndarray,
    ) -> int | None:
        length = len(pitch)
        first_split = max(context_frames, length - max_frames)
        for split in range(first_split, length - minimum_frames + 1):
            interior_start = max(0, split - context_frames)
            interior_pitch = pitch[interior_start:split]
            edge_pitch = pitch[split:length]
            if len(interior_pitch) < 5 or len(edge_pitch) < minimum_frames:
                continue
            interior_center = float(np.nanmedian(interior_pitch))
            edge_center = float(np.nanmedian(edge_pitch))
            if (
                not np.isfinite(interior_center)
                or not np.isfinite(edge_center)
                or abs(edge_center - interior_center)
                < config.edge_excursion_min_pitch_difference
            ):
                continue
            interior_rms = float(np.nanmedian(energy[interior_start:split]))
            edge_rms = float(np.nanmedian(energy[split:length]))
            if (
                interior_rms <= 0
                or edge_rms / interior_rms > config.edge_excursion_max_rms_ratio
            ):
                continue
            edge_secondary_support = (
                np.isfinite(secondary_pitch[split:length])
                & (
                    secondary_score[split:length]
                    >= config.edge_excursion_secondary_confidence
                )
                & (
                    np.abs(secondary_pitch[split:length] - edge_center)
                    <= config.edge_excursion_secondary_tolerance
                )
            )
            if (
                float(np.mean(edge_secondary_support))
                > config.edge_excursion_max_secondary_fraction
            ):
                continue
            evidence_slice = slice(interior_start, length)
            interior_secondary_support = (
                np.isfinite(secondary_pitch[evidence_slice])
                & (
                    secondary_score[evidence_slice]
                    >= config.edge_excursion_secondary_confidence
                )
                & (
                    np.abs(
                        secondary_pitch[evidence_slice] - interior_center
                    )
                    <= config.edge_excursion_secondary_tolerance
                )
                & (
                    np.abs(secondary_pitch[evidence_slice] - edge_center)
                    >= config.edge_excursion_min_pitch_difference
                )
            )
            if not np.any(interior_secondary_support):
                continue
            departure = np.flatnonzero(
                np.abs(pitch[split:length] - interior_center)
                >= config.edge_excursion_expansion_difference
            )
            return split + (int(departure[0]) if len(departure) else 0)
        return None

    removed_edges = 0
    segment_edges = np.flatnonzero(np.diff(np.r_[False, result, False]))
    for start, end in zip(segment_edges[::2], segment_edges[1::2]):
        suffix_start = find_suffix_start(
            midi[start:end],
            rms[start:end],
            secondary_midi[start:end],
            secondary_confidence[start:end],
        )
        if suffix_start is not None:
            absolute_start = start + suffix_start
            result[absolute_start:end] = False
            removed[absolute_start:end] = True
            removed_edges += 1

        prefix_end_reversed = find_suffix_start(
            midi[start:end][::-1],
            rms[start:end][::-1],
            secondary_midi[start:end][::-1],
            secondary_confidence[start:end][::-1],
        )
        if prefix_end_reversed is not None:
            absolute_end = end - prefix_end_reversed
            result[start:absolute_end] = False
            removed[start:absolute_end] = True
            removed_edges += 1

    return result, removed, {
        "maxSeconds": max_seconds,
        "minimumPitchDifference": config.edge_excursion_min_pitch_difference,
        "maximumRmsRatio": config.edge_excursion_max_rms_ratio,
        "removedEdges": removed_edges,
        "removedFrames": int(np.sum(removed)),
    }


def prune_unsettled_edges(
    times: np.ndarray,
    keep: np.ndarray,
    midi: np.ndarray,
    secondary_midi: np.ndarray,
    secondary_confidence: np.ndarray,
    raw_midi: np.ndarray,
    raw_confidence: np.ndarray,
    config: ContourPipelineConfig = BASELINE_CONFIG,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | float]]:
    """Trim boundary frames that spike or ramp before the pitch settles.

    Interior leave-and-return spikes are repaired by the excursion stage, but
    it requires context on both sides, so an artifact in the first or last
    frames of a segment is structurally unreachable; the weak-edge rule only
    fires on quiet edges at least five semitones away. This rule closes that
    gap: it removes a short unsettled prefix/suffix that deviates from the
    settled boundary level, unless the deviating frames look like a real
    sung approach — a flat sustained pickup note (stability guard) or pitch
    corroborated by pYIN or the unsmoothed CREPE output (evidence guard) is
    preserved. Only pYIN is an independent estimator.
    """
    result = np.asarray(keep, dtype=bool).copy()
    removed = np.zeros(len(result), dtype=bool)
    max_seconds = config.unsettled_edge_max_seconds
    stats: dict[str, int | float] = {
        "maxSeconds": max_seconds,
        "trimmedEdges": 0,
        "removedFrames": 0,
        "stabilityProtectedEdges": 0,
        "evidenceProtectedEdges": 0,
    }
    if max_seconds <= 0 or not len(result):
        return result, removed, stats

    times = np.asarray(times, dtype=float)
    midi = np.asarray(midi, dtype=float)
    secondary_midi = np.asarray(secondary_midi, dtype=float)
    secondary_confidence = np.asarray(secondary_confidence, dtype=float)
    raw_midi = np.asarray(raw_midi, dtype=float)
    raw_confidence = np.asarray(raw_confidence, dtype=float)
    hop = float(np.median(np.diff(times))) if len(times) > 1 else 0.01
    max_frames = max(1, int(round(max_seconds / hop)))
    context_frames = max(
        5, int(round(config.unsettled_edge_context_seconds / hop))
    )
    short_prefix_frames = max(
        1, int(round(config.unsettled_edge_short_prefix_seconds / hop))
    )
    settle_debounce = 3

    def find_settle(pitch: np.ndarray, evidence: np.ndarray, evidence_score,
                    raw: np.ndarray, raw_score: np.ndarray) -> tuple[int | None, str]:
        """Return (frames to trim from the front, reason) for one direction."""
        length = len(pitch)
        window = min(max_frames, length - context_frames)
        if window <= 0:
            return None, "short"
        reference = float(np.nanmedian(pitch[window : window + context_frames]))
        if not np.isfinite(reference):
            return None, "no-reference"
        deviations = np.abs(pitch - reference)
        with np.errstate(invalid="ignore"):
            settled = deviations <= config.unsettled_edge_settle_tolerance_st
        settle = None
        for index in range(0, window + 1):
            run = settled[index : index + settle_debounce]
            if len(run) and bool(np.all(run)):
                settle = index
                break
        if settle is None:
            settle = window
        if settle == 0:
            return None, "settled"
        prefix = pitch[:settle]
        finite = prefix[np.isfinite(prefix)]
        if not len(finite):
            return settle, "trim"
        if np.nanmax(np.abs(finite - reference)) < config.unsettled_edge_min_deviation_st:
            return None, "small"
        span = float(np.max(finite) - np.min(finite))
        if (
            span < config.unsettled_edge_min_prefix_span_st
            and settle > short_prefix_frames
        ):
            return None, "stable-note"
        supported = (
            np.isfinite(evidence[:settle])
            & (evidence_score[:settle] >= config.unsettled_edge_secondary_confidence)
            & (np.abs(evidence[:settle] - prefix) <= config.unsettled_edge_support_tolerance_st)
        ) | (
            np.isfinite(raw[:settle])
            & (raw_score[:settle] >= config.unsettled_edge_raw_confidence)
            & (np.abs(raw[:settle] - prefix) <= config.unsettled_edge_support_tolerance_st)
        )
        if float(np.mean(supported)) >= config.unsettled_edge_max_support_fraction:
            return None, "evidence"
        return settle, "trim"

    segment_edges = np.flatnonzero(np.diff(np.r_[False, result, False]))
    for start, end in zip(segment_edges[::2], segment_edges[1::2]):
        directions = (False, True) if config.unsettled_edge_trim_offsets else (False,)
        for reverse in directions:
            view = slice(start, end)
            pitch = midi[view][::-1] if reverse else midi[view]
            evidence = secondary_midi[view][::-1] if reverse else secondary_midi[view]
            evidence_score = (
                secondary_confidence[view][::-1] if reverse else secondary_confidence[view]
            )
            raw = raw_midi[view][::-1] if reverse else raw_midi[view]
            raw_score = raw_confidence[view][::-1] if reverse else raw_confidence[view]
            trim, reason = find_settle(pitch, evidence, evidence_score, raw, raw_score)
            if reason == "stable-note":
                stats["stabilityProtectedEdges"] += 1
            elif reason == "evidence":
                stats["evidenceProtectedEdges"] += 1
            if trim is None:
                continue
            if reverse:
                result[end - trim : end] = False
                removed[end - trim : end] = True
            else:
                result[start : start + trim] = False
                removed[start : start + trim] = True
            stats["trimmedEdges"] += 1

    stats["removedFrames"] = int(np.sum(removed))
    return result, removed, stats


def prune_unstable_unconfirmed_segments(
    times: np.ndarray,
    keep: np.ndarray,
    midi: np.ndarray,
    primary_confidence: np.ndarray,
    secondary_midi: np.ndarray,
    secondary_confidence: np.ndarray,
    config: ContourPipelineConfig = BASELINE_CONFIG,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | float]]:
    """Remove short, pitch-unstable regions unsupported by both estimators."""
    result = np.asarray(keep, dtype=bool).copy()
    removed = np.zeros(len(result), dtype=bool)
    max_seconds = config.unstable_segment_max_seconds
    if max_seconds <= 0 or not len(result):
        return result, removed, {
            "maxSeconds": max_seconds,
            "removedSegments": 0,
            "removedFrames": 0,
        }

    times = np.asarray(times, dtype=float)
    midi = np.asarray(midi, dtype=float)
    primary_confidence = np.asarray(primary_confidence, dtype=float)
    secondary_midi = np.asarray(secondary_midi, dtype=float)
    secondary_confidence = np.asarray(secondary_confidence, dtype=float)
    hop = float(np.median(np.diff(times))) if len(times) > 1 else 0.01
    segment_edges = np.flatnonzero(np.diff(np.r_[False, result, False]))
    removed_segments = 0
    for start, end in zip(segment_edges[::2], segment_edges[1::2]):
        if (end - start) * hop > max_seconds:
            continue
        values = midi[start:end]
        finite_values = values[np.isfinite(values)]
        if len(finite_values) < 2:
            continue
        pitch_span = float(
            np.percentile(finite_values, 95) - np.percentile(finite_values, 5)
        )
        if pitch_span < config.unstable_segment_min_pitch_span:
            continue
        primary_seed_fraction = float(
            np.mean(primary_confidence[start:end] >= config.seed_confidence)
        )
        if (
            primary_seed_fraction
            > config.unstable_segment_max_primary_seed_fraction
        ):
            continue
        secondary_support = (
            np.isfinite(secondary_midi[start:end])
            & (
                secondary_confidence[start:end]
                >= config.unstable_segment_secondary_confidence
            )
        )
        if (
            float(np.mean(secondary_support))
            > config.unstable_segment_max_secondary_fraction
        ):
            continue
        result[start:end] = False
        removed[start:end] = True
        removed_segments += 1

    return result, removed, {
        "maxSeconds": max_seconds,
        "minimumPitchSpan": config.unstable_segment_min_pitch_span,
        "maximumPrimarySeedFraction": (
            config.unstable_segment_max_primary_seed_fraction
        ),
        "maximumSecondaryFraction": config.unstable_segment_max_secondary_fraction,
        "removedSegments": removed_segments,
        "removedFrames": int(np.sum(removed)),
    }


def score_mask(data, notes, keep):
    times = data["time_s"]
    midi = data["midi"]
    in_note = np.zeros(len(times), dtype=bool)
    note_core = np.zeros(len(times), dtype=bool)
    target = np.full(len(times), np.nan)
    per_note_coverage = []
    for note in notes:
        selection = (times >= note["t0"]) & (times <= note["t1"])
        in_note |= selection
        duration = note["t1"] - note["t0"]
        trim = min(0.08, duration * 0.2)
        core = (times >= note["t0"] + trim) & (times <= note["t1"] - trim)
        note_core |= core
        target[core] = note["midi"]
        if np.any(selection):
            per_note_coverage.append(float(np.mean(keep[selection])))

    first = min(note["t0"] for note in notes)
    last = max(note["t1"] for note in notes)
    reference_span = (times >= first) & (times <= last)
    # Count only sustained reference rests, trimming note-transition edges.
    # Tiny gaps between discrete MIDI bars often contain intentional scoops or
    # consonants, so treating them as silence exaggerates false-voicing.
    rest_candidates = reference_span & ~in_note
    rests = np.zeros(len(times), dtype=bool)
    hop = float(times[1] - times[0])
    indexes = np.flatnonzero(np.diff(np.r_[0, rest_candidates.view(np.int8), 0]))
    for start, end in zip(indexes[::2], indexes[1::2]):
        if (end - start) * hop < 0.25:
            continue
        edge = int(round(0.08 / hop))
        if end - start > 2 * edge:
            rests[start + edge : end - edge] = True

    kept_notes = keep & note_core
    errors = np.abs(midi[kept_notes] - target[kept_notes])
    return {
        "frame_coverage": float(np.mean(keep[in_note])) * 100,
        "mean_note_coverage": float(np.mean(per_note_coverage)) * 100,
        "notes_ge_80": float(np.mean(np.array(per_note_coverage) >= 0.8)) * 100,
        "rest_false": float(np.mean(keep[rests])) * 100,
        "mae": float(np.mean(errors)),
        "within_1": float(np.mean(errors <= 1.0)) * 100,
        "beyond_2": float(np.mean(errors > 2.0)) * 100,
        "beyond_octave": float(np.mean(errors > 6.0)) * 100,
    }


def report(song_root: Path, song: str):
    data = load_csv(song_root / "contour_out" / f"{song}_contour.csv")
    notes = json.loads(
        (song_root / "contour_out" / f"{song}_ref.json").read_text()
    )["notes"]
    browser, floor, scale = browser_gate(data)
    stored = data["voiced"] > 0.5
    flat = (
        (data["confidence"] >= AUTO_CONF)
        & (data["rms"] >= floor)
        & np.isfinite(data["midi"])
    )
    print(f"\n{song}")
    print(f"auto conf={AUTO_CONF:.2f} rms floor={floor:.6f} ({scale:.2f}x p20)")
    for label, mask in (("flat", flat), ("browser", browser), ("stored", stored)):
        values = score_mask(data, notes, mask)
        formatted = " ".join(f"{key}={value:.2f}" for key, value in values.items())
        print(f"{label:8s} {formatted}")


if __name__ == "__main__":
    root = Path(sys.argv[1])
    if len(sys.argv) < 3:
        raise SystemExit("usage: analyze_pitch_gate.py <root> <song> [song ...]")
    for song_name in sys.argv[2:]:
        report(root, song_name)
