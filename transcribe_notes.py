#!/usr/bin/env python3
"""Reference-free singing contour -> canonical note transcription experiment.

This is deliberately separate from ``melody_clean.py``.  Smoothing answers
"where was the instantaneous F0?"; transcription must instead decide how many
musical notes were intended, their pitches, and their boundaries.

Inputs used during inference are audio-derived only:

* CREPE F0, confidence and RMS from ``contour_out/<song>_contour.csv``;
* the separated vocal stem, when available, for spectral onset evidence.

The hand-authored MIDIs/JSON files are never loaded here.  They are consumed
only by ``evaluate_transcription.py`` after an experiment has been written.

The decoder is a candidate-boundary semi-Markov model:

1. Use the validated hysteresis gate to identify usable vocal frames.
2. Propose boundaries from robust left/right pitch changes, vocal spectral
   onsets, energy attacks and voiced-region edges.
3. Estimate the song's tuning offset, then use global dynamic programming to
   select a parsimonious sequence of piecewise-constant semitone notes.
   Segment costs are confidence- and stability-weighted, so vibrato and glides
   do not become extra notes.
4. Split repeated notes at strong articulations, or at medium articulations
   corroborated by an independent pitch-confidence dip and recovery. Re-fit
   every child segment and attach a confidence score to every result.

These notes feed an optional diagnostic overlay, not the gameplay target.

Examples:
  .venv/bin/python transcribe_notes.py lemon
  .venv/bin/python transcribe_notes.py readymade --plot
  .venv/bin/python transcribe_notes.py --csv path/to/song_contour.csv \
      --vocals path/to/vocals.wav --out path/to/notes.json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, median_filter

from work.analyze_pitch_gate import browser_gate


@dataclass
class Note:
    t0: float
    t1: float
    midi: int
    confidence: float
    pitch_median: float
    pitch_mad: float
    voiced_coverage: float
    boundary_confidence: float
    estimator_agreement: float = 1.0
    octave_corrected_from: int | None = None


@dataclass
class DecoderConfig:
    bridge_gap_s: float = 0.12
    min_phrase_s: float = 0.09
    min_note_s: float = 0.07
    max_note_s: float = 4.0
    safeguard_s: float = 1.25
    pitch_context_s: float = 0.055
    pitch_candidate_st: float = 0.45
    onset_candidate: float = 0.32
    onset_repeat_split: float = 1.40
    # Medium onset splits failed the reference-free complexity/fit gate and
    # recovered no additional audited repeated boundaries. Keep the machinery
    # available for future phoneme cues, but ship strong-only by default.
    onset_repeat_candidate: float = 1.40
    repeat_confidence_recovery: float = 0.10
    corroborated_repeat_min_s: float = 0.09
    consonant_repeat_threshold: float = 1.15
    consonant_rms_attack_min: float = 0.12
    consonant_boundary_weight: float = 0.0
    secondary_confidence_min: float = 0.35
    secondary_agreement_sigma_st: float = 0.65
    secondary_weight: float = 0.30
    candidate_nms_s: float = 0.045
    note_penalty: float = 1.00
    max_boundary_reward: float = 0.72
    beat_weight: float = 0.12
    beat_sigma_s: float = 0.045
    pitch_sigma_st: float = 0.42
    edge_trim_s: float = 0.025


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    edges = np.flatnonzero(np.diff(np.r_[0, mask.view(np.int8), 0]))
    return list(zip(edges[::2], edges[1::2]))


def bridge_short_gaps(mask: np.ndarray, frames: int) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    if frames <= 0:
        return result
    for start, end in runs(~result):
        if start > 0 and end < len(result) and end - start <= frames:
            result[start:end] = True
    return result


def robust_scale(values: np.ndarray, percentile: float = 95.0) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return 1.0
    value = float(np.percentile(finite, percentile))
    return value if value > 1e-9 else 1.0


def resample_loaded_audio(
    loaded_audio: tuple[np.ndarray, int], target_sample_rate: int
) -> tuple[np.ndarray, int]:
    """Create a feature-specific rate from one native-rate decode."""
    audio, native_sample_rate = loaded_audio
    if native_sample_rate == target_sample_rate:
        return audio, native_sample_rate
    return (
        librosa.resample(
            audio,
            orig_sr=native_sample_rate,
            target_sr=target_sample_rate,
        ),
        target_sample_rate,
    )


def load_contour(path: Path) -> dict[str, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    required = ("time_s", "midi", "confidence", "rms", "voiced")
    missing = [name for name in required if name not in (data.dtype.names or ())]
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
    result = {name: np.asarray(data[name], dtype=float) for name in required}
    for optional in (
        "f0_hz_raw",
        "midi_raw",
        "confidence_raw",
        "f0_hz_pyin",
        "midi_pyin",
        "confidence_pyin",
    ):
        if optional in (data.dtype.names or ()):
            result[optional] = np.asarray(data[optional], dtype=float)
    return result


def attach_pitch_anchor(
    data: dict[str, np.ndarray], anchor: dict[str, np.ndarray]
) -> None:
    """Attach a filtered full-model contour as a pitch-center stabilizer.

    Raw tiny-model F0 preserves boundaries; the existing full-model filtered
    contour is usually more reliable for register and center pitch. Both are
    audio-derived. Interpolation keeps this usable when their final frame
    counts differ by one.
    """
    source_times = anchor["time_s"]
    target_times = data["time_s"]
    valid_pitch = np.isfinite(anchor["midi"])
    if np.any(valid_pitch):
        data["midi_anchor"] = np.interp(
            target_times,
            source_times[valid_pitch],
            anchor["midi"][valid_pitch],
            left=np.nan,
            right=np.nan,
        )
    else:
        data["midi_anchor"] = np.full(len(target_times), np.nan)
    data["confidence_anchor"] = np.interp(
        target_times,
        source_times,
        anchor["confidence"],
        left=0.0,
        right=0.0,
    )
    data["rms_anchor"] = np.interp(
        target_times,
        source_times,
        anchor["rms"],
        left=0.0,
        right=0.0,
    )


def vocal_onset_envelopes(
    vocal_path: Path | None,
    times: np.ndarray,
    include_consonant: bool = True,
    loaded_audio: tuple[np.ndarray, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if vocal_path is None or not vocal_path.exists():
        zeros = np.zeros(len(times), dtype=float)
        return zeros, zeros
    if loaded_audio is None:
        audio, sample_rate = librosa.load(vocal_path, sr=16_000, mono=True)
    else:
        audio, sample_rate = resample_loaded_audio(loaded_audio, 16_000)
    hop_length = 160  # 10 ms at 16 kHz, matching the contour by default.
    envelope = librosa.onset.onset_strength(
        y=audio,
        sr=sample_rate,
        hop_length=hop_length,
        n_fft=1024,
        aggregate=np.median,
    )
    onset_times = librosa.frames_to_time(
        np.arange(len(envelope)), sr=sample_rate, hop_length=hop_length
    )
    aligned = np.interp(times, onset_times, envelope, left=0.0, right=0.0)
    normalized_onset = np.clip(aligned / robust_scale(aligned), 0.0, 2.5)
    if not include_consonant:
        return normalized_onset, np.zeros(len(times), dtype=float)

    # A separate high-frequency flux emphasizes consonant re-articulation.
    # It is deliberately only corroborating evidence: accompaniment leakage
    # and sibilants can create transients that are not musical note onsets.
    high_mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=1024,
        hop_length=hop_length,
        win_length=1024,
        n_mels=24,
        fmin=1_500.0,
        fmax=7_800.0,
        power=1.0,
    )
    high = np.log1p(20.0 * high_mel)
    del high_mel
    flux = np.r_[0.0, np.mean(np.maximum(0.0, np.diff(high, axis=1)), axis=0)]
    flux_times = librosa.frames_to_time(
        np.arange(len(flux)), sr=sample_rate, hop_length=hop_length
    )
    aligned_flux = np.interp(times, flux_times, flux, left=0.0, right=0.0)
    return (
        normalized_onset,
        np.clip(aligned_flux / robust_scale(aligned_flux), 0.0, 2.5),
    )


def vocal_onset_envelope(
    vocal_path: Path | None,
    times: np.ndarray,
) -> np.ndarray:
    """Backward-compatible access to the broadband onset envelope."""
    return vocal_onset_envelopes(vocal_path, times)[0]


def metrical_prior(
    audio_path: Path | None,
    times: np.ndarray,
    config: DecoderConfig,
    loaded_audio: tuple[np.ndarray, int] | None = None,
) -> tuple[np.ndarray, float | None]:
    """Soft proximity to an audio-derived sixteenth-note grid.

    It never creates a boundary by itself. The prior only breaks ties between
    boundaries already supported by pitch/onset evidence, because expressive
    vocals and imperfect beat tracking make hard quantization unsafe.
    """
    if audio_path is None or not audio_path.exists():
        return np.zeros(len(times), dtype=float), None
    if loaded_audio is None:
        audio, sample_rate = librosa.load(audio_path, sr=22_050, mono=True)
    else:
        audio, sample_rate = resample_loaded_audio(loaded_audio, 22_050)
    hop_length = 512
    tempo, beat_frames = librosa.beat.beat_track(
        y=audio,
        sr=sample_rate,
        hop_length=hop_length,
        sparse=True,
    )
    beats = librosa.frames_to_time(beat_frames, sr=sample_rate, hop_length=hop_length)
    if len(beats) < 2:
        return np.zeros(len(times), dtype=float), None

    grid: list[float] = []
    for left, right in zip(beats, beats[1:]):
        interval = float(right - left)
        if 0.2 <= interval <= 1.5:
            grid.extend(float(left + interval * division / 4) for division in range(4))
    grid.append(float(beats[-1]))
    grid_array = np.asarray(sorted(grid), dtype=float)
    positions = np.searchsorted(grid_array, times)
    left_index = np.clip(positions - 1, 0, len(grid_array) - 1)
    right_index = np.clip(positions, 0, len(grid_array) - 1)
    distance = np.minimum(
        np.abs(times - grid_array[left_index]),
        np.abs(times - grid_array[right_index]),
    )
    prior = np.exp(-0.5 * (distance / config.beat_sigma_s) ** 2)
    tempo_value = float(np.asarray(tempo).reshape(-1)[0]) if np.size(tempo) else None
    return prior, tempo_value


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    midpoint = 0.5 * float(np.sum(sorted_weights))
    index = int(np.searchsorted(np.cumsum(sorted_weights), midpoint, side="left"))
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def pitch_candidate_losses(
    values: np.ndarray,
    weights: np.ndarray,
    pitch_candidates: np.ndarray,
    tuning_offset: float,
    pitch_sigma_st: float,
) -> np.ndarray:
    """Return capped weighted losses for every candidate without a Python loop."""
    residuals = values[np.newaxis, :] - (
        pitch_candidates[:, np.newaxis] + tuning_offset
    )
    return np.average(
        np.minimum((residuals / pitch_sigma_st) ** 2, 16.0),
        axis=1,
        weights=weights,
    )


def estimate_tuning_offset(
    midi: np.ndarray,
    confidence: np.ndarray,
    velocity: np.ndarray,
    keep: np.ndarray,
) -> float:
    """Estimate a song's stable-frame offset from the equal-tempered grid."""
    selection = (
        keep
        & np.isfinite(midi)
        & (confidence >= 0.5)
        & (velocity <= 8.0)
    )
    if np.sum(selection) < 50:
        return 0.0
    values = midi[selection]
    weights = confidence[selection] ** 2 / (1.0 + (velocity[selection] / 5.0) ** 2)
    offsets = np.linspace(-0.5, 0.5, 501)
    losses = np.asarray(
        [
            np.average(
                np.minimum(((values - offset) - np.round(values - offset)) ** 2, 0.16),
                weights=weights,
            )
            for offset in offsets
        ]
    )
    return float(offsets[int(np.argmin(losses))])


def boundary_features(
    data: dict[str, np.ndarray],
    keep: np.ndarray,
    onset: np.ndarray,
    config: DecoderConfig,
    metrical: np.ndarray | None = None,
    consonant_onset: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    times = data["time_s"]
    midi = data.get("midi_raw", data["midi"])
    rms = data["rms"]
    hop = float(np.median(np.diff(times)))
    valid = np.flatnonzero(keep & np.isfinite(midi))
    if not len(valid):
        zeros = np.zeros(len(times), dtype=float)
        return {"smooth": zeros, "velocity": zeros, "pitch_change": zeros,
                "rms_attack": zeros, "consonant_onset": zeros,
                "estimator_agreement": np.ones(len(times), dtype=float),
                "secondary_available": np.zeros(len(times), dtype=bool),
                "tuning_offset": 0.0, "boundary": zeros}

    filled = np.interp(np.arange(len(midi)), valid, midi[valid])
    smoothing_s = 0.03 if "midi_raw" in data else 0.05
    smooth_frames = max(3, int(round(smoothing_s / hop)) | 1)
    smooth = median_filter(filled, size=smooth_frames, mode="nearest")

    context = max(2, int(round(config.pitch_context_s / hop)))
    before = median_filter(
        np.r_[np.repeat(smooth[0], context), smooth[:-context]],
        size=2 * context + 1,
        mode="nearest",
    )
    after = median_filter(
        np.r_[smooth[context:], np.repeat(smooth[-1], context)],
        size=2 * context + 1,
        mode="nearest",
    )
    pitch_change = np.abs(after - before)
    velocity = np.abs(np.gradient(smooth, hop))
    pitch_confidence = data.get("confidence_raw", data["confidence"])
    if "midi_anchor" in data:
        anchor = data["midi_anchor"]
        anchor_valid = np.flatnonzero(np.isfinite(anchor))
        anchor_filled = np.interp(np.arange(len(anchor)), anchor_valid, anchor[anchor_valid])
        anchor_velocity = np.abs(np.gradient(anchor_filled, hop))
        tuning_offset = estimate_tuning_offset(
            anchor_filled, data["confidence_anchor"], anchor_velocity, keep
        )
    else:
        tuning_offset = estimate_tuning_offset(
            smooth, pitch_confidence, velocity, keep
        )

    estimator_agreement = np.ones(len(times), dtype=float)
    secondary_available = np.zeros(len(times), dtype=bool)
    if "midi_pyin" in data and "confidence_pyin" in data:
        secondary = data["midi_pyin"]
        secondary_confidence = data["confidence_pyin"]
        secondary_available = (
            np.isfinite(secondary)
            & (secondary_confidence >= config.secondary_confidence_min)
            & keep
        )
        difference = np.abs(secondary[secondary_available] - smooth[secondary_available])
        estimator_agreement[secondary_available] = np.exp(
            -0.5 * (difference / config.secondary_agreement_sigma_st) ** 2
        )

    log_rms = np.log(np.maximum(rms, 1e-8))
    rms_attack = np.maximum(
        0.0,
        gaussian_filter1d(log_rms, 2) - gaussian_filter1d(log_rms, 10),
    )
    rms_attack = np.clip(rms_attack / robust_scale(rms_attack), 0.0, 2.5)

    starts = keep & ~np.r_[False, keep[:-1]]
    stops = keep & ~np.r_[keep[1:], False]
    pitch_score = np.clip(
        (pitch_change - config.pitch_candidate_st * 0.6) / 2.0, 0.0, 1.0
    )
    onset_score = np.clip((onset - 0.18) / 1.1, 0.0, 1.0)
    energy_score = np.clip((rms_attack - 0.12) / 0.9, 0.0, 1.0)
    if consonant_onset is None:
        consonant_onset = np.zeros(len(times), dtype=float)
    consonant_score = np.clip((np.asarray(consonant_onset) - 0.25) / 1.25, 0.0, 1.0)
    acoustic_boundary = np.maximum.reduce(
        [
            0.70 * pitch_score
            + 0.45 * onset_score
            + 0.15 * energy_score
            + config.consonant_boundary_weight * consonant_score,
            starts.astype(float),
            stops.astype(float),
        ]
    )
    if metrical is None:
        metrical = np.zeros(len(times), dtype=float)
    # Meter is allowed to strengthen an acoustic candidate, never to invent
    # one in an otherwise featureless held note.
    beat_support = config.beat_weight * np.asarray(metrical) * (acoustic_boundary > 0.12)
    boundary = acoustic_boundary + beat_support
    return {
        "smooth": smooth,
        "velocity": velocity,
        "pitch_change": pitch_change,
        "rms_attack": rms_attack,
        "consonant_onset": np.asarray(consonant_onset),
        "estimator_agreement": estimator_agreement,
        "secondary_available": secondary_available,
        "metrical": np.asarray(metrical),
        "tuning_offset": tuning_offset,
        "boundary": np.clip(boundary, 0.0, 1.5),
    }


def candidate_boundaries(
    start: int,
    end: int,
    features: dict[str, np.ndarray],
    onset: np.ndarray,
    hop: float,
    config: DecoderConfig,
) -> list[int]:
    pitch_change = features["pitch_change"]
    boundary = features["boundary"]
    radius = max(1, int(round(config.candidate_nms_s / hop)))
    size = radius * 2 + 1
    pitch_peaks = pitch_change >= maximum_filter1d(pitch_change, size=size, mode="nearest") - 1e-9
    onset_peaks = onset >= maximum_filter1d(onset, size=size, mode="nearest") - 1e-9

    selected = (
        (pitch_peaks & (pitch_change >= config.pitch_candidate_st))
        | (onset_peaks & (onset >= config.onset_candidate))
    )
    indexes = list(np.flatnonzero(selected[start + 1 : end - 1]) + start + 1)

    # A pitch detector can stay perfectly flat for a very long note.  Sparse
    # safeguards keep the graph connected without themselves rewarding a split.
    safeguard = max(1, int(round(config.safeguard_s / hop)))
    indexes.extend(range(start + safeguard, end, safeguard))
    indexes.extend((start, end))

    # Collapse near-duplicate candidates, keeping the strongest musical cue.
    indexes = sorted(set(indexes))
    merged: list[int] = []
    for index in indexes:
        if not merged or index - merged[-1] > radius:
            merged.append(index)
            continue
        previous = merged[-1]
        previous_strength = boundary[previous]
        current_strength = boundary[index]
        # Always retain exact phrase endpoints.
        if previous == start or index == end:
            if index == end:
                merged.append(index)
            continue
        if current_strength > previous_strength:
            merged[-1] = index
    if merged[0] != start:
        merged.insert(0, start)
    if merged[-1] != end:
        merged.append(end)
    return merged


def segment_summary(
    left: int,
    right: int,
    data: dict[str, np.ndarray],
    keep: np.ndarray,
    features: dict[str, np.ndarray],
    hop: float,
    config: DecoderConfig,
) -> tuple[float, int, float, float, float, float] | None:
    trim = min(
        int(round(config.edge_trim_s / hop)),
        max(0, (right - left - 2) // 4),
    )
    core_left = left + trim
    core_right = right - trim
    selection = np.arange(core_left, core_right)
    usable = keep[selection] & np.isfinite(data["midi"][selection])
    indexes = selection[usable]
    if len(indexes) < max(3, int(round(0.025 / hop))):
        return None

    pitch_confidence = data.get("confidence_raw", data["confidence"])
    confidence = np.clip(pitch_confidence[indexes], 0.0, 1.0)
    stability = 1.0 / (1.0 + features["velocity"][indexes] / 12.0)
    secondary_available = features["secondary_available"][indexes]
    estimator_agreement = features["estimator_agreement"][indexes]
    # Use the local robust observation rather than the legacy 90 ms mean. This
    # preserves short-note evidence while rejecting isolated raw-CREPE spikes.
    values = features["smooth"][indexes].copy()
    if "midi_anchor" in data:
        anchor = data["midi_anchor"][indexes]
        anchor_confidence = data["confidence_anchor"][indexes]
        anchor_available = (
            np.isfinite(anchor)
            & (anchor_confidence >= 0.25)
        )
        # The tiny raw contour is a boundary detector. The existing filtered
        # full-model contour is the pitch-center estimator: this restores its
        # stronger register accuracy while raw transitions still determine
        # where notes may begin and end. pYIN handles the rare shared CREPE
        # octave island afterward.
        anchor_weight = 1.0
        values[anchor_available] = (
            (1.0 - anchor_weight) * values[anchor_available]
            + anchor_weight * anchor[anchor_available]
        )
        confidence[anchor_available] = np.maximum(
            confidence[anchor_available], anchor_confidence[anchor_available]
        )
    weights = (0.35 + 0.65 * confidence) * (0.25 + 0.75 * stability)
    secondary_factor = np.ones(len(indexes), dtype=float)
    secondary_factor[secondary_available] = (
        1.0
        - config.secondary_weight
        + 2.0 * config.secondary_weight * estimator_agreement[secondary_available]
    )
    weights *= secondary_factor
    center = weighted_median(values, weights)
    tuning_offset = float(features.get("tuning_offset", 0.0))
    candidate_low = int(np.floor(np.percentile(values, 5) - tuning_offset)) - 1
    candidate_high = int(np.ceil(np.percentile(values, 95) - tuning_offset)) + 1
    pitch_candidates = np.arange(candidate_low, candidate_high + 1)
    # Evaluate every integer pitch in one broadcasted operation.  This is
    # mathematically identical to averaging one candidate at a time, but this
    # function sits inside the O(boundaries²) DP loop, so avoiding thousands
    # of Python-level np.average calls saves meaningful decoder time.
    candidate_losses = pitch_candidate_losses(
        values,
        weights,
        pitch_candidates,
        tuning_offset,
        config.pitch_sigma_st,
    )
    pitch = int(pitch_candidates[int(np.argmin(candidate_losses))])
    residual = np.abs(values - (pitch + tuning_offset))
    # Capped quadratic loss: octave mistakes are bad, but one bad CREPE frame
    # must not force the global decoder to invent several notes.
    loss = np.minimum((residual / config.pitch_sigma_st) ** 2, 16.0)
    fit_cost = float(np.sum(weights * loss) / np.sum(weights)) * (right - left) * hop
    coverage = float(np.mean(keep[left:right]))
    pitch_mad = float(np.median(np.abs(values - np.median(values))))
    # Prefer well-supported regions, but allow brief consonant gaps inside a
    # note instead of splitting the phrase around every unvoiced phoneme.
    fit_cost += max(0.0, 0.55 - coverage) * (right - left) * hop * 2.0
    agreement = (
        float(np.mean(estimator_agreement[secondary_available]))
        if np.any(secondary_available)
        else 1.0
    )
    return fit_cost, pitch, center, pitch_mad, coverage, agreement


def decode_phrase(
    start: int,
    end: int,
    data: dict[str, np.ndarray],
    keep: np.ndarray,
    onset: np.ndarray,
    features: dict[str, np.ndarray],
    config: DecoderConfig,
) -> list[Note]:
    times = data["time_s"]
    hop = float(np.median(np.diff(times)))
    candidates = candidate_boundaries(start, end, features, onset, hop, config)
    count = len(candidates)
    costs = np.full(count, np.inf)
    previous = np.full(count, -1, dtype=int)
    summaries: dict[
        tuple[int, int], tuple[float, int, float, float, float, float]
    ] = {}
    costs[0] = 0.0

    min_frames = max(1, int(round(config.min_note_s / hop)))
    max_frames = max(min_frames + 1, int(round(config.max_note_s / hop)))
    for right_pos in range(1, count):
        right = candidates[right_pos]
        for left_pos in range(right_pos - 1, -1, -1):
            left = candidates[left_pos]
            length = right - left
            if length > max_frames:
                break
            if length < min_frames or not np.isfinite(costs[left_pos]):
                continue
            summary = segment_summary(left, right, data, keep, features, hop, config)
            if summary is None:
                continue
            summaries[(left_pos, right_pos)] = summary
            reward = 0.0
            if left_pos > 0:
                reward = min(
                    config.max_boundary_reward,
                    float(features["boundary"][left]) * config.max_boundary_reward,
                )
            candidate_cost = costs[left_pos] + summary[0] + config.note_penalty - reward
            if candidate_cost < costs[right_pos]:
                costs[right_pos] = candidate_cost
                previous[right_pos] = left_pos

    if previous[-1] < 0:
        return []
    edges: list[tuple[int, int]] = []
    right_pos = count - 1
    while right_pos > 0:
        left_pos = int(previous[right_pos])
        if left_pos < 0:
            return []
        edges.append((left_pos, right_pos))
        right_pos = left_pos
    edges.reverse()

    notes: list[Note] = []
    for left_pos, right_pos in edges:
        left = candidates[left_pos]
        right = candidates[right_pos]
        fit_cost, pitch, center, pitch_mad, coverage, agreement = summaries[
            (left_pos, right_pos)
        ]
        boundary_confidence = 1.0 if left == start else min(1.0, float(features["boundary"][left]))
        support = math.exp(-min(5.0, fit_cost / max((right - left) * hop, 0.05)) * 0.18)
        confidence = float(
            np.clip(
                support * (0.55 + 0.45 * coverage) * (0.8 + 0.2 * agreement),
                0.0,
                1.0,
            )
        )
        notes.append(
            Note(
                t0=float(times[left]),
                t1=float(times[min(right, len(times) - 1)] + (hop if right >= len(times) else 0.0)),
                midi=pitch,
                confidence=confidence,
                pitch_median=center,
                pitch_mad=pitch_mad,
                voiced_coverage=coverage,
                boundary_confidence=boundary_confidence,
                estimator_agreement=agreement,
            )
        )

    return split_repeated_notes(notes, onset, times, config, data, keep, features)


def split_repeated_notes(
    notes: list[Note],
    onset: np.ndarray,
    times: np.ndarray,
    config: DecoderConfig,
    data: dict[str, np.ndarray],
    keep: np.ndarray,
    features: dict[str, np.ndarray],
) -> list[Note]:
    if not notes:
        return notes
    hop = float(np.median(np.diff(times)))
    min_frames = max(1, int(round(config.min_note_s / hop)))
    corroborated_min_frames = max(
        min_frames, int(round(config.corroborated_repeat_min_s / hop))
    )
    pre_outer = max(1, int(round(0.08 / hop)))
    pre_inner = max(1, int(round(0.03 / hop)))
    post_inner = max(1, int(round(0.02 / hop)))
    post_outer = max(post_inner + 1, int(round(0.08 / hop)))

    def has_centered_confidence_valley(
        values: np.ndarray, candidate: int, left: int, right: int
    ) -> bool:
        pre = values[max(left, candidate - pre_outer) : candidate - pre_inner]
        valley = values[
            max(left, candidate - pre_inner) : min(right, candidate + post_inner)
        ]
        post = values[
            min(right, candidate + post_inner) : min(right, candidate + post_outer)
        ]
        if not len(pre) or not len(valley) or not len(post):
            return False
        floor = float(np.percentile(valley, 20))
        return (
            min(float(np.median(pre)), float(np.median(post))) - floor
            >= config.repeat_confidence_recovery
            and float(np.median(post)) - floor
            >= config.repeat_confidence_recovery
        )
    result: list[Note] = []
    peak_mask = onset >= maximum_filter1d(onset, size=7, mode="nearest") - 1e-9
    for note in notes:
        left = int(np.searchsorted(times, note.t0, side="left"))
        right = int(np.searchsorted(times, note.t1, side="left"))
        if right - left < 2 * min_frames:
            result.append(note)
            continue
        candidates = np.flatnonzero(
            peak_mask[left + min_frames : right - min_frames]
            & (
                onset[left + min_frames : right - min_frames]
                >= config.onset_repeat_candidate
            )
        ) + left + min_frames
        cuts: list[int] = []
        last = left
        for candidate in candidates:
            strong = onset[candidate] >= config.onset_repeat_split
            required_frames = min_frames if strong else corroborated_min_frames
            if candidate - last < required_frames or right - candidate < required_frames:
                continue
            if not strong:
                pitch_confidence = data.get("confidence_raw", data["confidence"])
                recovered = has_centered_confidence_valley(
                    pitch_confidence, candidate, left, right
                )
                if "confidence_pyin" in data:
                    recovered = recovered or has_centered_confidence_valley(
                        data["confidence_pyin"], candidate, left, right
                    )
                consonant_supported = (
                    features["consonant_onset"][candidate]
                    >= config.consonant_repeat_threshold
                    or features["rms_attack"][candidate]
                    >= config.consonant_rms_attack_min
                )
                if not (recovered and consonant_supported):
                    continue
            if candidate - last >= required_frames and right - candidate >= required_frames:
                cuts.append(int(candidate))
                last = int(candidate)
        if not cuts:
            result.append(note)
            continue
        boundaries = [left, *cuts, right]
        for a, b in zip(boundaries, boundaries[1:]):
            summary = segment_summary(a, b, data, keep, features, hop, config)
            if summary is None:
                clone = Note(**asdict(note))
                clone.t0 = float(times[a])
                clone.t1 = float(
                    times[min(b, len(times) - 1)] + (hop if b >= len(times) else 0.0)
                )
                clone.boundary_confidence = (
                    min(1.0, float(onset[a])) if a != left else note.boundary_confidence
                )
                result.append(clone)
                continue

            fit_cost, pitch, center, pitch_mad, coverage, agreement = summary
            duration = max((b - a) * hop, 0.05)
            support = math.exp(-min(5.0, fit_cost / duration) * 0.18)
            confidence = float(
                np.clip(
                    support
                    * (0.55 + 0.45 * coverage)
                    * (0.8 + 0.2 * agreement),
                    0.0,
                    1.0,
                )
            )
            result.append(
                Note(
                    t0=float(times[a]),
                    t1=float(
                        times[min(b, len(times) - 1)] + (hop if b >= len(times) else 0.0)
                    ),
                    midi=pitch,
                    confidence=confidence,
                    pitch_median=center,
                    pitch_mad=pitch_mad,
                    voiced_coverage=coverage,
                    boundary_confidence=(
                        min(1.0, float(onset[a]))
                        if a != left
                        else note.boundary_confidence
                    ),
                    estimator_agreement=agreement,
                )
            )
    return result


def merge_adjacent(notes: list[Note], max_gap_s: float = 0.035) -> list[Note]:
    merged: list[Note] = []
    for note in notes:
        previous = merged[-1] if merged else None
        if (
            previous
            and previous.midi == note.midi
            and note.t0 - previous.t1 <= max_gap_s
            and note.boundary_confidence < 0.72
        ):
            left_duration = max(previous.t1 - previous.t0, 0.0)
            right_duration = max(note.t1 - note.t0, 0.0)
            total = max(left_duration + right_duration, 1e-6)
            left_weight = left_duration / total
            right_weight = right_duration / total
            previous.t1 = note.t1
            previous.confidence = left_weight * previous.confidence + right_weight * note.confidence
            previous.pitch_median = left_weight * previous.pitch_median + right_weight * note.pitch_median
            previous.pitch_mad = max(previous.pitch_mad, note.pitch_mad)
            previous.voiced_coverage = left_weight * previous.voiced_coverage + right_weight * note.voiced_coverage
            previous.estimator_agreement = (
                left_weight * previous.estimator_agreement
                + right_weight * note.estimator_agreement
            )
            if previous.octave_corrected_from is None:
                previous.octave_corrected_from = note.octave_corrected_from
        else:
            merged.append(note)
    return merged


def rescue_isolated_octaves(
    notes: list[Note],
    data: dict[str, np.ndarray],
    config: DecoderConfig,
) -> list[Note]:
    """Correct only pYIN-confirmed, jump-and-return octave islands."""
    if not notes or "midi_pyin" not in data or "confidence_pyin" not in data:
        return notes
    times = data["time_s"]
    secondary = data["midi_pyin"]
    secondary_confidence = data["confidence_pyin"]
    for index, note in enumerate(notes):
        note_frames = (times >= note.t0) & (times < note.t1)
        selection = (
            note_frames
            & np.isfinite(secondary)
            & (secondary_confidence >= max(0.55, config.secondary_confidence_min))
        )
        if np.sum(selection) < 3:
            continue
        secondary_pitch = float(np.median(secondary[selection]))
        candidates = (note.midi - 12, note.midi + 12)
        alternative = min(candidates, key=lambda pitch: abs(pitch - secondary_pitch))
        if (
            abs(secondary_pitch - alternative) > 0.55
            or abs(secondary_pitch - note.midi) < 6.0
        ):
            continue

        selected_confidence = float(np.median(secondary_confidence[selection]))
        secondary_coverage = float(np.sum(selection) / max(np.sum(note_frames), 1))
        independent_support = (
            secondary_coverage >= 0.15 and selected_confidence >= 0.65
        )
        neighbor_support = False
        if 0 < index < len(notes) - 1:
            previous, following = notes[index - 1], notes[index + 1]
            contiguous = (
                note.t0 - previous.t1 <= 0.15
                and following.t0 - note.t1 <= 0.15
            )
            neighbor_support = (
                contiguous
                and abs(previous.midi - following.midi) <= 4
                and abs(alternative - 0.5 * (previous.midi + following.midi)) <= 4
                and abs(note.midi - previous.midi) >= 9
                and abs(note.midi - following.midi) >= 9
            )
        if not (independent_support or neighbor_support):
            continue

        original = note.midi
        note.midi = int(alternative)
        note.pitch_median += note.midi - original
        note.estimator_agreement = float(
            np.exp(
                -0.5
                * (
                    (secondary_pitch - note.midi)
                    / config.secondary_agreement_sigma_st
                )
                ** 2
            )
        )
        note.octave_corrected_from = original
    return notes


def transcribe(
    data: dict[str, np.ndarray],
    vocal_path: Path | None,
    config: DecoderConfig,
    audio_path: Path | None = None,
) -> tuple[list[Note], dict[str, np.ndarray], np.ndarray, float, float | None]:
    times = data["time_s"]
    hop = float(np.median(np.diff(times)))
    if "midi_anchor" in data:
        gate_data = {
            "midi": data["midi_anchor"],
            "confidence": data["confidence_anchor"],
            "rms": data["rms_anchor"],
        }
    else:
        gate_data = data
    keep, rms_floor, _ = browser_gate(gate_data)
    phrase_mask = bridge_short_gaps(keep, int(round(config.bridge_gap_s / hop)))
    use_consonant = (
        config.consonant_boundary_weight > 0
        or config.onset_repeat_candidate < config.onset_repeat_split
    )
    shared_audio = None
    if (
        vocal_path is not None
        and audio_path is not None
        and vocal_path.exists()
        and audio_path.exists()
        and vocal_path.resolve() == audio_path.resolve()
    ):
        # Decode a shared source once at its native rate.  librosa.load's
        # default target-rate path performs the same resampling internally;
        # making it explicit lets both feature extractors reuse the decode.
        shared_audio = librosa.load(vocal_path, sr=None, mono=True)
    onset, consonant_onset = vocal_onset_envelopes(
        vocal_path,
        times,
        include_consonant=use_consonant,
        loaded_audio=shared_audio,
    )
    metrical, tempo = metrical_prior(
        audio_path, times, config, loaded_audio=shared_audio
    )
    features = boundary_features(
        data, keep, onset, config, metrical, consonant_onset
    )

    notes: list[Note] = []
    for start, end in runs(phrase_mask):
        if (end - start) * hop < config.min_phrase_s:
            continue
        notes.extend(decode_phrase(start, end, data, keep, onset, features, config))
    notes = rescue_isolated_octaves(notes, data, config)
    return merge_adjacent(notes), features, onset, float(rms_floor), tempo


def plot_diagnostics(
    output: Path,
    data: dict[str, np.ndarray],
    notes: list[Note],
    features: dict[str, np.ndarray],
    onset: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = data["time_s"]
    fig, axes = plt.subplots(3, 1, figsize=(18, 10), sharex=True)
    axes[0].plot(times, data["midi"], color="#999999", linewidth=0.5, alpha=0.5)
    for note in notes:
        color = "#1f77b4" if note.confidence >= 0.55 else "#d62728"
        axes[0].plot([note.t0, note.t1], [note.midi, note.midi], color=color, linewidth=2.2)
    axes[0].set_ylabel("MIDI")
    axes[0].set_title("Audio-only canonical-note decode (red = low confidence)")
    axes[1].plot(times, features["pitch_change"], label="pitch change")
    axes[1].plot(times, onset, label="vocal onset", alpha=0.8)
    axes[1].legend(loc="upper right")
    axes[1].set_ylabel("cue strength")
    axes[2].plot(times, features["boundary"], color="#9467bd")
    axes[2].set_ylabel("boundary")
    axes[2].set_xlabel("seconds")
    fig.tight_layout()
    fig.savefig(output, dpi=130)
    plt.close(fig)


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path | None, Path | None, Path]:
    if args.song:
        csv_path = Path("contour_out") / f"{args.song}_contour.csv"
        default_vocals = Path("contour_out/demucs/htdemucs") / args.song / "vocals.wav"
        vocal_path = Path(args.vocals) if args.vocals else default_vocals
        audio_path = None if args.no_beat else (Path(args.audio) if args.audio else Path("public/audio") / f"{args.song}.mp3")
        output = Path(args.out) if args.out else Path("contour_out") / f"{args.song}_notes_auto.json"
    else:
        if not args.csv:
            raise SystemExit("provide a song id or --csv")
        csv_path = Path(args.csv)
        vocal_path = Path(args.vocals) if args.vocals else None
        audio_path = None if args.no_beat else (Path(args.audio) if args.audio else None)
        output = Path(args.out) if args.out else csv_path.with_name(f"{csv_path.stem}_notes.json")
    return csv_path, vocal_path, audio_path, output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("song", nargs="?", help="song id using contour_out/<song> assets")
    parser.add_argument("--csv", help="explicit contour CSV path")
    parser.add_argument(
        "--pitch-anchor",
        help="optional filtered full-model contour CSV for stable pitch centers",
    )
    parser.add_argument("--vocals", help="optional separated-vocal WAV for onset evidence")
    parser.add_argument("--audio", help="optional full mix for a soft metrical prior")
    parser.add_argument("--no-beat", action="store_true", help="disable the soft metrical prior")
    parser.add_argument(
        "--consonant-cue",
        action="store_true",
        help="experimental: enable the rejected high-frequency boundary cue",
    )
    parser.add_argument("--out", help="output JSON path")
    parser.add_argument("--plot", action="store_true", help="also write a diagnostic PNG")
    parser.add_argument(
        "--note-penalty",
        type=float,
        help="override the per-note complexity penalty for an ablation",
    )
    args = parser.parse_args()

    csv_path, vocal_path, audio_path, output = resolve_paths(args)
    data = load_contour(csv_path)
    pitch_anchor_path = Path(args.pitch_anchor) if args.pitch_anchor else None
    if pitch_anchor_path is not None:
        attach_pitch_anchor(data, load_contour(pitch_anchor_path))
    config = DecoderConfig()
    if args.consonant_cue:
        config.consonant_boundary_weight = 0.20
    if args.note_penalty is not None:
        config.note_penalty = args.note_penalty
    notes, features, onset, rms_floor, tempo = transcribe(data, vocal_path, config, audio_path)
    payload = {
        "schema": 2,
        "source": {
            "contour": str(csv_path),
            "vocals": str(vocal_path) if vocal_path and vocal_path.exists() else None,
            "audio": str(audio_path) if audio_path and audio_path.exists() else None,
            "pitch_anchor": (
                str(pitch_anchor_path)
                if pitch_anchor_path is not None and pitch_anchor_path.exists()
                else None
            ),
        },
        "method": "reference-free-ensemble-semi-markov-final-v1",
        "config": asdict(config),
        "rms_floor": rms_floor,
        "estimated_tempo_bpm": round(tempo, 3) if tempo is not None else None,
        "estimated_tuning_offset_st": round(
            float(features.get("tuning_offset", 0.0)), 4
        ),
        "secondary_f0_frame_coverage": round(
            float(np.mean(features.get("secondary_available", np.zeros(1)))), 4
        ),
        "octave_corrections": sum(
            note.octave_corrected_from is not None for note in notes
        ),
        "notes": [
            {
                **asdict(note),
                "t0": round(note.t0, 4),
                "t1": round(note.t1, 4),
                "confidence": round(note.confidence, 4),
                "pitch_median": round(note.pitch_median, 4),
                "pitch_mad": round(note.pitch_mad, 4),
                "voiced_coverage": round(note.voiced_coverage, 4),
                "boundary_confidence": round(note.boundary_confidence, 4),
            }
            for note in notes
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(output.stem + ".tmp" + output.suffix)
    temporary_output.write_text(json.dumps(payload, indent=2) + "\n")
    temporary_output.replace(output)
    duration = float(data["time_s"][-1] - data["time_s"][0])
    low_confidence = sum(note.confidence < 0.55 for note in notes)
    print(
        f"{csv_path.stem}: {len(notes)} notes over {duration:.1f}s; "
        f"{low_confidence} low-confidence; wrote {output}"
    )
    if args.plot:
        plot_path = output.with_suffix(".png")
        plot_diagnostics(plot_path, data, notes, features, onset)
        print(f"wrote {plot_path}")


if __name__ == "__main__":
    main()
