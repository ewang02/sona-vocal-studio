"""Shared features and safe inference for reviewed contour-region models.

The annotation queue labels short *regions*, not isolated frames.  Both
classifiers therefore operate on region summaries:

* ``breath`` decides whether a proposed region should be unvoiced.
* ``pitchSpike`` decides whether a voiced region contains a tracker error that
  should be repaired from independent pitch evidence.

This module is deliberately independent of scikit-learn.  Training serializes
standardized logistic models to JSON, while production inference needs only
NumPy.  A bundle must explicitly record ``productionEligible: true`` before it
can alter a contour.
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:  # Support package imports and ``python work/<script>.py``.
    from .breath_classifier import predict_breath_probabilities
    from .breath_features import FEATURE_NAMES as BREATH_FEATURE_NAMES
    from .breath_features import extract_breath_features, load_audio_16k
except ImportError:  # pragma: no cover - exercised by script entry points
    from breath_classifier import predict_breath_probabilities
    from breath_features import FEATURE_NAMES as BREATH_FEATURE_NAMES
    from breath_features import extract_breath_features, load_audio_16k


HOP_SECONDS = 0.01
MIN_REGION_SECONDS = 0.03
CONTEXT_SECONDS = 0.25
BASE_BREATH_THRESHOLD = 0.80

MODEL_FEATURE_NAMES = (
    "duration",
    "visible_fraction",
    "broad_confidence_median",
    "lead_confidence_median",
    "pyin_confidence_median",
    "pyin_support_fraction",
    "secondary_agreement_fraction",
    "log_lead_rms_median",
    "lead_rms_context_log_ratio",
    "final_pitch_span",
    "maximum_frame_jump",
    "final_pyin_difference_median",
    "final_lead_difference_median",
    "final_broad_difference_median",
    "context_pitch_deviation_median",
    "context_pitch_deviation_max",
    "base_breath_probability_median",
    "base_breath_probability_max",
    "base_breath_probability_fraction_high",
    "audio_log_rms_median",
    "audio_log_flatness_median",
    "audio_periodicity_median",
    "audio_zcr_median",
    "audio_flux_median",
)

_AUDIO_FEATURE_INDEX = {
    name: BREATH_FEATURE_NAMES.index(name)
    for name in ("log_rms", "log_flatness", "periodicity", "zcr", "flux")
}


def _as_float(values: Iterable[Any]) -> np.ndarray:
    return np.asarray(
        [
            float(value)
            if value is not None and math.isfinite(float(value))
            else math.nan
            for value in values
        ],
        dtype=float,
    )


def task_frame_arrays(payload: dict[str, Any]) -> dict[str, np.ndarray]:
    """Normalize one review-task JSON payload onto absolute seconds."""
    clip_start = float(payload["clipStart"])
    return {
        "time": _as_float(payload["time"]) + clip_start,
        "finalMidi": _as_float(payload["finalMidi"]),
        "broadMidi": _as_float(payload["broadMidi"]),
        "broadConfidence": _as_float(payload["broadConfidence"]),
        "leadMidi": _as_float(payload["leadMidi"]),
        "leadConfidence": _as_float(payload["leadConfidence"]),
        "pyinMidi": _as_float(payload["pyinMidi"]),
        "pyinConfidence": _as_float(payload["pyinConfidence"]),
        "leadRms": _as_float(payload["leadRms"]),
    }


def acoustic_evidence(
    audio_path: Path,
    *,
    absolute_start: float,
    base_breath_model: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """Extract frame-aligned audio evidence for one clip or full-song stem."""
    features = extract_breath_features(load_audio_16k(audio_path))
    probabilities = (
        predict_breath_probabilities(features, base_breath_model)
        if base_breath_model is not None
        else np.full(len(features), np.nan, dtype=np.float32)
    )
    return {
        "time": absolute_start + np.arange(len(features), dtype=float) * HOP_SECONDS,
        "features": features,
        "baseBreathProbability": probabilities,
    }


def _finite_percentile(values: np.ndarray, high: float, low: float = 50.0) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return math.nan
    if high == low:
        return float(np.percentile(finite, high))
    return float(np.percentile(finite, high) - np.percentile(finite, low))


def _finite_median(values: np.ndarray) -> float:
    return _finite_percentile(values, 50.0, 50.0)


def _finite_max(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.max(finite)) if len(finite) else math.nan


def _paired_median_difference(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    return (
        float(np.median(np.abs(left[valid] - right[valid])))
        if np.any(valid)
        else math.nan
    )


def _context_pitch(
    frame_data: dict[str, np.ndarray],
    start: float,
    end: float,
) -> float:
    times = frame_data["time"]
    final = frame_data["finalMidi"]
    context = (
        (times >= start - CONTEXT_SECONDS)
        & (times < end + CONTEXT_SECONDS)
        & ((times < start) | (times >= end))
        & np.isfinite(final)
    )
    return _finite_median(final[context])


def region_feature_vector(
    frame_data: dict[str, np.ndarray],
    start: float,
    end: float,
    acoustic: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Return the stable feature vector for one absolute-time region."""
    if end - start < MIN_REGION_SECONDS - 1e-9:
        raise ValueError(
            f"Region {start:.3f}-{end:.3f} is shorter than "
            f"{MIN_REGION_SECONDS:.2f} seconds"
        )
    times = frame_data["time"]
    selected = (times >= start) & (times < end)
    if not np.any(selected):
        raise ValueError(f"Region {start:.3f}-{end:.3f} contains no contour frames")

    final = frame_data["finalMidi"][selected]
    broad = frame_data["broadMidi"][selected]
    lead = frame_data["leadMidi"][selected]
    pyin = frame_data["pyinMidi"][selected]
    broad_confidence = frame_data["broadConfidence"][selected]
    lead_confidence = frame_data["leadConfidence"][selected]
    pyin_confidence = frame_data["pyinConfidence"][selected]
    lead_rms = frame_data["leadRms"][selected]
    visible = np.isfinite(final)
    supported = np.isfinite(pyin) & (pyin_confidence >= 0.35)
    agreement = (
        visible
        & supported
        & (np.abs(final - pyin) <= 1.0)
    )
    finite_final = final[np.isfinite(final)]
    jumps = np.abs(np.diff(finite_final)) if len(finite_final) >= 2 else np.array([])

    context_pitch = _context_pitch(frame_data, start, end)
    deviations = (
        np.abs(finite_final - context_pitch)
        if len(finite_final) and math.isfinite(context_pitch)
        else np.array([], dtype=float)
    )
    context_mask = (
        (times >= start - CONTEXT_SECONDS)
        & (times < end + CONTEXT_SECONDS)
        & ((times < start) | (times >= end))
    )
    inside_rms = _finite_median(lead_rms)
    outside_rms = _finite_median(frame_data["leadRms"][context_mask])
    rms_ratio = (
        math.log10((inside_rms + 1e-8) / (outside_rms + 1e-8))
        if math.isfinite(inside_rms) and math.isfinite(outside_rms)
        else math.nan
    )

    audio_values = np.full((0, len(BREATH_FEATURE_NAMES)), np.nan)
    base_probability = np.array([], dtype=float)
    if acoustic is not None:
        audio_mask = (acoustic["time"] >= start) & (acoustic["time"] < end)
        audio_values = acoustic["features"][audio_mask]
        base_probability = acoustic["baseBreathProbability"][audio_mask]

    def audio_median(name: str) -> float:
        if not len(audio_values):
            return math.nan
        return _finite_median(audio_values[:, _AUDIO_FEATURE_INDEX[name]])

    base_high = (
        float(np.mean(base_probability[np.isfinite(base_probability)] >= BASE_BREATH_THRESHOLD))
        if np.any(np.isfinite(base_probability))
        else math.nan
    )
    values = (
        end - start,
        float(np.mean(visible)),
        _finite_median(broad_confidence),
        _finite_median(lead_confidence),
        _finite_median(pyin_confidence),
        float(np.mean(supported)),
        float(np.mean(agreement)),
        math.log10(inside_rms + 1e-8) if math.isfinite(inside_rms) else math.nan,
        rms_ratio,
        _finite_percentile(final, 95.0, 5.0),
        _finite_max(jumps),
        _paired_median_difference(final, pyin),
        _paired_median_difference(final, lead),
        _paired_median_difference(final, broad),
        _finite_median(deviations),
        _finite_max(deviations),
        _finite_median(base_probability),
        _finite_max(base_probability),
        base_high,
        audio_median("log_rms"),
        audio_median("log_flatness"),
        audio_median("periodicity"),
        audio_median("zcr"),
        audio_median("flux"),
    )
    return np.asarray(values, dtype=np.float32)


def predict_model(features: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    """Run a serialized standardized logistic head."""
    if not model.get("trained", False):
        raise ValueError(f"Model head {model.get('name', '<unknown>')!r} is not trained")
    if tuple(model.get("featureNames", ())) != MODEL_FEATURE_NAMES:
        raise ValueError("Reviewed contour model feature order does not match the extractor")
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    if matrix.ndim != 2 or matrix.shape[1] != len(MODEL_FEATURE_NAMES):
        raise ValueError(
            f"Expected (regions, {len(MODEL_FEATURE_NAMES)}) features, got {matrix.shape}"
        )
    impute = np.asarray(model["impute"], dtype=float)
    matrix = np.where(np.isfinite(matrix), matrix, impute)
    mean = np.asarray(model["mean"], dtype=float)
    std = np.asarray(model["std"], dtype=float)
    coefficients = np.asarray(model["coefficients"], dtype=float)
    logits = ((matrix - mean) / std) @ coefficients + float(model["intercept"])
    logits = np.clip(logits, -40.0, 40.0)
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)


def load_model_bundle(
    path: Path,
    *,
    require_production_eligible: bool = True,
) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("type") != "reviewed-contour-region-classifiers":
        raise ValueError(f"Unsupported reviewed contour model type: {payload.get('type')!r}")
    if require_production_eligible and not payload.get("productionEligible", False):
        raise ValueError(
            "Reviewed contour model is audit-only; retrain and explicitly promote it "
            "after the readiness checks pass"
        )
    for name in ("breath", "pitchSpike"):
        head = payload.get("heads", {}).get(name)
        if not isinstance(head, dict):
            raise ValueError(f"Reviewed contour bundle is missing the {name!r} head")
        if head.get("trained") and tuple(head.get("featureNames", ())) != MODEL_FEATURE_NAMES:
            raise ValueError(f"Reviewed contour {name!r} feature order is incompatible")
    return payload


def _contiguous_regions(
    mask: np.ndarray,
    times: np.ndarray,
    hop: float,
    *,
    maximum_seconds: float,
) -> list[tuple[float, float]]:
    edges = np.flatnonzero(np.diff(np.r_[False, np.asarray(mask, dtype=bool), False]))
    result = []
    for start, end in zip(edges[::2], edges[1::2]):
        duration = (end - start) * hop
        if MIN_REGION_SECONDS <= duration <= maximum_seconds:
            result.append((float(times[start]), float(times[end - 1] + hop)))
    return result


def propose_breath_regions(
    times: np.ndarray,
    keep: np.ndarray,
    broad_confidence: np.ndarray,
    lead_confidence: np.ndarray,
    pyin_midi: np.ndarray,
    pyin_confidence: np.ndarray,
    *,
    acoustic_times: np.ndarray | None = None,
    base_breath_probability: np.ndarray | None = None,
) -> list[tuple[float, float]]:
    """Propose surviving low-support or acoustically breath-like regions."""
    hop = float(np.median(np.diff(times)))
    unsupported = (
        np.asarray(keep, dtype=bool)
        & (np.asarray(broad_confidence) < 0.55)
        & (np.asarray(lead_confidence) < 0.58)
        & (~np.isfinite(pyin_midi) | (np.asarray(pyin_confidence) < 0.35))
    )
    if acoustic_times is not None and base_breath_probability is not None:
        indexes = np.searchsorted(acoustic_times, times, side="left")
        indexes = np.clip(indexes, 0, len(acoustic_times) - 1)
        previous = np.maximum(0, indexes - 1)
        use_previous = np.abs(acoustic_times[previous] - times) < np.abs(
            acoustic_times[indexes] - times
        )
        indexes[use_previous] = previous[use_previous]
        # The legacy classifier is only a proposal source here. Its 0.50
        # threshold favors recall; the user-trained head makes the decision.
        acoustic_candidate = np.asarray(base_breath_probability)[indexes] >= 0.50
        unsupported |= np.asarray(keep, dtype=bool) & acoustic_candidate
    return _contiguous_regions(unsupported, times, hop, maximum_seconds=1.0)


def _rolling_nanmedian(values: np.ndarray, frames: int) -> np.ndarray:
    radius = frames // 2
    padded = np.pad(np.asarray(values, dtype=float), (radius, radius), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, frames)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(windows, axis=1)


def propose_pitch_spike_regions(
    times: np.ndarray,
    keep: np.ndarray,
    midi: np.ndarray,
    pyin_midi: np.ndarray,
    pyin_confidence: np.ndarray,
) -> list[tuple[float, float]]:
    """Propose brief gross deviations while leaving ordinary slides alone."""
    hop = float(np.median(np.diff(times)))
    active_midi = np.where(keep, midi, np.nan)
    local = _rolling_nanmedian(active_midi, 21)
    deviation = np.abs(active_midi - local)
    independent_alternative = (
        np.isfinite(pyin_midi)
        & (np.asarray(pyin_confidence) >= 0.45)
        & (np.abs(active_midi - pyin_midi) >= 2.5)
    )
    adjacent = np.asarray(keep[1:] & keep[:-1], dtype=bool)
    sharp = np.r_[False, adjacent & (np.abs(np.diff(midi)) >= 2.5)]
    sharp = np.convolve(sharp.astype(np.int8), np.ones(5, dtype=np.int8), mode="same") > 0
    candidate = np.asarray(keep, dtype=bool) & (
        ((deviation >= 2.5) & independent_alternative) | sharp
    )
    return _contiguous_regions(candidate, times, hop, maximum_seconds=0.45)


def _region_mask(times: np.ndarray, start: float, end: float) -> np.ndarray:
    return (times >= start) & (times < end)


def apply_reviewed_contour_models(
    *,
    times: np.ndarray,
    keep: np.ndarray,
    midi: np.ndarray,
    broad_midi: np.ndarray,
    broad_confidence: np.ndarray,
    lead_midi: np.ndarray,
    lead_confidence: np.ndarray,
    lead_rms: np.ndarray,
    pyin_midi: np.ndarray,
    pyin_confidence: np.ndarray,
    lead_audio_path: Path,
    model_path: Path,
    base_breath_model_path: Path | None = None,
    require_production_eligible: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply both reviewed-region heads to conservative automatic proposals."""
    bundle = load_model_bundle(
        model_path, require_production_eligible=require_production_eligible
    )
    base_model = (
        json.loads(base_breath_model_path.read_text())
        if base_breath_model_path is not None and base_breath_model_path.exists()
        else None
    )
    acoustic = acoustic_evidence(
        lead_audio_path, absolute_start=0.0, base_breath_model=base_model
    )
    frame_data = {
        "time": np.asarray(times, dtype=float),
        "finalMidi": np.asarray(midi, dtype=float),
        "broadMidi": np.asarray(broad_midi, dtype=float),
        "broadConfidence": np.asarray(broad_confidence, dtype=float),
        "leadMidi": np.asarray(lead_midi, dtype=float),
        "leadConfidence": np.asarray(lead_confidence, dtype=float),
        "pyinMidi": np.asarray(pyin_midi, dtype=float),
        "pyinConfidence": np.asarray(pyin_confidence, dtype=float),
        "leadRms": np.asarray(lead_rms, dtype=float),
    }
    result_keep = np.asarray(keep, dtype=bool).copy()
    result_midi = np.asarray(midi, dtype=float).copy()
    breath_removed = np.zeros(len(times), dtype=bool)
    pitch_repaired = np.zeros(len(times), dtype=bool)

    breath_head = bundle["heads"]["breath"]
    breath_regions = propose_breath_regions(
        times,
        result_keep,
        broad_confidence,
        lead_confidence,
        pyin_midi,
        pyin_confidence,
        acoustic_times=acoustic["time"],
        base_breath_probability=acoustic["baseBreathProbability"],
    )
    for start, end in breath_regions:
        features = region_feature_vector(frame_data, start, end, acoustic)
        probability = float(predict_model(features, breath_head)[0])
        if probability >= float(breath_head["threshold"]):
            selected = _region_mask(times, start, end) & result_keep
            result_keep[selected] = False
            breath_removed[selected] = True

    spike_head = bundle["heads"]["pitchSpike"]
    spike_regions = propose_pitch_spike_regions(
        times, result_keep, result_midi, pyin_midi, pyin_confidence
    )
    for start, end in spike_regions:
        frame_data["finalMidi"] = result_midi
        features = region_feature_vector(frame_data, start, end, acoustic)
        probability = float(predict_model(features, spike_head)[0])
        if probability < float(spike_head["threshold"]):
            continue
        selected = _region_mask(times, start, end) & result_keep
        supported = selected & np.isfinite(pyin_midi) & (pyin_confidence >= 0.45)
        repair = supported & (np.abs(result_midi - pyin_midi) >= 2.0)
        if np.any(repair):
            result_midi[repair] = pyin_midi[repair]
            pitch_repaired[repair] = True

    stats = {
        "modelSchemaVersion": bundle["schemaVersion"],
        "breathCandidates": len(breath_regions),
        "breathRemovedFrames": int(np.sum(breath_removed)),
        "pitchSpikeCandidates": len(spike_regions),
        "pitchRepairedFrames": int(np.sum(pitch_repaired)),
    }
    return result_keep, result_midi, breath_removed, pitch_repaired, stats
