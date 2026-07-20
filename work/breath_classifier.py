"""Portable inference helpers for the experimental MIR-1K breath model.

The production contour pipeline does not call this module yet.  The serialized
model carries an explicit ``productionEligible`` flag so an evaluation model
cannot accidentally become a destructive voicing veto.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

try:  # Support package imports and direct ``python work/<script>.py`` usage.
    from .breath_features import FEATURE_NAMES, extract_breath_features
except ImportError:  # pragma: no cover - exercised by script entry points
    from breath_features import FEATURE_NAMES, extract_breath_features


def smooth_probabilities(probabilities: np.ndarray, frames: int) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    if frames <= 1 or not len(values):
        return values.copy()
    from scipy.ndimage import median_filter

    return median_filter(values, size=int(frames), mode="nearest")


def load_breath_model(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("type") != "standardized-logistic":
        raise ValueError(f"Unsupported breath model type: {payload.get('type')!r}")
    if tuple(payload.get("featureNames", ())) != FEATURE_NAMES:
        raise ValueError("Breath model feature order does not match the extractor")
    return payload


def predict_breath_probabilities(
    features: np.ndarray,
    model: dict[str, Any],
    *,
    smooth: bool = True,
) -> np.ndarray:
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            f"Expected (frames, {len(FEATURE_NAMES)}) features, got {matrix.shape}"
        )
    mean = np.asarray(model["mean"], dtype=float)
    std = np.asarray(model["std"], dtype=float)
    coefficients = np.asarray(model["coefficients"], dtype=float)
    logits = ((matrix - mean) / std) @ coefficients + float(model["intercept"])
    # Clipping keeps exp stable without changing useful probabilities.
    logits = np.clip(logits, -40.0, 40.0)
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    if smooth:
        probabilities = smooth_probabilities(
            probabilities, int(model.get("smoothFrames", 1))
        )
    return probabilities.astype(np.float32)


def predict_audio_breath_probabilities(
    audio: np.ndarray, model: dict[str, Any]
) -> np.ndarray:
    return predict_breath_probabilities(extract_breath_features(audio), model)
