"""Robust, frame-aligned pitch filters used by extraction and ablations."""

from __future__ import annotations

import numpy as np


def _odd_frames(milliseconds: float, hop_ms: float) -> int:
    frames = max(3, int(round(milliseconds / hop_ms)))
    return frames if frames % 2 else frames + 1


def _windows(values: np.ndarray, size: int) -> np.ndarray:
    radius = size // 2
    padded = np.pad(np.asarray(values, dtype=float), (radius, radius), mode="edge")
    return np.lib.stride_tricks.sliding_window_view(padded, size)


def median_filter(values: np.ndarray, size: int) -> np.ndarray:
    return np.nanmedian(_windows(values, size), axis=1)


def mean_filter(values: np.ndarray, size: int) -> np.ndarray:
    return np.nanmean(_windows(values, size), axis=1)


def weighted_median_filter(
    values: np.ndarray, weights: np.ndarray, size: int
) -> np.ndarray:
    value_windows = _windows(values, size)
    weight_windows = _windows(np.clip(weights, 0.0, 1.0) ** 2, size)
    result = np.empty(len(values), dtype=float)
    for index, (window, window_weights) in enumerate(
        zip(value_windows, weight_windows)
    ):
        valid = np.isfinite(window) & np.isfinite(window_weights)
        if not np.any(valid):
            result[index] = np.nan
            continue
        ordered = np.argsort(window[valid])
        ordered_values = window[valid][ordered]
        ordered_weights = window_weights[valid][ordered]
        total = float(np.sum(ordered_weights))
        if total <= 0:
            result[index] = float(np.median(ordered_values))
            continue
        result[index] = float(
            ordered_values[np.searchsorted(np.cumsum(ordered_weights), total / 2.0)]
        )
    return result


def hz_to_midi(frequency: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return 69.0 + 12.0 * np.log2(np.asarray(frequency, dtype=float) / 440.0)


def midi_to_hz(midi: np.ndarray) -> np.ndarray:
    return 440.0 * 2.0 ** ((np.asarray(midi, dtype=float) - 69.0) / 12.0)


def filter_f0(
    raw_f0: np.ndarray,
    raw_confidence: np.ndarray,
    hop_ms: float,
    method: str,
) -> np.ndarray:
    raw_f0 = np.asarray(raw_f0, dtype=float)
    if method == "mean_hz_90":
        return mean_filter(raw_f0, _odd_frames(90.0, hop_ms))

    midi = hz_to_midi(raw_f0)
    if method == "median_midi_90":
        filtered = median_filter(midi, _odd_frames(90.0, hop_ms))
    elif method == "median_midi_70":
        filtered = median_filter(midi, _odd_frames(70.0, hop_ms))
    elif method == "median_midi_70_mean_midi_30":
        filtered = median_filter(midi, _odd_frames(70.0, hop_ms))
        filtered = mean_filter(filtered, _odd_frames(30.0, hop_ms))
    elif method == "weighted_median_midi_70":
        filtered = weighted_median_filter(
            midi, raw_confidence, _odd_frames(70.0, hop_ms)
        )
    else:
        raise ValueError(f"Unknown pitch filter: {method}")
    return midi_to_hz(filtered)
