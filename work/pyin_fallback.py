"""Trust lead-only pYIN wherever the anomaly miner would flag the contour.

The stage recomputes the three signal-derived flags used by
prepare_contour_anomaly_review on the corrected contour, then applies pYIN's
verdict inside flagged territory only:

* override — flagged frames where pYIN is confident take pYIN's pitch;
* bridge   — short flagged remainders without confident pYIN are linearly
  interpolated between kept neighbors, removing spike interiors whose noise
  burst also collapsed pYIN's own confidence;
* unvoice  — low-support regions where pYIN's voicing probability stays below
  an active-rejection floor are dropped, bounded in duration so a sustained
  quiet note is surfaced for review instead of silently deleted;
* edges    — the first and last window of every surviving voiced segment takes
  confident pYIN directly, because CREPE settling plus the display smoothing
  owns exactly those frames (the onset/offset spike shape).

Frames outside flagged territory are never modified, and unvoiced frames are
never resurrected here (recover_secondary_voicing already owns recovery).
"""

from __future__ import annotations

import numpy as np


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.flatnonzero(np.diff(np.r_[False, np.asarray(mask, dtype=bool), False]))
    return list(zip(edges[::2], edges[1::2]))


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or not np.any(mask):
        return mask
    window = np.ones(2 * radius + 1, dtype=int)
    return np.convolve(mask.astype(int), window, mode="same") > 0


def apply_pyin_flagged_fallback(
    times: np.ndarray,
    midi: np.ndarray,
    keep: np.ndarray,
    broad_confidence: np.ndarray,
    lead_confidence: np.ndarray,
    pyin_midi: np.ndarray,
    pyin_confidence: np.ndarray,
    config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Return (midi, keep, pitch_changed_mask, unvoiced_mask, stats)."""

    times = np.asarray(times, dtype=float)
    midi = np.asarray(midi, dtype=float).copy()
    keep = np.asarray(keep, dtype=bool).copy()
    broad_confidence = np.asarray(broad_confidence, dtype=float)
    lead_confidence = np.asarray(lead_confidence, dtype=float)
    pyin_midi = np.asarray(pyin_midi, dtype=float)
    pyin_confidence = np.asarray(pyin_confidence, dtype=float)
    hop = float(times[1] - times[0]) if len(times) > 1 else 0.01

    finite_pyin = np.isfinite(pyin_midi)
    confident = finite_pyin & (pyin_confidence >= config.pyin_fallback_confidence)
    visible = keep & np.isfinite(midi)

    # Flag 1: sustained disagreement with confident pYIN.
    disagreement = (
        visible
        & confident
        & (np.abs(midi - pyin_midi) >= config.pyin_fallback_disagreement_st)
    )
    minimum_disagreement = max(
        1, int(round(config.pyin_fallback_min_disagreement_seconds / hop))
    )
    flag_pitch = np.zeros(len(times), dtype=bool)
    for start, end in _runs(disagreement):
        if end - start >= minimum_disagreement:
            flag_pitch[start:end] = True

    # Flag 2: sharp frame jumps inside a continuous visible run, dilated the
    # same way the miner widens its jump candidates.
    jump = np.zeros(len(times), dtype=bool)
    if len(midi) > 1:
        adjacent = visible[1:] & visible[:-1]
        frame_jump = np.abs(np.diff(midi))
        hits = adjacent & (frame_jump >= config.pyin_fallback_jump_st)
        jump[1:][hits] = True
        jump[:-1][hits] = True
    jump = _dilate(
        jump, int(round(config.pyin_fallback_jump_dilation_seconds / hop))
    )
    flag_pitch |= jump & visible

    # Flag 3: visible voicing weak in both CREPE views with no pYIN support.
    low_support = (
        visible
        & (broad_confidence < config.pyin_fallback_low_support_broad_confidence)
        & (lead_confidence < config.pyin_fallback_low_support_lead_confidence)
        & (~finite_pyin | (pyin_confidence < config.pyin_fallback_support_confidence))
    )
    minimum_low_support = max(
        1, int(round(config.pyin_fallback_min_low_support_seconds / hop))
    )
    flag_low = np.zeros(len(times), dtype=bool)
    for start, end in _runs(low_support):
        if end - start >= minimum_low_support:
            flag_low[start:end] = True

    # Override: inside pitch-flagged territory pYIN's confident value wins.
    override = flag_pitch & keep & confident
    pitch_changed = np.zeros(len(times), dtype=bool)
    changed = override & (np.abs(midi - pyin_midi) >= 0.25)
    midi[override] = pyin_midi[override]
    pitch_changed |= changed

    # Bridge: flagged remainders too noisy even for pYIN, interpolated between
    # kept neighbors when short enough to be an artifact rather than a phrase.
    bridge_limit = int(round(config.pyin_fallback_bridge_max_seconds / hop))
    bridged_frames = 0
    for start, end in _runs(flag_pitch & keep & ~confident):
        if end - start > bridge_limit:
            continue
        left = start - 1
        right = end
        if left < 0 or right >= len(times):
            continue
        if not (keep[left] and np.isfinite(midi[left])):
            continue
        if not (keep[right] and np.isfinite(midi[right])):
            continue
        span = right - left
        ramp = midi[left] + (midi[right] - midi[left]) * (
            np.arange(1, end - start + 1) / span
        )
        changed = np.abs(midi[start:end] - ramp) >= 0.25
        midi[start:end] = ramp
        pitch_changed[start:end] |= changed
        bridged_frames += int(np.sum(changed))

    # Unvoice: low-support regions pYIN actively rejects. Long regions are
    # deliberately left visible so a sustained quiet note reaches review.
    maximum_unvoice = int(round(config.pyin_fallback_max_unvoice_seconds / hop))
    unvoiced = np.zeros(len(times), dtype=bool)
    if config.pyin_fallback_unvoice_enabled:
        for start, end in _runs(flag_low & keep):
            if end - start > maximum_unvoice:
                continue
            region_confidence = pyin_confidence[start:end]
            if np.any(region_confidence >= config.pyin_fallback_reject_confidence):
                continue
            keep[start:end] = False
            unvoiced[start:end] = True

    # Edges: on the final voicing, substitute confident pYIN through each
    # segment's leading and trailing window. pYIN gaps stay CREPE.
    edge_frames = 0
    edge_window = int(round(config.pyin_fallback_edge_seconds / hop))
    if edge_window > 0:
        edge_confident = finite_pyin & (
            pyin_confidence >= config.pyin_fallback_edge_confidence
        )
        edge_mask = np.zeros(len(times), dtype=bool)
        for start, end in _runs(keep):
            edge_mask[start : min(start + edge_window, end)] = True
            edge_mask[max(start, end - edge_window) : end] = True
        substitute = (
            edge_mask & edge_confident & (np.abs(midi - pyin_midi) >= 0.25)
        )
        midi[substitute] = pyin_midi[substitute]
        pitch_changed |= substitute
        edge_frames = int(np.sum(substitute))

    stats = {
        "confidence": config.pyin_fallback_confidence,
        "flaggedFrames": int(np.sum(flag_pitch | flag_low)),
        "overriddenFrames": int(np.sum(pitch_changed & override)),
        "bridgedFrames": bridged_frames,
        "unvoicedFrames": int(np.sum(unvoiced)),
        "edgeFrames": edge_frames,
    }
    return midi, keep, pitch_changed, unvoiced, stats
