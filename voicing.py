"""Voicing gates shared by contour extraction and diagnostics."""

from __future__ import annotations

import numpy as np


def drop_short_runs(mask, min_frames):
    out = np.asarray(mask, dtype=bool).copy()
    i = 0
    while i < len(out):
        if not out[i]:
            i += 1
            continue
        j = i
        while j < len(out) and out[j]:
            j += 1
        if j - i < min_frames:
            out[i:j] = False
        i = j
    return out


def running_median(x, win):
    x = np.asarray(x, dtype=float)
    if win <= 1 or len(x) < 2:
        return x.copy()
    if win % 2 == 0:
        win += 1
    half = win // 2
    out = np.empty_like(x)
    for k in range(len(x)):
        a = max(0, k - half)
        b = min(len(x), k + half + 1)
        out[k] = np.median(x[a:b])
    return out


def adaptive_conf_threshold(conf, rms_pass, default=0.55, lo=0.45, hi=0.62):
    """Per-song reliable-anchor threshold from the RMS-active confidence distribution.

    This is intentionally clipped: it can loosen for gritty voices, but it
    should still identify frames reliable enough to anchor low-confidence
    recovery rather than becoming the recovery gate itself.
    """
    active = np.asarray(conf)[np.asarray(rms_pass, dtype=bool)]
    active = active[np.isfinite(active)]
    if len(active) < 50:
        return default
    return float(np.clip(np.percentile(active, 35), lo, hi))


def _anchor_coherent(midi, primary, candidates, hop, max_gap_s, tolerance_st):
    n = len(midi)
    idx = np.arange(n)
    anchor_idx = np.where(primary)[0]
    out = np.zeros(n, dtype=bool)
    if len(anchor_idx) == 0:
        return out

    left = np.full(n, -1, dtype=int)
    right = np.full(n, n, dtype=int)
    left[anchor_idx] = anchor_idx
    right[anchor_idx] = anchor_idx
    left = np.maximum.accumulate(left)
    right = np.minimum.accumulate(right[::-1])[::-1]

    max_gap = max(1, int(round(max_gap_s / hop)))
    cand_idx = np.where(candidates)[0]
    for k in cand_idx:
        l = left[k]
        r = right[k]
        pred = None
        if l >= 0 and r < n and r > l and r - l <= max_gap:
            frac = (k - l) / (r - l)
            pred = midi[l] + frac * (midi[r] - midi[l])
        elif l >= 0 and k - l <= max_gap:
            pred = midi[l]
        elif r < n and r - k <= max_gap:
            pred = midi[r]
        if pred is not None and abs(midi[k] - pred) <= tolerance_st:
            out[k] = True
    return out


def _internal_coherent(midi, low_mask, candidates, hop, window_s, tolerance_st,
                       max_jump_st, min_run_s):
    out = np.zeros(len(midi), dtype=bool)
    win = max(3, int(round(window_s / hop)))
    if win % 2 == 0:
        win += 1
    min_run = max(1, int(round(min_run_s / hop)))

    i = 0
    while i < len(midi):
        if not low_mask[i]:
            i += 1
            continue
        j = i
        while j < len(midi) and low_mask[j]:
            j += 1

        run = midi[i:j]
        if len(run) >= min_run and np.any(candidates[i:j]):
            smooth = running_median(run, min(win, len(run) | 1))
            residual = np.abs(run - smooth)
            jump = np.zeros(len(run), dtype=float)
            if len(run) > 1:
                d = np.abs(np.diff(smooth))
                jump[:-1] = d
                jump[1:] = np.maximum(jump[1:], d)
            keep = (residual <= tolerance_st) & (jump <= max_jump_st)
            out[i:j] = keep & candidates[i:j]
        i = j
    return out


def coherent_voicing_mask(times, midi, conf, rms, rms_floor, conf_hi=0.55,
                          conf_lo=0.30, recover=True, adaptive=False,
                          coherence_st=1.0, anchor_gap_s=0.45,
                          coherence_window_s=0.12, max_jump_st=1.8,
                          min_recover_s=0.08, min_run_s=0.06):
    """Return voiced mask plus diagnostics.

    Primary frames must pass RMS and the reliable confidence threshold.
    Recovery frames must pass RMS and a lower confidence threshold, then pass
    either local-anchor coherence or internal pitch-stability coherence.
    """
    times = np.asarray(times, dtype=float)
    midi = np.asarray(midi, dtype=float)
    conf = np.asarray(conf, dtype=float)
    rms = np.asarray(rms, dtype=float)
    hop = times[1] - times[0] if len(times) > 1 else 0.01
    rms_pass = rms >= rms_floor

    if adaptive:
        conf_hi = adaptive_conf_threshold(conf, rms_pass, default=conf_hi)
        conf_lo = min(conf_lo, max(0.20, conf_hi - 0.25))

    primary = rms_pass & (conf >= conf_hi)
    low_mask = rms_pass & (conf >= conf_lo)
    low_conf_rms = rms_pass & ~primary
    candidates = low_mask & ~primary

    anchor_recovered = np.zeros(len(midi), dtype=bool)
    internal_recovered = np.zeros(len(midi), dtype=bool)
    if recover:
        anchor_recovered = _anchor_coherent(
            midi, primary, candidates, hop, anchor_gap_s, coherence_st)
        internal_recovered = _internal_coherent(
            midi, low_mask, candidates, hop, coherence_window_s, coherence_st,
            max_jump_st, min_recover_s)

    recovered = anchor_recovered | internal_recovered
    pre_min = primary | recovered
    min_frames = max(1, int(round(min_run_s / hop)))
    voiced = drop_short_runs(pre_min, min_frames)

    diagnostics = {
        "conf_hi": conf_hi,
        "conf_lo": conf_lo,
        "rms_pass": rms_pass,
        "primary": primary,
        "low_conf_rms": low_conf_rms,
        "low_candidate": candidates,
        "below_recover_conf": low_conf_rms & ~low_mask,
        "anchor_recovered": anchor_recovered,
        "internal_recovered": internal_recovered,
        "recovered": recovered,
        "pre_min_run": pre_min,
        "pitchless_low_conf": low_conf_rms & ~recovered,
    }
    return voiced, diagnostics
