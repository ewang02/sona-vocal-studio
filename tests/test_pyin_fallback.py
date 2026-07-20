"""Unit tests for the flagged-region pYIN fallback stage.

The synthetic fixtures replicate the two measured Gunjou defects (the 25.0 s
offset ramp and the 26.2 s two-octave mountain) plus the protections the stage
must preserve.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "work"))

from contour_pipeline_config import get_config
from pyin_fallback import apply_pyin_flagged_fallback

CONFIG = get_config("pyin-fallback-flagged")
HOP = 0.01


def build(n, midi=60.0, broad_conf=0.9, lead_conf=0.9, pyin=60.0, pyin_conf=0.9):
    return {
        "times": np.arange(n) * HOP,
        "midi": np.full(n, float(midi)),
        "keep": np.ones(n, dtype=bool),
        "broad_confidence": np.full(n, float(broad_conf)),
        "lead_confidence": np.full(n, float(lead_conf)),
        "pyin_midi": np.full(n, float(pyin)),
        "pyin_confidence": np.full(n, float(pyin_conf)),
    }


def run(arrays):
    return apply_pyin_flagged_fallback(
        arrays["times"],
        arrays["midi"],
        arrays["keep"],
        arrays["broad_confidence"],
        arrays["lead_confidence"],
        arrays["pyin_midi"],
        arrays["pyin_confidence"],
        CONFIG,
    )


def test_agreeing_contour_untouched():
    arrays = build(60, midi=60.0, pyin=60.1)
    midi, keep, pitch_mask, unvoiced_mask, stats = run(arrays)
    assert np.array_equal(midi, arrays["midi"])
    assert keep.all()
    assert not pitch_mask.any() and not unvoiced_mask.any()
    assert stats["overriddenFrames"] == 0
    assert stats["unvoicedFrames"] == 0


def test_sustained_disagreement_takes_pyin():
    # Wrong-harmonic tracking: contour a fifth up while confident pYIN holds.
    arrays = build(60, midi=58.0, pyin=58.0)
    arrays["midi"][20:35] = 65.0
    midi, keep, pitch_mask, _, stats = run(arrays)
    assert np.allclose(midi[22:33], 58.0)
    assert pitch_mask[22:33].all()
    assert keep.all()
    assert stats["overriddenFrames"] > 0


def test_mountain_spike_bridged_through_pyin_dropout():
    # Gunjou 26.2 s: broad CREPE leaps two octaves for ~6 frames while the
    # noise burst also collapses pYIN's confidence at the exact peak.
    arrays = build(60, midi=58.0, pyin=57.9)
    arrays["midi"][30:36] = [67.0, 77.0, 87.0, 87.0, 78.0, 68.0]
    arrays["pyin_confidence"][30:36] = 0.02
    midi, keep, pitch_mask, _, _ = run(arrays)
    assert np.max(np.abs(midi[30:36] - 57.9)) <= 0.5
    assert pitch_mask[30:36].all()
    assert keep.all()


def test_offset_tail_unvoiced_when_pyin_rejects():
    # Gunjou 25.0 s: a weak segment tail ramps toward a phantom high octave
    # while every estimator is unconfident and pYIN says unvoiced.
    arrays = build(40, midi=58.0, pyin=58.0)
    arrays["broad_confidence"][30:] = 0.12
    arrays["lead_confidence"][30:] = 0.30
    arrays["pyin_confidence"][30:] = 0.01
    arrays["pyin_midi"][30:] = np.nan
    arrays["midi"][38:] = [66.3, 74.6]
    midi, keep, _, unvoiced_mask, stats = run(arrays)
    assert not keep[30:].any()
    assert unvoiced_mask[30:].all()
    assert keep[:30].all()
    assert np.allclose(midi[:30], 58.0)
    assert stats["unvoicedFrames"] == 10


def test_unvoice_disabled_keeps_voicing_but_still_corrects():
    from dataclasses import replace

    config = replace(CONFIG, pyin_fallback_unvoice_enabled=False)
    # Same shape as the offset-tail unvoice case, plus a correctable spike.
    arrays = build(50, midi=58.0, pyin=58.0)
    arrays["broad_confidence"][40:] = 0.12
    arrays["lead_confidence"][40:] = 0.30
    arrays["pyin_confidence"][40:] = 0.01
    arrays["pyin_midi"][40:] = np.nan
    arrays["midi"][48:] = [66.3, 74.6]
    arrays["midi"][20:24] = 65.0  # wrong-harmonic block pYIN disagrees with
    midi, keep, pitch_mask, unvoiced_mask, stats = apply_pyin_flagged_fallback(
        arrays["times"], arrays["midi"], arrays["keep"],
        arrays["broad_confidence"], arrays["lead_confidence"],
        arrays["pyin_midi"], arrays["pyin_confidence"], config,
    )
    assert keep.all()                       # nothing unvoiced
    assert not unvoiced_mask.any()
    assert stats["unvoicedFrames"] == 0
    assert np.allclose(midi[21:23], 58.0)   # override still fires


def test_quiet_note_with_pyin_support_survives():
    # A quiet held note keeps mild pYIN backing: voicing must not be dropped.
    arrays = build(40, midi=58.0, pyin=58.0)
    arrays["broad_confidence"][10:30] = 0.20
    arrays["lead_confidence"][10:30] = 0.30
    arrays["pyin_confidence"][10:30] = 0.20
    midi, keep, _, unvoiced_mask, _ = run(arrays)
    assert keep.all()
    assert not unvoiced_mask.any()
    assert np.allclose(midi, 58.0)


def test_long_low_support_region_left_for_review():
    n = 200
    arrays = build(n, midi=58.0)
    arrays["broad_confidence"][20:180] = 0.10
    arrays["lead_confidence"][20:180] = 0.10
    arrays["pyin_confidence"][20:180] = 0.01
    arrays["pyin_midi"][20:180] = np.nan
    _, keep, _, unvoiced_mask, _ = run(arrays)
    assert keep.all()
    assert not unvoiced_mask.any()


def test_onset_and_offset_edges_take_confident_pyin():
    # Classic onset/offset spike: the smoothed contour ramps in from a false
    # level over the first frames while pYIN already sits on the note.
    arrays = build(60, midi=58.0, pyin=58.0)
    arrays["midi"][:3] = [62.0, 60.5, 59.0]
    arrays["midi"][-3:] = [59.0, 60.5, 62.0]
    midi, keep, pitch_mask, _, stats = run(arrays)
    assert np.allclose(midi[:3], 58.0)
    assert np.allclose(midi[-3:], 58.0)
    assert pitch_mask[:3].all() and pitch_mask[-3:].all()
    assert keep.all()
    assert stats["edgeFrames"] == 6


def test_edge_without_confident_pyin_untouched():
    # Gunjou 25.19s shape: at this onset the broad model was right and pYIN
    # was still ramping up its confidence — no substitution allowed.
    arrays = build(60, midi=60.7, pyin=60.3, pyin_conf=0.30)
    before = arrays["midi"].copy()
    midi, _, _, _, stats = run(arrays)
    assert np.array_equal(midi, before)
    assert stats["edgeFrames"] == 0


def test_interior_untouched_beyond_edge_window():
    # A mild 1 st disagreement mid-segment is below every flag threshold and
    # outside the edge window: the contour keeps its own value.
    arrays = build(60, midi=58.0, pyin=58.0)
    arrays["pyin_midi"][20:30] = 59.0
    midi, _, _, _, _ = run(arrays)
    assert np.allclose(midi[20:30], 58.0)


def test_bridge_requires_kept_anchor_on_both_sides():
    # A spike at the very edge of voicing has no right anchor: leave it to the
    # edge prunes rather than inventing pitch.
    arrays = build(30, midi=58.0, pyin=58.0)
    arrays["midi"][26:] = [70.0, 80.0, 80.0, 70.0]
    arrays["pyin_confidence"][24:] = 0.02
    arrays["keep"][:2] = True
    before = arrays["midi"].copy()
    midi, _, _, _, _ = run(arrays)
    assert np.array_equal(midi[26:], before[26:])
