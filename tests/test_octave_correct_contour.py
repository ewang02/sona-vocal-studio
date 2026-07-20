import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "work"))

from octave_correct_contour import (
    correct_octaves,
    recover_secondary_voicing,
    repair_isolated_excursions,
)


class OctaveCorrectionTests(unittest.TestCase):
    def test_explicit_note_rescue_shifts_whole_region(self):
        times = np.arange(6) * 0.01
        midi = np.full(6, 72.1)
        notes = [{"t0": 0.01, "t1": 0.05, "midi": 60, "octave_corrected_from": 72}]
        result, mask, stats = correct_octaves(times, midi, notes=notes)
        np.testing.assert_allclose(result[1:5], 60.1)
        self.assertFalse(mask[0])
        self.assertEqual(stats["note_regions"], 1)

    def test_sustained_estimator_octave_is_repaired(self):
        times = np.arange(6) * 0.01
        midi = np.full(6, 72.0)
        evidence = np.array([72, 60, 60, 60, 72, 72], dtype=float)
        result, mask, stats = correct_octaves(
            times, midi, evidence_times=times, evidence_midi=evidence,
            evidence_confidence=np.ones(6)
        )
        np.testing.assert_allclose(result[1:4], 60.0)
        self.assertEqual(int(mask.sum()), 3)
        self.assertEqual(stats["estimator_runs"], 1)

    def test_short_run_and_non_octave_difference_are_ignored(self):
        times = np.arange(5) * 0.01
        midi = np.full(5, 72.0)
        short = np.array([72, 60, 60, 72, 72], dtype=float)
        result, mask, _ = correct_octaves(
            times, midi, evidence_times=times, evidence_midi=short,
            evidence_confidence=np.ones(5)
        )
        np.testing.assert_allclose(result, midi)
        self.assertFalse(mask.any())

    def test_short_leave_and_return_excursion_is_repaired(self):
        times = np.arange(80) * 0.01
        midi = np.full(80, 60.0)
        midi[35:43] = [61, 64, 68, 68, 68, 65, 62, 61]
        result, mask, stats = repair_isolated_excursions(
            times, midi, np.ones(80, dtype=bool)
        )
        np.testing.assert_allclose(result[35:43], 60.0)
        self.assertTrue(mask[35:43].all())
        self.assertEqual(stats["repaired_regions"], 1)

    def test_sustained_pitch_change_is_not_repaired(self):
        times = np.arange(80) * 0.01
        midi = np.r_[np.full(40, 60.0), np.full(40, 67.0)]
        result, mask, stats = repair_isolated_excursions(
            times, midi, np.ones(80, dtype=bool)
        )
        np.testing.assert_allclose(result, midi)
        self.assertFalse(mask.any())
        self.assertEqual(stats["repaired_regions"], 0)

    def test_near_octave_disagreement_and_secondary_voicing_recovery(self):
        times = np.arange(8) * 0.01
        midi = np.full(8, 83.0)
        evidence = np.full(8, 71.8)
        result, mask, _ = correct_octaves(
            times, midi, evidence_times=times, evidence_midi=evidence,
            evidence_confidence=np.full(8, 0.8)
        )
        np.testing.assert_allclose(result, 71.0)
        self.assertTrue(mask.all())

        recovered, added = recover_secondary_voicing(
            times, np.zeros(8, dtype=bool), np.full(8, 0.1), 0.01,
            evidence, np.full(8, 0.8)
        )
        self.assertTrue(recovered.all())
        self.assertEqual(added, 8)

        # A low-confidence gross-error edge beside the confirmed octave run is
        # replaced by the stable independent estimate instead of left as a spike.
        source = np.array([72, 78, 83, 83, 83, 78, 72, 72], dtype=float)
        confidence = np.array([0.1, 0.3, 0.8, 0.8, 0.8, 0.3, 0.1, 0.1])
        result, _, stats = correct_octaves(
            times, source, evidence_times=times, evidence_midi=np.full(8, 71.0),
            evidence_confidence=confidence
        )
        np.testing.assert_allclose(result[1:6], 71.0)
        self.assertEqual(stats["gross_error_frames"], 2)

        wrong_interval = np.full(8, 65.0)
        result, mask, _ = correct_octaves(
            times, midi, evidence_times=times, evidence_midi=wrong_interval,
            evidence_confidence=np.ones(8)
        )
        np.testing.assert_allclose(result, midi)
        self.assertFalse(mask.any())


if __name__ == "__main__":
    unittest.main()
