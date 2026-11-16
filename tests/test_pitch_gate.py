import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "work"))

from analyze_pitch_gate import choose_rms_floor  # noqa: E402
from contour_pipeline_config import PRODUCTION_CONFIG  # noqa: E402


def sung_and_silent(silence_level: float, seed: int = 3) -> np.ndarray:
    """RMS frames for a song: quiet-to-loud singing with gaps between phrases."""
    random = np.random.default_rng(seed)
    singing = 10 ** random.normal(-0.85, 0.25, 6000)  # around -17 dBFS
    soft_edges = 10 ** random.uniform(-2.2, -1.2, 600)  # onsets and tails
    silence = np.full(2000, silence_level)
    frames = np.concatenate([singing, soft_edges, silence])
    random.shuffle(frames)
    return frames


def test_digital_silence_counts_as_silence_not_missing_data() -> None:
    # A stem with a faint residual noise floor between phrases...
    noisy = sung_and_silent(silence_level=2e-5)
    # ...and the same performance from a separator that leaves exact zeros.
    clean = sung_and_silent(silence_level=0.0)

    noisy_floor, _ = choose_rms_floor(noisy, PRODUCTION_CONFIG)
    clean_floor, diagnostics = choose_rms_floor(clean, PRODUCTION_CONFIG)

    # Both floors sit between silence and the softest singing, so soft note
    # edges survive the gate either way.
    for floor in (noisy_floor, clean_floor):
        assert 2e-5 < floor < 10 ** -2.2
    assert diagnostics["method"] == "otsu"
    # Without the silence class Otsu would split the singing itself.
    assert clean_floor < 0.01


def test_floor_is_unchanged_for_stems_without_exact_zeros() -> None:
    frames = sung_and_silent(silence_level=2e-5)
    nonzero = frames[frames > 0]
    assert len(nonzero) == len(frames)
    first, _ = choose_rms_floor(frames, PRODUCTION_CONFIG)
    # Appending nothing but finite positive values keeps the legacy path:
    # the result depends only on those values.
    second, _ = choose_rms_floor(nonzero.copy(), PRODUCTION_CONFIG)
    assert first == second


def test_all_silent_input_has_no_floor() -> None:
    floor, diagnostics = choose_rms_floor(np.zeros(100), PRODUCTION_CONFIG)
    assert floor == 0.0
    assert diagnostics["method"] == "empty"
