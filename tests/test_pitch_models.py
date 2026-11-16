import json
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from work import process_song
from work.contour_pipeline_config import PRESETS
from work.pitch_models import (
    CONFIDENCE_LEVELS,
    DEFAULT_PITCH_MODEL,
    PITCH_MODELS,
    RMVPE_BIN_CENTS,
    RMVPE_CENTS_ORIGIN,
    calibrate_confidence,
    decode_rmvpe_salience,
    estimate_rmvpe,
    estimate_swiftf0,
    get_pitch_model,
)

ROOT = Path(__file__).resolve().parents[1]


def harmonic_tone(frequency: float, seconds: float = 1.5) -> np.ndarray:
    time = np.arange(int(16_000 * seconds)) / 16_000
    tone = sum(np.sin(2 * np.pi * frequency * k * time) / k**1.2 for k in range(1, 9))
    return (0.2 * tone).astype(np.float32)


def cents_between(measured: np.ndarray, expected: float) -> np.ndarray:
    return 1200 * np.log2(measured / expected)


def test_every_model_carries_complete_independent_tuning() -> None:
    assert DEFAULT_PITCH_MODEL == "crepe"
    assert set(PITCH_MODELS) == {"crepe", "rmvpe", "swiftf0"}
    presets = [model.contour_preset for model in PITCH_MODELS.values()]
    # One preset each, so tuning one model's export never moves another's.
    assert len(set(presets)) == len(presets)
    for name, model in PITCH_MODELS.items():
        assert model.name == name
        assert model.contour_preset in PRESETS
        assert 0 < model.voicing_recover < model.voicing_reliable < 1
        for role in ("anchor", "lead"):
            knots = model.confidence_knots(role)
            if knots is None:
                continue
            assert len(knots) == len(CONFIDENCE_LEVELS)
            assert all(0 < low < high < 1 for low, high in zip(knots, knots[1:]))
    # CREPE is the reference scale and must stay exactly as it was.
    crepe = PITCH_MODELS["crepe"]
    assert crepe.anchor_confidence is None and crepe.lead_confidence is None
    assert (crepe.anchor_variant, crepe.lead_variant) == ("full", "tiny")
    assert (crepe.voicing_reliable, crepe.voicing_recover) == (0.55, 0.30)
    assert crepe.contour_preset == "production-v6r"


def test_unknown_model_and_role_are_rejected() -> None:
    assert get_pitch_model(None).name == "crepe"
    assert get_pitch_model(" RMVPE ").name == "rmvpe"
    with pytest.raises(ValueError, match="Unknown pitch model"):
        get_pitch_model("yin")
    with pytest.raises(ValueError, match="role"):
        PITCH_MODELS["rmvpe"].confidence_knots("backing")


def test_calibration_puts_each_native_threshold_on_its_level() -> None:
    for model in (PITCH_MODELS["rmvpe"], PITCH_MODELS["swiftf0"]):
        for role in ("anchor", "lead"):
            knots = model.confidence_knots(role)
            calibrated = calibrate_confidence(np.array(knots), knots)
            assert np.allclose(calibrated, CONFIDENCE_LEVELS)
            # Monotonic: ranking frames by confidence never changes.
            sweep = calibrate_confidence(np.linspace(0, 1, 501), knots)
            assert np.all(np.diff(sweep) >= 0)
            assert sweep[0] == 0.0 and sweep[-1] == 1.0
    identity = np.array([0.0, 0.2, 0.55, 1.0])
    assert np.array_equal(calibrate_confidence(identity, None), identity)
    with pytest.raises(ValueError, match="increase strictly"):
        calibrate_confidence(identity, tuple([0.5] * len(CONFIDENCE_LEVELS)))


def test_a_models_own_voicing_threshold_marks_the_recoverable_level() -> None:
    rmvpe = PITCH_MODELS["rmvpe"]
    recover = CONFIDENCE_LEVELS.index(0.30)
    for role in ("anchor", "lead"):
        assert rmvpe.confidence_knots(role)[recover] == rmvpe.native_voicing_floor
    # Below its own threshold RMVPE considers a frame unvoiced; so must we.
    below = calibrate_confidence(np.array([0.02]), rmvpe.lead_confidence)[0]
    assert below < rmvpe.voicing_recover


def test_tuning_fingerprint_changes_with_any_threshold() -> None:
    model = PITCH_MODELS["swiftf0"]
    retuned = replace(model, voicing_reliable=0.6)
    knots = list(model.lead_confidence)
    knots[4] += 0.01
    reknotted = replace(model, lead_confidence=tuple(knots))
    assert len({model.fingerprint(), retuned.fingerprint(), reknotted.fingerprint()}) == 3
    assert model.fingerprint() == replace(model).fingerprint()


def test_rmvpe_salience_decodes_to_sub_bin_pitch() -> None:
    salience = np.zeros((3, 360))
    # Frame 0: a clean peak on bin 100.
    salience[0, 99:102] = [0.3, 0.9, 0.3]
    # Frame 1: energy split between two bins reads halfway between them.
    salience[1, 200:202] = [0.8, 0.8]
    # Frame 2: nothing salient.
    f0, confidence = decode_rmvpe_salience(salience)
    expected_0 = 10 * 2 ** ((RMVPE_BIN_CENTS * 100 + RMVPE_CENTS_ORIGIN) / 1200)
    expected_1 = 10 * 2 ** ((RMVPE_BIN_CENTS * 200.5 + RMVPE_CENTS_ORIGIN) / 1200)
    assert abs(cents_between(f0[:1], expected_0)[0]) < 0.01
    assert abs(cents_between(f0[1:2], expected_1)[0]) < 0.01
    assert np.allclose(confidence, [0.9, 0.8, 0.0])
    assert np.isfinite(f0[2])


def test_swiftf0_reports_on_the_pipeline_frame_grid() -> None:
    audio = harmonic_tone(220.0)
    f0, confidence = estimate_swiftf0(audio, 160, 65.0, 1500.0)
    assert len(f0) == len(confidence) == len(audio) // 160 + 1
    steady = slice(20, -20)
    assert np.all(np.abs(cents_between(f0[steady], 220.0)) < 25)
    assert np.median(confidence[steady]) > 0.9


def test_rmvpe_tracks_a_tone_when_its_weights_are_installed() -> None:
    weights = process_song.MODEL_DIR / "rmvpe.onnx"
    if not weights.is_file():
        pytest.skip("RMVPE weights are downloaded on first use")
    audio = harmonic_tone(330.0)
    f0, confidence = estimate_rmvpe(audio, 160, weights)
    assert len(f0) == len(confidence) == len(audio) // 160 + 1
    steady = slice(20, -20)
    assert np.all(np.abs(cents_between(f0[steady], 330.0)) < 25)
    assert np.median(confidence[steady]) > 0.5
    with pytest.raises(ValueError, match="fixed"):
        estimate_rmvpe(audio, 256, weights)
    with pytest.raises(FileNotFoundError, match="RMVPE weights"):
        estimate_rmvpe(audio, 160, weights.with_name("missing.onnx"))


def test_downloadable_pitch_weights_are_pinned_in_the_manifest() -> None:
    manifest = json.loads(process_song.MODEL_ASSET_MANIFEST.read_text())
    for name, model in PITCH_MODELS.items():
        pinned = set(process_song.PITCH_MODEL_HASHES.get(name, {}))
        assert pinned == ({model.asset} if model.asset else set())
    for entry in manifest["pitchModels"].values():
        for asset in entry["assets"]:
            assert asset["url"].startswith("https://")
            assert re.fullmatch(r"[a-f0-9]{64}", asset["sha256"])


def test_missing_pitch_weights_name_the_fix(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(process_song, "MODEL_DIR", tmp_path)
    assert process_song.validate_pitch_model("crepe") == {}
    with pytest.raises(RuntimeError, match="model-assets.mjs pitch rmvpe"):
        process_song.validate_pitch_model("rmvpe")
    (tmp_path / "rmvpe.onnx").write_bytes(b"not the model")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        process_song.validate_pitch_model("rmvpe")


def test_companion_and_interface_offer_exactly_the_pipeline_models() -> None:
    server = (ROOT / "server" / "pitch-models.mjs").read_text()
    listed = re.search(r"PITCH_MODELS = \[([^\]]*)\]", server).group(1)
    assert sorted(re.findall(r'"([a-z0-9]+)"', listed)) == sorted(PITCH_MODELS)
    interface = (ROOT / "app" / "library-data.ts").read_text()
    assert sorted(re.findall(r'id: "([a-z0-9]+)",', interface)) == sorted(PITCH_MODELS)
    for model in PITCH_MODELS.values():
        assert f'label: "{model.label}"' in interface
