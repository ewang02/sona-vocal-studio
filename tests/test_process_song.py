import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event

from work import build_final_transcriptions
from work import process_song
from work.pipeline_cache import StageCache
from work.separate_lead_vocals import resolve_source


ROOT = Path(__file__).resolve().parents[1]


def test_production_model_files_match_the_pinned_hashes() -> None:
    assert process_song.validate_models() == process_song.MODEL_HASHES


def test_separator_commands_use_the_selected_model_for_each_product_stem(tmp_path) -> None:
    executable = tmp_path / "audio-separator"
    source = tmp_path / "song.mp3"
    output = tmp_path / "stems"

    instrumental = process_song.separator_command(
        executable,
        source,
        process_song.KIM_INST_MODEL,
        "Instrumental",
        "instrumental",
        output,
    )
    vocals = process_song.separator_command(
        executable,
        source,
        process_song.KIM_VOCAL_MODEL,
        "Vocals",
        "vocals",
        output,
    )

    assert instrumental[instrumental.index("--model") + 1] == "Kim_Inst.onnx"
    assert vocals[vocals.index("--model") + 1] == "Kim_Vocal_2.onnx"
    assert instrumental[instrumental.index("--stem") + 1] == "Instrumental"
    assert vocals[vocals.index("--stem") + 1] == "Vocals"
    assert instrumental[instrumental.index("--output-name") + 1] == "instrumental"
    assert vocals[vocals.index("--output-name") + 1] == "vocals"
    assert instrumental[instrumental.index("--batch-size") + 1] == "1"
    assert vocals[vocals.index("--batch-size") + 1] == "1"


def test_one_kim_product_replaces_only_its_own_stem_atomically(
    tmp_path, monkeypatch
) -> None:
    python = tmp_path / "bin" / "python"
    python.parent.mkdir()
    python.touch()
    source = tmp_path / "song.mp3"
    source.touch()
    destination = tmp_path / "separators" / "song" / "vocals.wav"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"old-vocal")
    sibling = destination.with_name("instrumental.wav")
    sibling.write_bytes(b"unchanged-instrumental")

    def fake_run(command: list[str]) -> None:
        output_dir = Path(command[command.index("--output-dir") + 1])
        output_name = command[command.index("--output-name") + 1]
        (output_dir / f"{output_name}.wav").write_bytes(b"new-vocal")

    monkeypatch.setattr(process_song, "PYTHON", str(python))
    monkeypatch.setattr(process_song, "run", fake_run)

    result = process_song.separate_kim_product(
        source,
        destination,
        model=process_song.KIM_VOCAL_MODEL,
        stem="Vocals",
        output_name="vocals",
    )

    assert result == destination
    assert destination.read_bytes() == b"new-vocal"
    assert sibling.read_bytes() == b"unchanged-instrumental"
    assert not destination.parent.joinpath(".vocals.tmp").exists()


def test_lead_source_prefers_new_stems_but_supports_demucs_legacy(tmp_path) -> None:
    new_source = tmp_path / "contour_out" / "separators" / "song" / "vocals.wav"
    legacy_source = (
        tmp_path / "contour_out" / "demucs" / "htdemucs" / "song" / "vocals.wav"
    )
    legacy_source.parent.mkdir(parents=True)
    legacy_source.touch()
    assert resolve_source(tmp_path, "song", None) == legacy_source

    new_source.parent.mkdir(parents=True)
    new_source.touch()
    assert resolve_source(tmp_path, "song", None) == new_source
    assert resolve_source(tmp_path, "song", Path("custom.wav")) == tmp_path / "custom.wav"


def test_stage_cache_reuses_only_integrity_checked_outputs(tmp_path) -> None:
    output = tmp_path / "artifacts" / "result.bin"
    messages: list[str] = []
    calls = 0

    def build() -> None:
        nonlocal calls
        calls += 1
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"exact-output")

    cache = StageCache(tmp_path, "song", messages.append)
    payload = {"source": "abc", "settings": {"model": "same"}}

    assert not cache.run("extract", payload, [output], build)
    assert cache.run("extract", payload, [output], build)
    assert calls == 1
    assert "::cache:hit stage=extract" in messages

    output.write_bytes(b"tampered")
    assert not cache.run("extract", payload, [output], build)
    assert calls == 2
    assert output.read_bytes() == b"exact-output"


def test_stage_cache_invalidates_changed_inputs_and_force(tmp_path) -> None:
    output = tmp_path / "result.bin"
    calls = 0

    def build() -> None:
        nonlocal calls
        calls += 1
        output.write_text(str(calls))

    cache = StageCache(tmp_path, "song")
    assert not cache.run("stage", {"input": "one"}, [output], build)
    assert cache.run("stage", {"input": "one"}, [output], build)
    assert not cache.run("stage", {"input": "two"}, [output], build)
    assert not cache.run("stage", {"input": "two"}, [output], build, force=True)
    assert calls == 3


def test_stage_cache_commits_parallel_branches_without_losing_a_record(
    tmp_path,
) -> None:
    cache = StageCache(tmp_path, "song")
    barrier = Barrier(2, timeout=2)

    def run_stage(stage: str) -> None:
        output = tmp_path / f"{stage}.bin"

        def build() -> None:
            output.write_text(stage)
            barrier.wait()

        cache.run(stage, {"stage": stage}, [output], build)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run_stage, stage) for stage in ("anchor", "lead")]
        for future in futures:
            future.result()

    reloaded = StageCache(tmp_path, "song")
    assert set(reloaded.data["stages"]) == {"anchor", "lead"}


def test_process_finishes_vocals_then_overlaps_instrumental_with_pitch_dag(
    tmp_path, monkeypatch
) -> None:
    song = "song"
    source = tmp_path / f"{song}.mp3"
    source.write_bytes(b"source")
    kim_helper = tmp_path / "work" / "separate_kim_stem.py"
    kim_helper.parent.mkdir()
    kim_helper.write_text("helper")
    stages: list[str] = []
    vocal_done = Event()
    instrumental_started = Event()
    pitch_started = Event()

    class FakeCache:
        def __init__(self, root, cache_song, cache_marker=print):
            self.root = root

        def file_hash(self, path: Path) -> str:
            return process_song.sha256(path)

        def code_hashes(self, paths) -> dict[str, str]:
            return {}

        def run(self, stage, payload, outputs, action, *, force=False):
            stages.append(stage)
            action()
            assert all(path.is_file() for path in outputs)
            return False

    def fake_run(command: list[str]) -> None:
        if len(command) > 1 and command[1] == str(kim_helper):
            output_dir = Path(command[command.index("--output-dir") + 1])
            output_name = command[command.index("--output-name") + 1]
            if output_name == "vocals":
                assert not instrumental_started.is_set()
                (output_dir / "vocals.wav").write_bytes(b"vocals")
                vocal_done.set()
            else:
                assert vocal_done.is_set()
                instrumental_started.set()
                assert pitch_started.wait(timeout=2)
                (output_dir / "instrumental.wav").write_bytes(b"instrumental")
            return
        if "work/separate_lead_vocals.py" in command:
            assert instrumental_started.wait(timeout=2)
            pitch_started.set()
            output = tmp_path / "contour_out" / "lead_vocals" / song
            output.mkdir(parents=True, exist_ok=True)
            (output / "lead.wav").write_bytes(b"lead")
            (output / "backing.wav").write_bytes(b"backing")
            return
        if "contour.py" in command:
            pitch_started.set()
            output = tmp_path / "contour_out" / f"{song}_contour.csv"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("anchor")
            return
        if "work/build_final_transcriptions.py" in command:
            if "--extract-only" in command:
                output = (
                    tmp_path
                    / "contour_out"
                    / "lead_contours"
                    / f"{song}_contour.csv"
                )
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text("lead contour")
            else:
                output = (
                    tmp_path
                    / "experiments"
                    / "transcription_final"
                    / f"{song}_notes_final.json"
                )
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text("{}")
            return
        if "work/export_contour.py" in command:
            output = tmp_path / "public" / "data" / f"{song}-contour.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("{}")
            (output.parent / f"{song}-pyin.json").write_text("{}")
            return
        if command[0] == "ffmpeg":
            Path(command[-1]).write_bytes(b"encoded instrumental")
            return
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(process_song, "ROOT", tmp_path)
    monkeypatch.setattr(
        process_song, "SEPARATOR_ROOT", tmp_path / "contour_out" / "separators"
    )
    monkeypatch.setattr(process_song, "KIM_SEPARATOR", kim_helper)
    monkeypatch.setattr(process_song, "PYTHON", "python")
    monkeypatch.setattr(process_song, "StageCache", FakeCache)
    monkeypatch.setattr(
        process_song,
        "validate_models",
        lambda hash_file: dict(process_song.MODEL_HASHES),
    )
    monkeypatch.setattr(process_song, "installed_versions", lambda *packages: {})
    monkeypatch.setattr(process_song, "command_version", lambda command: "ffmpeg")
    monkeypatch.setattr(process_song, "ffprobe_duration", lambda path: 30.0)
    monkeypatch.setattr(process_song, "run", fake_run)

    duration = process_song.process(song, source)

    assert duration == 30.0
    assert stages.index("vocal_separation") < stages.index(
        "instrumental_separation"
    )
    assert stages.index("vocal_separation") < stages.index("anchor_extraction")
    assert (
        tmp_path / "public" / "audio" / f"{song}-instrumental.mp3"
    ).read_bytes() == b"encoded instrumental"


def test_extract_only_keeps_the_exact_tiny_crepe_and_pyin_contract(
    tmp_path, monkeypatch
) -> None:
    song = "song"
    lead = tmp_path / "contour_out" / "lead_vocals" / song / "lead.wav"
    anchor = tmp_path / "contour_out" / f"{song}_contour.csv"
    lead.parent.mkdir(parents=True)
    lead.write_bytes(b"lead")
    anchor.parent.mkdir(parents=True, exist_ok=True)
    anchor.write_text("time,pitch\n")
    commands: list[list[str]] = []

    def fake_run(command: list[str]) -> None:
        commands.append(command)
        contour = tmp_path / "contour_out" / "lead_contours" / f"{song}_contour.csv"
        contour.parent.mkdir(parents=True, exist_ok=True)
        contour.write_text("time,pitch\n")

    monkeypatch.setattr(build_final_transcriptions, "ROOT", tmp_path)
    monkeypatch.setattr(build_final_transcriptions, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["build_final_transcriptions.py", song, "--model", "tiny", "--extract-only"],
    )

    build_final_transcriptions.main()

    assert len(commands) == 1
    command = commands[0]
    assert command[1] == "contour.py"
    expected_options = {
        "--model": "tiny",
        "--fmax": "1500",
        "--batch-size": "64",
        "--torch-threads": "2",
        "--chunk-seconds": "20",
        "--decoder": "weighted_argmax",
        "--pitch-filter": process_song.PRODUCTION_CONFIG.lead_pitch_filter,
        "--secondary-f0": "pyin",
        "--pyin-chunk-seconds": "20",
        "--pyin-workers": "0",
    }
    for option, expected in expected_options.items():
        assert command[command.index(option) + 1] == expected
    assert "--data-only" in command


def test_skip_piano_keeps_note_transcription_but_omits_unused_preview(
    tmp_path, monkeypatch
) -> None:
    song = "song"
    lead = tmp_path / "contour_out" / "lead_vocals" / song / "lead.wav"
    anchor = tmp_path / "contour_out" / f"{song}_contour.csv"
    contour = tmp_path / "contour_out" / "lead_contours" / f"{song}_contour.csv"
    for path in (lead, anchor, contour):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ready")
    commands: list[list[str]] = []

    monkeypatch.setattr(build_final_transcriptions, "ROOT", tmp_path)
    monkeypatch.setattr(build_final_transcriptions, "run", commands.append)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_final_transcriptions.py",
            song,
            "--skip-extraction",
            "--skip-piano",
        ],
    )

    build_final_transcriptions.main()

    assert len(commands) == 1
    assert commands[0][1] == "transcribe_notes.py"
    assert "--pitch-anchor" in commands[0]
    assert "--no-beat" in commands[0]
