import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "work"))

from export_contour import export_pyin_repair_source


def test_pyin_repair_export_preserves_empty_frames(tmp_path):
    source = (
        tmp_path
        / "contour_out"
        / "lead_contours"
        / "sample_contour.csv"
    )
    source.parent.mkdir(parents=True)
    source.write_text(
        "time_s,midi_pyin\n"
        "0.00,60.0\n"
        "0.01,60.2\n"
        "0.02,nan\n"
        "0.03,61.0\n"
    )
    output = tmp_path / "public" / "data"
    output.mkdir(parents=True)

    path = export_pyin_repair_source(tmp_path, "sample", output)
    payload = json.loads(path.read_text())

    assert payload["hop"] == 0.01
    assert payload["duration"] == 0.04
    assert payload["segments"] == [
        {"t0": 0.0, "midi": [60.0, 60.2]},
        {"t0": 0.03, "midi": [61.0]},
    ]
