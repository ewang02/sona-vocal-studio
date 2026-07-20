"""Content-addressed, integrity-checked cache for the local song pipeline.

The cache never substitutes approximate artifacts.  A stage is reusable only
when its complete fingerprint matches and every recorded output still has the
same SHA-256 digest.  Manifests are committed atomically after a successful
stage, which also makes an interrupted song safe to resume.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable


SCHEMA_VERSION = 1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(payload: dict) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class StageCache:
    def __init__(
        self,
        root: Path,
        song: str,
        marker: Callable[[str], None] = print,
    ) -> None:
        self.root = root.resolve()
        self.song = song
        self.marker = marker
        self.path = self.root / "contour_out" / "pipeline_cache" / f"{song}.json"
        # Independent DAG branches may finish at the same time.  Keep manifest
        # updates atomic without holding the lock while an expensive stage is
        # running, otherwise the cache itself would serialize the pipeline.
        self._lock = threading.RLock()
        self._file_hashes: dict[tuple[str, int, int], str] = {}
        self.data = self._load()

    def _load(self) -> dict:
        if not self.path.exists():
            return {"schemaVersion": SCHEMA_VERSION, "songId": self.song, "stages": {}}
        try:
            value = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError):
            return {"schemaVersion": SCHEMA_VERSION, "songId": self.song, "stages": {}}
        if (
            value.get("schemaVersion") != SCHEMA_VERSION
            or value.get("songId") != self.song
            or not isinstance(value.get("stages"), dict)
        ):
            return {"schemaVersion": SCHEMA_VERSION, "songId": self.song, "stages": {}}
        return value

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.data, indent=2, sort_keys=True) + "\n")
        temporary.replace(self.path)

    def file_hash(self, path: Path) -> str:
        resolved = path.resolve()
        stat = resolved.stat()
        key = (str(resolved), stat.st_size, stat.st_mtime_ns)
        with self._lock:
            value = self._file_hashes.get(key)
        if value is None:
            value = sha256(resolved)
            with self._lock:
                self._file_hashes[key] = value
        return value

    def code_hashes(self, paths: Iterable[Path]) -> dict[str, str]:
        return {
            str(path.resolve().relative_to(self.root)): self.file_hash(path)
            for path in paths
        }

    def _relative(self, path: Path) -> str:
        return str(path.resolve().relative_to(self.root))

    def is_hit(self, stage: str, payload: dict, outputs: Iterable[Path]) -> bool:
        with self._lock:
            expected_fingerprint = fingerprint(payload)
            record = self.data["stages"].get(stage)
            output_paths = tuple(outputs)
            if (
                not isinstance(record, dict)
                or record.get("fingerprint") != expected_fingerprint
            ):
                return False
            recorded_outputs = record.get("outputs")
            expected_names = {self._relative(path) for path in output_paths}
            if (
                not isinstance(recorded_outputs, dict)
                or set(recorded_outputs) != expected_names
            ):
                return False
            for path in output_paths:
                if not path.exists() or not path.is_file():
                    return False
                name = self._relative(path)
                saved = recorded_outputs[name]
                if not isinstance(saved, dict):
                    return False
                stat = path.stat()
                if int(saved.get("size", -1)) != stat.st_size:
                    return False
                if saved.get("sha256") != self.file_hash(path):
                    return False
            return True

    def record(self, stage: str, payload: dict, outputs: Iterable[Path]) -> None:
        with self._lock:
            output_paths = tuple(outputs)
            missing = [str(path) for path in output_paths if not path.is_file()]
            if missing:
                raise RuntimeError(
                    f"stage {stage} did not produce required output(s): "
                    f"{', '.join(missing)}"
                )
            self.data["stages"][stage] = {
                "fingerprint": fingerprint(payload),
                "completedAt": datetime.now(timezone.utc).isoformat(),
                "outputs": {
                    self._relative(path): {
                        "size": path.stat().st_size,
                        "sha256": self.file_hash(path),
                    }
                    for path in output_paths
                },
            }
            self._save()

    def run(
        self,
        stage: str,
        payload: dict,
        outputs: Iterable[Path],
        action: Callable[[], None],
        *,
        force: bool = False,
    ) -> bool:
        """Run or reuse one stage; return True on an integrity-checked hit."""
        output_paths = tuple(outputs)
        started = time.perf_counter()
        if not force and self.is_hit(stage, payload, output_paths):
            elapsed = time.perf_counter() - started
            self.marker(f"::cache:hit stage={stage}")
            self.marker(f"::timing stage={stage} seconds={elapsed:.3f} cache=hit")
            return True

        self.marker(f"::cache:miss stage={stage}")
        try:
            action()
            self.record(stage, payload, output_paths)
        except BaseException:
            elapsed = time.perf_counter() - started
            self.marker(
                f"::timing stage={stage} seconds={elapsed:.3f} cache=miss status=error"
            )
            raise
        elapsed = time.perf_counter() - started
        self.marker(f"::timing stage={stage} seconds={elapsed:.3f} cache=miss")
        return False
