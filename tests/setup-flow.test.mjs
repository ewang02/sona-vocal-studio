import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { pythonCandidates } from "../scripts/setup-pipeline.mjs";
import {
  markPipelineReady,
  pipelineReadiness,
} from "../scripts/pipeline-state.mjs";
import {
  unsafeHostedOutput,
  unsafePublicAssets,
} from "../scripts/check-public-assets.mjs";

function seedPipelineInputs(root) {
  for (const relative of [
    "requirements-pipeline.txt",
    "scripts/setup-pipeline.mjs",
    "scripts/select-inference-backend.mjs",
    "contour_out/models/audio-separator/model-assets.json",
  ]) {
    const path = join(root, relative);
    mkdirSync(join(path, ".."), { recursive: true });
    writeFileSync(path, relative);
  }
}

test("keeps Windows Python launcher arguments separate from the executable", () => {
  assert.deepEqual(
    pythonCandidates({ platform: "win32", env: { PYTHON: "py -3.13" } })[0],
    { command: "py", prefixArgs: ["-3.13"] },
  );
  assert.deepEqual(
    pythonCandidates({ platform: "win32", env: { PYTHON: "C:\\Program Files\\Python313\\python.exe" } })[0],
    { command: "C:\\Program Files\\Python313\\python.exe", prefixArgs: [] },
  );
});

test("pipeline readiness is versioned and invalidates changed setup inputs", () => {
  const root = mkdtempSync(join(tmpdir(), "sona-ready-"));
  try {
    seedPipelineInputs(root);
    const python = join(root, ".venv", "bin", "python");
    const marker = join(root, ".venv", "sona-ready.json");
    mkdirSync(join(root, ".venv", "bin"), { recursive: true });
    writeFileSync(python, "python");

    assert.equal(pipelineReadiness({ root, python, marker, checkCommands: false }).ready, false);
    markPipelineReady(marker, root);
    assert.equal(pipelineReadiness({ root, python, marker, checkCommands: false }).ready, true);
    writeFileSync(join(root, "requirements-pipeline.txt"), "changed");
    assert.equal(pipelineReadiness({ root, python, marker, checkCommands: false }).ready, false);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test("hosted builds reject local media, contours, and model weights", () => {
  const root = mkdtempSync(join(tmpdir(), "sona-build-"));
  try {
    mkdirSync(join(root, "public", "audio"), { recursive: true });
    mkdirSync(join(root, "public", "data"), { recursive: true });
    writeFileSync(join(root, "public", "library.json"), '{"songs":[]}');
    assert.deepEqual(unsafePublicAssets(root), []);

    writeFileSync(join(root, "public", "audio", "song.mp3"), "private");
    writeFileSync(join(root, "public", "data", "song-contour.json"), "{}");
    assert.deepEqual(
      unsafePublicAssets(root).map((path) => path.slice(root.length + 1)),
      ["public/audio/song.mp3", "public/data/song-contour.json"],
    );

    mkdirSync(join(root, "dist", "client", "audio"), { recursive: true });
    writeFileSync(join(root, "dist", "client", "audio", "song.mp3"), "private");
    assert.deepEqual(
      unsafeHostedOutput(root).map((path) => path.slice(root.length + 1)),
      ["dist/client/audio/song.mp3"],
    );
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test("hosted builds reject symlinks instead of following local targets", (context) => {
  if (process.platform === "win32") {
    context.skip("creating symlinks requires optional Windows privileges");
    return;
  }
  const root = mkdtempSync(join(tmpdir(), "sona-build-link-"));
  try {
    const privateRoot = join(root, "private-library");
    mkdirSync(join(root, "public"), { recursive: true });
    mkdirSync(privateRoot, { recursive: true });
    writeFileSync(join(privateRoot, "song.mp3"), "private");
    symlinkSync(privateRoot, join(root, "public", "linked-library"));

    assert.deepEqual(
      unsafePublicAssets(root).map((path) => path.slice(root.length + 1)),
      ["public/linked-library"],
    );
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});
