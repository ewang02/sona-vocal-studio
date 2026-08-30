/** Versioned readiness marker shared by setup and the lightweight companion. */

import { createHash } from "node:crypto";
import { existsSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
export const VENV_PYTHON =
  process.platform === "win32"
    ? join(ROOT, ".venv", "Scripts", "python.exe")
    : join(ROOT, ".venv", "bin", "python");
export const PIPELINE_READY_PATH = join(ROOT, ".venv", "sona-ready.json");

const INPUTS = [
  "requirements-pipeline.txt",
  "scripts/setup-pipeline.mjs",
  "scripts/select-inference-backend.mjs",
  "contour_out/models/audio-separator/model-assets.json",
];

export function pipelineFingerprint(root = ROOT) {
  const hash = createHash("sha256");
  hash.update(`${process.platform}:${process.arch}\n`);
  for (const relative of INPUTS) {
    hash.update(`${relative}\0`);
    hash.update(readFileSync(join(root, relative)));
    hash.update("\0");
  }
  return hash.digest("hex");
}

export function clearPipelineReady(path = PIPELINE_READY_PATH) {
  rmSync(path, { force: true });
}

export function markPipelineReady(path = PIPELINE_READY_PATH, root = ROOT) {
  writeFileSync(
    path,
    `${JSON.stringify({ version: 1, fingerprint: pipelineFingerprint(root) }, null, 2)}\n`,
    { mode: 0o600 },
  );
}

function commandAvailable(command) {
  return spawnSync(command, ["-version"], {
    stdio: "ignore",
    timeout: 5_000,
    windowsHide: true,
  }).status === 0;
}

export function pipelineReadiness({
  root = ROOT,
  python = VENV_PYTHON,
  marker = PIPELINE_READY_PATH,
  checkCommands = true,
} = {}) {
  if (!existsSync(python)) {
    return { ready: false, reason: "Song processing has not been installed yet." };
  }
  let saved;
  try {
    saved = JSON.parse(readFileSync(marker, "utf8"));
  } catch {
    return { ready: false, reason: "Song processing needs to finish setup." };
  }
  let expectedFingerprint;
  try {
    expectedFingerprint = pipelineFingerprint(root);
  } catch {
    return { ready: false, reason: "Song-processing files are incomplete." };
  }
  if (saved?.version !== 1 || saved.fingerprint !== expectedFingerprint) {
    return { ready: false, reason: "Song processing needs an update." };
  }
  if (checkCommands && (!commandAvailable("ffmpeg") || !commandAvailable("ffprobe"))) {
    return { ready: false, reason: "FFmpeg and FFprobe are required." };
  }
  return { ready: true, reason: null };
}
