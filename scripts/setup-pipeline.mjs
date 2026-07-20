#!/usr/bin/env node

/** Create the local Python environment used by `npm run pipeline`. */

import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import {
  ACCELERATED_ONNX_PROVIDERS,
  selectInferenceBackend,
} from "./select-inference-backend.mjs";
import { ensureModelAssets } from "./model-assets.mjs";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const VENV_PYTHON =
  process.platform === "win32"
    ? join(ROOT, ".venv", "Scripts", "python.exe")
    : join(ROOT, ".venv", "bin", "python");

function run(command, args, options = {}) {
  const result = spawnSync(command, args, {
    cwd: ROOT,
    stdio: "inherit",
    ...options,
  });
  if (result.status !== 0) {
    throw new Error(`${command} ${args.join(" ")} failed with exit code ${result.status}`);
  }
}

function runResult(command, args, options = {}) {
  return spawnSync(command, args, {
    cwd: ROOT,
    encoding: "utf8",
    ...options,
  });
}

function findPython() {
  const candidates = [process.env.PYTHON, "python3", "python"].filter(Boolean);
  for (const candidate of candidates) {
    const result = spawnSync(candidate, ["--version"], { encoding: "utf8" });
    if (result.status === 0) return candidate;
  }
  throw new Error("Python 3 was not found on PATH. Install Python 3.13 and retry.");
}

function nvidiaAvailable() {
  return runResult("nvidia-smi", ["--query-gpu=name", "--format=csv,noheader"]).status === 0;
}

function installedOnnxProviders() {
  const result = runResult(VENV_PYTHON, [
    "-c",
    "import json, onnxruntime as ort; print(json.dumps(ort.get_available_providers()))",
  ]);
  if (result.status !== 0) return [];
  try {
    return JSON.parse(result.stdout.trim());
  } catch {
    return [];
  }
}

function installInferenceBackend() {
  const requested = process.env.SONA_ACCELERATOR || "auto";
  const existing = installedOnnxProviders();
  if (
    requested.trim().toLowerCase() === "auto" &&
    existing.some((provider) => ACCELERATED_ONNX_PROVIDERS.has(provider))
  ) {
    console.log(`[pipeline-setup] preserving accelerated ONNX providers: ${existing.join(", ")}`);
    return;
  }

  const selection = selectInferenceBackend({
    platform: process.platform,
    requested,
    nvidiaAvailable: nvidiaAvailable(),
  });
  const alternatives = ["onnxruntime", "onnxruntime-gpu", "onnxruntime-directml"]
    .filter((name) => name !== selection.packageName);
  run(VENV_PYTHON, ["-m", "pip", "uninstall", "-y", ...alternatives]);

  const requirement = `${selection.packageName}==${selection.version}`;
  const installed = runResult(
    VENV_PYTHON,
    ["-m", "pip", "install", "--upgrade", requirement],
    { stdio: "inherit" },
  );
  if (installed.status !== 0) {
    if (selection.backend === "cpu" || selection.backend === "coreml") {
      throw new Error(`Could not install ${requirement}`);
    }
    console.warn(
      `[pipeline-setup] ${selection.backend} runtime installation failed; falling back to CPU.`,
    );
    run(VENV_PYTHON, ["-m", "pip", "uninstall", "-y", selection.packageName]);
    run(VENV_PYTHON, ["-m", "pip", "install", "--upgrade", "onnxruntime==1.27.0"]);
  }

  const providers = installedOnnxProviders();
  console.log(`[pipeline-setup] ONNX providers: ${providers.join(", ") || "unavailable"}`);
}

try {
  if (!existsSync(VENV_PYTHON)) {
    run(findPython(), ["-m", "venv", ".venv"]);
  }
  run(VENV_PYTHON, ["-m", "pip", "install", "--upgrade", "pip"]);
  run(VENV_PYTHON, ["-m", "pip", "install", "-r", "requirements-pipeline.txt"]);
  installInferenceBackend();
  await ensureModelAssets();
  for (const command of ["ffmpeg", "ffprobe"]) {
    const result = spawnSync(command, ["-version"], { stdio: "ignore" });
    if (result.status !== 0) {
      throw new Error(`${command} is required by the song pipeline but was not found on PATH.`);
    }
  }
  run(VENV_PYTHON, ["-c", "from work.process_song import validate_models; validate_models(); print('Pipeline models verified.')"]);
  run(VENV_PYTHON, [
    "-c",
    "import json; from work.hardware_acceleration import inference_hardware; print('Selected hardware: ' + json.dumps(inference_hardware()))",
  ]);
  console.log("Pipeline environment is ready. Run `npm run pipeline` beside `npm run dev`.");
} catch (error) {
  console.error(`[pipeline-setup] ${error instanceof Error ? error.message : error}`);
  process.exitCode = 1;
}
