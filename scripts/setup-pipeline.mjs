#!/usr/bin/env node

/** Create the optional local Python environment used by the Sona companion. */

import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import {
  ACCELERATED_ONNX_PROVIDERS,
  selectInferenceBackend,
} from "./select-inference-backend.mjs";
import { ensureModelAssets, selectedSeparatorProfile } from "./model-assets.mjs";
import {
  clearPipelineReady,
  markPipelineReady,
} from "./pipeline-state.mjs";

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

export function pythonCandidates({ platform = process.platform, env = process.env } = {}) {
  const configured = [];
  if (env.PYTHON) {
    // Older versions of bootstrap.ps1 used the string "py -3.13". Preserve
    // that upgrade path without treating a general space-containing path as a
    // shell command.
    const launcher = /^(py(?:\.exe)?)\s+(-\d+(?:\.\d+)?)$/i.exec(env.PYTHON.trim());
    configured.push(
      launcher
        ? { command: launcher[1], prefixArgs: [launcher[2]] }
        : {
            command: env.PYTHON,
            prefixArgs: env.PYTHON_ARGS ? env.PYTHON_ARGS.split(/\s+/).filter(Boolean) : [],
          },
    );
  }
  return [
    ...configured,
    ...(platform === "win32" ? [{ command: "py", prefixArgs: ["-3.13"] }] : []),
    { command: "python3.13", prefixArgs: [] },
    { command: "python3", prefixArgs: [] },
    { command: "python", prefixArgs: [] },
  ];
}

export function findPython(options = {}) {
  for (const candidate of pythonCandidates(options)) {
    const result = spawnSync(
      candidate.command,
      [
        ...candidate.prefixArgs,
        "-c",
        "import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 13) else 1)",
      ],
      { encoding: "utf8", windowsHide: true },
    );
    if (result.status === 0) return candidate;
  }
  throw new Error("Python 3.13 or newer was not found. Run `npm run setup` and retry.");
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

export async function setupPipeline() {
  try {
    clearPipelineReady();
    for (const command of ["ffmpeg", "ffprobe"]) {
      const result = spawnSync(command, ["-version"], { stdio: "ignore", windowsHide: true });
      if (result.status !== 0) {
        throw new Error(`${command} is required. Run \`npm run setup\` and retry.`);
      }
    }
    if (!existsSync(VENV_PYTHON)) {
      const python = findPython();
      run(python.command, [...python.prefixArgs, "-m", "venv", ".venv"]);
    }
    run(VENV_PYTHON, ["-m", "pip", "install", "--upgrade", "pip"]);
    run(VENV_PYTHON, ["-m", "pip", "install", "-r", "requirements-pipeline.txt"]);
    installInferenceBackend();
    // Decided after the Python environment exists: the choice depends on
    // whether the installed Torch can use this machine's accelerator.
    await ensureModelAssets(selectedSeparatorProfile(VENV_PYTHON));
    run(VENV_PYTHON, ["-c", "from work.process_song import validate_models; validate_models(); print('Pipeline models verified.')"]);
    run(VENV_PYTHON, [
      "-c",
      "import json; from work.hardware_acceleration import inference_hardware; print('Selected hardware: ' + json.dumps(inference_hardware()))",
    ]);
    markPipelineReady();
    console.log("Pipeline environment is ready. Start Sona with `npm start`.");
  } catch (error) {
    console.error(`[pipeline-setup] ${error instanceof Error ? error.message : error}`);
    process.exitCode = 1;
  }
}

const invokedPath = process.argv[1]
  ? pathToFileURL(resolve(process.argv[1])).href
  : "";
if (invokedPath === import.meta.url) {
  await setupPipeline();
}
