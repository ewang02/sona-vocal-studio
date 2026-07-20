#!/usr/bin/env node

import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const python =
  process.platform === "win32"
    ? join(root, ".venv", "Scripts", "python.exe")
    : join(root, ".venv", "bin", "python");

if (!existsSync(python)) {
  console.error("Pipeline environment is missing. Run `npm run pipeline:setup` first.");
  process.exitCode = 1;
} else {
  const tests = [
    "tests/test_process_song.py",
    "tests/test_hardware_acceleration.py",
    "tests/test_export_contour_repair_source.py",
    "tests/test_pyin_fallback.py",
    "tests/test_octave_correct_contour.py",
  ];
  const result = spawnSync(python, ["-m", "pytest", ...tests], {
    cwd: root,
    stdio: "inherit",
  });
  process.exitCode = result.status ?? 1;
}
