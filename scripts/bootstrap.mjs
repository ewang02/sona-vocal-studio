#!/usr/bin/env node

/**
 * Dispatch `npm run setup` to the right platform bootstrap script.
 *
 * This only helps once Node is already installed. On a machine with no Node at
 * all, run the platform script directly instead:
 *
 *   macOS/Linux   ./scripts/bootstrap.sh
 *   Windows       powershell -ExecutionPolicy Bypass -File scripts\bootstrap.ps1
 */

import { spawnSync } from "node:child_process";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const scripts = dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);

const { command, commandArgs } = process.platform === "win32"
  ? {
      command: "powershell",
      commandArgs: [
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        join(scripts, "bootstrap.ps1"),
        ...args,
      ],
    }
  : { command: "bash", commandArgs: [join(scripts, "bootstrap.sh"), ...args] };

const result = spawnSync(command, commandArgs, { stdio: "inherit" });
if (result.error) {
  console.error(`Could not run the setup script: ${result.error.message}`);
  process.exit(1);
}
process.exit(result.status ?? 1);
