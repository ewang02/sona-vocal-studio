#!/usr/bin/env node

/**
 * Run the web interface and the local processing companion together.
 *
 *   npm start            development server + pipeline
 *   npm start -- --prod  production server + pipeline (run `npm run build` first)
 *
 * Output from both is prefixed so it stays readable in one terminal. The web
 * interface stays usable if the optional companion cannot start. Ctrl-C stops
 * both, including any Python workers spawned by the companion.
 */

import { spawn } from "node:child_process";
import { createServer } from "node:net";

const production = process.argv.slice(2).some((a) => a === "--prod" || a === "--production");
const webScript = production ? "serve" : "dev";
const windows = process.platform === "win32";

const COLORS = { pipeline: "[36m", web: "[35m", dim: "[2m", reset: "[0m" };
const useColor = process.stdout.isTTY && !process.env.NO_COLOR;
const paint = (color, text) => (useColor ? `${COLORS[color]}${text}${COLORS.reset}` : text);

const WEB_PORT = 3000;
const PIPELINE_PORT = Number(process.env.PIPELINE_PORT || 4599);

/**
 * Report a busy port before spawning anything. Both halves bind fixed ports,
 * so without this the user sees a raw Node/Vite stack trace instead of the
 * one thing that matters: something is already listening.
 */
function portInUse(port, host) {
  return new Promise((resolve) => {
    const probe = createServer();
    probe.once("error", (error) => resolve(error.code === "EADDRINUSE"));
    probe.once("listening", () => probe.close(() => resolve(false)));
    probe.listen(port, host);
  });
}

async function checkPorts() {
  // The web server binds loopback via ::1; the companion binds 127.0.0.1.
  const busy = [];
  if (await portInUse(WEB_PORT, "::1")) busy.push([WEB_PORT, "web interface"]);
  if (await portInUse(PIPELINE_PORT, "127.0.0.1")) busy.push([PIPELINE_PORT, "local companion"]);
  if (!busy.length) return;

  const lines = busy.map(([port, role]) => `  port ${port} (${role})`).join("\n");
  const ports = busy.map(([port]) => port).join(" ");
  process.stderr.write(
    `\nSona cannot start because these ports are already in use:\n${lines}\n\n` +
    "This usually means another copy of Sona is still running, possibly from a\n" +
    "terminal that was closed without stopping it. Stop it with:\n\n" +
    `  lsof -ti tcp:${ports.split(" ").join(" -ti tcp:")} | xargs kill\n\n` +
    "Then run npm start again.\n\n",
  );
  process.exit(1);
}

const children = new Map();
let shuttingDown = false;
let exitCode = 0;

function log(name, line) {
  if (!line) return;
  process.stdout.write(`${paint(name === "web" ? "web" : "pipeline", `[${name}]`)} ${line}\n`);
}

/** Split a chunk into lines without losing a partial trailing line. */
function lineWriter(name) {
  let remainder = "";
  return {
    push(chunk) {
      remainder += chunk.toString();
      const lines = remainder.split(/\r?\n/);
      remainder = lines.pop() ?? "";
      for (const line of lines) log(name, line);
    },
    flush() {
      if (remainder) log(name, remainder);
      remainder = "";
    },
  };
}

function start(name, script) {
  const child = spawn(windows ? "npm.cmd" : "npm", ["run", script], {
    // Detaching on POSIX puts each child in its own process group so a single
    // signal reaches the Python and FFmpeg workers the pipeline starts.
    detached: !windows,
    env: { ...process.env, FORCE_COLOR: process.env.FORCE_COLOR ?? "1" },
    shell: windows,
    stdio: ["ignore", "pipe", "pipe"],
    windowsHide: true,
  });

  const out = lineWriter(name);
  const err = lineWriter(name);
  child.stdout?.on("data", (chunk) => out.push(chunk));
  child.stderr?.on("data", (chunk) => err.push(chunk));

  child.on("error", (error) => {
    log(name, `failed to start: ${error.message}`);
    if (name === "web") {
      exitCode = 1;
      void stop();
    } else {
      log(name, "The web interface will keep running; restart the companion with `npm run companion`.");
    }
  });

  child.on("exit", (code, signal) => {
    out.flush();
    err.flush();
    children.delete(name);
    if (shuttingDown) return;
    log(name, `exited (${signal ? `signal ${signal}` : `code ${code}`})`);
    if (name === "web") {
      if (code) exitCode = code;
      void stop();
    } else {
      log(name, "The web interface is still available without local song processing.");
    }
  });

  children.set(name, child);
  return child;
}

function terminate(child, signal) {
  if (!child || child.exitCode !== null || child.signalCode !== null) return;
  if (windows) {
    spawn("taskkill", ["/pid", String(child.pid), "/t", "/f"], {
      stdio: "ignore",
      windowsHide: true,
    }).on("error", () => {
      try {
        child.kill();
      } catch {
        // Already gone between the check and the signal.
      }
    });
    return;
  }
  try {
    process.kill(-child.pid, signal);
  } catch {
    try {
      child.kill(signal);
    } catch {
      // Already gone.
    }
  }
}

async function stop() {
  if (shuttingDown) return;
  shuttingDown = true;
  process.stdout.write(paint("dim", "\nStopping…\n"));

  for (const child of children.values()) terminate(child, "SIGTERM");

  const deadline = Date.now() + 5_000;
  while (children.size > 0 && Date.now() < deadline) {
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  for (const child of children.values()) terminate(child, "SIGKILL");

  process.exit(exitCode);
}

for (const signal of ["SIGINT", "SIGTERM"]) process.on(signal, () => void stop());

process.stdout.write(
  paint("dim", `Starting the ${production ? "production" : "development"} server and local companion.\n`) +
    paint("dim", "Press Ctrl-C to stop both.\n\n"),
);

await checkPorts();

start("pipeline", "pipeline");
start("web", webScript);
