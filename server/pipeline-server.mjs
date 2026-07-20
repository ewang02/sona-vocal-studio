/**
 * Local companion server for the Sona karaoke app.
 *
 * The app is served by the Cloudflare Worker runtime (vinext), which cannot
 * spawn Python or run the separator/CREPE toolchain. This tiny Node server runs ALONGSIDE
 * `npm run dev` and does the heavy lifting: it accepts an uploaded mp3, runs
 * the real offline pipeline (work/process_song.py) as a background job, writes
 * the resulting assets into public/, and appends the song to public/library.json
 * so the running dev server serves it with no rebuild.
 *
 * Endpoints (CORS-open for localhost):
 *   POST /api/songs?title=…&artist=…   body = raw mp3 bytes  → { jobId, songId }
 *   GET  /api/jobs/:id                  → { status, step, steps, error, songId }
 *   GET  /api/library                   → the manifest (convenience/debug)
 *   GET  /api/health                    → { ok: true }
 *
 * Run:  npm run pipeline      (defaults to port 4599; override with PORT=…)
 */

import { spawn } from "node:child_process";
import { createHash, randomUUID } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { applyPipelineCors } from "./pipeline-cors.mjs";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const PORT = Number(process.env.PORT || 4599);
const MAX_UPLOAD_BYTES = 80 * 1024 * 1024; // generous ceiling for a single song
const RESERVED_IDS = new Set();
const MANIFEST_PATH = join(ROOT, "public", "library.json");
const PUBLIC_DATA = join(ROOT, "public", "data");
const PUBLIC_AUDIO = join(ROOT, "public", "audio");
// Ordered pipeline steps the wrapper emits, so the client can show N/total.
const STEP_ORDER = ["separating", "isolating", "tracking", "transcribing", "finalizing"];

const unixVenvPython = join(ROOT, ".venv", "bin", "python");
const windowsVenvPython = join(ROOT, ".venv", "Scripts", "python.exe");
const pythonBin = existsSync(unixVenvPython)
  ? unixVenvPython
  : existsSync(windowsVenvPython)
    ? windowsVenvPython
    : process.platform === "win32"
      ? "python"
      : "python3";

/** @type {Map<string, any>} */
const jobs = new Map();
const queue = [];
let processing = false;

// ---------------------------------------------------------------------------
// Manifest helpers (server owns public/library.json)
// ---------------------------------------------------------------------------

function readManifest() {
  try {
    return JSON.parse(readFileSync(MANIFEST_PATH, "utf8"));
  } catch {
    return { songs: [] };
  }
}

function writeManifest(manifest) {
  mkdirSync(dirname(MANIFEST_PATH), { recursive: true });
  const temporary = `${MANIFEST_PATH}.tmp`;
  writeFileSync(temporary, `${JSON.stringify(manifest, null, 2)}\n`);
  renameSync(temporary, MANIFEST_PATH);
}

function slugify(title) {
  const base = String(title)
    .toLowerCase()
    .normalize("NFKD")
    .replace(/[̀-ͯ]/g, "") // strip combining diacritics
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 40);
  return base || "song";
}

function sha256Bytes(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function matchesIncompleteUpload(songId, body) {
  try {
    return sha256Bytes(readFileSync(join(ROOT, `${songId}.mp3`))) === sha256Bytes(body);
  } catch {
    return false;
  }
}

function uniqueId(title, body) {
  const taken = new Set([
    ...RESERVED_IDS,
    ...readManifest().songs.map((song) => song.id),
    ...[...jobs.values()]
      .filter((job) => job.status === "queued" || job.status === "running")
      .map((job) => job.songId),
  ]);
  const base = slugify(title);
  let candidate = base;
  let suffix = 2;
  while (true) {
    const hasPartialOutput =
      existsSync(join(PUBLIC_DATA, `${candidate}-contour.json`)) ||
      existsSync(join(PUBLIC_AUDIO, `${candidate}.mp3`));
    if (!taken.has(candidate) && (!hasPartialOutput || matchesIncompleteUpload(candidate, body))) {
      return candidate;
    }
    candidate = `${base}-${suffix++}`;
  }
}

function upsertManifestSong(entry) {
  const manifest = readManifest();
  manifest.songs = manifest.songs.filter((song) => song.id !== entry.id);
  manifest.songs.push(entry);
  writeManifest(manifest);
}

// ---------------------------------------------------------------------------
// Job queue — one song at a time (torchcrepe caches model state; parallel runs
// multiply memory), matching work/build_final_transcriptions.py's design.
// ---------------------------------------------------------------------------

function enqueue(job) {
  jobs.set(job.jobId, job);
  queue.push(job);
  pump();
}

function pump() {
  if (processing || queue.length === 0) return;
  processing = true;
  const job = queue.shift();
  runJob(job).finally(() => {
    processing = false;
    pump();
  });
}

function runJob(job) {
  return new Promise((resolvePromise) => {
    job.status = "running";
    const child = spawn(pythonBin, ["work/process_song.py", job.songId], {
      cwd: ROOT,
      env: process.env,
    });
    let stdoutTail = "";
    let stdoutBuffer = "";
    let stderrTail = "";

    const handleLine = (line) => {
      const text = line.trim();
      if (text.startsWith("::step:")) {
        job.step = text.slice("::step:".length);
        if (!job.steps.includes(job.step)) job.steps.push(job.step);
      } else if (text.startsWith("::done")) {
        const match = text.match(/duration=([\d.]+)/);
        job.duration = match ? Number(match[1]) : 0;
      } else if (text.startsWith("::cache:")) {
        const match = text.match(/^::cache:(hit|miss) stage=([^\s]+)/);
        if (match) job.cache[match[2]] = match[1];
      } else if (text.startsWith("::timing")) {
        const match = text.match(/stage=([^\s]+) seconds=([\d.]+)/);
        if (match) job.timings[match[1]] = Number(match[2]);
      } else if (text.startsWith("::error")) {
        job.error = text.slice("::error".length).trim() || "pipeline error";
      }
    };

    child.stdout.on("data", (chunk) => {
      stdoutTail = (stdoutTail + chunk).split("\n").slice(-40).join("\n");
      stdoutBuffer += chunk.toString();
      const lines = stdoutBuffer.split(/\r?\n/);
      stdoutBuffer = lines.pop() || "";
      for (const line of lines) if (line) handleLine(line);
      process.stdout.write(chunk);
    });
    child.stderr.on("data", (chunk) => {
      stderrTail = (stderrTail + chunk).split("\n").slice(-40).join("\n");
      process.stderr.write(chunk);
    });

    child.on("error", (error) => {
      job.status = "error";
      job.error = job.error || String(error.message || error);
      resolvePromise();
    });

    child.on("close", (code) => {
      if (stdoutBuffer) handleLine(stdoutBuffer);
      if (code === 0 && !job.error) {
        upsertManifestSong({
          id: job.songId,
          title: job.title,
          artist: job.artist,
          duration: job.duration || 0,
          audio: `/audio/${job.songId}.mp3`,
          instrumental: `/audio/${job.songId}-instrumental.mp3`,
          contourUrl: `/data/${job.songId}-contour.json`,
          createdAt: new Date().toISOString(),
        });
        job.status = "done";
        job.step = "done";
      } else {
        job.status = "error";
        job.error =
          job.error ||
          stderrTail.split("\n").filter(Boolean).pop() ||
          `pipeline exited with code ${code}`;
      }
      resolvePromise();
    });
  });
}

// ---------------------------------------------------------------------------
// HTTP plumbing
// ---------------------------------------------------------------------------

function sendJson(request, response, status, body) {
  applyPipelineCors(request, response);
  response.writeHead(status, { "Content-Type": "application/json" });
  response.end(JSON.stringify(body));
}

function readBody(request, limit) {
  return new Promise((resolvePromise, reject) => {
    const chunks = [];
    let size = 0;
    request.on("data", (chunk) => {
      size += chunk.length;
      if (size > limit) {
        reject(new Error("upload too large"));
        request.destroy();
        return;
      }
      chunks.push(chunk);
    });
    request.on("end", () => resolvePromise(Buffer.concat(chunks)));
    request.on("error", reject);
  });
}

async function handleUpload(request, response, url) {
  const title = (url.searchParams.get("title") || "").trim();
  const artist = (url.searchParams.get("artist") || "").trim();
  if (!title) return sendJson(request, response, 400, { error: "A title is required." });

  let body;
  try {
    body = await readBody(request, MAX_UPLOAD_BYTES);
  } catch (error) {
    return sendJson(request, response, 413, { error: String(error.message || error) });
  }
  if (body.length < 1024) {
    return sendJson(request, response, 400, { error: "No audio was uploaded." });
  }

  const songId = uniqueId(title, body);
  mkdirSync(PUBLIC_AUDIO, { recursive: true });
  mkdirSync(PUBLIC_DATA, { recursive: true });
  writeFileSync(join(ROOT, `${songId}.mp3`), body);

  const job = {
    jobId: randomUUID(),
    songId,
    title,
    artist,
    status: "queued",
    step: null,
    steps: [],
    stepOrder: STEP_ORDER,
    duration: 0,
    cache: {},
    timings: {},
    error: null,
  };
  enqueue(job);
  sendJson(request, response, 202, { jobId: job.jobId, songId });
}

const server = createServer(async (request, response) => {
  const url = new URL(request.url, `http://localhost:${PORT}`);
  if (request.method === "OPTIONS") {
    applyPipelineCors(request, response);
    response.writeHead(204);
    return response.end();
  }

  if (request.method === "POST" && url.pathname === "/api/songs") {
    return handleUpload(request, response, url);
  }
  if (request.method === "GET" && url.pathname.startsWith("/api/jobs/")) {
    const job = jobs.get(url.pathname.slice("/api/jobs/".length));
    if (!job) return sendJson(request, response, 404, { error: "unknown job" });
    const {
      jobId, songId, title, artist, status, step, steps, stepOrder,
      duration, cache, timings, error,
    } = job;
    return sendJson(request, response, 200, {
      jobId, songId, title, artist, status, step, steps, stepOrder,
      duration, cache, timings, error,
    });
  }
  if (request.method === "GET" && url.pathname === "/api/library") {
    return sendJson(request, response, 200, readManifest());
  }
  if (request.method === "GET" && url.pathname === "/api/health") {
    return sendJson(request, response, 200, { ok: true });
  }
  sendJson(request, response, 404, { error: "not found" });
});

if (!existsSync(MANIFEST_PATH)) writeManifest({ songs: [] });

server.listen(PORT, () => {
  console.log(`[sona] pipeline server on http://localhost:${PORT}  (python: ${pythonBin})`);
});
