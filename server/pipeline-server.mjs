/**
 * Local companion server for the Sona karaoke app.
 *
 * The app is served by the Cloudflare Worker runtime (vinext), which cannot
 * spawn Python or run the separator/CREPE toolchain. This tiny Node server runs ALONGSIDE
 * `npm run dev` and does the heavy lifting: it accepts an uploaded mp3, runs
 * the real offline pipeline (work/process_song.py) as a background job, and
 * keeps the user's source audio and generated library in private local storage.
 *
 * Endpoints (CORS-limited to the Sona UI):
 *   POST /api/songs?title=…&artist=…&pitchModel=…   body = raw mp3 bytes  → { jobId, songId }
 *   POST /api/songs/:id/analysis?pitchModel=…   re-analyse a library song  → { jobId, songId }
 *   GET  /api/jobs/:id                  → { status, step, steps, error, songId }
 *   GET  /api/library                   → the local manifest
 *   GET  /api/songs/:id/:kind           → source/instrumental/contour/repair
 *   POST /api/setup                     → install the optional local ML pipeline
 *   GET  /api/setup                     → setup status
 *   GET  /api/health                    → companion + pipeline readiness
 *
 * Run:  npm run companion     (defaults to port 4599; override with PORT=…)
 */

import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import {
  createReadStream,
  existsSync,
  mkdirSync,
  rmSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { createServer } from "node:http";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import { applyPipelineCors, allowedPipelineOrigin } from "./pipeline-cors.mjs";
import {
  createLocalLibrary,
  migrateLegacyLibrary,
  parseByteRange,
  SONG_ID_PATTERN,
} from "./local-library.mjs";
import {
  ensurePitchModelAssets,
  MODEL_DIRECTORY,
  pitchModelAssets,
  readModelManifest,
} from "../scripts/model-assets.mjs";
import { pipelineReadiness } from "../scripts/pipeline-state.mjs";
import { DEFAULT_PITCH_MODEL, resolvePitchModel } from "./pitch-models.mjs";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const PORT = Number(process.env.PORT || 4599);
const HOST = "127.0.0.1";
const MAX_UPLOAD_BYTES = 80 * 1024 * 1024; // generous ceiling for a single song
const RESERVED_IDS = new Set();
const library = createLocalLibrary(process.env.SONA_DATA_DIR || join(ROOT, ".sona-data"));
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
let readiness = pipelineReadiness();
const setup = {
  status: readiness.ready ? "done" : "idle",
  message: readiness.ready ? "Song processing is ready." : readiness.reason,
  error: null,
};

// ---------------------------------------------------------------------------
// Manifest helpers (server owns the private local library)
// ---------------------------------------------------------------------------

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

function uniqueId(title) {
  const taken = new Set([
    ...RESERVED_IDS,
    ...library.readManifest().songs.map((song) => song.id),
    ...[...jobs.values()]
      .filter((job) => job.status === "queued" || job.status === "running")
      .map((job) => job.songId),
  ]);
  const base = slugify(title);
  let candidate = base;
  let suffix = 2;
  while (true) {
    const hasPartialOutput =
      existsSync(join(library.dataDir, `${candidate}-contour.json`)) ||
      existsSync(join(library.audioDir, `${candidate}.mp3`));
    if (!taken.has(candidate) && !hasPartialOutput) {
      return candidate;
    }
    candidate = `${base}-${suffix++}`;
  }
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

/** A re-analysis reads the library's own copy of the song; never delete it. */
function discardJobSource(job) {
  if (!job.keepSource) rmSync(job.uploadPath, { force: true });
}

/**
 * Fetch a pitch model's weights the first time a song asks for that model.
 * Choosing the model is the explicit request; nothing is downloaded for
 * models that are never selected.
 */
async function preparePitchModel(job) {
  const assets = pitchModelAssets(readModelManifest(), job.pitchModel);
  if (assets.every((asset) => existsSync(join(MODEL_DIRECTORY, asset.filename)))) return;
  job.step = "downloading";
  job.steps.push("downloading");
  await ensurePitchModelAssets(job.pitchModel);
}

async function runJob(job) {
  job.status = "running";
  try {
    await preparePitchModel(job);
  } catch (error) {
    job.status = "error";
    job.error = `The ${job.pitchModel} pitch model could not be downloaded: ${error.message || error}`;
    discardJobSource(job);
    return;
  }
  await new Promise((resolvePromise) => {
    const child = spawn(pythonBin, [
      "work/process_song.py",
      job.songId,
      "--mp3",
      job.uploadPath,
      "--pitch-model",
      job.pitchModel,
      "--output-dir",
      library.libraryRoot,
      // Every upload gets a fresh id, so the four full-length WAV stems a
      // song leaves behind (about 190 MB) would never be reused.
      ...(process.env.SONA_KEEP_INTERMEDIATES === "1" ? [] : ["--clean-intermediates"]),
    ], {
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
        const analyzedAt = new Date().toISOString();
        if (job.kind === "reanalyze") {
          library.updateSong(job.songId, { pitchModel: job.pitchModel, analyzedAt });
        } else {
          library.upsertSong({
            id: job.songId,
            title: job.title,
            artist: job.artist,
            duration: job.duration || 0,
            createdAt: analyzedAt,
            pitchModel: job.pitchModel,
            analyzedAt,
          });
        }
        job.status = "done";
        job.step = "done";
      } else {
        job.status = "error";
        job.error =
          job.error ||
          stderrTail.split("\n").filter(Boolean).pop() ||
          `pipeline exited with code ${code}`;
      }
      discardJobSource(job);
      resolvePromise();
    });
  });
}

// ---------------------------------------------------------------------------
// Optional first-use pipeline setup
// ---------------------------------------------------------------------------

function setupSnapshot() {
  return {
    status: setup.status,
    message: setup.message,
    error: setup.error,
  };
}

function updateSetupMessage(line) {
  const text = line.trim();
  if (!text) return;
  if (text.startsWith("[pipeline-setup]")) {
    setup.error = text.slice("[pipeline-setup]".length).trim();
  } else if (text.includes("downloading") || text.includes("verified")) {
    setup.message = "Downloading and verifying local model files…";
  } else if (text.includes("pip") || text.includes("Installing collected packages")) {
    setup.message = "Installing the local audio-processing tools…";
  } else if (text.includes("Selected hardware") || text.includes("ONNX providers")) {
    setup.message = "Checking hardware acceleration…";
  }
}

function refreshReadiness() {
  readiness = pipelineReadiness();
  if (readiness.ready && setup.status !== "running") {
    setup.status = "done";
    setup.message = "Song processing is ready.";
    setup.error = null;
  } else if (!readiness.ready && setup.status === "done") {
    setup.status = "idle";
    setup.message = readiness.reason;
    setup.error = null;
  }
  return readiness;
}

function beginSetup() {
  if (setup.status === "running") return;
  refreshReadiness();
  if (readiness.ready) {
    setup.status = "done";
    setup.message = "Song processing is ready.";
    setup.error = null;
    return;
  }

  setup.status = "running";
  setup.message = "Preparing private, on-device song processing…";
  setup.error = null;
  const child = spawn(process.execPath, ["scripts/setup-pipeline.mjs"], {
    cwd: ROOT,
    env: process.env,
    stdio: ["ignore", "pipe", "pipe"],
    windowsHide: true,
  });
  child.stdout.on("data", (chunk) => {
    process.stdout.write(chunk);
    for (const line of chunk.toString().split(/\r?\n/)) updateSetupMessage(line);
  });
  child.stderr.on("data", (chunk) => {
    process.stderr.write(chunk);
    for (const line of chunk.toString().split(/\r?\n/)) updateSetupMessage(line);
  });
  child.on("error", () => {
    setup.status = "error";
    setup.message = "Song-processing setup did not start.";
    setup.error = "Run `npm run setup` in a terminal, then try again.";
  });
  child.on("close", (code) => {
    readiness = pipelineReadiness();
    if (code === 0 && readiness.ready) {
      setup.status = "done";
      setup.message = "Song processing is ready.";
      setup.error = null;
    } else {
      setup.status = "error";
      setup.message = "Song-processing setup did not finish.";
      setup.error = setup.error || readiness.reason || "Check the companion terminal for details.";
    }
  });
}

// ---------------------------------------------------------------------------
// HTTP plumbing
// ---------------------------------------------------------------------------

function sendJson(request, response, status, body) {
  applyPipelineCors(request, response);
  response.writeHead(status, {
    "Cache-Control": "no-store",
    "Content-Type": "application/json; charset=utf-8",
    "X-Content-Type-Options": "nosniff",
  });
  response.end(JSON.stringify(body));
}

function requestOriginAllowed(request) {
  return allowedPipelineOrigin(request.headers.origin) !== null;
}

function requireBrowserOrigin(request, response) {
  if (requestOriginAllowed(request)) return true;
  sendJson(request, response, 403, { error: "This page is not allowed to use Sona's local companion." });
  return false;
}

function readBody(request, limit) {
  return new Promise((resolvePromise, reject) => {
    const chunks = [];
    let size = 0;
    let rejected = false;
    request.on("data", (chunk) => {
      if (rejected) return;
      size += chunk.length;
      if (size > limit) {
        rejected = true;
        reject(new Error("upload too large"));
        return;
      }
      chunks.push(chunk);
    });
    request.on("end", () => resolvePromise(Buffer.concat(chunks)));
    request.on("error", reject);
  });
}

async function handleUpload(request, response, url) {
  if (!requireBrowserOrigin(request, response)) return;
  if (request.headers["content-type"]?.split(";", 1)[0].trim().toLowerCase() !== "application/octet-stream") {
    return sendJson(request, response, 415, { error: "Upload an MP3 file as application/octet-stream." });
  }
  const contentLength = Number(request.headers["content-length"]);
  if (Number.isFinite(contentLength) && contentLength > MAX_UPLOAD_BYTES) {
    return sendJson(request, response, 413, { error: "upload too large" });
  }
  refreshReadiness();
  if (!readiness.ready) {
    return sendJson(request, response, 503, {
      error: readiness.reason || "Song processing is not ready.",
      code: "pipeline_not_ready",
    });
  }
  const title = (url.searchParams.get("title") || "").trim();
  const artist = (url.searchParams.get("artist") || "").trim();
  if (!title) return sendJson(request, response, 400, { error: "A title is required." });
  if (title.length > 120 || artist.length > 120) {
    return sendJson(request, response, 400, { error: "Title and artist must be 120 characters or fewer." });
  }
  const pitchModel = resolvePitchModel(url.searchParams.get("pitchModel"));
  if (!pitchModel) return sendJson(request, response, 400, { error: "Unknown pitch model." });

  let body;
  try {
    body = await readBody(request, MAX_UPLOAD_BYTES);
  } catch (error) {
    return sendJson(request, response, 413, { error: String(error.message || error) });
  }
  if (body.length < 1024) {
    return sendJson(request, response, 400, { error: "No audio was uploaded." });
  }

  const songId = uniqueId(title);
  mkdirSync(library.uploadsDir, { recursive: true });
  const uploadPath = join(library.uploadsDir, `${randomUUID()}.mp3`);
  writeFileSync(uploadPath, body, { mode: 0o600 });

  const job = {
    jobId: randomUUID(),
    kind: "upload",
    songId,
    title,
    artist,
    pitchModel,
    status: "queued",
    step: null,
    steps: [],
    stepOrder: STEP_ORDER,
    duration: 0,
    cache: {},
    timings: {},
    error: null,
    uploadPath,
  };
  enqueue(job);
  sendJson(request, response, 202, { jobId: job.jobId, songId });
}

/** Re-run pitch analysis for a song already in the library with another model. */
function handleReanalysis(request, response, url, songId) {
  if (!requireBrowserOrigin(request, response)) return;
  // A custom header forces a CORS preflight, so no cross-site form can start
  // minutes of processing.
  if (request.headers["x-sona-action"] !== "reanalyze") {
    return sendJson(request, response, 400, { error: "Missing re-analysis confirmation." });
  }
  const pitchModel = resolvePitchModel(url.searchParams.get("pitchModel"));
  if (!pitchModel) return sendJson(request, response, 400, { error: "Unknown pitch model." });
  const source = SONG_ID_PATTERN.test(songId) ? library.resolveAsset(songId, "source") : null;
  const song = library.readManifest().songs.find((entry) => entry.id === songId);
  if (!source || !song) return sendJson(request, response, 404, { error: "Song not found." });
  refreshReadiness();
  if (!readiness.ready) {
    return sendJson(request, response, 503, {
      error: readiness.reason || "Song processing is not ready.",
      code: "pipeline_not_ready",
    });
  }
  const busy = [...jobs.values()].some(
    (job) => job.songId === songId && (job.status === "queued" || job.status === "running"),
  );
  if (busy) return sendJson(request, response, 409, { error: "This song is already being processed." });

  const job = {
    jobId: randomUUID(),
    kind: "reanalyze",
    songId,
    title: String(song.title || "Untitled"),
    artist: String(song.artist || ""),
    pitchModel,
    status: "queued",
    step: null,
    steps: [],
    stepOrder: STEP_ORDER,
    duration: 0,
    cache: {},
    timings: {},
    error: null,
    uploadPath: source.path,
    keepSource: true,
  };
  enqueue(job);
  sendJson(request, response, 202, { jobId: job.jobId, songId });
}

function serveAsset(request, response, asset) {
  applyPipelineCors(request, response);
  const size = statSync(asset.path).size;
  const range = asset.ranged ? parseByteRange(request.headers.range, size) : null;
  const headers = {
    "Accept-Ranges": asset.ranged ? "bytes" : "none",
    "Cache-Control": "private, no-store",
    "Content-Type": asset.contentType,
    "X-Content-Type-Options": "nosniff",
  };
  if (range === false) {
    response.writeHead(416, {
      ...headers,
      "Content-Length": "0",
      "Content-Range": `bytes */${size}`,
    });
    return response.end();
  }
  if (range) {
    const length = range.end - range.start + 1;
    response.writeHead(206, {
      ...headers,
      "Content-Length": String(length),
      "Content-Range": `bytes ${range.start}-${range.end}/${size}`,
    });
    if (request.method === "HEAD") return response.end();
    return createReadStream(asset.path, range).pipe(response);
  }
  response.writeHead(200, { ...headers, "Content-Length": String(size) });
  if (request.method === "HEAD") return response.end();
  createReadStream(asset.path).pipe(response);
}

const server = createServer(async (request, response) => {
  const url = new URL(request.url, `http://localhost:${PORT}`);
  if (request.method === "OPTIONS") {
    if (!requestOriginAllowed(request)) {
      return sendJson(request, response, 403, { error: "Origin not allowed." });
    }
    applyPipelineCors(request, response);
    response.writeHead(204);
    return response.end();
  }

  if (request.method === "POST" && url.pathname === "/api/setup") {
    if (!requireBrowserOrigin(request, response)) return;
    if (request.headers["x-sona-action"] !== "setup") {
      return sendJson(request, response, 400, { error: "Missing setup confirmation." });
    }
    beginSetup();
    return sendJson(request, response, setup.status === "done" ? 200 : 202, {
      ready: readiness.ready,
      setup: setupSnapshot(),
    });
  }
  if (request.method === "POST" && url.pathname === "/api/songs") {
    return handleUpload(request, response, url);
  }
  const analysisMatch = /^\/api\/songs\/([a-z0-9-]+)\/analysis$/.exec(url.pathname);
  if (request.method === "POST" && analysisMatch) {
    return handleReanalysis(request, response, url, analysisMatch[1]);
  }
  if (request.method === "GET" && url.pathname.startsWith("/api/jobs/")) {
    if (!requireBrowserOrigin(request, response)) return;
    const job = jobs.get(url.pathname.slice("/api/jobs/".length));
    if (!job) return sendJson(request, response, 404, { error: "unknown job" });
    const {
      jobId, kind, songId, title, artist, pitchModel, status, step, steps, stepOrder,
      duration, cache, timings, error,
    } = job;
    return sendJson(request, response, 200, {
      jobId, kind, songId, title, artist, pitchModel, status, step, steps, stepOrder,
      duration, cache, timings, error,
    });
  }
  if (request.method === "GET" && url.pathname === "/api/library") {
    if (!requireBrowserOrigin(request, response)) return;
    const songs = library.readManifest().songs
      .filter((song) => SONG_ID_PATTERN.test(song?.id || ""))
      .map(({ id, title, artist, duration, createdAt, pitchModel, analyzedAt }) => ({
        id,
        title: String(title || "Untitled"),
        artist: String(artist || ""),
        duration: Number.isFinite(Number(duration)) ? Number(duration) : 0,
        // Songs added before the model was selectable were analysed with CREPE.
        pitchModel: resolvePitchModel(pitchModel) ?? DEFAULT_PITCH_MODEL,
        ...(createdAt ? { createdAt: String(createdAt) } : {}),
        ...(analyzedAt ? { analyzedAt: String(analyzedAt) } : {}),
      }));
    return sendJson(request, response, 200, { songs });
  }
  if (request.method === "GET" && url.pathname === "/api/setup") {
    if (!requireBrowserOrigin(request, response)) return;
    refreshReadiness();
    return sendJson(request, response, 200, {
      ready: readiness.ready,
      setup: setupSnapshot(),
    });
  }
  if ((request.method === "GET" || request.method === "HEAD") && url.pathname.startsWith("/api/songs/")) {
    if (!requireBrowserOrigin(request, response)) return;
    const match = /^\/api\/songs\/([^/]+)\/(source|instrumental|contour|repair)$/.exec(url.pathname);
    let songId = "";
    try {
      songId = match ? decodeURIComponent(match[1]) : "";
    } catch {
      songId = "";
    }
    const asset = SONG_ID_PATTERN.test(songId) && match
      ? library.resolveAsset(songId, match[2])
      : null;
    if (!asset) return sendJson(request, response, 404, { error: "Song asset not found." });
    return serveAsset(request, response, asset);
  }
  if (request.method === "GET" && url.pathname === "/api/health") {
    if (!requireBrowserOrigin(request, response)) return;
    refreshReadiness();
    return sendJson(request, response, 200, {
      ok: true,
      ready: readiness.ready,
      reason: readiness.reason,
      setup: setupSnapshot(),
    });
  }
  sendJson(request, response, 404, { error: "not found" });
});

const migrated = migrateLegacyLibrary(ROOT, library);
if (migrated) console.log(`[sona] migrated ${migrated} song(s) into private local storage`);

export { server };

const invokedPath = process.argv[1]
  ? pathToFileURL(resolve(process.argv[1])).href
  : "";
if (invokedPath === import.meta.url) {
  server.on("error", (error) => {
    if (error.code === "EADDRINUSE") {
      // Usually a companion from an earlier run that outlived its terminal.
      console.error(
        `[sona] port ${PORT} is already in use, so the companion did not start.\n` +
        "[sona] Another Sona companion is probably still running. Stop it with:\n" +
        `[sona]   npx kill-port ${PORT}      (or)  lsof -ti tcp:${PORT} | xargs kill\n` +
        "[sona] Then run npm start again.",
      );
      process.exit(1);
    }
    throw error;
  });
  server.listen(PORT, HOST, () => {
    console.log(`[sona] local companion on http://${HOST}:${PORT}`);
    console.log(`[sona] private library: ${library.libraryRoot}`);
    if (!readiness.ready) console.log(`[sona] ${readiness.reason}`);
  });
}
