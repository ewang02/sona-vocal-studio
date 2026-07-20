import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

const root = new URL("../", import.meta.url);

async function read(path) {
  return readFile(new URL(path, root), "utf8");
}

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("https://sona.test/", {
      headers: { accept: "text/html", host: "sona.test", "x-forwarded-proto": "https" },
    }),
    {
      ASSETS: {
        fetch: async () => new Response("Not found", { status: 404 }),
      },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );
}

test("renders a content-free song library as the landing view", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>Sona — Find your note<\/title>/i);
  assert.match(html, /Choose a song to practice/);
  assert.match(html, /Add a song/);
  assert.match(html, /https:\/\/sona\.test\/og\.png/);
  assert.doesNotMatch(html, /codex-preview|Your site is taking shape/i);
});

test("ships no songs, contours, or repair sources", async () => {
  const manifest = JSON.parse(await read("public/library.json"));
  assert.deepEqual(manifest.songs, []);

  await Promise.all([
    access(new URL("public/pitch-worklet.js", root)),
    access(new URL("public/soundtouch-processor.js", root)),
    access(new URL("public/og.png", root)),
  ]);

  const songs = await read("app/songs.ts");
  assert.match(songs, /export const SONGS: Song\[\] = \[\]/);
});

test("ships the three-mode editor, practice, and recording behavior", async () => {
  const [studio, songs, pitchUtils, worklet, packageJson] = await Promise.all([
    read("app/StudioPlayer.tsx"),
    read("app/songs.ts"),
    read("app/pitch-utils.ts"),
    read("public/pitch-worklet.js"),
    read("package.json"),
  ]);

  assert.match(studio, /\["editing", "practicing", "recording"\]/);
  assert.match(studio, /applyContourEdit\("remove"\)/);
  assert.match(studio, /applyContourEdit\("smooth"\)/);
  assert.match(studio, /applyContourEdit\("repair"\)/);
  assert.match(studio, /saveEditedDefault/);
  assert.match(studio, /restoreOriginalContour/);

  assert.match(studio, /createAdaptivePitchFilter/);
  assert.match(studio, /adaptiveFilterRef\.current\.process/);
  assert.doesNotMatch(studio, /autoFilter|effectiveLaneMode|autoNotes|manualNotes/);
  assert.doesNotMatch(songs, /autoNotes|manualNotes|-auto\.json|-ref\.json/);

  assert.match(studio, /guideVocalsRef\.current = true/);
  assert.match(studio, /studioMode !== "recording" && guideVocals/);
  assert.match(studio, /instrumentCaptureRef/);
  assert.match(studio, /Raw vocals/);
  assert.match(studio, /Instrumental/);
  assert.match(studio, /Combined mix/);

  assert.match(studio, /songTranspose/);
  assert.match(studio, /contourTranspose/);
  assert.match(studio, /transpositionsLinked/);
  assert.match(studio, /const TRANSPOSE_MIN = -16/);
  assert.match(studio, /const TRANSPOSE_MAX = 16/);
  assert.match(studio, /studioMode === "practicing" \|\| studioMode === "recording"/);
  assert.match(studio, /sona\.micNudgeMs/);

  assert.match(studio, /livePitchSampleTimeRef/);
  assert.match(studio, /outputLatencyRef/);
  assert.match(studio, /captureLatencyRef/);
  assert.match(studio, /echoCancellation: false/);
  assert.doesNotMatch(studio, /detail\?\.framesBuffered/);
  assert.match(worklet, /analysisLatency/);
  assert.match(worklet, /contextTimeOrigin/);
  assert.match(worklet, /CONTINUITY_MEMORY_S/);
  assert.match(pitchUtils, /UPWARD_OCTAVE_FOLD_LIMIT/);
  assert.match(worklet, /registerProcessor\(["']sona-pitch-processor["']/);

  assert.match(packageJson, /@soundtouchjs\/audio-worklet/);
  assert.match(packageJson, /"pipeline"/);
  assert.doesNotMatch(packageJson, /drizzle|react-loading-skeleton/);
});
