import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, test } from "node:test";

const dataRoot = mkdtempSync(join(tmpdir(), "sona-companion-"));
process.env.SONA_DATA_DIR = dataRoot;
const { server } = await import(`../server/pipeline-server.mjs?test=${Date.now()}`);
let base;
let listenError = null;

const allowedOrigin = "http://127.0.0.1:3000";
const request = (path, init = {}) => fetch(`${base}${path}`, {
  ...init,
  headers: { origin: allowedOrigin, ...init.headers },
});

try {
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  const address = server.address();
  base = `http://127.0.0.1:${address.port}`;

  const libraryRoot = join(dataRoot, "library");
  mkdirSync(join(libraryRoot, "audio"), { recursive: true });
  mkdirSync(join(libraryRoot, "data"), { recursive: true });
  writeFileSync(join(libraryRoot, "library.json"), JSON.stringify({
    songs: [{
      id: "test-song",
      title: "Test Song",
      artist: "Singer",
      duration: 12,
      internalPath: "/private/do-not-expose",
    }],
  }));
  writeFileSync(join(libraryRoot, "audio", "test-song.mp3"), "0123456789");
  writeFileSync(join(libraryRoot, "audio", "test-song-instrumental.mp3"), "abcdefghij");
  writeFileSync(join(libraryRoot, "data", "test-song-contour.json"), "{\"segments\":[]}");
  writeFileSync(join(libraryRoot, "data", "test-song-pyin.json"), "{\"segments\":[]}");
} catch (error) {
  listenError = error;
}

if (listenError && process.env.CI) throw listenError;

const networkOptions = listenError ? { skip: `loopback unavailable: ${listenError.code || listenError}` } : {};

after(async () => {
  if (server.listening) await new Promise((resolve) => server.close(resolve));
  rmSync(dataRoot, { recursive: true, force: true });
});

test("reports companion and processing readiness separately", networkOptions, async () => {
  const response = await request("/api/health");
  assert.equal(response.status, 200);
  assert.equal(response.headers.get("access-control-allow-origin"), allowedOrigin);
  const health = await response.json();
  assert.equal(health.ok, true);
  assert.equal(typeof health.ready, "boolean");
  assert.equal(typeof health.setup.status, "string");
});

test("answers allowed local-network preflight without reading a body", networkOptions, async () => {
  const response = await request("/api/songs", {
    method: "OPTIONS",
    headers: {
      "access-control-request-method": "POST",
      "access-control-request-headers": "content-type",
      "access-control-request-private-network": "true",
    },
  });
  assert.equal(response.status, 204);
  assert.equal(response.headers.get("access-control-allow-origin"), allowedOrigin);
  assert.equal(response.headers.get("access-control-allow-private-network"), "true");
  assert.match(response.headers.get("access-control-allow-methods") ?? "", /POST/);
});

test("returns a sanitized private library", networkOptions, async () => {
  const response = await request("/api/library");
  assert.equal(response.status, 200);
  const manifest = await response.json();
  assert.deepEqual(manifest.songs, [{
    id: "test-song",
    title: "Test Song",
    artist: "Singer",
    duration: 12,
  }]);
  assert.doesNotMatch(JSON.stringify(manifest), /private|internalPath/);
});

test("streams full, ranged, suffix, and HEAD audio responses", networkOptions, async () => {
  const full = await request("/api/songs/test-song/source");
  assert.equal(full.status, 200);
  assert.equal(full.headers.get("accept-ranges"), "bytes");
  assert.equal(await full.text(), "0123456789");

  const bounded = await request("/api/songs/test-song/source", {
    headers: { range: "bytes=2-5" },
  });
  assert.equal(bounded.status, 206);
  assert.equal(bounded.headers.get("content-range"), "bytes 2-5/10");
  assert.equal(await bounded.text(), "2345");

  const suffix = await request("/api/songs/test-song/source", {
    headers: { range: "bytes=-4" },
  });
  assert.equal(suffix.status, 206);
  assert.equal(await suffix.text(), "6789");

  const head = await request("/api/songs/test-song/instrumental", { method: "HEAD" });
  assert.equal(head.status, 200);
  assert.equal(head.headers.get("content-length"), "10");
  assert.equal(await head.text(), "");
});

test("returns 416 and CORS headers for invalid ranges", networkOptions, async () => {
  const response = await request("/api/songs/test-song/source", {
    headers: { range: "bytes=99-" },
  });
  assert.equal(response.status, 416);
  assert.equal(response.headers.get("content-range"), "bytes */10");
  assert.equal(response.headers.get("access-control-allow-origin"), allowedOrigin);
});

test("rejects traversal, missing assets, hostile origins, and simple upload CSRF", networkOptions, async () => {
  assert.equal((await request("/api/songs/%2e%2e/source")).status, 404);
  assert.equal((await request("/api/songs/test-song/unknown")).status, 404);

  const hostile = await fetch(`${base}/api/library`, {
    headers: { origin: "https://attacker.example" },
  });
  assert.equal(hostile.status, 403);
  assert.equal(hostile.headers.get("access-control-allow-origin"), null);

  const upload = await fetch(`${base}/api/songs?title=Bad`, {
    method: "POST",
    headers: { origin: "https://attacker.example", "content-type": "text/plain" },
    body: "x".repeat(2048),
  });
  assert.equal(upload.status, 403);
  assert.deepEqual(readdirSync(join(dataRoot, "uploads")), []);
});
