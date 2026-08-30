import assert from "node:assert/strict";
import {
  mkdtempSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import {
  createLocalLibrary,
  parseByteRange,
  resolveSonaDataRoot,
} from "../server/local-library.mjs";

test("resolves conventional private data directories with an explicit override", () => {
  assert.equal(
    resolveSonaDataRoot({ env: { SONA_DATA_DIR: "/tmp/sona-portable" }, platform: "linux", home: "/home/me" }),
    "/tmp/sona-portable",
  );
  assert.equal(
    resolveSonaDataRoot({ env: {}, platform: "darwin", home: "/Users/me" }),
    "/Users/me/Library/Application Support/Sona",
  );
  assert.equal(
    resolveSonaDataRoot({ env: { XDG_DATA_HOME: "/data" }, platform: "linux", home: "/home/me" }),
    "/data/sona",
  );
});

test("local library serves only fixed assets for manifest-listed song ids", () => {
  const root = mkdtempSync(join(tmpdir(), "sona-library-"));
  try {
    const library = createLocalLibrary(root);
    library.writeManifest({ songs: [{ id: "my-song", title: "My Song" }] });
    writeFileSync(join(library.audioDir, "my-song.mp3"), "source");
    writeFileSync(join(library.audioDir, "my-song-instrumental.mp3"), "instrumental");
    writeFileSync(join(library.dataDir, "my-song-contour.json"), "{}");
    writeFileSync(join(library.dataDir, "my-song-pyin.json"), "{}");

    assert.equal(library.resolveAsset("my-song", "source")?.contentType, "audio/mpeg");
    assert.equal(library.resolveAsset("my-song", "repair")?.ranged, false);
    assert.equal(library.resolveAsset("other-song", "source"), null);
    assert.equal(library.resolveAsset("../my-song", "source"), null);
    assert.equal(library.resolveAsset("my-song", "unknown"), null);

    const outside = join(root, "outside.mp3");
    writeFileSync(outside, "outside");
    rmSync(join(library.audioDir, "my-song.mp3"));
    symlinkSync(outside, join(library.audioDir, "my-song.mp3"));
    assert.equal(library.resolveAsset("my-song", "source"), null);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test("parses bounded, open, and suffix byte ranges", () => {
  assert.deepEqual(parseByteRange("bytes=2-5", 10), { start: 2, end: 5 });
  assert.deepEqual(parseByteRange("bytes=6-", 10), { start: 6, end: 9 });
  assert.deepEqual(parseByteRange("bytes=-4", 10), { start: 6, end: 9 });
  assert.deepEqual(parseByteRange("bytes=8-99", 10), { start: 8, end: 9 });
  assert.equal(parseByteRange(null, 10), null);
  assert.equal(parseByteRange("bytes=10-", 10), false);
  assert.equal(parseByteRange("bytes=5-2", 10), false);
  assert.equal(parseByteRange("bytes=1-2,4-5", 10), false);
  assert.equal(parseByteRange("not-a-range", 10), false);
});
