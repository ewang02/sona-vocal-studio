import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import {
  MODEL_MANIFEST_PATH,
  readModelManifest,
  sha256File,
  verifyModelAsset,
} from "../scripts/model-assets.mjs";

test("model manifest contains canonical HTTPS assets and pinned hashes", () => {
  const manifest = readModelManifest();
  assert.deepEqual(
    manifest.assets.map((asset) => asset.filename).sort(),
    ["5_HP-Karaoke-UVR.pth", "Kim_Inst.onnx", "Kim_Vocal_2.onnx"],
  );
  for (const asset of manifest.assets) {
    assert.match(asset.url, /^https:\/\/github\.com\/TRvlvr\/model_repo\//);
    assert.match(asset.sha256, /^[a-f0-9]{64}$/);
    assert.ok(asset.size > 50 * 1024 * 1024);
  }
  assert.doesNotThrow(() => JSON.parse(readFileSync(MODEL_MANIFEST_PATH, "utf8")));
});

test("model verification rejects changed bytes", async () => {
  const directory = mkdtempSync(join(tmpdir(), "sona-model-test-"));
  const asset = {
    filename: "small.bin",
    sha256: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
  };
  try {
    writeFileSync(join(directory, asset.filename), "abc");
    assert.equal(await sha256File(join(directory, asset.filename)), asset.sha256);
    assert.equal(await verifyModelAsset(asset, directory), true);
    writeFileSync(join(directory, asset.filename), "changed");
    assert.equal(await verifyModelAsset(asset, directory), false);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});
