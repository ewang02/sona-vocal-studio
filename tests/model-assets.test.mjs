import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import {
  MODEL_MANIFEST_PATH,
  pitchModelAssets,
  profileAssets,
  readModelManifest,
  sha256File,
  verifyModelAsset,
} from "../scripts/model-assets.mjs";

test("model manifest pins every asset of both separator profiles", () => {
  const manifest = readModelManifest();
  assert.deepEqual(Object.keys(manifest.profiles).sort(), ["compact", "roformer"]);
  assert.deepEqual(
    profileAssets(manifest, "compact").map((asset) => asset.filename).sort(),
    ["5_HP-Karaoke-UVR.pth", "Kim_Inst.onnx", "Kim_Vocal_2.onnx"],
  );
  assert.deepEqual(
    profileAssets(manifest, "roformer").map((asset) => asset.filename).sort(),
    [
      "config_mel_band_roformer_karaoke_gabox.yaml",
      "mel_band_roformer_karaoke_gabox.ckpt",
      "vocals_mel_band_roformer.ckpt",
      "vocals_mel_band_roformer.yaml",
    ],
  );
  for (const asset of profileAssets(manifest, "compact")) {
    assert.match(asset.url, /^https:\/\/github\.com\/TRvlvr\/model_repo\//);
    assert.ok(asset.size > 50 * 1024 * 1024);
  }
  for (const asset of profileAssets(manifest, "roformer")) {
    assert.match(asset.url, /^https:\/\/github\.com\/nomadkaraoke\/python-audio-separator\//);
    // Weights are large; each one's small config is pinned alongside it so
    // nothing is fetched outside the verified path at first run.
    if (asset.filename.endsWith(".ckpt")) assert.ok(asset.size > 500 * 1024 * 1024);
    else assert.ok(asset.filename.endsWith(".yaml") && asset.size < 64 * 1024);
  }
  for (const profile of Object.keys(manifest.profiles)) {
    for (const asset of profileAssets(manifest, profile)) {
      assert.match(asset.sha256, /^[a-f0-9]{64}$/);
    }
  }
  assert.throws(() => profileAssets(manifest, "missing"), /Unknown separator profile/);
  assert.doesNotThrow(() => JSON.parse(readFileSync(MODEL_MANIFEST_PATH, "utf8")));
});

test("only pitch models that ship as a download list assets", () => {
  const manifest = readModelManifest();
  const rmvpe = pitchModelAssets(manifest, "rmvpe");
  assert.deepEqual(rmvpe.map((asset) => asset.filename), ["rmvpe.onnx"]);
  assert.match(rmvpe[0].url, /^https:\/\/huggingface\.co\/lj1995\/VoiceConversionWebUI\//);
  assert.match(rmvpe[0].sha256, /^[a-f0-9]{64}$/);
  assert.ok(rmvpe[0].size > 100 * 1024 * 1024);
  // CREPE ships with its Python package and SwiftF0 inside its own.
  assert.deepEqual(pitchModelAssets(manifest, "crepe"), []);
  assert.deepEqual(pitchModelAssets(manifest, "swiftf0"), []);
  assert.deepEqual(pitchModelAssets(manifest, "unknown"), []);
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
