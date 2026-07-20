#!/usr/bin/env node

/** Download the public separator weights without committing them to Git. */

import { createHash } from "node:crypto";
import {
  createReadStream,
  createWriteStream,
  existsSync,
  mkdirSync,
  readFileSync,
  renameSync,
  rmSync,
} from "node:fs";
import { dirname, join, resolve } from "node:path";
import { Readable, Transform } from "node:stream";
import { pipeline } from "node:stream/promises";
import { fileURLToPath, pathToFileURL } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
export const MODEL_DIRECTORY = join(
  ROOT,
  "contour_out",
  "models",
  "audio-separator",
);
export const MODEL_MANIFEST_PATH = join(MODEL_DIRECTORY, "model-assets.json");

export function readModelManifest(path = MODEL_MANIFEST_PATH) {
  const manifest = JSON.parse(readFileSync(path, "utf8"));
  if (manifest.version !== 1 || !Array.isArray(manifest.assets)) {
    throw new Error("Unsupported separator model manifest.");
  }
  for (const asset of manifest.assets) {
    if (
      typeof asset.filename !== "string" ||
      !/^[A-Za-z0-9_.-]+$/.test(asset.filename) ||
      asset.filename === "." ||
      asset.filename === ".." ||
      typeof asset.url !== "string" ||
      !asset.url.startsWith("https://") ||
      !Number.isSafeInteger(asset.size) ||
      asset.size <= 0 ||
      !/^[a-f0-9]{64}$/.test(asset.sha256)
    ) {
      throw new Error(`Invalid separator model manifest entry: ${asset.filename}`);
    }
  }
  return manifest;
}

export async function sha256File(path) {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest("hex");
}

export async function verifyModelAsset(asset, directory = MODEL_DIRECTORY) {
  const path = join(directory, asset.filename);
  if (!existsSync(path)) return false;
  const hash = await sha256File(path);
  return hash === asset.sha256;
}

async function downloadModelAsset(asset, directory = MODEL_DIRECTORY) {
  mkdirSync(directory, { recursive: true });
  const destination = join(directory, asset.filename);
  const temporary = `${destination}.download`;
  rmSync(temporary, { force: true });
  console.log(
    `[pipeline-setup] downloading ${asset.filename} (${(asset.size / 1024 / 1024).toFixed(1)} MiB)`,
  );

  const response = await fetch(asset.url, {
    redirect: "follow",
    signal: AbortSignal.timeout(30 * 60 * 1000),
  });
  if (!response.ok || !response.body) {
    throw new Error(`download of ${asset.filename} returned HTTP ${response.status}`);
  }

  const hash = createHash("sha256");
  let bytes = 0;
  let nextProgress = 25;
  const meter = new Transform({
    transform(chunk, _encoding, callback) {
      bytes += chunk.length;
      hash.update(chunk);
      const percent = Math.floor((bytes / asset.size) * 100);
      if (percent >= nextProgress && nextProgress < 100) {
        console.log(`[pipeline-setup] ${asset.filename}: ${percent}%`);
        nextProgress += 25;
      }
      callback(null, chunk);
    },
  });

  try {
    await pipeline(
      Readable.fromWeb(response.body),
      meter,
      createWriteStream(temporary, { mode: 0o644 }),
    );
    const actualHash = hash.digest("hex");
    if (bytes !== asset.size) {
      throw new Error(
        `${asset.filename} size mismatch: expected ${asset.size}, received ${bytes}`,
      );
    }
    if (actualHash !== asset.sha256) {
      throw new Error(
        `${asset.filename} hash mismatch: expected ${asset.sha256}, received ${actualHash}`,
      );
    }
    rmSync(destination, { force: true });
    renameSync(temporary, destination);
  } finally {
    rmSync(temporary, { force: true });
  }
  console.log(`[pipeline-setup] verified ${asset.filename}`);
}

export async function ensureModelAssets(directory = MODEL_DIRECTORY) {
  const manifest = readModelManifest();
  for (const asset of manifest.assets) {
    if (await verifyModelAsset(asset, directory)) {
      console.log(`[pipeline-setup] ${asset.filename} is present and verified.`);
      continue;
    }
    await downloadModelAsset(asset, directory);
  }
}

const invokedPath = process.argv[1]
  ? pathToFileURL(resolve(process.argv[1])).href
  : "";
if (invokedPath === import.meta.url) {
  try {
    await ensureModelAssets();
  } catch (error) {
    console.error(
      `[model-assets] ${error instanceof Error ? error.message : error}`,
    );
    process.exitCode = 1;
  }
}
