#!/usr/bin/env node

/** Refuse to place user media or model weights in a hosted build. */

import { existsSync, lstatSync, readFileSync, readdirSync, statSync } from "node:fs";
import { dirname, extname, join, relative, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const BLOCKED_EXTENSIONS = new Set([".mp3", ".wav", ".onnx", ".pth"]);

function walk(directory) {
  if (!existsSync(directory)) return [];
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const path = join(directory, entry.name);
    return entry.isDirectory() ? walk(path) : [path];
  });
}

function isPrivateAsset(path, assetRoot) {
  const name = relative(assetRoot, path);
  const segments = name.split(/[\\/]/);
  return (
    lstatSync(path).isSymbolicLink() ||
    BLOCKED_EXTENSIONS.has(extname(path).toLowerCase()) ||
    segments.includes("audio") ||
    segments.includes("data")
  );
}

export function unsafePublicAssets(root = ROOT) {
  const publicRoot = join(root, "public");
  const unsafe = walk(publicRoot).filter((path) => isPrivateAsset(path, publicRoot));
  const manifest = join(publicRoot, "library.json");
  if (existsSync(manifest) && statSync(manifest).isFile()) {
    try {
      const value = JSON.parse(readFileSync(manifest, "utf8"));
      if (Array.isArray(value?.songs) && value.songs.length) unsafe.push(manifest);
    } catch {
      unsafe.push(manifest);
    }
  }
  return [...new Set(unsafe)].sort();
}

/** Scan the static deployment output after a build as a second safety net. */
export function unsafeHostedOutput(root = ROOT) {
  const clientRoot = join(root, "dist", "client");
  const unsafe = walk(clientRoot).filter((path) => isPrivateAsset(path, clientRoot));
  const manifest = join(clientRoot, "library.json");
  if (existsSync(manifest) && statSync(manifest).isFile()) {
    try {
      const value = JSON.parse(readFileSync(manifest, "utf8"));
      if (Array.isArray(value?.songs) && value.songs.length) unsafe.push(manifest);
    } catch {
      unsafe.push(manifest);
    }
  }
  return [...new Set(unsafe)].sort();
}

export function checkPublicAssets(root = ROOT) {
  const unsafe = unsafePublicAssets(root);
  if (!unsafe.length) return;
  const names = unsafe.map((path) => relative(root, path)).join(", ");
  throw new Error(
    `Refusing to build with local song/model assets in public/: ${names}. ` +
    "Move them to Sona's private local library before deploying.",
  );
}

export function checkHostedOutput(root = ROOT) {
  const unsafe = unsafeHostedOutput(root);
  if (!unsafe.length) return;
  const names = unsafe.map((path) => relative(root, path)).join(", ");
  throw new Error(
    `Refusing to deploy local song/model assets from dist/: ${names}. ` +
    "Remove the hosted output and rebuild from an empty public library.",
  );
}

const invokedPath = process.argv[1]
  ? pathToFileURL(resolve(process.argv[1])).href
  : "";
if (invokedPath === import.meta.url) {
  try {
    if (process.argv.includes("--dist")) checkHostedOutput();
    else checkPublicAssets();
  } catch (error) {
    console.error(`[build-safety] ${error instanceof Error ? error.message : error}`);
    process.exitCode = 1;
  }
}
