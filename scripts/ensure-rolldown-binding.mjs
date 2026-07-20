#!/usr/bin/env node

/**
 * Repair npm's optional-dependency omission for Rolldown's native package.
 *
 * Vite 8 installs Rolldown through an optional platform package. Some npm
 * installs leave that package absent (or create an empty directory) even when
 * it is present in the lockfile. This postinstall resolves the exact binding
 * required by the installed Rolldown version, downloads the integrity-pinned
 * lockfile tarball directly, extracts it using Node built-ins, and verifies
 * the native entry point. It deliberately does not invoke npm recursively.
 */

import { createHash } from "node:crypto";
import {
  chmodSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, relative, resolve, sep } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { gunzipSync } from "node:zlib";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");

export function detectLinuxLibc(report = process.report?.getReport?.()) {
  if (report?.header?.glibcVersionRuntime) return "gnu";
  if (report?.sharedObjects?.some((entry) => /musl/i.test(entry))) return "musl";
  try {
    if (/musl/i.test(readFileSync("/usr/bin/ldd", "utf8"))) return "musl";
  } catch {
    // Minimal containers may not include ldd. The Cloudflare/most glibc Linux
    // builders use GNU libc, so it is the conservative final fallback.
  }
  return "gnu";
}

export function bindingNameFor(
  platform = process.platform,
  arch = process.arch,
  linuxLibc = platform === "linux" ? detectLinuxLibc() : null,
) {
  const key = `${platform}-${arch}${platform === "linux" ? `-${linuxLibc}` : ""}`;
  const names = {
    "darwin-arm64": "@rolldown/binding-darwin-arm64",
    "darwin-x64": "@rolldown/binding-darwin-x64",
    "win32-arm64": "@rolldown/binding-win32-arm64-msvc",
    "win32-x64": "@rolldown/binding-win32-x64-msvc",
    "freebsd-x64": "@rolldown/binding-freebsd-x64",
    "openharmony-arm64": "@rolldown/binding-openharmony-arm64",
    "android-arm64": "@rolldown/binding-android-arm64",
    "linux-arm-gnu": "@rolldown/binding-linux-arm-gnueabihf",
    "linux-arm64-gnu": "@rolldown/binding-linux-arm64-gnu",
    "linux-arm64-musl": "@rolldown/binding-linux-arm64-musl",
    "linux-x64-gnu": "@rolldown/binding-linux-x64-gnu",
    "linux-x64-musl": "@rolldown/binding-linux-x64-musl",
    "linux-ppc64-gnu": "@rolldown/binding-linux-ppc64-gnu",
    "linux-s390x-gnu": "@rolldown/binding-linux-s390x-gnu",
  };
  return names[key] ?? null;
}

function packageDirectory(packageName) {
  return join(ROOT, "node_modules", ...packageName.split("/"));
}

function readJson(path) {
  return JSON.parse(readFileSync(path, "utf8"));
}

function validBinding(packageName, version) {
  const directory = packageDirectory(packageName);
  try {
    const manifest = readJson(join(directory, "package.json"));
    return (
      manifest.name === packageName &&
      manifest.version === version &&
      typeof manifest.main === "string" &&
      existsSync(join(directory, manifest.main))
    );
  } catch {
    return false;
  }
}

function parseOctal(header, start, length) {
  const value = header
    .subarray(start, start + length)
    .toString("utf8")
    .replace(/\0.*$/, "")
    .trim();
  return value ? Number.parseInt(value, 8) : 0;
}

function headerString(header, start, length) {
  return header
    .subarray(start, start + length)
    .toString("utf8")
    .replace(/\0.*$/, "");
}

/** Extract the regular files in an npm `package/` tarball without shell tools. */
export function extractNpmTarball(tarball, destination) {
  const archive = gunzipSync(readFileSync(tarball));
  let offset = 0;
  while (offset + 512 <= archive.length) {
    const header = archive.subarray(offset, offset + 512);
    if (header.every((byte) => byte === 0)) break;

    const name = headerString(header, 0, 100);
    const prefix = headerString(header, 345, 155);
    const archivePath = prefix ? `${prefix}/${name}` : name;
    const size = parseOctal(header, 124, 12);
    const mode = parseOctal(header, 100, 8);
    const type = String.fromCharCode(header[156] || 48);
    const dataStart = offset + 512;
    const dataEnd = dataStart + size;

    if (archivePath.startsWith("package/")) {
      const stripped = archivePath.slice("package/".length);
      if (stripped) {
        const output = resolve(destination, stripped);
        const pathFromDestination = relative(destination, output);
        if (
          pathFromDestination === ".." ||
          pathFromDestination.startsWith(`..${sep}`)
        ) {
          throw new Error(`unsafe path in Rolldown tarball: ${archivePath}`);
        }
        if (type === "5") {
          mkdirSync(output, { recursive: true });
        } else if (type === "0" || type === "\0") {
          mkdirSync(dirname(output), { recursive: true });
          writeFileSync(output, archive.subarray(dataStart, dataEnd));
          if (mode) chmodSync(output, mode);
        }
      }
    }
    offset = dataStart + Math.ceil(size / 512) * 512;
  }
}

function bindingLockEntry(packageName) {
  try {
    const lockfile = readJson(join(ROOT, "package-lock.json"));
    return lockfile.packages?.[`node_modules/${packageName}`] ?? null;
  } catch {
    return null;
  }
}

async function registryDownload(packageName, version) {
  const lockEntry = bindingLockEntry(packageName);
  if (lockEntry?.resolved) {
    return { url: lockEntry.resolved, integrity: lockEntry.integrity ?? null };
  }

  const registry = (process.env.npm_config_registry || "https://registry.npmjs.org/")
    .replace(/\/+$/, "");
  const encodedName = packageName.replace("/", "%2F");
  const metadataResponse = await fetch(`${registry}/${encodedName}/${version}`);
  if (!metadataResponse.ok) {
    throw new Error(
      `registry metadata for ${packageName}@${version} returned ${metadataResponse.status}`,
    );
  }
  const metadata = await metadataResponse.json();
  if (!metadata.dist?.tarball) {
    throw new Error(`registry metadata for ${packageName}@${version} has no tarball`);
  }
  return { url: metadata.dist.tarball, integrity: metadata.dist.integrity ?? null };
}

function verifyIntegrity(bytes, integrity) {
  if (!integrity) return;
  const candidate = integrity
    .split(/\s+/)
    .map((entry) => entry.split("-", 2))
    .find(([algorithm]) => algorithm === "sha512" || algorithm === "sha256");
  if (!candidate) throw new Error(`unsupported package integrity: ${integrity}`);
  const [algorithm, expected] = candidate;
  const actual = createHash(algorithm).update(bytes).digest("base64");
  if (actual !== expected) throw new Error(`integrity check failed (${algorithm})`);
}

async function downloadTarball(packageName, version, destination) {
  const { url, integrity } = await registryDownload(packageName, version);
  const response = await fetch(url, { redirect: "follow" });
  if (!response.ok) {
    throw new Error(`download of ${packageName}@${version} returned ${response.status}`);
  }
  const bytes = Buffer.from(await response.arrayBuffer());
  verifyIntegrity(bytes, integrity);
  const tarball = join(destination, "binding.tgz");
  writeFileSync(tarball, bytes);
  return tarball;
}

export async function ensureRolldownBinding() {
  const rolldownManifestPath = join(ROOT, "node_modules", "rolldown", "package.json");
  if (!existsSync(rolldownManifestPath)) {
    throw new Error("Rolldown is not installed; npm did not finish installing dependencies.");
  }
  const rolldown = readJson(rolldownManifestPath);
  const packageName = bindingNameFor();
  if (!packageName) {
    throw new Error(
      `Rolldown has no supported native binding for ${process.platform}/${process.arch}.`,
    );
  }
  const version = rolldown.optionalDependencies?.[packageName] ?? rolldown.version;
  if (validBinding(packageName, version)) {
    console.log(`[rolldown-binding] ${packageName}@${version} is present.`);
    return { packageName, version, repaired: false };
  }

  const temporary = mkdtempSync(join(tmpdir(), "sona-rolldown-"));
  const destination = packageDirectory(packageName);
  try {
    const tarball = await downloadTarball(packageName, version, temporary);
    rmSync(destination, { recursive: true, force: true });
    mkdirSync(destination, { recursive: true });
    extractNpmTarball(tarball, destination);
    if (!validBinding(packageName, version)) {
      throw new Error(`downloaded ${packageName}@${version}, but its native entry point is missing`);
    }
  } finally {
    rmSync(temporary, { recursive: true, force: true });
  }
  console.log(`[rolldown-binding] repaired ${packageName}@${version}.`);
  return { packageName, version, repaired: true };
}

const invokedPath = process.argv[1] ? pathToFileURL(resolve(process.argv[1])).href : "";
if (invokedPath === import.meta.url) {
  try {
    await ensureRolldownBinding();
  } catch (error) {
    console.error(`[rolldown-binding] ${error instanceof Error ? error.message : error}`);
    process.exitCode = 1;
  }
}
