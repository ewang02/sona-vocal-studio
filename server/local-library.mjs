/** Local, non-deployable storage for user-provided songs and generated assets. */

import {
  copyFileSync,
  existsSync,
  mkdirSync,
  readFileSync,
  realpathSync,
  renameSync,
  writeFileSync,
} from "node:fs";
import { homedir } from "node:os";
import { basename, dirname, join, resolve, sep } from "node:path";

export const SONG_ID_PATTERN = /^[a-z0-9](?:[a-z0-9-]{0,78}[a-z0-9])?$/;

/** Resolve the normal per-user data directory, with an override for portable installs/tests. */
export function resolveSonaDataRoot({
  env = process.env,
  platform = process.platform,
  home = homedir(),
} = {}) {
  if (env.SONA_DATA_DIR) return resolve(env.SONA_DATA_DIR);
  if (platform === "win32") {
    return join(env.LOCALAPPDATA || join(home, "AppData", "Local"), "Sona");
  }
  if (platform === "darwin") return join(home, "Library", "Application Support", "Sona");
  return join(env.XDG_DATA_HOME || join(home, ".local", "share"), "sona");
}

export function createLocalLibrary(dataRoot = resolveSonaDataRoot()) {
  const root = resolve(dataRoot);
  const libraryRoot = join(root, "library");
  const audioDir = join(libraryRoot, "audio");
  const dataDir = join(libraryRoot, "data");
  const uploadsDir = join(root, "uploads");
  const manifestPath = join(libraryRoot, "library.json");

  function ensureDirectories() {
    for (const path of [libraryRoot, audioDir, dataDir, uploadsDir]) {
      mkdirSync(path, { recursive: true });
    }
    if (!existsSync(manifestPath)) writeManifest({ songs: [] });
  }

  function readManifest() {
    try {
      const value = JSON.parse(readFileSync(manifestPath, "utf8"));
      return Array.isArray(value?.songs) ? value : { songs: [] };
    } catch {
      return { songs: [] };
    }
  }

  function writeManifest(manifest) {
    mkdirSync(dirname(manifestPath), { recursive: true });
    const temporary = `${manifestPath}.tmp`;
    writeFileSync(temporary, `${JSON.stringify(manifest, null, 2)}\n`, { mode: 0o600 });
    renameSync(temporary, manifestPath);
  }

  function upsertSong(entry) {
    const manifest = readManifest();
    manifest.songs = manifest.songs.filter((song) => song.id !== entry.id);
    manifest.songs.push(entry);
    writeManifest(manifest);
  }

  function containsSong(songId) {
    return readManifest().songs.some((song) => song.id === songId);
  }

  /** Map a public API kind to a fixed filename; callers never supply a path. */
  function resolveAsset(songId, kind) {
    if (!SONG_ID_PATTERN.test(songId) || !containsSong(songId)) return null;
    const assets = {
      source: {
        path: join(audioDir, `${songId}.mp3`),
        contentType: "audio/mpeg",
        ranged: true,
      },
      instrumental: {
        path: join(audioDir, `${songId}-instrumental.mp3`),
        contentType: "audio/mpeg",
        ranged: true,
      },
      contour: {
        path: join(dataDir, `${songId}-contour.json`),
        contentType: "application/json; charset=utf-8",
        ranged: false,
      },
      repair: {
        path: join(dataDir, `${songId}-pyin.json`),
        contentType: "application/json; charset=utf-8",
        ranged: false,
      },
    };
    const asset = assets[kind];
    if (!asset || !existsSync(asset.path)) return null;
    const realDirectory = realpathSync(asset.ranged ? audioDir : dataDir);
    const realAsset = realpathSync(asset.path);
    if (!realAsset.startsWith(`${realDirectory}${sep}`)) return null;
    return { ...asset, path: realAsset };
  }

  ensureDirectories();
  return {
    root,
    libraryRoot,
    audioDir,
    dataDir,
    uploadsDir,
    manifestPath,
    readManifest,
    writeManifest,
    upsertSong,
    containsSong,
    resolveAsset,
  };
}

/**
 * Preserve libraries created by older versions without deleting their source.
 * Copy only files named by the manifest; never walk arbitrary paths.
 */
export function migrateLegacyLibrary(projectRoot, library) {
  const legacyManifest = join(projectRoot, "public", "library.json");
  if (library.readManifest().songs.length > 0 || !existsSync(legacyManifest)) return 0;
  let legacy;
  try {
    legacy = JSON.parse(readFileSync(legacyManifest, "utf8"));
  } catch {
    return 0;
  }
  const songs = Array.isArray(legacy?.songs) ? legacy.songs : [];
  const migrated = [];
  for (const song of songs) {
    if (!SONG_ID_PATTERN.test(song?.id || "")) continue;
    const copies = [
      [join(projectRoot, "public", "audio", `${song.id}.mp3`), library.audioDir],
      [join(projectRoot, "public", "audio", `${song.id}-instrumental.mp3`), library.audioDir],
      [join(projectRoot, "public", "data", `${song.id}-contour.json`), library.dataDir],
      [join(projectRoot, "public", "data", `${song.id}-pyin.json`), library.dataDir],
    ];
    if (!copies.every(([source]) => existsSync(source))) continue;
    for (const [source, destination] of copies) {
      copyFileSync(source, join(destination, basename(source)));
    }
    migrated.push(song);
  }
  if (migrated.length) library.writeManifest({ songs: migrated });
  return migrated.length;
}

/** Parse one RFC 9110 byte range. Multi-range responses are intentionally unsupported. */
export function parseByteRange(header, size) {
  if (!header) return null;
  const match = /^bytes=(\d*)-(\d*)$/.exec(header.trim());
  if (!match || size <= 0 || (!match[1] && !match[2])) return false;
  let start;
  let end;
  if (!match[1]) {
    const suffix = Number(match[2]);
    if (!Number.isSafeInteger(suffix) || suffix <= 0) return false;
    start = Math.max(0, size - suffix);
    end = size - 1;
  } else {
    start = Number(match[1]);
    end = match[2] ? Number(match[2]) : size - 1;
    if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end)) return false;
    if (start >= size || end < start) return false;
    end = Math.min(end, size - 1);
  }
  return { start, end };
}
