/**
 * Library data layer.
 *
 * Built-in songs stay on the static import path (`songs.ts`). Songs created by
 * the local companion pipeline are written to `public/`, listed in
 * `public/library.json`, and loaded from the same origin so WebAudio never sees
 * cross-origin song assets.
 */

import { SONGS } from "./songs";
import type { ContourData, Song } from "./songs";

const DIRECT_PIPELINE_BASE = "http://localhost:4599";
const LOOPBACK_HOSTS = new Set(["localhost", "127.0.0.1", "[::1]"]);

/** Use Vite's same-origin proxy in development and loopback elsewhere. */
export function pipelineBaseForHostname(hostname: string): string {
  return LOOPBACK_HOSTS.has(hostname) ? "/api/pipeline" : DIRECT_PIPELINE_BASE;
}

export const PIPELINE_BASE =
  typeof window === "undefined"
    ? DIRECT_PIPELINE_BASE
    : pipelineBaseForHostname(window.location.hostname);

/** Lightweight descriptor for a grid card — no heavy contour arrays. */
export type LibraryCard = {
  id: string;
  title: string;
  artist: string;
  duration: number;
  builtin: boolean;
};

type ManifestSong = {
  id: string;
  title: string;
  artist: string;
  duration: number;
};

const builtinById = new Map(SONGS.map((song) => [song.id, song]));

/** Static cards for the built-in songs, available before the manifest fetch. */
export const BUILTIN_CARDS: LibraryCard[] = SONGS.map((song) => ({
  id: song.id,
  title: song.title,
  artist: song.artist,
  duration: song.contour.duration,
  builtin: true,
}));

/** Built-in cards followed by additional packaged songs from the manifest. */
export async function fetchLibrary(): Promise<LibraryCard[]> {
  let packaged: LibraryCard[] = [];
  try {
    const response = await fetch("/library.json", { cache: "no-store" });
    if (response.ok) {
      const data = (await response.json()) as { songs?: ManifestSong[] };
      packaged = (data.songs ?? []).map((song) => ({
        id: song.id,
        title: song.title,
        artist: song.artist,
        duration: song.duration,
        builtin: false,
      }));
    }
  } catch {
    // A missing manifest leaves only any explicitly bundled registry entries.
  }
  const builtinIds = new Set(BUILTIN_CARDS.map((card) => card.id));
  return [...BUILTIN_CARDS, ...packaged.filter((card) => !builtinIds.has(card.id))];
}

/** Resolve a card to a full playable Song (built-ins are synchronous imports). */
export async function resolveSong(card: LibraryCard): Promise<Song> {
  const builtin = builtinById.get(card.id);
  if (builtin) return builtin;

  const contourResponse = await fetch(`/data/${card.id}-contour.json`, {
    cache: "no-store",
  });
  if (!contourResponse.ok) throw new Error(`Could not load "${card.title}".`);
  const contour = (await contourResponse.json()) as ContourData;
  return {
    id: card.id,
    title: card.title,
    artist: card.artist,
    audio: `/audio/${card.id}.mp3`,
    instrumental: `/audio/${card.id}-instrumental.mp3`,
    repairSource: `/data/${card.id}-pyin.json`,
    contour,
  };
}

// ---------------------------------------------------------------------------
// Per-song background image — downscaled JPEG kept in localStorage (per device,
// no backend), keyed by song id.
// ---------------------------------------------------------------------------

const BG_PREFIX = "sona.bg.";

export function getBackground(id: string): string | null {
  try {
    return window.localStorage.getItem(BG_PREFIX + id);
  } catch {
    return null;
  }
}

export function setBackground(id: string, dataUrl: string | null): void {
  try {
    if (dataUrl) window.localStorage.setItem(BG_PREFIX + id, dataUrl);
    else window.localStorage.removeItem(BG_PREFIX + id);
  } catch {
    // Private mode or quota exceeded: the choice simply does not persist.
  }
}

/** Downscale an image file to a bounded JPEG data URL so storage stays small. */
export async function fileToBackground(file: File, maxDim = 1200): Promise<string> {
  const bitmap = await createImageBitmap(file);
  const scale = Math.min(1, maxDim / Math.max(bitmap.width, bitmap.height));
  const width = Math.max(1, Math.round(bitmap.width * scale));
  const height = Math.max(1, Math.round(bitmap.height * scale));
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  if (!context) throw new Error("Canvas is unavailable.");
  context.drawImage(bitmap, 0, 0, width, height);
  bitmap.close?.();
  return canvas.toDataURL("image/jpeg", 0.82);
}

// ---------------------------------------------------------------------------
// Upload + job polling against the local companion server.
// ---------------------------------------------------------------------------

export type JobStatus = {
  jobId: string;
  songId: string;
  title?: string;
  artist?: string;
  status: "queued" | "running" | "done" | "error";
  step: string | null;
  steps: string[];
  stepOrder?: string[];
  duration: number;
  cache?: Record<string, "hit" | "miss">;
  timings?: Record<string, number>;
  error: string | null;
};

export const STEP_LABELS: Record<string, string> = {
  separating: "Separating vocals",
  isolating: "Isolating lead vocal",
  tracking: "Tracking pitch",
  transcribing: "Transcribing notes",
  finalizing: "Finalizing",
  done: "Done",
};

export async function pipelineAvailable(): Promise<boolean> {
  try {
    const response = await fetch(`${PIPELINE_BASE}/api/health`, { cache: "no-store" });
    return response.ok;
  } catch {
    return false;
  }
}

export async function uploadSong(
  file: File,
  title: string,
  artist: string,
): Promise<{ jobId: string; songId: string }> {
  const params = new URLSearchParams({ title, artist });
  const response = await fetch(`${PIPELINE_BASE}/api/songs?${params.toString()}`, {
    method: "POST",
    headers: { "Content-Type": "application/octet-stream" },
    body: file,
  });
  if (!response.ok) {
    const detail = (await response.json().catch(() => null)) as { error?: string } | null;
    throw new Error(detail?.error || "Upload failed.");
  }
  return response.json();
}

export async function fetchJob(jobId: string): Promise<JobStatus> {
  const response = await fetch(`${PIPELINE_BASE}/api/jobs/${jobId}`, { cache: "no-store" });
  if (!response.ok) throw new Error("Lost track of the processing job.");
  return response.json();
}
