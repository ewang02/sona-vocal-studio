/**
 * Library data layer.
 *
 * Built-in songs stay on the static import path (`songs.ts`). User-provided
 * songs live behind the loopback-only companion API and never become hosted
 * build assets.
 */

import { SONGS } from "./songs";
import type { ContourData, Song } from "./songs";

const DIRECT_PIPELINE_BASE = "http://127.0.0.1:4599";
// Use the same explicit loopback address everywhere. Besides avoiding
// localhost IPv4/IPv6 differences, the cross-origin request carries the
// browser Origin header that the companion verifies before serving local data.
export const PIPELINE_BASE = DIRECT_PIPELINE_BASE;

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

/** Built-in cards followed by songs exposed by the private local companion. */
export async function fetchLibrary(): Promise<LibraryCard[]> {
  let packaged: LibraryCard[] = [];
  try {
    const response = await fetch(`${PIPELINE_BASE}/api/library`, { cache: "no-store" });
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

  const contourResponse = await fetch(`${PIPELINE_BASE}/api/songs/${encodeURIComponent(card.id)}/contour`, {
    cache: "no-store",
  });
  if (!contourResponse.ok) throw new Error(`Could not load "${card.title}".`);
  const contour = (await contourResponse.json()) as ContourData;
  return {
    id: card.id,
    title: card.title,
    artist: card.artist,
    audio: `${PIPELINE_BASE}/api/songs/${encodeURIComponent(card.id)}/source`,
    instrumental: `${PIPELINE_BASE}/api/songs/${encodeURIComponent(card.id)}/instrumental`,
    repairSource: `${PIPELINE_BASE}/api/songs/${encodeURIComponent(card.id)}/repair`,
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

export type SetupStatus = {
  status: "idle" | "running" | "done" | "error";
  message: string | null;
  error: string | null;
};

export type PipelineHealth = {
  ok: true;
  ready: boolean;
  reason: string | null;
  setup: SetupStatus;
};

export const STEP_LABELS: Record<string, string> = {
  separating: "Separating vocals",
  isolating: "Isolating lead vocal",
  tracking: "Tracking pitch",
  transcribing: "Transcribing notes",
  finalizing: "Finalizing",
  done: "Done",
};

export async function fetchPipelineHealth(): Promise<PipelineHealth | null> {
  try {
    const response = await fetch(`${PIPELINE_BASE}/api/health`, { cache: "no-store" });
    return response.ok ? response.json() : null;
  } catch {
    return null;
  }
}

export async function pipelineAvailable(): Promise<boolean> {
  return (await fetchPipelineHealth()) !== null;
}

export async function startPipelineSetup(): Promise<{ ready: boolean; setup: SetupStatus }> {
  const response = await fetch(`${PIPELINE_BASE}/api/setup`, {
    method: "POST",
    headers: { "X-Sona-Action": "setup" },
  });
  if (!response.ok) throw new Error("Local song-processing setup could not start.");
  return response.json();
}

export async function fetchPipelineSetup(): Promise<{ ready: boolean; setup: SetupStatus }> {
  const response = await fetch(`${PIPELINE_BASE}/api/setup`, { cache: "no-store" });
  if (!response.ok) throw new Error("Lost connection to local song-processing setup.");
  return response.json();
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
