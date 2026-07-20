/** Types and optional compile-time registry for playable songs. */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type ContourSegmentData = { t0: number; midi: number[] };

export type ContourData = {
  hop: number;
  duration: number;
  range: number[];
  segments: ContourSegmentData[];
};

export type Song = {
  /** Unique slug used as a key and in file paths. */
  id: string;
  title: string;
  artist: string;
  /** Path to full mix, served from /public. */
  audio: string;
  /** Path to instrumental (no-vocals) stem, served from /public. */
  instrumental: string;
  /** Frame-aligned lead-pYIN used by the browser's explicit repair tool. */
  repairSource: string;
  /** Continuous target-pitch contour. */
  contour: ContourData;
};

// ---------------------------------------------------------------------------
// Optional compile-time registry. The distributed source contains no songs.
// ---------------------------------------------------------------------------

export const SONGS: Song[] = [];
