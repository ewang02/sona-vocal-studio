#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# add-song.sh — place audio files for a new song
#
# Usage:  ./work/add-song.sh <song-id>
#
# Prerequisites (run these first):
#   1. Place <song-id>.mp3 (the full mix) in the project root.
#   2. Run the selected separator pipeline:
#        .venv/bin/python work/process_song.py <song-id>
#   3. Extract the pitch contour:
#        .venv/bin/python contour.py <song-id>.mp3 --fmax 1500
#   4. Export the contour JSON (the end-to-end command already does this):
#        .venv/bin/python work/export_contour.py . public/data <song-id>
#
# This script then:
#   • Copies <song-id>.mp3              → public/audio/<song-id>.mp3
#   • Encodes the Kim_Inst stem          → public/audio/<song-id>-instrumental.mp3
#
# The companion upload server normally adds the library manifest entry.
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <song-id>"
  echo "Example: $0 my-song"
  exit 1
fi

SONG="$1"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

MIX="$ROOT/$SONG.mp3"
STEM="$ROOT/contour_out/separators/$SONG/instrumental.wav"
CONTOUR="$ROOT/public/data/$SONG-contour.json"

OUT_MIX="$ROOT/public/audio/$SONG.mp3"
OUT_INST="$ROOT/public/audio/$SONG-instrumental.mp3"

# ── Preflight checks ────────────────────────────────────────────────────────

errors=0

if [[ ! -f "$MIX" ]]; then
  echo "✗  Missing full mix: $MIX"
  echo "   Place the song's mp3 in the project root first."
  errors=1
fi

if [[ ! -f "$STEM" ]]; then
  echo "✗  Missing selected instrumental stem: $STEM"
  echo "   Run:  .venv/bin/python work/process_song.py $SONG"
  errors=1
fi

if [[ ! -f "$CONTOUR" ]]; then
  echo "✗  Missing contour JSON: $CONTOUR"
  echo "   Run:  .venv/bin/python work/process_song.py $SONG"
  errors=1
fi

if [[ $errors -ne 0 ]]; then
  echo ""
  echo "Fix the above issues and re-run."
  exit 1
fi

# ── Copy / encode ───────────────────────────────────────────────────────────

mkdir -p "$ROOT/public/audio"

echo "→  Copying full mix to $OUT_MIX"
cp "$MIX" "$OUT_MIX"

echo "→  Encoding instrumental stem to $OUT_INST"
ffmpeg -y -i "$STEM" -codec:a libmp3lame -b:a 192k "$OUT_INST" 2>/dev/null

echo ""
echo "✓  Audio files placed for '$SONG'."
echo "The companion upload server normally adds this song to public/library.json."
