#!/usr/bin/env bash
#
# Fetch / generate the sample's test + demo assets (kept out of git — they're
# regenerable fixtures used only by the headless self-tests and the LITERT_DEMO
# scripted demo; the interactive chat doesn't need them).
#
# Run once after cloning if you want those:
#   ./fetch-test-assets.sh
#
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)/Resources"
BASE="https://raw.githubusercontent.com/google-ai-edge/litert-lm/main"
mkdir -p "$DIR"

echo "→ apple.png + have_a_wonderful_day.wav (LiteRT-LM testdata, Apache-2.0)"
curl -fsSL "$BASE/runtime/components/preprocessor/testdata/apple.png" -o "$DIR/apple.png"
curl -fsSL "$BASE/runtime/testdata/have_a_wonderful_day.wav" -o "$DIR/have_a_wonderful_day.wav"

if command -v ffmpeg >/dev/null 2>&1; then
  # A red → green → blue sequence (1.5 s each). Time-varying on purpose: the
  # video self-test can only prove the model saw *multiple frames in order*
  # if the frames differ over time — a static clip can't distinguish "saw the
  # sequence" from "saw one frame". Expected description: three color blocks,
  # red, green, and blue.
  echo "→ sample.mp4 (red→green→blue sequence, for the video self-test)"
  ffmpeg -y \
    -f lavfi -i "color=red:size=512x512:duration=1.5:rate=15" \
    -f lavfi -i "color=green:size=512x512:duration=1.5:rate=15" \
    -f lavfi -i "color=blue:size=512x512:duration=1.5:rate=15" \
    -filter_complex "[0][1][2]concat=n=3:v=1:a=0" -pix_fmt yuv420p \
    "$DIR/sample.mp4" >/dev/null 2>&1
else
  echo "  (skipping sample.mp4 — ffmpeg not found)"
fi

if command -v say >/dev/null 2>&1 && command -v ffmpeg >/dev/null 2>&1; then
  echo "→ question.wav (spoken question, for the audio self-test)"
  say -o /tmp/_q.aiff "What is the capital of France?"
  ffmpeg -y -i /tmp/_q.aiff -ar 16000 -ac 1 -c:a pcm_s16le "$DIR/question.wav" >/dev/null 2>&1
  rm -f /tmp/_q.aiff
else
  echo "  (skipping question.wav — needs macOS 'say' + ffmpeg)"
fi

echo "Done → $DIR"
