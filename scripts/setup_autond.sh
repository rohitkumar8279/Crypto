#!/usr/bin/env bash
# Clone the official AutoND repository at the audited commit (unmodified).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMMIT=8d66e0a43a9e6fbbb3624d68296c1b77aa2da779
mkdir -p "$ROOT/third_party"
if [ ! -d "$ROOT/third_party/AutoND/.git" ]; then
  git clone https://github.com/Crypto-TII/AutoND "$ROOT/third_party/AutoND"
fi
git -C "$ROOT/third_party/AutoND" fetch --quiet origin
git -C "$ROOT/third_party/AutoND" checkout --quiet "$COMMIT"
git -C "$ROOT/third_party/AutoND" log -1 --format='AutoND pinned at %H (%cd)'
