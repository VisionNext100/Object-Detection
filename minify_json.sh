#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "Usage: $(basename "$0") INPUT_JSON [OUTPUT_JSON]" >&2
  exit 2
fi

INPUT_JSON="$1"
OUTPUT_JSON="${2:-}"

python - "$INPUT_JSON" "$OUTPUT_JSON" <<'PY'
import json
import sys
from pathlib import Path

src = Path(sys.argv[1]).expanduser()
dst_arg = sys.argv[2] if len(sys.argv) > 2 else ""

if not src.is_file():
    raise SystemExit(f"Input JSON not found: {src}")

if dst_arg:
    dst = Path(dst_arg).expanduser()
else:
    dst = src.with_name(f"{src.stem}_min{src.suffix or '.json'}")

with src.open("r", encoding="utf-8") as f:
    data = json.load(f)

dst.parent.mkdir(parents=True, exist_ok=True)
with dst.open("w", encoding="utf-8") as f:
    json.dump(data, f, separators=(",", ":"), ensure_ascii=False)

old_size = src.stat().st_size
new_size = dst.stat().st_size
saved = old_size - new_size
ratio = saved / old_size * 100 if old_size else 0.0

print(f"Input:  {src}")
print(f"Output: {dst}")
print(f"Size:   {old_size} -> {new_size} bytes ({ratio:.1f}% smaller)")
PY
