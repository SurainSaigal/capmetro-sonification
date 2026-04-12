#!/usr/bin/env bash
set -e

DATES=(
  2026_03_05 2026_03_06 2026_03_07
  2026_03_08 2026_03_09 2026_03_10 2026_03_11 2026_03_12
  2026_03_13 2026_03_14 2026_03_15 2026_03_16
)

cd "$(dirname "$0")/src"

for date in "${DATES[@]}"; do
  echo "=== Rendering $date ==="
  python matplot-map.py "$date" --interval 50 --gradient --prerender
done

echo "Done."
