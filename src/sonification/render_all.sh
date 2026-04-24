#!/usr/bin/env bash
set -e

DATES=(
  2026_03_05 2026_03-06 2026_03_07
  2026_03_08 2026_03_09 2026_03_10 2026_03_11 2026_03_12
  2026_03_13 2026_03_14 2026_03_15
)

cd "$(dirname "$0")/src"

for date in "${DATES[@]}"; do
  echo "=== Rendering $date (crosspath) ==="
  python matplot-map.py "$date" --interval 100 --gradient --sonification crosspath --routes 1,801,803,20,7,300,800,837 --prerender --output-dir ../web/public/renders
done

DATES=(
  2026_03_04 2026_03_05 2026_03_06 2026_03_07
  2026_03_08 2026_03_09 2026_03_10 2026_03_11 2026_03_12
  2026_03_13 2026_03_14 2026_03_15
)

for date in "${DATES[@]}"; do
  echo "=== Rendering $date (buscount) ==="
  python matplot-map.py "$date" --interval 50 --gradient --sonification buscount --prerender --output-dir ../web/public/renders
done

echo "=== Rendering complete ==="
