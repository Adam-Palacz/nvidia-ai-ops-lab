#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

streams="${1:-2}"
output="benchmarks/profiles/resnet18_fp16_${streams}streams"

mkdir -p benchmarks/profiles

nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --force-overwrite=true \
  --output="$output" \
  trtexec \
    --loadEngine=resnet18_fp16.engine \
    --warmUp=1000 \
    --duration=10 \
    --infStreams="$streams" \
    --useSpinWait

echo "Report: ${output}.nsys-rep"
