#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

trtexec \
  --onnx=resnet18.onnx \
  --saveEngine=resnet18_fp32.engine \
  --noTF32 \
  --skipInference

trtexec \
  --onnx=resnet18.onnx \
  --saveEngine=resnet18_fp16.engine \
  --fp16 \
  --skipInference
