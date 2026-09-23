# Phase 1 inference performance

Measured on 2026-09-23 on an NVIDIA Jetson Orin Nano Super Developer Kit.
These are local measurements from this repository, not vendor specifications.

## Test setup

- Jetson Linux R39.2.1
- TensorRT 10.16.2
- Nsight Systems 2025.6.3
- 25 W power mode
- clocks controlled by DVFS (`jetson_clocks` was not locked)
- ResNet18, batch 1, input `1x3x224x224`
- 2 s warmup followed by a 10 s measurement
- H2D and D2H transfers enabled
- `--useSpinWait`
- strict FP32 engine built with `--noTF32`
- FP16 engine built with `--fp16`

GPU utilization is sampled from `tegrastats` every 100 ms across the complete
run, including warmup. Engine plans and profiler reports are generated locally
and excluded from Git.

## FP32 and FP16

| Precision | Streams | Throughput (QPS) | Mean latency (ms) | p95 latency (ms) | Mean GPU compute (ms) | Mean GPU util. | Peak GPU util. |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP32 | 1 | 291.5 | 3.468 | 3.478 | 3.428 | 95.6% | 99% |
| FP32 | 2 | 321.6 | 6.263 | 6.702 | 6.212 | 97.0% | 99% |
| FP32 | 4 | 329.4 | 12.173 | 12.772 | 12.123 | 96.8% | 99% |
| FP32 | 8 | 324.2 | 24.674 | 25.396 | 24.622 | 96.1% | 99% |
| FP16 | 1 | 1106.7 | 0.943 | 0.947 | 0.902 | 95.7% | 98% |
| FP16 | 2 | 1356.9 | 1.527 | 1.620 | 1.470 | 96.5% | 99% |
| FP16 | 4 | **1402.4** | 2.908 | 3.070 | 2.844 | 96.5% | 99% |
| FP16 | 8 | 1317.9 | 6.120 | 6.342 | 6.058 | 96.8% | 99% |

At one stream, FP16 delivers 3.8x the FP32 throughput and reduces mean GPU
compute time from 3.43 ms to 0.90 ms. The serialized engine also falls from
approximately 44 MiB to 22 MiB.

Two FP16 streams are the best latency/throughput operating point for the
service: throughput increases by 22.6% over one stream, sampled GPU utilization
averages 96.5% and reaches 99%, while mean latency remains 1.53 ms. Four streams
produce the maximum throughput, but add only 3.4% over two streams while almost
doubling latency. Eight streams oversubscribe this workload: throughput falls
by 6.0% from the four-stream peak and latency more than doubles.

TensorRT warns that per-inference latency with multiple streams includes
overlap and contention. Throughput is therefore the primary scaling metric.

## Nsight Systems findings

A fresh FP16, two-stream profile was captured with:

```bash
./scripts/profile_nsys.sh 2
nsys stats \
  --report cuda_gpu_kern_sum,cuda_api_sum \
  benchmarks/profiles/resnet18_fp16_2streams.nsys-rep
```

Findings:

- Tensor Core FP16 convolution kernels (`xmma_fprop`) dominate GPU time. The
  five largest kernel variants account for about 55% of all kernel time.
- The first convolution alone accounts for another 6%; pooling, reformat, and
  the final GEMM are comparatively small.
- On the CUDA API timeline, `cuLaunchKernelEx` accounts for 62.1% of API time,
  event synchronization for 13.0%, event recording for 8.9%, and async copies
  for 6.1%.
- The profiled run reaches 1337 QPS versus 1357 QPS without profiling, an
  expected profiler overhead of roughly 1.4%.
- The API distribution shows that launch and synchronization overhead is the
  main remaining host-side optimization opportunity; H2D and D2H are not the
  throughput bottleneck.

## CUDA Graph experiment

CUDA Graph capture succeeded for both one and two execution contexts.

| FP16 mode | Throughput (QPS) | Mean latency (ms) | Mean enqueue (ms) | Mean GPU compute (ms) |
|---|---:|---:|---:|---:|
| 1 stream, baseline | 1107.3 | 0.943 | 0.293 | 0.902 |
| 1 stream, CUDA Graph | 1249.8 | 0.839 | 0.006 | 0.799 |
| 2 streams, baseline | 1356.9 | 1.527 | — | 1.470 |
| 2 streams, CUDA Graph | 1381.6 | 1.497 | 0.006 | 1.445 |

At one stream, CUDA Graph reduces enqueue time by 97.9% and improves throughput
by 12.9%. At two streams, where the GPU is already close to saturation, the
throughput gain is only 1.8%. CUDA Graph is therefore a useful next optimization
for the low-latency single-stream path, but less important for saturated
multi-stream throughput.

The current Python service does not use CUDA Graph: preprocessing shapes are
fixed, so capture is technically feasible, but it requires persistent device
buffers and a captured execution path rather than per-request allocation.

## HTTP serving baseline

The included load test measures client request latency and returns a server-side
breakdown. A local 30-request smoke test at concurrency 4 measured:

```text
request latency        46.490 ms
├── preprocessing       9.530 ms
├── H2D                 0.039 ms
├── TensorRT GPU        0.937 ms
├── D2H                 0.073 ms
├── postprocessing      0.322 ms
└── other/queue         35.589 ms
```

This is a functional smoke test, not the controlled TensorRT benchmark above.
It demonstrates that API queueing, image decoding, allocations, and framework
overhead dominate end-to-end latency once GPU execution is below 1 ms.

## Reproduction

```bash
./scripts/build_engines.sh
python3 benchmarks/run_trtexec_benchmarks.py
./scripts/profile_nsys.sh 2
```

Machine-readable benchmark data is committed in
`benchmarks/results/phase1.json`.
