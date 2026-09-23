# ResNet18 TensorRT inference on Jetson

This project exports pretrained torchvision ResNet18 to ONNX, builds TensorRT
engines on the target Jetson, exposes inference through FastAPI, and provisions
Prometheus and Grafana for latency monitoring.

TensorRT engine plans are hardware- and version-specific. Build them on the
target device; do not copy an engine from another GPU or TensorRT release.

## Pipeline

```text
PyTorch weights
  -> ONNX (NCHW FP32 input, 1x3x224x224)
  -> TensorRT strict FP32 and FP16 engines
  -> CUDA inference on Jetson
  -> FastAPI /infer and /metrics
  -> Prometheus
  -> Grafana
```

## 1. Export ONNX

Create a Python environment with the export dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python export.py
```

`export.py` uses `ResNet18_Weights.DEFAULT`. The service applies the matching
ImageNet resize, center crop, and normalization.

## 2. Build TensorRT engines

TensorRT 10.16.2 is provided by JetPack and should not be replaced with the
unrelated latest package from PyPI.

```bash
chmod +x scripts/build_engines.sh
./scripts/build_engines.sh
```

The script produces:

- `resnet18_fp32.engine`: strict FP32 (`--noTF32`)
- `resnet18_fp16.engine`: FP16 tactics with FP32 input/output

## 3. Run the observable service

The Docker build imports the Jetson host's TensorRT Python binding and runtime
libraries through Docker additional build contexts.

```bash
sudo docker compose up --build -d
sudo docker compose ps
```

Test the endpoints:

```bash
curl http://localhost:8001/health
curl -F "file=@/path/to/image.jpg" http://localhost:8001/infer
curl http://localhost:8001/metrics
```

The inference response contains top-5 class IDs, probabilities, and per-stage
timings. Prometheus scrapes the following custom series:

- `resnet18_request_duration_seconds`
- `resnet18_inference_component_duration_seconds`
- `resnet18_inference_requests_total`
- `resnet18_inferences_in_progress`
- `resnet18_upload_bytes`

Grafana is available at `http://localhost:3000` with the initial credentials
`admin` / `admin`. The provisioned dashboard is under `NVIDIA AI Ops Lab`.

Stop the stack with:

```bash
sudo docker compose down
```

## 4. Load test

Run the test from the host after the service is healthy:

```bash
python3 load_test.py --image /path/to/image.jpg --requests 200 --concurrency 1
python3 load_test.py --image /path/to/image.jpg --requests 500 --concurrency 4
```

The report separates client-observed request latency from preprocessing, H2D,
TensorRT GPU, D2H, and postprocessing time. `other/queue` includes HTTP,
multipart parsing, CUDA allocation, framework overhead, and time waiting for
the serialized TensorRT execution context.

## 5. Reproduce performance results

For stable comparisons, stop unrelated GPU workloads. Locking clocks with
`sudo jetson_clocks` is recommended when available.

```bash
python3 benchmarks/run_trtexec_benchmarks.py
./scripts/profile_nsys.sh 2
```

The benchmark runner uses a 2-second warmup, a 10-second measured interval,
enabled H2D/D2H transfers, spin waiting, and 1/2/4/8 inference streams. It also
samples `tegrastats` and writes machine-readable results to
`benchmarks/results/phase1.json`.

See [RESULTS.md](RESULTS.md) for the measured Phase 1 baseline.
