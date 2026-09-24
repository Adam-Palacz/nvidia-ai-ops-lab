# NVIDIA AI Ops Lab

Hands-on NVIDIA GPU, inference, and Kubernetes operations across two different
architectures:

```text
Jetson Orin Nano (ARM64, K3s control-plane)
                    │
             Tailscale underlay
                    │
Legion WSL2 (AMD64, RTX 5070, K3s agent)
                    │
                   K3s
                    │
          NVIDIA Device Plugin
                    │
              GPU workloads
```

The Tailscale link is the underlay for Flannel VXLAN. It replaces direct VXLAN
over WSL mirrored networking, which did not deliver inbound UDP 8472 traffic
from the LAN to WSL.

## Current status

- ResNet18 TensorRT FP32/FP16 benchmarks on Jetson are complete and reproducible.
- The two-node, mixed-architecture K3s cluster is connected through Tailscale.
- Cross-node pod connectivity and bidirectional iperf3 traffic are validated.
- NVIDIA runtime, device-plugin, and GPU workload manifests are prepared for
  validation on the Legion worker.

## Repository

```text
cuda/vector-add/              CUDA kernel and CPU/GPU benchmark
docs/k3s-cluster.md           Cluster architecture, decisions, and runbook
k8s/                          GPU and networking Kubernetes manifests
scripts/                      K3s, Tailscale, and NVIDIA runtime bootstrap
inference/resnet18/           Phase 1 TensorRT inference pipeline
```

Generated models, TensorRT engines, kubeconfigs, credentials, virtual
environments, and profiler reports are intentionally excluded from Git.

## Cluster quick start

The scripts discover the local Tailscale IPv4 address. The agent token is passed
only through the environment and must not be committed.

```bash
# Jetson control-plane
sudo ./scripts/setup-tailscale.sh
sudo ./scripts/bootstrap-k3s-server.sh

# Legion WSL agent
sudo ./scripts/setup-tailscale.sh
sudo env \
  K3S_URL=https://100.101.234.11:6443 \
  K3S_TOKEN='<read from the server at runtime>' \
  ./scripts/join-k3s-agent.sh
sudo ./scripts/setup-nvidia-runtime.sh
```

Install the device plugin and run the GPU checks by following
[`k8s/nvidia-device-plugin/README.md`](k8s/nvidia-device-plugin/README.md).
Operational details and troubleshooting are in
[`docs/k3s-cluster.md`](docs/k3s-cluster.md).

## Phase 1: TensorRT inference

Phase 1 exports ResNet18 to ONNX, builds strict FP32 and FP16 TensorRT engines,
serves inference through FastAPI, and exposes Prometheus/Grafana telemetry.

```bash
cd inference/resnet18
python3 export.py
./scripts/build_engines.sh
sudo docker compose up --build -d
```

See the [ResNet18 guide](inference/resnet18/README.md) and
[measured results](inference/resnet18/RESULTS.md).
