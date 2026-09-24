# NVIDIA Device Plugin

This directory contains environment-specific Helm values for the Legion WSL
worker. Install NVIDIA Container Toolkit on that worker first:

```bash
sudo ./scripts/setup-nvidia-runtime.sh
```

Confirm that K3s discovered the runtime:

```bash
sudo k3s kubectl get runtimeclass nvidia
sudo grep -n nvidia \
  /var/lib/rancher/k3s/agent/etc/containerd/config.toml
```

Install the official NVIDIA Device Plugin chart from the control-plane:

```bash
helm repo add nvidia-device-plugin \
  https://nvidia.github.io/k8s-device-plugin
helm repo update

sudo helm upgrade --install nvidia-device-plugin \
  nvidia-device-plugin/nvidia-device-plugin \
  --kubeconfig /etc/rancher/k3s/k3s.yaml \
  --namespace nvidia-device-plugin \
  --create-namespace \
  --values k8s/nvidia-device-plugin/values.yaml
```

Verify resource discovery:

```bash
sudo k3s kubectl get pods -n nvidia-device-plugin -o wide
sudo k3s kubectl get node apailegion \
  -o jsonpath='{.status.allocatable.nvidia\.com/gpu}{"\n"}'
```

Run the lightweight CUDA check first:

```bash
sudo k3s kubectl apply -f k8s/gpu-test.yaml
sudo k3s kubectl wait -n gpu-lab \
  --for=jsonpath='{.status.phase}'=Succeeded \
  pod/cuda-gpu-test \
  --timeout=300s
sudo k3s kubectl logs -n gpu-lab cuda-gpu-test
```

Then run the larger PyTorch container:

```bash
sudo k3s kubectl apply -f k8s/pytorch-gpu.yaml
sudo k3s kubectl logs -n gpu-lab -f pytorch-gpu
```

Delete completed test pods before rerunning them:

```bash
sudo k3s kubectl delete -f k8s/gpu-test.yaml --ignore-not-found
sudo k3s kubectl delete -f k8s/pytorch-gpu.yaml --ignore-not-found
```

Neither the chart nor these manifests install an NVIDIA driver. Under WSL, the
GPU driver is supplied by Windows and exposed through `/dev/dxg`.
