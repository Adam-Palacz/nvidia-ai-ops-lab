# K3s cluster

## Architecture

The lab combines two Linux environments with different CPU architectures and
GPU stacks:

- `adamp`: Jetson Orin Nano, ARM64, K3s control-plane.
- `apailegion`: Ubuntu 24.04 under WSL2, AMD64, K3s agent, RTX 5070 exposed by
  the Windows NVIDIA driver.
- Tailscale provides stable, bidirectional node connectivity.
- Flannel VXLAN carries the K3s pod network over `tailscale0`.

Current observed node addresses:

- Jetson Tailscale: `100.101.234.11`
- Legion Tailscale: `100.112.18.72`
- Cluster pod CIDR: `10.42.0.0/16`
- Jetson pod CIDR: `10.42.0.0/24`
- Legion pod CIDR: `10.42.2.0/24`

The addresses are operational metadata, not credentials. Scripts discover the
local Tailscale address instead of embedding it.

## Why Tailscale is required

Direct Flannel VXLAN over the LAN was tested first. Both nodes created correct
routes, FDB entries, and `flannel.1` interfaces, and the Jetson transmitted UDP
8472 packets to the Windows LAN address. Those packets did not appear on any WSL
interface, even with both Windows and Hyper-V firewall allowances and with the
WSL firewall disabled for diagnosis.

The `host-gw` backend was also unsuitable. It requires each node to route the
remote pod CIDR through the other node's LAN address. WSL mirrored networking
does not act as a normal L3 router for the pod subnet.

Binding Flannel to `tailscale0` provides an underlay that supports bidirectional
traffic. After the change:

- Jetson-to-WSL pod ping succeeded at about 9 ms.
- Cross-node iperf3 succeeded in both directions.
- A measured 20-second pod test reached about 164 Mbit/s in one direction and
  132 Mbit/s in reverse, with Wi-Fi variability.

These are connectivity checks, not stable performance baselines.

## K3s configuration

The server configuration is equivalent to:

```yaml
flannel-backend: vxlan
flannel-iface: tailscale0
node-ip: <Jetson Tailscale IPv4>
tls-san:
  - <Jetson Tailscale IPv4>
```

The agent joins through the server's Tailscale address and uses:

```text
--node-ip <Legion Tailscale IPv4>
--flannel-iface tailscale0
```

Use the scripts in `scripts/` to generate these settings. Do not store the K3s
join token or an administrator kubeconfig in this repository.

## Validation

Check node addresses and pod placement:

```bash
sudo k3s kubectl get nodes -o wide
sudo k3s kubectl get pods -A -o wide
```

Apply the cross-node connectivity test:

```bash
sudo k3s kubectl apply -f k8s/networking/cross-node-test.yaml
sudo k3s kubectl wait -n netlab \
  --for=condition=Ready pod/iperf-server-wsl \
  --timeout=180s
sudo k3s kubectl logs -n netlab job/iperf-client-jetson
```

Remove the test:

```bash
sudo k3s kubectl delete -f k8s/networking/cross-node-test.yaml
```

## GPU enablement

WSL uses the Windows NVIDIA driver projection. Do not install a Linux display
driver inside WSL. Install only NVIDIA Container Toolkit, restart `k3s-agent`,
then install NVIDIA Device Plugin.

Expected validation sequence:

```bash
nvidia-smi
sudo ./scripts/setup-nvidia-runtime.sh

helm repo add nvidia-device-plugin \
  https://nvidia.github.io/k8s-device-plugin
helm repo update

sudo helm upgrade --install nvidia-device-plugin \
  nvidia-device-plugin/nvidia-device-plugin \
  --kubeconfig /etc/rancher/k3s/k3s.yaml \
  --namespace nvidia-device-plugin \
  --create-namespace \
  --values k8s/nvidia-device-plugin/values.yaml

sudo k3s kubectl get runtimeclass nvidia
sudo k3s kubectl get node apailegion \
  -o jsonpath='{.status.allocatable.nvidia\.com/gpu}{"\n"}'
sudo k3s kubectl apply -f k8s/gpu-test.yaml
sudo k3s kubectl logs -n gpu-lab pod/cuda-gpu-test
```

The GPU path is considered validated only after the node reports
`nvidia.com/gpu` and both test manifests complete successfully.

## Operational notes

- A WSL shutdown stops the K3s agent. The node returns after WSL and
  `k3s-agent` start again.
- Tailscale authentication is external state. Never commit auth keys or login
  URLs.
- TensorRT engine plans are tied to GPU architecture and TensorRT version.
  Build separate engines on Jetson and RTX hardware.
- The Jetson and Legion use different CPU architectures. Workload images must
  be multi-architecture or explicitly pinned to a compatible node.
- The Tailscale health warning about `CONNMARK --nfmask` observed on the Jetson
  did not prevent direct Tailscale or cross-node pod connectivity, but should be
  revisited before treating this setup as production-like.

## Troubleshooting

Confirm the underlay:

```bash
tailscale status
tailscale ping apailegion
ip -brief address show tailscale0
```

Confirm Flannel selected the tunnel:

```bash
ip -d link show flannel.1
ip route | grep 10.42
sudo journalctl -u k3s --since "10 minutes ago" | grep -Ei 'flannel|vxlan'
```

On the WSL worker, use `k3s-agent` instead of `k3s` in the journal command.

If `kubectl` on the agent tries `localhost:8080`, no kubeconfig is configured
there. Run administrative commands on the control-plane or provide a protected,
out-of-repository kubeconfig whose server address uses Tailscale.
