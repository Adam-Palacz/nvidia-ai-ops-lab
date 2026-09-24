#!/usr/bin/env bash
set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
  echo "Run this script with sudo." >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable. Under WSL, update the Windows NVIDIA driver;" >&2
  echo "do not install a Linux display driver inside WSL." >&2
  exit 1
fi

apt-get update
apt-get install -y --no-install-recommends ca-certificates curl gnupg

keyring=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
list_file=/etc/apt/sources.list.d/nvidia-container-toolkit.list

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey |
  gpg --dearmor --yes -o "${keyring}"

curl -fsSL \
  https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list |
  sed "s#deb https://#deb [signed-by=${keyring}] https://#" \
    >"${list_file}"

apt-get update
apt-get install -y nvidia-container-toolkit

if systemctl cat k3s-agent.service >/dev/null 2>&1; then
  service_name=k3s-agent
elif systemctl cat k3s.service >/dev/null 2>&1; then
  service_name=k3s
else
  echo "K3s is not installed. Install the server or agent first." >&2
  exit 1
fi

# K3s discovers nvidia-container-runtime from PATH and regenerates its managed
# containerd configuration on restart. Do not edit config.toml directly.
systemctl restart "${service_name}"

config=/var/lib/rancher/k3s/agent/etc/containerd/config.toml
for _ in {1..30}; do
  if [[ -f ${config} ]] && grep -q nvidia "${config}"; then
    echo "NVIDIA runtime discovered in ${config}."
    nvidia-smi
    exit 0
  fi
  sleep 1
done

echo "K3s restarted, but the generated containerd config has no NVIDIA runtime." >&2
echo "Check: journalctl -u ${service_name} --since '5 minutes ago'" >&2
exit 1
