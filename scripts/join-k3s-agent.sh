#!/usr/bin/env bash
set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
  echo "Run this script with sudo while preserving K3S_URL and K3S_TOKEN." >&2
  exit 1
fi

: "${K3S_URL:?Set K3S_URL to the server Tailscale URL, for example https://100.x.y.z:6443}"
: "${K3S_TOKEN:?Read K3S_TOKEN from the server at runtime; never store it in Git}"

if ! command -v tailscale >/dev/null 2>&1; then
  echo "Tailscale is required. Run scripts/setup-tailscale.sh first." >&2
  exit 1
fi

TAILSCALE_IP=${TAILSCALE_IP:-$(tailscale ip -4 | sed -n '1p')}
if [[ -z ${TAILSCALE_IP} ]]; then
  echo "No Tailscale IPv4 address found. Run 'sudo tailscale up' first." >&2
  exit 1
fi

curl -sfL https://get.k3s.io | \
  K3S_URL="${K3S_URL}" \
  K3S_TOKEN="${K3S_TOKEN}" \
  sh -s - agent \
    --node-ip "${TAILSCALE_IP}" \
    --flannel-iface tailscale0

echo "K3s agent joined through Tailscale as ${TAILSCALE_IP}."
