#!/usr/bin/env bash
set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
  echo "Run this script with sudo." >&2
  exit 1
fi

if ! command -v tailscale >/dev/null 2>&1; then
  curl -fsSL https://tailscale.com/install.sh | sh
fi

systemctl enable --now tailscaled

if ! tailscale ip -4 >/dev/null 2>&1; then
  echo "Authenticate this node with Tailscale."
  tailscale up
fi

echo "Tailscale IPv4: $(tailscale ip -4 | sed -n '1p')"
tailscale status
