#!/usr/bin/env bash
set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
  echo "Run this script with sudo." >&2
  exit 1
fi

if ! command -v tailscale >/dev/null 2>&1; then
  echo "Tailscale is required. Run scripts/setup-tailscale.sh first." >&2
  exit 1
fi

TAILSCALE_IP=${TAILSCALE_IP:-$(tailscale ip -4 | sed -n '1p')}
if [[ -z ${TAILSCALE_IP} ]]; then
  echo "No Tailscale IPv4 address found. Run 'sudo tailscale up' first." >&2
  exit 1
fi

install -d -m 0755 /etc/rancher/k3s

config_file=/etc/rancher/k3s/config.yaml
if [[ -f ${config_file} ]]; then
  cp --preserve=mode,ownership,timestamps \
    "${config_file}" "${config_file}.backup.$(date +%Y%m%d%H%M%S)"
fi

temp_file=$(mktemp)
trap 'rm -f "${temp_file}"' EXIT

cat >"${temp_file}" <<EOF
flannel-backend: vxlan
flannel-iface: tailscale0
node-ip: ${TAILSCALE_IP}
advertise-address: ${TAILSCALE_IP}
tls-san:
  - ${TAILSCALE_IP}
EOF

install -m 0600 "${temp_file}" "${config_file}"

curl -sfL https://get.k3s.io | sh -s - server

echo "K3s server configured on ${TAILSCALE_IP}."
echo "Join token remains only on this host:"
echo "  /var/lib/rancher/k3s/server/node-token"
