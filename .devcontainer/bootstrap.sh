#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# bootstrap.sh — runs as root inside the devcontainer (postCreateCommand).
# No sudo, no vscode user, no docker-group juggling.
# ---------------------------------------------------------------------------
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

# ===========================================================================
# 1. Base packages
# ===========================================================================
echo "==> Installing base packages..."
apt-get update -qq
apt-get install -y -qq --no-install-recommends \
  ca-certificates curl gnupg lsb-release unzip vim make tree jq \
  python3-pip python3-venv \
  libicu-dev qemu-user-binfmt

# ===========================================================================
# 2. Azure CLI (pinned)
# ===========================================================================
install_azure_cli() {
  local AZ_VERSION="2.87.0"
  # Microsoft's official docs list Ubuntu 26.04 (Resolute Raccoon) as tested,
  # but practical reports confirm the apt repo still publishes up to "noble".
  # "noble" packages are compatible with 26.04 and are the reliable choice.
  local AZ_DIST="noble"
  local AZ_PKG="azure-cli=${AZ_VERSION}-1~${AZ_DIST}"
  local SOURCES_FILE="/etc/apt/sources.list.d/azure-cli.sources"
  local KEYRING="/etc/apt/keyrings/microsoft.gpg"

  if command -v az >/dev/null 2>&1; then
    local current
    current="$(az version --query '"azure-cli"' -o tsv 2>/dev/null || true)"
    if [[ "$current" == "$AZ_VERSION" ]]; then
      echo "Azure CLI $AZ_VERSION already installed."
      return 0
    fi
    echo "Azure CLI $current found; installing $AZ_VERSION..."
  else
    echo "Installing Azure CLI $AZ_VERSION..."
  fi

  if [[ ! -f "$SOURCES_FILE" ]]; then
    mkdir -p /etc/apt/keyrings
    curl -sLS https://packages.microsoft.com/keys/microsoft.asc \
      | gpg --dearmor > "$KEYRING"
    chmod go+r "$KEYRING"

    cat > "$SOURCES_FILE" <<EOF
Types: deb
URIs: https://packages.microsoft.com/repos/azure-cli/
Suites: ${AZ_DIST}
Components: main
Architectures: $(dpkg --print-architecture)
Signed-by: ${KEYRING}
EOF
  fi

  apt-get update -qq
  if ! apt-get install -y --allow-downgrades --allow-change-held-packages "$AZ_PKG"; then
    echo "ERROR: $AZ_PKG not found in the ${AZ_DIST} suite." >&2
    echo "       Check available versions with: apt-cache policy azure-cli" >&2
    return 1
  fi

  echo "Azure CLI $(az version --query '"azure-cli"' -o tsv) installed."
}

install_azure_cli

# ===========================================================================
# 3. Bicep CLI (pinned, direct binary download)
# ===========================================================================
install_bicep() {
  local BICEP_VERSION="0.48.1"
  local BICEP_BIN="bicep-linux-x64"
  local INSTALL_DIR="/usr/local/bin"
  local BASE_URL="https://github.com/Azure/bicep/releases/download/v${BICEP_VERSION}"

  if command -v bicep >/dev/null 2>&1; then
    local current
    current="$(bicep --version 2>/dev/null | awk '{print $4}' | head -1)"
    if [[ "$current" == "$BICEP_VERSION" ]]; then
      echo "Bicep $BICEP_VERSION already installed."
      return 0
    fi
    echo "Bicep $current found; installing $BICEP_VERSION..."
  else
    echo "Installing Bicep $BICEP_VERSION..."
  fi

  local TMP_DIR
  TMP_DIR="$(mktemp -d)"
  trap 'rm -rf "$TMP_DIR"' RETURN

  curl -fsSL "${BASE_URL}/${BICEP_BIN}" -o "${TMP_DIR}/${BICEP_BIN}"

  # Checksum file name varies between releases; skip silently if absent.
  if curl -fsSL "${BASE_URL}/${BICEP_BIN}.sha256" -o "${TMP_DIR}/${BICEP_BIN}.sha256" 2>/dev/null; then
    (cd "$TMP_DIR" && sha256sum -c "${BICEP_BIN}.sha256")
  fi

  chmod +x "${TMP_DIR}/${BICEP_BIN}"
  mv "${TMP_DIR}/${BICEP_BIN}" "${INSTALL_DIR}/bicep"
  echo "Bicep $(bicep --version | head -1) installed."
}

install_bicep

# ===========================================================================
# 4. Python tooling (pinned)
# ===========================================================================
echo "==> Installing Python tooling..."
python3 -m pip install --break-system-packages --ignore-installed -q \
  pytest==9.0.3 \
  pre-commit==4.2.0 \
  huggingface_hub==2.4.0 \

apt-get update

apt-get install -y python3.14-dev


# ===========================================================================
# 5. Pre-commit hooks
# ===========================================================================
if [[ -f .pre-commit-config.yaml ]]; then
  echo "==> Installing pre-commit hooks..."
  pre-commit install --install-hooks
else
  echo "==> No .pre-commit-config.yaml; skipping pre-commit install."
fi

curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sudo sh -s -- -b /usr/local/bin v0.76.2


echo "==> bootstrap.sh complete."
