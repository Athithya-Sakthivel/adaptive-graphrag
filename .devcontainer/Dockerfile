# PREREQUISITES: Docker running, enabled on boot, and must be accessible without sudo(sudo usermod -aG docker $USER && newgrp docker).

FROM mcr.microsoft.com/devcontainers/base:ubuntu26.04@sha256:b1e3b08610ae0b2599f57c3a492eebc93d1d26dbaefb4a0bad0caa7149e4ec95

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates curl gnupg lsb-release sudo openssh-client socat \
       gh jq apt-transport-https unzip vim make tree \
       python3-pip python3-venv \
       postgresql-client-common postgresql-client \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
         | gpg --dearmor -o /etc/apt/keyrings/docker.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
         > /etc/apt/sources.list.d/docker.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends docker-ce-cli qemu-user-binfmt \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
