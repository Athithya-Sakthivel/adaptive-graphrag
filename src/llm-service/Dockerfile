# syntax=docker/dockerfile:1.7

# Use the same base image as the devcontainer, which has a working CA trust store.
FROM mcr.microsoft.com/devcontainers/base:ubuntu26.04@sha256:b1e3b08610ae0b2599f57c3a492eebc93d1d26dbaefb4a0bad0caa7149e4ec95 AS build

ARG IK_LLAMA_REF=39fc3128ccc428314a64b98de09af24bd7bb4990
ARG BUILD_JOBS=1

ENV DEBIAN_FRONTEND=noninteractive
ENV LC_ALL=C.UTF-8

# APT bootstrap: we still force HTTPS and use the host CA bundle for maximum reliability.
RUN --mount=type=secret,id=host_ca,target=/run/secrets/host-ca.pem,required=true,mode=0444 \
    set -eux; \
    find /etc/apt -type f \
        \( -name '*.sources' -o -name 'sources.list' \) \
        -exec sed -i \
            -e 's|http://archive.ubuntu.com/ubuntu|https://archive.ubuntu.com/ubuntu|g' \
            -e 's|http://security.ubuntu.com/ubuntu|https://security.ubuntu.com/ubuntu|g' \
            {} +; \
    apt-get \
        -o Acquire::https::CAInfo=/run/secrets/host-ca.pem \
        -o Acquire::Retries=5 \
        update; \
    apt-get \
        -o Acquire::https::CAInfo=/run/secrets/host-ca.pem \
        -o Acquire::Retries=5 \
        install -y --no-install-recommends \
            ca-certificates \
            build-essential \
            cmake \
            git \
            pkg-config \
            libcurl4-openssl-dev; \
    rm -rf /var/lib/apt/lists/*

# Fetch the exact upstream version.
RUN --mount=type=secret,id=host_ca,target=/run/secrets/host-ca.pem,required=true,mode=0444 \
    set -eux; \
    git init /src; \
    git -C /src remote add origin \
        https://github.com/ikawrakow/ik_llama.cpp.git; \
    GIT_SSL_CAINFO=/run/secrets/host-ca.pem \
        git -C /src fetch --depth 1 origin "${IK_LLAMA_REF}"; \
    git -C /src checkout --detach FETCH_HEAD; \
    test "$(git -C /src rev-parse HEAD)" = "${IK_LLAMA_REF}"

WORKDIR /src

# Ryzen 7 5700U native CPU build.
RUN set -eux; \
    cmake -S /src -B /src/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_NATIVE=ON \
        -DGGML_OPENMP=ON \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_CURL=ON; \
    cmake --build /src/build \
        --config Release \
        --parallel "${BUILD_JOBS}"; \
    install -D -m 0755 \
        /src/build/bin/llama-server \
        /opt/ik/bin/llama-server; \
    mkdir -p /opt/ik/lib; \
    find /src/build -type f -name '*.so*' \
        -exec cp -a -t /opt/ik/lib/ '{}' +; \
    find /src/build -type l -name '*.so*' \
        -exec cp -a -t /opt/ik/lib/ '{}' +; \
    test -x /opt/ik/bin/llama-server

# ============================================================
# RUNTIME STAGE
# ============================================================

FROM mcr.microsoft.com/devcontainers/base:ubuntu26.04@sha256:b1e3b08610ae0b2599f57c3a492eebc93d1d26dbaefb4a0bad0caa7149e4ec95 AS runtime

ARG IK_LLAMA_REF

ENV DEBIAN_FRONTEND=noninteractive
ENV LC_ALL=C.UTF-8
ENV LD_LIBRARY_PATH=/opt/ik/lib
ENV HOME=/tmp
ENV TMPDIR=/tmp

# The runtime image gets the same trusted CA bundle during APT.
RUN --mount=type=secret,id=host_ca,target=/run/secrets/host-ca.pem,required=true,mode=0444 \
    set -eux; \
    find /etc/apt -type f \
        \( -name '*.sources' -o -name 'sources.list' \) \
        -exec sed -i \
            -e 's|http://archive.ubuntu.com/ubuntu|https://archive.ubuntu.com/ubuntu|g' \
            -e 's|http://security.ubuntu.com/ubuntu|https://security.ubuntu.com/ubuntu|g' \
            {} +; \
    apt-get \
        -o Acquire::https::CAInfo=/run/secrets/host-ca.pem \
        -o Acquire::Retries=5 \
        update; \
    apt-get \
        -o Acquire::https::CAInfo=/run/secrets/host-ca.pem \
        -o Acquire::Retries=5 \
        install -y --no-install-recommends \
            ca-certificates \
            curl \
            libcurl4t64 \
            libgomp1 \
            libstdc++6 \
            libgcc-s1 \
            passwd; \
    rm -rf /var/lib/apt/lists/*; \
    groupadd --gid 10001 ik; \
    useradd \
        --uid 10001 \
        --gid 10001 \
        --home-dir /tmp \
        --no-create-home \
        --shell /usr/sbin/nologin ik; \
    mkdir -p /models /opt/ik

# Copy executable and required build libraries.
COPY --from=build --chown=10001:10001 /opt/ik/ /opt/ik/

# Embed the model into the image.
COPY --chown=10001:10001 \
    models/llm/Qwen3.5-0.8B-Q4_K_M.gguf \
    /models/Qwen3.5-0.8B-Q4_K_M.gguf

# Runtime dependency validation.
RUN set -eux; \
    chmod 0444 /models/Qwen3.5-0.8B-Q4_K_M.gguf; \
    ldd /opt/ik/bin/llama-server > /tmp/ik-ldd.txt; \
    cat /tmp/ik-ldd.txt; \
    if grep -q 'not found' /tmp/ik-ldd.txt; then \
        echo "Missing runtime shared library"; \
        exit 1; \
    fi; \
    rm -f /tmp/ik-ldd.txt

USER 10001:10001

WORKDIR /opt/ik

LABEL org.opencontainers.image.title="Qwen3.5 IK LLM Service"
LABEL org.opencontainers.image.revision="${IK_LLAMA_REF}"

EXPOSE 8099

STOPSIGNAL SIGTERM

HEALTHCHECK \
    --interval=30s \
    --timeout=5s \
    --start-period=180s \
    --retries=3 \
    CMD ["curl", "--fail", "--silent", "http://127.0.0.1:8099/health"]

ENTRYPOINT ["/opt/ik/bin/llama-server"]

# Tested AMD Ryzen 7 configuration.
CMD /opt/ik/bin/llama-server -m /models/Qwen3.5-0.8B-Q4_K_M.gguf -c 4096 -t 4 -tb 7 -b 2048 -ub 1024 -np 1 -fa on --host 0.0.0.0 --port 8099 -rtr --api-key "$LLM_API_KEY"
