#!/usr/bin/env bash
set -Eeuo pipefail

# Install/build current CPU-only llama.cpp + ik_llama.cpp and download the GGUF.
# Intended location: /workspace/models/bootstrap_llm.sh
# The script derives all paths from its own directory.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ENGINE_ROOT="${SCRIPT_DIR}/engines"
MODEL_DIR="${SCRIPT_DIR}/llm"
MODEL_REPO="${MODEL_REPO:-Athithya117/Qwen3.5-0.8B-GGUF}"
MODEL_FILE="${MODEL_FILE:-Qwen3.5-0.8B-Q4_K_M.gguf}"
BUILD_JOBS="${BUILD_JOBS:-4}"

LLAMA_REPO="https://github.com/ggml-org/llama.cpp.git"
LLAMA_BRANCH="master"
IK_REPO="https://github.com/ikawrakow/ik_llama.cpp.git"
IK_BRANCH="main"

log() { printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

[[ "${BUILD_JOBS}" =~ ^[1-9][0-9]*$ ]] || die "BUILD_JOBS must be a positive integer (got: ${BUILD_JOBS})."
install_system_dependencies() {
    if ! command -v apt-get >/dev/null 2>&1; then
        log "apt-get not found; skipping system package installation. Ensure CMake, GCC/G++, Git, curl, OpenMP and libcurl development files are installed."
        return
    fi

    local apt_prefix=()
    if [[ "${EUID}" -ne 0 ]]; then
        command -v sudo >/dev/null 2>&1 || die "Run as root or install dependencies manually (sudo is unavailable)."
        apt_prefix=(sudo)
    fi

    log "Installing Ubuntu/Debian build dependencies"
    "${apt_prefix[@]}" apt-get update
    DEBIAN_FRONTEND=noninteractive "${apt_prefix[@]}" apt-get install -y \
        build-essential git cmake pkg-config curl ca-certificates \
        libcurl4-openssl-dev libssl-dev libgomp1 python3 python3-venv python3-pip
}

sync_repo() {
    local url="$1"
    local dest="$2"
    local label="$3"
    local branch="$4"

    if [[ -d "${dest}/.git" ]]; then
        log "Updating ${label} source"
        if [[ -n "$(git -C "${dest}" status --porcelain)" ]]; then
            die "${dest} has local changes. Commit/stash them or move the directory before rerunning; bootstrap will not overwrite edits."
        fi
        git -C "${dest}" fetch --depth 1 origin "${branch}"
        git -C "${dest}" checkout -B "${branch}" FETCH_HEAD
    elif [[ -e "${dest}" ]]; then
        die "${dest} exists but is not a Git repository. Move it aside and rerun."
    else
        log "Cloning ${label}"
        git clone --depth 1 --branch "${branch}" --single-branch "${url}" "${dest}"
    fi
}

build_cpu_server() {
    local src="$1"
    local label="$2"
    local build_dir="${src}/build"
    local binary="${build_dir}/bin/llama-server"

    log "Configuring ${label} for the native CPU (GGML_NATIVE=ON)"
    cmake -S "${src}" -B "${build_dir}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_NATIVE=ON \
        -DGGML_OPENMP=ON \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=ON

    log "Building ${label} llama-server using ${BUILD_JOBS} job(s)"
    cmake --build "${build_dir}" --config Release --target llama-server -j "${BUILD_JOBS}"

    [[ -x "${binary}" ]] || die "Build completed but server binary was not found at ${binary}. Inspect ${build_dir}."
    "${binary}" --version >/dev/null
    log "${label} server verified: ${binary}"
}

install_system_dependencies

# Verify tools only after the package-install step has had a chance to provide them.
for tool in git cmake python3 g++ pkg-config; do
    command -v "${tool}" >/dev/null 2>&1 || die "Required build tool '${tool}' is missing. Install it and rerun."
done

mkdir -p "${ENGINE_ROOT}" "${MODEL_DIR}"

MODEL_PATH="${MODEL_DIR}/${MODEL_FILE}"
MODEL_MIN_BYTES="${MODEL_MIN_BYTES:-300000000}"
if [[ ! -s "${MODEL_PATH}" ]] || [[ "$(stat -c '%s' "${MODEL_PATH}" 2>/dev/null || echo 0)" -lt "${MODEL_MIN_BYTES}" ]] || [[ "$(head -c 4 "${MODEL_PATH}" 2>/dev/null || true)" != "GGUF" ]]; then
    # Prefer an already-installed hf CLI; otherwise use /workspace/data/.venv
    # when it already contains hf, and finally a private venv under models/.
    if [[ -x "${WORKSPACE_ROOT}/data/.venv/bin/hf" ]]; then
        HF_BIN="${WORKSPACE_ROOT}/data/.venv/bin/hf"
    elif command -v hf >/dev/null 2>&1; then
        HF_BIN="$(command -v hf)"
    else
        HF_VENV="${SCRIPT_DIR}/.venv"
        if [[ ! -x "${HF_VENV}/bin/python" ]]; then
            log "Creating a private Hugging Face CLI environment: ${HF_VENV}"
            python3 -m venv "${HF_VENV}"
        fi
        log "Installing huggingface_hub into ${HF_VENV}"
        "${HF_VENV}/bin/python" -m pip install --upgrade pip huggingface_hub
        HF_BIN="${HF_VENV}/bin/hf"
    fi

    [[ -x "${HF_BIN}" ]] || die "Hugging Face CLI not found/executable: ${HF_BIN}"
    log "Downloading ${MODEL_REPO}/${MODEL_FILE}"
    HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}" \
        "${HF_BIN}" download "${MODEL_REPO}" "${MODEL_FILE}" --local-dir "${MODEL_DIR}"
fi

[[ -s "${MODEL_PATH}" ]] || die "Model download did not produce ${MODEL_PATH}."
MODEL_BYTES="$(stat -c '%s' "${MODEL_PATH}")"
[[ "${MODEL_BYTES}" -ge "${MODEL_MIN_BYTES}" ]] || die "Model file appears incomplete (${MODEL_BYTES} bytes): ${MODEL_PATH}"
[[ "$(head -c 4 "${MODEL_PATH}")" == "GGUF" ]] || die "Model file does not have the GGUF magic header: ${MODEL_PATH}"
log "Model ready: ${MODEL_PATH} ($(numfmt --to=iec-i --suffix=B "${MODEL_BYTES}" 2>/dev/null || echo "${MODEL_BYTES} bytes"))"

sync_repo "${LLAMA_REPO}" "${ENGINE_ROOT}/llama.cpp" "llama.cpp" "${LLAMA_BRANCH}"
sync_repo "${IK_REPO}" "${ENGINE_ROOT}/ik_llama.cpp" "ik_llama.cpp" "${IK_BRANCH}"

build_cpu_server "${ENGINE_ROOT}/llama.cpp" "llama.cpp"
build_cpu_server "${ENGINE_ROOT}/ik_llama.cpp" "ik_llama.cpp"

log "Build revisions"
printf 'llama.cpp:    %s\n' "$(git -C "${ENGINE_ROOT}/llama.cpp" rev-parse --short HEAD)"
printf 'ik_llama.cpp: %s\n' "$(git -C "${ENGINE_ROOT}/ik_llama.cpp" rev-parse --short HEAD)"

cat <<EOF

Bootstrap complete.

Model:
  ${MODEL_PATH}

Servers:
  ${ENGINE_ROOT}/llama.cpp/build/bin/llama-server
  ${ENGINE_ROOT}/ik_llama.cpp/build/bin/llama-server

Next:
  python3 "${SCRIPT_DIR}/compare_llm_engines.py" --repeats 3

Optional to use more compile parallelism (more RAM/heat):
  BUILD_JOBS=6 bash "${SCRIPT_DIR}/bootstrap_llm.sh"
EOF
