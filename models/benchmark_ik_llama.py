#!/usr/bin/env python3
"""IK-only CPU prefill/TTFT benchmark for ik_llama.cpp.

Targets the supplied layout:
  model:  /workspace/models/llm/Qwen3.5-0.8B-Q4_K_M.gguf
  server: /workspace/models/engines/ik_llama.cpp/build/bin/llama-server

What it measures
----------------
* no-prefix-reuse prefill: POST /completion, n_predict=0, cache_prompt=false
* no-prefix-reuse TTFT:   streaming POST /completion, cache_prompt=false
* client request latency, server timing fields, server-process CPU core use,
  cgroup throttling deltas, CPU frequency snapshots when exposed by the VM

Important accounting rule
-------------------------
The IK server API documents `tokens_evaluated` as the total prompt token count.
`tokens_cached` describes the slot's cached/past-token state and must NOT be
added to `timings.prompt_n`. Doing so can double-count the same prompt. This
script keeps every raw field and treats each field according to its semantics.

This is a benchmark/diagnostic tool, not a guarantee of a 3-second SLA. It never
starts mainline llama.cpp and never kills a process it did not start.
Python standard library only.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
import re
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = SCRIPT_DIR / "llm" / "Qwen3.5-0.8B-Q4_K_M.gguf"
DEFAULT_SERVER = SCRIPT_DIR / "engines" / "ik_llama.cpp" / "build" / "bin" / "llama-server"
DEFAULT_OUTPUT = SCRIPT_DIR / "ik_llama_benchmark.json"
DEFAULT_LOG_DIR = SCRIPT_DIR / "logs_ik"
DEFAULT_TARGETS = (1024, 2048, 5120)

PASSAGES = (
    "The retrieved record states that an application is reviewed against the published eligibility criteria, the submission date, and the supporting documents listed in the applicable schedule.",
    "The policy excerpt explains that an exception requires written reasons, approval by the designated reviewer, and a record of the decision in the case file.",
    "The procedure describes a notice period before a decision takes effect. The period may vary with the category of request and the date on which the complete submission is received.",
    "The guidance distinguishes mandatory evidence from optional material. Missing mandatory evidence can result in a request for clarification or a decision based on the available record.",
    "The review notes require the decision-maker to identify the governing section, explain how the relevant facts satisfy that section, and cite the documents considered.",
    "A subsequent review may be available where the applicant identifies a material factual error, supplies new evidence, or demonstrates that the required procedure was not followed.",
    "The schedule assigns responsibility to the receiving office for registering the request, confirming receipt, and preserving the supporting materials with the official record.",
    "The record cautions that an indicative target date is not a guaranteed completion date. Delays may occur where further verification is needed.",
)


def log(message: str = "") -> None:
    print(message, flush=True)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    """Convert values to JSON-safe types and replace non-finite floats with null."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps(json_safe(data), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def read_text(path: Path, limit: int = 1_000_000) -> str | None:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return handle.read(limit)
    except OSError:
        return None


def read_tail(path: Path, count: int = 100) -> str:
    text = read_text(path, limit=2_000_000)
    return "\n".join(text.splitlines()[-count:]) if text else "<log unavailable>"


def as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def first_float(*values: Any) -> float | None:
    for value in values:
        number = as_float(value)
        if number is not None:
            return number
    return None


def nearest_rank(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p * len(ordered)) - 1)]


def stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "median": None, "mean": None, "p90": None, "min": None, "max": None}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "median": float(statistics.median(ordered)),
        "mean": float(statistics.fmean(ordered)),
        "p90": nearest_rank(ordered, 0.90),
        "min": ordered[0],
        "max": ordered[-1],
    }


def http_json(url: str, payload: dict[str, Any] | None = None, timeout: float = 120.0) -> dict[str, Any]:
    headers = {"Accept": "application/json"}
    body = None
    method = "GET"
    if payload is not None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        headers["Content-Type"] = "application/json"
        method = "POST"
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail[:2500]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Request failed for {url}: {exc}") from exc
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Non-JSON response from {url}: {raw[:500]!r}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Expected JSON object from {url}; got {type(parsed).__name__}")
    return parsed


def tokenize(base_url: str, prompt: str, timeout: float) -> list[int]:
    result = http_json(
        f"{base_url}/tokenize",
        {"content": prompt, "add_special": True, "parse_special": True},
        timeout=timeout,
    )
    tokens = result.get("tokens")
    if not isinstance(tokens, list) or any(not isinstance(x, int) for x in tokens):
        raise RuntimeError(f"Unexpected /tokenize response: {json.dumps(result)[:1500]}")
    return tokens


def make_evidence_blob(chars_per_passage: int) -> str:
    chunks: list[str] = []
    for i in range(1, 11):
        template = PASSAGES[(i - 1) % len(PASSAGES)]
        source = f"policy-record-{i:03d}"
        section = f"section-{i:02d}"
        date = f"202{(i - 1) % 6}-0{(i - 1) % 9 + 1}-1{(i - 1) % 9}"
        seed = (
            f"{template} The evidence item is {source}, filed under {section}, with record date {date}. "
            f"The evidence reference is RAG-{i:05d}. The reviewer must distinguish the recorded requirement "
            "from an explanatory note, identify the relevant facts, and cite this source when relying on the statement."
        )
        repetitions = max(1, chars_per_passage // len(seed) + 1)
        body = ((seed + " ") * repetitions)[:chars_per_passage]
        chunks.append(f"[RAG CHUNK {i:02d}/10 | source={source} | {section} | date={date}]\n{body}")
    return "\n\n".join(chunks)


def build_prompt(base_url: str, target: int, timeout: float) -> tuple[str, list[int], float]:
    prefix = (
        "<|im_start|>system\n"
        "You are an evidence-grounded retrieval assistant. Answer only from the retrieved passages. "
        "Do not invent facts. Cite passage identifiers. Be concise and directly answer the question.\n"
        "<|im_end|>\n<|im_start|>user\n"
        "Question: Based only on the retrieved passages, summarize the relevant policy conditions and cite the passage identifiers.\n\n"
        "Retrieved passages:\n"
    )
    suffix = (
        "\n\nProvide a concise answer. Cite the passage identifiers supporting each factual statement.\n"
        "<|im_end|>\n<|im_start|>assistant\n"
    )
    started = time.perf_counter()
    low, high = 0, max(5000, target * 3)
    best_prompt: str | None = None
    best_tokens: list[int] | None = None
    best_delta = float("inf")
    while low <= high:
        chars = (low + high) // 2
        candidate = prefix + make_evidence_blob(chars) + suffix
        ids = tokenize(base_url, candidate, timeout)
        delta = abs(len(ids) - target)
        if delta < best_delta:
            best_prompt, best_tokens, best_delta = candidate, ids, delta
        if len(ids) < target:
            low = chars + 1
        elif len(ids) > target:
            high = chars - 1
        else:
            break
    tolerance = max(20, int(target * 0.015))
    if best_prompt is None or best_tokens is None or best_delta > tolerance:
        raise RuntimeError(f"Could not build {target}-token prompt within ±{tolerance}; closest={len(best_tokens) if best_tokens else None}")
    return best_prompt, best_tokens, (time.perf_counter() - started) * 1000.0


def extract_accounting(response: dict[str, Any], expected: int, tolerance: int) -> dict[str, Any]:
    timings = response.get("timings") if isinstance(response.get("timings"), dict) else {}
    prompt_n = as_float(timings.get("prompt_n"))
    cache_n = as_float(timings.get("cache_n"))
    tokens_evaluated = as_float(response.get("tokens_evaluated"))
    tokens_cached = as_float(response.get("tokens_cached"))

    # Per IK's server API: tokens_evaluated is the total prompt-token count.
    # tokens_cached is the slot's reusable/past token state, not an additive
    # count of tokens reused during this request. Never add it to prompt_n.
    if tokens_evaluated is not None:
        total = tokens_evaluated
        source = "response.tokens_evaluated (documented total)"
    elif prompt_n is not None and cache_n is not None:
        total = prompt_n + cache_n
        source = "timings.prompt_n + timings.cache_n"
    elif prompt_n is not None:
        total = prompt_n
        source = "timings.prompt_n (cache count unavailable; tokens_cached not added)"
    else:
        total = None
        source = "unavailable"

    if total is None:
        status = "unavailable"
    elif abs(total - expected) <= tolerance:
        status = "match"
    else:
        status = "mismatch"

    return {
        "expected_prompt_tokens": expected,
        "reported_total_prompt_tokens": total,
        "accounting_source": source,
        "accounting_status": status,
        "timings_prompt_n": prompt_n,
        "timings_cache_n": cache_n,
        "response_tokens_evaluated": tokens_evaluated,
        "response_tokens_cached_diagnostic_only": tokens_cached,
        "timings": timings,
    }


def proc_cpu_seconds(pid: int) -> float | None:
    """Total user+system CPU seconds for process (all its threads)."""
    text = read_text(Path(f"/proc/{pid}/stat"), limit=10000)
    if not text:
        return None
    # comm is parenthesized and can contain spaces; split only after final ')'.
    end = text.rfind(")")
    if end < 0:
        return None
    fields = text[end + 2:].split()  # starts at field 3 (state)
    try:
        utime_ticks = int(fields[11])  # /proc stat field 14
        stime_ticks = int(fields[12])  # /proc stat field 15
        ticks_per_second = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
        return (utime_ticks + stime_ticks) / float(ticks_per_second)
    except (IndexError, ValueError, OSError, KeyError):
        return None


def cgroup_cpu_stat() -> dict[str, int]:
    result: dict[str, int] = {}
    text = read_text(Path("/sys/fs/cgroup/cpu.stat"), limit=10000)
    if text:
        for line in text.splitlines():
            parts = line.split()
            if len(parts) == 2:
                try:
                    result[parts[0]] = int(parts[1])
                except ValueError:
                    continue
    return result


def cpu_frequency_khz() -> dict[str, Any]:
    values: list[int] = []
    for path in Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_cur_freq"):
        text = read_text(path, limit=100)
        if text:
            try:
                values.append(int(text.strip()))
            except ValueError:
                pass
    return {
        "count": len(values),
        "median_khz": int(statistics.median(values)) if values else None,
        "min_khz": min(values) if values else None,
        "max_khz": max(values) if values else None,
    }


def request_resource_snapshot(pid: int) -> dict[str, Any]:
    return {
        "process_cpu_seconds": proc_cpu_seconds(pid),
        "cgroup_cpu_stat": cgroup_cpu_stat(),
        "cpu_frequency": cpu_frequency_khz(),
    }


def resource_delta(before: dict[str, Any], after: dict[str, Any], wall_seconds: float) -> dict[str, Any]:
    cpu_before = before.get("process_cpu_seconds")
    cpu_after = after.get("process_cpu_seconds")
    cpu_used = cpu_after - cpu_before if cpu_before is not None and cpu_after is not None else None
    stat_before = before.get("cgroup_cpu_stat") or {}
    stat_after = after.get("cgroup_cpu_stat") or {}
    cgroup_delta = {
        key: stat_after[key] - stat_before[key]
        for key in ("usage_usec", "nr_periods", "nr_throttled", "throttled_usec")
        if key in stat_before and key in stat_after
    }
    return {
        "server_process_cpu_seconds": cpu_used,
        "server_process_cpu_cores_used": cpu_used / wall_seconds if cpu_used is not None and wall_seconds > 0 else None,
        "cgroup_delta": cgroup_delta,
        "cpu_frequency_before": before.get("cpu_frequency"),
        "cpu_frequency_after": after.get("cpu_frequency"),
    }


def prefill_request(
    base_url: str,
    prompt: str,
    token_ids: list[int],
    expected: int,
    input_mode: str,
    timeout: float,
    pid: int,
) -> dict[str, Any]:
    prompt_value: str | list[int] = token_ids if input_mode == "tokens" else prompt
    payload = {
        "prompt": prompt_value,
        "n_predict": 0,
        "temperature": 0.0,
        "top_k": 1,
        "cache_prompt": False,
        "stream": False,
        "timings_per_token": True,
    }
    resources_before = request_resource_snapshot(pid)
    started = time.perf_counter()
    response = http_json(f"{base_url}/completion", payload, timeout=timeout)
    wall_seconds = time.perf_counter() - started
    resources_after = request_resource_snapshot(pid)

    timings = response.get("timings") if isinstance(response.get("timings"), dict) else {}
    prompt_ms = as_float(timings.get("prompt_ms"))
    prompt_n = as_float(timings.get("prompt_n"))
    server_tps = as_float(timings.get("prompt_per_second"))
    derived_expected_tps = expected * 1000.0 / prompt_ms if prompt_ms and prompt_ms > 0 else None
    derived_prompt_n_tps = prompt_n * 1000.0 / prompt_ms if prompt_n is not None and prompt_ms and prompt_ms > 0 else None
    accounting = extract_accounting(response, expected, max(8, int(expected * 0.005)))

    return {
        "client_e2e_ms": wall_seconds * 1000.0,
        "server_prompt_ms": prompt_ms,
        "server_prompt_tps": server_tps,
        "derived_tps_using_tokenize_count": derived_expected_tps,
        "derived_tps_using_timings_prompt_n": derived_prompt_n_tps,
        "server_prompt_n": prompt_n,
        "server_cache_n": as_float(timings.get("cache_n")),
        "server_generated_tokens": first_float(timings.get("predicted_n"), response.get("tokens_predicted")),
        "response_truncated": response.get("truncated"),
        "response_stop_type": response.get("stop_type"),
        "response_keys": sorted(response.keys()),
        "accounting": accounting,
        "raw_timings": timings,
        "resource_use": resource_delta(resources_before, resources_after, wall_seconds),
    }


def stream_ttft(
    base_url: str,
    prompt: str,
    token_ids: list[int],
    input_mode: str,
    output_tokens: int,
    timeout: float,
    pid: int,
) -> dict[str, Any]:
    prompt_value: str | list[int] = token_ids if input_mode == "tokens" else prompt
    payload = {
        "prompt": prompt_value,
        "n_predict": output_tokens,
        "temperature": 0.0,
        "top_k": 1,
        "ignore_eos": True,
        "cache_prompt": False,
        "stream": True,
        "timings_per_token": True,
    }
    request = urllib.request.Request(
        f"{base_url}/completion",
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    resources_before = request_resource_snapshot(pid)
    started = time.perf_counter()
    first_byte_at: float | None = None
    first_event_at: float | None = None
    first_content_at: float | None = None
    content_chars = 0
    event_count = 0
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            while True:
                raw = response.readline()
                if not raw:
                    break
                now = time.perf_counter()
                if first_byte_at is None:
                    first_byte_at = now
                line = raw.decode("utf-8", errors="replace").strip()
                if not line or line.startswith(":") or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, dict):
                    continue
                event_count += 1
                if first_event_at is None:
                    first_event_at = now
                content = event.get("content")
                if isinstance(content, str) and content:
                    if first_content_at is None:
                        first_content_at = now
                    content_chars += len(content)
                if event.get("stop") is True:
                    break
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Streaming HTTP {exc.code}: {detail[:2000]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Streaming request failed: {exc}") from exc
    wall_seconds = time.perf_counter() - started
    resources_after = request_resource_snapshot(pid)
    if first_content_at is None:
        raise RuntimeError("Streaming response ended without a non-empty content delta; TTFT unavailable.")
    return {
        "first_byte_ms": (first_byte_at - started) * 1000.0 if first_byte_at is not None else None,
        "first_event_ms": (first_event_at - started) * 1000.0 if first_event_at is not None else None,
        "ttft_ms": (first_content_at - started) * 1000.0,
        "stream_e2e_ms": wall_seconds * 1000.0,
        "stream_content_chars": content_chars,
        "stream_events": event_count,
        "resource_use": resource_delta(resources_before, resources_after, wall_seconds),
    }


def port_is_free(host: str, port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
        return True
    except OSError:
        return False


def process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


class IKServer:
    def __init__(self, binary: Path, model: Path, host: str, port: int, log_path: Path,
                 threads: int, batch_threads: int, batch_size: int, ubatch_size: int,
                 ctx_size: int, flash_attn: str, runtime_repack: bool,
                 mlock: bool, stop_timeout: float):
        self.binary = binary
        self.model = model
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.log_path = log_path
        self.threads = threads
        self.batch_threads = batch_threads
        self.batch_size = batch_size
        self.ubatch_size = ubatch_size
        self.ctx_size = ctx_size
        self.flash_attn = flash_attn
        self.runtime_repack = runtime_repack
        self.mlock = mlock
        self.stop_timeout = stop_timeout
        self.command: list[str] = []
        self.process: subprocess.Popen[bytes] | None = None
        self.log_handle: Any = None
        self.startup_seconds: float | None = None

    def start(self, startup_timeout: float) -> None:
        if not self.binary.is_file() or not os.access(self.binary, os.X_OK):
            raise FileNotFoundError(f"ik_llama.cpp server missing/not executable: {self.binary}")
        if not self.model.is_file():
            raise FileNotFoundError(f"Model not found: {self.model}")
        if not port_is_free(self.host, self.port):
            raise RuntimeError(f"Port {self.port} is occupied. Inspect: ss -ltnp 'sport = :{self.port}'")

        self.command = [
            str(self.binary), "-m", str(self.model),
            "-c", str(self.ctx_size),
            "-t", str(self.threads),
            "-tb", str(self.batch_threads),
            "-b", str(self.batch_size),
            "-ub", str(self.ubatch_size),
            "-np", "1",
            "-fa", self.flash_attn,
            "--host", self.host,
            "--port", str(self.port),
        ]
        if self.runtime_repack:
            self.command.append("-rtr")
        if self.mlock:
            self.command.append("--mlock")

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_handle = self.log_path.open("wb")
        log("\nStarting ik_llama.cpp")
        log("Command: " + " ".join(self.command))
        started = time.perf_counter()
        try:
            self.process = subprocess.Popen(
                self.command,
                cwd=str(self.binary.parent),
                stdout=self.log_handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        except Exception:
            self.log_handle.close()
            self.log_handle = None
            raise

        deadline = time.monotonic() + startup_timeout
        last_error: str | None = None
        while time.monotonic() < deadline:
            assert self.process is not None
            exit_code = self.process.poll()
            if exit_code is not None:
                raise RuntimeError(
                    f"ik_llama.cpp exited during startup with code {exit_code}.\n"
                    f"Log: {self.log_path}\n{read_tail(self.log_path)}"
                )
            try:
                health = http_json(f"{self.base_url}/health", timeout=2.0)
                if health.get("status") == "ok":
                    self.startup_seconds = time.perf_counter() - started
                    log(f"ik_llama.cpp ready in {self.startup_seconds:.2f}s")
                    return
                last_error = str(health)
            except Exception as exc:
                last_error = str(exc)
            time.sleep(0.2)
        raise TimeoutError(
            f"Timed out waiting for ik_llama.cpp. Last result: {last_error}\n"
            f"Log: {self.log_path}\n{read_tail(self.log_path)}"
        )

    def stop(self) -> None:
        proc = self.process
        if proc is None:
            if self.log_handle is not None:
                self.log_handle.close()
                self.log_handle = None
            return
        pgid = proc.pid  # start_new_session => leader PID is its process-group ID.
        log(f"Stopping IK process group {pgid}")
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=self.stop_timeout)
        except subprocess.TimeoutExpired:
            log("SIGTERM timeout; sending SIGKILL to the benchmark-created process group")
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"Could not reap IK server PID {proc.pid}") from exc

        if process_group_exists(pgid):
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        group_deadline = time.monotonic() + 3.0
        while time.monotonic() < group_deadline and process_group_exists(pgid):
            time.sleep(0.1)

        port_deadline = time.monotonic() + 5.0
        while time.monotonic() < port_deadline and not port_is_free(self.host, self.port):
            time.sleep(0.1)
        still_busy = not port_is_free(self.host, self.port)

        self.process = None
        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None
        if still_busy:
            raise RuntimeError(
                f"IK process group exited but {self.host}:{self.port} remains occupied. "
                "The script will not kill another listener. Inspect with: "
                f"ss -ltnp 'sport = :{self.port}'"
            )
        log(f"ik_llama.cpp stopped; port {self.port} is free")

    def __enter__(self) -> "IKServer":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        try:
            self.stop()
        except Exception as cleanup_error:
            if exc is None:
                raise
            log(f"WARNING: cleanup also failed: {cleanup_error}")


def command_output(command: list[str], timeout: float = 8.0) -> str | None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
        return (result.stdout or result.stderr).strip() or None
    except (OSError, subprocess.TimeoutExpired):
        return None


def build_metadata(binary: Path) -> dict[str, Any]:
    build_dir = binary.parents[1]
    repo_dir = binary.parents[2]
    result: dict[str, Any] = {"build_dir": str(build_dir), "cmake_options": {}, "compiler_flags_evidence": []}
    cache = build_dir / "CMakeCache.txt"
    text = read_text(cache, limit=4_000_000)
    if text:
        keys = (
            "CMAKE_BUILD_TYPE", "CMAKE_CXX_COMPILER", "CMAKE_CXX_FLAGS", "GGML_NATIVE",
            "GGML_AVX", "GGML_AVX2", "GGML_AVXVNNI", "GGML_AVX512", "GGML_AVX512_VNNI",
            "GGML_OPENMP", "GGML_BLAS", "GGML_BLAS_VENDOR",
        )
        for key in keys:
            match = re.search(rf"^{re.escape(key)}:[^=]+=([^\n]*)$", text, flags=re.MULTILINE)
            if match:
                result["cmake_options"][key] = match.group(1).strip()
    evidence: list[str] = []
    candidates = [build_dir / "compile_commands.json"]
    candidates.extend(build_dir.rglob("flags.make"))
    candidates.extend(build_dir.rglob("build.ninja"))
    pattern = re.compile(r"-march=[^\s\"]+|-mcpu=[^\s\"]+|-mavx2|-mavxvnni|-mavx512[^\s\"]*|-mfma")
    total_read = 0
    for path in candidates:
        if not path.is_file() or total_read > 8_000_000:
            continue
        body = read_text(path, limit=2_000_000)
        if not body:
            continue
        total_read += len(body)
        hits = sorted(set(pattern.findall(body)))
        if hits:
            evidence.append(f"{path}: {' '.join(hits)}")
    result["compiler_flags_evidence"] = evidence[:30]
    try:
        result["git_revision"] = subprocess.check_output(
            ["git", "-C", str(repo_dir), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True, timeout=5,
        ).strip()
    except Exception:
        result["git_revision"] = None
    result["version"] = command_output([str(binary), "--version"], timeout=10)
    result["help_relevant"] = None
    help_text = command_output([str(binary), "--help"], timeout=10)
    if help_text:
        result["help_relevant"] = "\n".join(
            line for line in help_text.splitlines()
            if any(word in line.lower() for word in ("flash", "repack", "threads", "batch", "affinity", "checkpoint"))
        )[:12000]
    return result


def cpu_metadata() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor() or None,
        "logical_cpu_count": os.cpu_count(),
        "model_name": None,
        "flags": [],
        "hypervisor_flag_present": False,
        "cgroup_cpu_max": None,
        "cgroup_cpu_stat": cgroup_cpu_stat(),
        "cpuset_cpus_effective": None,
        "process_affinity": None,
        "cpu_frequency": cpu_frequency_khz(),
        "lscpu": command_output(["lscpu"], timeout=5),
    }
    cpuinfo = read_text(Path("/proc/cpuinfo"), limit=200_000)
    if cpuinfo:
        match = re.search(r"^model name\s*:\s*(.+)$", cpuinfo, flags=re.MULTILINE)
        flag_match = re.search(r"^flags\s*:\s*(.+)$", cpuinfo, flags=re.MULTILINE)
        if match:
            info["model_name"] = match.group(1).strip()
        if flag_match:
            info["flags"] = sorted(set(flag_match.group(1).split()))
            info["hypervisor_flag_present"] = "hypervisor" in info["flags"]
    try:
        info["cgroup_cpu_max"] = Path("/sys/fs/cgroup/cpu.max").read_text().strip()
    except OSError:
        pass
    try:
        info["cpuset_cpus_effective"] = Path("/sys/fs/cgroup/cpuset.cpus.effective").read_text().strip()
    except OSError:
        pass
    try:
        info["process_affinity"] = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        pass
    return info


def summarize_runs(prefill_runs: list[dict[str, Any]], ttft_runs: list[dict[str, Any]]) -> dict[str, Any]:
    def values(rows: list[dict[str, Any]], key: str) -> list[float]:
        out: list[float] = []
        for row in rows:
            value = as_float(row.get(key))
            if value is not None:
                out.append(value)
        return out

    mismatch = sum(1 for x in prefill_runs if x.get("accounting", {}).get("accounting_status") == "mismatch")
    unavailable = sum(1 for x in prefill_runs if x.get("accounting", {}).get("accounting_status") == "unavailable")
    return {
        "prefill_server_ms": stats(values(prefill_runs, "server_prompt_ms")),
        "prefill_client_ms": stats(values(prefill_runs, "client_e2e_ms")),
        "prefill_tps_using_prompt_token_count": stats(values(prefill_runs, "derived_tps_using_tokenize_count")),
        "server_reported_prompt_tps": stats(values(prefill_runs, "server_prompt_tps")),
        "server_process_cpu_cores_used": stats([
            v for x in prefill_runs
            if (v := as_float((x.get("resource_use") or {}).get("server_process_cpu_cores_used"))) is not None
        ]),
        "ttft_ms": stats(values(ttft_runs, "ttft_ms")),
        "first_byte_ms": stats(values(ttft_runs, "first_byte_ms")),
        "stream_e2e_ms": stats(values(ttft_runs, "stream_e2e_ms")),
        "accounting_mismatches": mismatch,
        "accounting_unavailable": unavailable,
    }


def parse_int_list(value: str) -> list[int]:
    try:
        items = [int(x.strip()) for x in value.split(",") if x.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not items or any(x <= 0 for x in items) or len(set(items)) != len(items):
        raise argparse.ArgumentTypeError("values must be distinct positive integers")
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-user, IK-only CPU prefill and TTFT benchmark for ik_llama.cpp."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--ik-server", type=Path, default=DEFAULT_SERVER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8098)
    parser.add_argument("--threads", type=int, default=4, help="generation threads; not the prompt-processing thread count")
    parser.add_argument("--batch-threads", type=int, default=8, help="prompt/batch threads; default is 8 physical cores on the reported 5700U")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--ubatch-size", type=int, default=512)
    parser.add_argument("--ctx-size", type=int, default=8192)
    parser.add_argument("--prompt-tokens", type=parse_int_list, default=list(DEFAULT_TARGETS))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--ttft-tokens", type=int, default=1, help="generate this many tokens; TTFT stops at the first non-empty delta")
    parser.add_argument("--input-mode", choices=("text", "tokens"), default="text", help="tokens mode pre-tokenizes outside the timed request")
    parser.add_argument("--prefill-only", action="store_true", help="skip streaming TTFT to make batch sweeps much quicker")
    parser.add_argument("--flash-attn", choices=("on", "off", "auto"), default="on")
    parser.add_argument("--runtime-repack", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mlock", action="store_true")
    parser.add_argument("--sweep", action="store_true", help="test batch/thread configurations for an 8-core Zen2 CPU")
    parser.add_argument("--compare-repack", action="store_true", help="add a matched -rtr on/off A/B using the 6144/1024/8 configuration")
    parser.add_argument("--startup-timeout", type=float, default=600.0)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--stop-timeout", type=float, default=15.0)
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def build_configs(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.sweep:
        configs = [
            {"batch_size": 2048, "ubatch_size": 512, "batch_threads": 4, "runtime_repack": args.runtime_repack},
            {"batch_size": 2048, "ubatch_size": 512, "batch_threads": 8, "runtime_repack": args.runtime_repack},
            {"batch_size": 4096, "ubatch_size": 512, "batch_threads": 8, "runtime_repack": args.runtime_repack},
            {"batch_size": 6144, "ubatch_size": 512, "batch_threads": 8, "runtime_repack": args.runtime_repack},
            {"batch_size": 6144, "ubatch_size": 1024, "batch_threads": 8, "runtime_repack": args.runtime_repack},
            {"batch_size": 6144, "ubatch_size": 2048, "batch_threads": 8, "runtime_repack": args.runtime_repack},
        ]
    else:
        configs = [{
            "batch_size": args.batch_size,
            "ubatch_size": args.ubatch_size,
            "batch_threads": args.batch_threads,
            "runtime_repack": args.runtime_repack,
        }]
    if args.compare_repack:
        base = {"batch_size": 6144, "ubatch_size": 1024, "batch_threads": 8}
        configs.extend([{**base, "runtime_repack": True}, {**base, "runtime_repack": False}])
    # Deduplicate without relying on dict hashing.
    unique: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for cfg in configs:
        key = (cfg["batch_size"], cfg["ubatch_size"], cfg["batch_threads"], cfg["runtime_repack"])
        if key not in seen:
            seen.add(key)
            unique.append(cfg)
    return unique


def validate_args(args: argparse.Namespace) -> None:
    for name in ("threads", "batch_threads", "batch_size", "ubatch_size", "ctx_size", "repeats", "ttft_tokens"):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    for name in ("startup_timeout", "request_timeout", "stop_timeout"):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if not (1 <= args.port <= 65535):
        raise SystemExit("--port must be between 1 and 65535")
    if args.ctx_size < max(args.prompt_tokens) + args.ttft_tokens + 32:
        raise SystemExit("--ctx-size must exceed largest prompt + TTFT output cap + 32 tokens")


def print_summary(result: dict[str, Any]) -> None:
    log("\nIK LLAMA.CPP RESULTS (median values)")
    log("cfg  -b     -ub    -tb  -rtr  prompt  prefill-ms  prefill-t/s  TTFT-ms  server-CPU-cores  acct")
    log("---  -----  -----  ---  ----  ------  ----------  -----------  -------  ----------------  ----")
    for cfg in result.get("configurations", []):
        p = cfg.get("parameters", {})
        for row in cfg.get("prompt_sizes", []):
            s = row.get("summary", {})
            pre_ms = s.get("prefill_server_ms", {}).get("median")
            pre_tps = s.get("prefill_tps_using_prompt_token_count", {}).get("median")
            ttft = s.get("ttft_ms", {}).get("median")
            cpu_cores = s.get("server_process_cpu_cores_used", {}).get("median")
            mismatches = s.get("accounting_mismatches", 0)
            acct = "OK" if mismatches == 0 else f"mismatch:{mismatches}"
            log(
                f"{cfg.get('index', 0):>3}  {p.get('batch_size', 0):>5}  {p.get('ubatch_size', 0):>5}  "
                f"{p.get('batch_threads', 0):>3}  {str(p.get('runtime_repack')):>4}  "
                f"{row.get('target_tokens', 0):>6}  "
                f"{pre_ms if pre_ms is not None else float('nan'):>10.1f}  "
                f"{pre_tps if pre_tps is not None else float('nan'):>11.1f}  "
                f"{ttft if ttft is not None else float('nan'):>7.1f}  "
                f"{cpu_cores if cpu_cores is not None else float('nan'):>16.2f}  {acct}"
            )


def main() -> int:
    args = parse_args()
    validate_args(args)
    args.model = args.model.expanduser().resolve()
    args.ik_server = args.ik_server.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.log_dir = args.log_dir.expanduser().resolve()
    if not args.model.is_file():
        raise SystemExit(f"Model not found: {args.model}")
    if not args.ik_server.is_file() or not os.access(args.ik_server, os.X_OK):
        raise SystemExit(f"ik_llama.cpp server missing/not executable: {args.ik_server}")
    if not port_is_free(args.host, args.port):
        raise SystemExit(f"Port {args.port} is occupied. Inspect with: ss -ltnp 'sport = :{args.port}'")

    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_dir = args.log_dir / run_id
    run_log_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output.with_name(f"{args.output.stem}_{run_id}_prompts.json")
    configs = build_configs(args)
    model_stat = args.model.stat()

    result: dict[str, Any] = {
        "schema_version": 2,
        "run_id": run_id,
        "created_at_utc": utc_now(),
        "status": "running",
        "engine": "ik_llama.cpp",
        "binary": str(args.ik_server),
        "build": build_metadata(args.ik_server),
        "hardware": {"python": sys.version, "cpu": cpu_metadata()},
        "model": {
            "path": str(args.model),
            "size_bytes": model_stat.st_size,
            "mtime_utc": dt.datetime.fromtimestamp(model_stat.st_mtime, dt.timezone.utc).isoformat(),
        },
        "settings": {
            "port": args.port,
            "generation_threads": args.threads,
            "context_size": args.ctx_size,
            "prompt_targets": args.prompt_tokens,
            "repeats": args.repeats,
            "ttft_output_tokens": args.ttft_tokens,
            "input_mode": args.input_mode,
            "prefill_only": args.prefill_only,
            "flash_attention": args.flash_attn,
            "runtime_repack": args.runtime_repack,
            "mlock": args.mlock,
            "warmup": args.warmup,
            "sweep": args.sweep,
            "compare_repack": args.compare_repack,
            "cache_prompt": False,
        },
        "prompt_manifest": str(manifest_path),
        "log_directory": str(run_log_dir),
        "configurations": [],
        "interpretation_notes": [
            "-np 1 means one server slot; -t and -tb determine inference worker counts.",
            "cache_prompt=false disables KV-prefix reuse for each timed request; OS file/page cache can still be warm.",
            "The /completion response field tokens_evaluated is used as the total prompt count.",
            "tokens_cached is stored for diagnosis only and is never added to prompt_n.",
            "n_predict=0 measures prompt evaluation without decoding output tokens.",
            "TTFT is measured from request start until the first non-empty streamed content delta.",
            "Server CPU-core use is process CPU seconds divided by wall seconds; values near 8 imply eight CPU cores were busy on average.",
            "cgroup throttling is recorded, but a VM hypervisor can still impose host scheduling contention when cpu.max is unlimited.",
        ],
    }
    atomic_json(args.output, result)

    prompts: list[dict[str, Any]] | None = None
    try:
        for config_index, config in enumerate(configs, start=1):
            log(f"\n========== IK CONFIG {config_index}/{len(configs)}: {config} ==========")
            log_path = run_log_dir / f"ik_config_{config_index:02d}.log"
            engine = IKServer(
                args.ik_server, args.model, args.host, args.port, log_path,
                args.threads, config["batch_threads"], config["batch_size"], config["ubatch_size"],
                args.ctx_size, args.flash_attn, config["runtime_repack"], args.mlock, args.stop_timeout,
            )
            config_record: dict[str, Any] = {
                "index": config_index,
                "parameters": {**config, "threads": args.threads, "flash_attn": args.flash_attn},
                "command": [],
                "log_file": str(log_path),
                "startup_seconds": None,
                "prompt_sizes": [],
                "status": "starting",
            }
            result["configurations"].append(config_record)
            atomic_json(args.output, result)

            with engine:
                engine.start(args.startup_timeout)
                config_record["command"] = engine.command
                config_record["startup_seconds"] = engine.startup_seconds
                config_record["status"] = "running"
                atomic_json(args.output, result)

                if prompts is None:
                    prompts = []
                    for target in args.prompt_tokens:
                        prompt, token_ids, builder_ms = build_prompt(engine.base_url, target, args.request_timeout)
                        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                        prompts.append({
                            "target_tokens": target,
                            "token_count": len(token_ids),
                            "tokenize_build_ms": builder_ms,
                            "prompt_sha256": digest,
                            "prompt": prompt,
                            "token_ids": token_ids,
                        })
                        log(f"Built {target}-target prompt: {len(token_ids)} tokens, sha256={digest[:12]}")
                    atomic_json(manifest_path, {
                        "created_at_utc": utc_now(),
                        "model": str(args.model),
                        "tokenizer_endpoint": "/tokenize (add_special=true, parse_special=true)",
                        "prompts": [{k: v for k, v in item.items() if k != "token_ids"} for item in prompts],
                    })

                # Re-tokenize with each new server process. This also validates
                # the prompt text/token-ID pair for the exact binary under test.
                for item in prompts:
                    check_ids = tokenize(engine.base_url, item["prompt"], args.request_timeout)
                    if check_ids != item["token_ids"]:
                        raise RuntimeError(
                            f"Tokenizer changed across server restart for prompt {item['target_tokens']}; "
                            "refusing to compare different token sequences."
                        )

                if args.warmup:
                    warm_prompt = "IK CPU benchmark warm-up. Evaluate this short prompt without generating output."
                    http_json(
                        f"{engine.base_url}/completion",
                        {"prompt": warm_prompt, "n_predict": 0, "temperature": 0.0,
                         "cache_prompt": False, "stream": False, "timings_per_token": True},
                        timeout=args.request_timeout,
                    )

                assert engine.process is not None
                for item in prompts:
                    expected = int(item["token_count"])
                    row: dict[str, Any] = {
                        "target_tokens": item["target_tokens"],
                        "actual_prompt_tokens": expected,
                        "prompt_sha256": item["prompt_sha256"],
                        "prefill_runs": [],
                        "ttft_runs": [],
                        "summary": {},
                    }
                    config_record["prompt_sizes"].append(row)
                    atomic_json(args.output, result)
                    log(f"\nIK config {config_index}: {expected} actual tokens; {args.repeats} repetitions; mode={args.input_mode}")

                    for repeat in range(args.repeats):
                        ttft: dict[str, Any] | None = None
                        if args.prefill_only or repeat % 2 == 0:
                            prefill = prefill_request(
                                engine.base_url, item["prompt"], item["token_ids"], expected,
                                args.input_mode, args.request_timeout, engine.process.pid,
                            )
                            row["prefill_runs"].append(prefill)
                            atomic_json(args.output, result)
                            if not args.prefill_only:
                                ttft = stream_ttft(
                                    engine.base_url, item["prompt"], item["token_ids"], args.input_mode,
                                    args.ttft_tokens, args.request_timeout, engine.process.pid,
                                )
                        else:
                            ttft = stream_ttft(
                                engine.base_url, item["prompt"], item["token_ids"], args.input_mode,
                                args.ttft_tokens, args.request_timeout, engine.process.pid,
                            )
                            row["ttft_runs"].append(ttft)
                            atomic_json(args.output, result)
                            prefill = prefill_request(
                                engine.base_url, item["prompt"], item["token_ids"], expected,
                                args.input_mode, args.request_timeout, engine.process.pid,
                            )
                            row["prefill_runs"].append(prefill)

                        if ttft is not None and repeat % 2 == 0:
                            row["ttft_runs"].append(ttft)
                        row["summary"] = summarize_runs(row["prefill_runs"], row["ttft_runs"])
                        atomic_json(args.output, result)

                        accounting = prefill["accounting"]
                        if accounting["accounting_status"] != "match":
                            log(
                                f"WARNING accounting={accounting['accounting_status']} expected={expected} "
                                f"reported={accounting['reported_total_prompt_tokens']} "
                                f"source={accounting['accounting_source']}"
                            )
                        resource = prefill.get("resource_use") or {}
                        ttft_text = f"TTFT={ttft['ttft_ms']:.1f} ms, " if ttft is not None else ""
                        log(
                            f"  {repeat + 1}/{args.repeats}: "
                            f"prefill={prefill.get('server_prompt_ms') if prefill.get('server_prompt_ms') is not None else float('nan'):.1f} ms, "
                            f"rate={prefill.get('derived_tps_using_tokenize_count') if prefill.get('derived_tps_using_tokenize_count') is not None else float('nan'):.1f} tok/s, "
                            f"{ttft_text}"
                            f"CPU={resource.get('server_process_cpu_cores_used') if resource.get('server_process_cpu_cores_used') is not None else float('nan'):.2f} cores, "
                            f"accounting={accounting['accounting_status']}"
                        )

                config_record["status"] = "complete"
                config_record["server_log_tail"] = read_tail(log_path, 80)
                atomic_json(args.output, result)

        result["status"] = "complete"
        result["completed_at_utc"] = utc_now()
        atomic_json(args.output, result)
        print_summary(result)
        log(f"\nResults: {args.output}")
        log(f"Prompt manifest: {manifest_path}")
        log(f"Server logs: {run_log_dir}")
        return 0

    except KeyboardInterrupt:
        result["status"] = "interrupted"
        result["updated_at_utc"] = utc_now()
        atomic_json(args.output, result)
        log("\nInterrupted. Completed requests and partial results were saved.")
        return 130
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["updated_at_utc"] = utc_now()
        if result.get("configurations"):
            active = result["configurations"][-1]
            if active.get("status") in ("starting", "running"):
                active["status"] = "failed"
                active["error"] = result["error"]
                log_path_value = active.get("log_file")
                if log_path_value:
                    active["server_log_tail"] = read_tail(Path(log_path_value), 100)
        atomic_json(args.output, result)
        log(f"\nERROR: {type(exc).__name__}: {exc}")
        log(f"Partial results saved: {args.output}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
