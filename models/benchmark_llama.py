#!/usr/bin/env python3
"""Reliable CPU benchmark runner for llama.cpp vs ik_llama.cpp.

Defaults target the user's /workspace layout:
  model:        /workspace/models/llm/Qwen3.5-0.8B-Q4_K_M.gguf
  llama.cpp:    /workspace/models/engines/llama.cpp/build/bin/llama-server
  ik_llama.cpp: /workspace/models/engines/ik_llama.cpp/build/bin/llama-server

Key design choices:
- Each engine has its own port (8097 and 8098 by default). No port hand-off race.
- Each server is started in a new process group; cleanup terminates the whole group.
- Both engines receive byte-identical prompts, created once and persisted to a manifest.
- Every measured request sets cache_prompt=false. Cache counters are recorded and
  reported as verified / detected / unavailable; missing counters do not erase results.
- Server timing fields are parsed defensively. Missing metrics are reported as n/a,
  not guessed. Prompt and generated token counts are checked when the server reports them.
- Partial JSON results are atomically saved after each completed measurement.

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
import shutil
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
DEFAULT_OUTPUT = SCRIPT_DIR / "llm_engines_compare.json"
DEFAULT_LOG_DIR = SCRIPT_DIR / "logs"
DEFAULT_LLAMA_SERVER = SCRIPT_DIR / "engines" / "llama.cpp" / "build" / "bin" / "llama-server"
DEFAULT_IK_SERVER = SCRIPT_DIR / "engines" / "ik_llama.cpp" / "build" / "bin" / "llama-server"
DEFAULT_TARGETS = (1024, 2048, 5120)

PROMPT_PREFIX = (
    "<|im_start|>system\n"
    "You are an evidence-grounded retrieval assistant. Answer only from the "
    "retrieved passages. Do not invent facts. Cite passage identifiers. "
    "Be concise and directly answer the question.\n"
    "<|im_end|>\n"
    "<|im_start|>user\n"
    "Question: Based only on the retrieved passages, summarize the relevant "
    "policy conditions and cite the passage identifiers.\n\n"
    "Retrieved passages:\n"
)
PROMPT_SUFFIX = (
    "\n\nProvide a concise answer. Cite the passage identifiers supporting "
    "each factual statement.\n"
    "<|im_end|>\n<|im_start|>assistant\n"
)
PASSAGE_TEMPLATES = (
    "The retrieved record states that an application is reviewed against the "
    "published eligibility criteria, the submission date, and the supporting "
    "documents listed in the applicable schedule.",
    "The policy excerpt explains that an exception requires written reasons, "
    "approval by the designated reviewer, and a record of the decision in the case file.",
    "The procedure describes a notice period before a decision takes effect. "
    "The period may vary with the category of request and the date on which "
    "the complete submission is received.",
    "The guidance distinguishes mandatory evidence from optional material. "
    "Missing mandatory evidence can result in a request for clarification or "
    "a decision based on the available record.",
    "The review notes require the decision-maker to identify the governing "
    "section, explain how the relevant facts satisfy that section, and cite "
    "the documents considered.",
    "A subsequent review may be available where the applicant identifies a "
    "material factual error, supplies new evidence, or demonstrates that "
    "the required procedure was not followed.",
    "The schedule assigns responsibility to the receiving office for "
    "registering the request, confirming receipt, and preserving the "
    "supporting materials with the official record.",
    "The record cautions that an indicative target date is not a guaranteed "
    "completion date. Delays may occur where further verification is needed.",
)


def log(message: str = "") -> None:
    print(message, flush=True)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def tail_file(path: Path, lines: int = 100) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return "<log unavailable>"


def as_number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else None


def first_number(*values: Any) -> float | None:
    for value in values:
        number = as_number(value)
        if number is not None:
            return number
    return None


def http_json(url: str, payload: dict[str, Any] | None = None, timeout: float = 120.0) -> dict[str, Any]:
    headers = {"Accept": "application/json"}
    body = None
    method = "GET"
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
        method = "POST"
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail[:2000]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Request failed for {url}: {exc}") from exc
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Non-JSON response from {url}: {raw[:500]!r}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Expected a JSON object from {url}; got {type(parsed).__name__}")
    return parsed


def tokenize(base_url: str, prompt: str, timeout: float = 180.0) -> int:
    result = http_json(
        f"{base_url}/tokenize",
        {"content": prompt, "add_special": True, "parse_special": True},
        timeout=timeout,
    )
    tokens = result.get("tokens")
    if not isinstance(tokens, list):
        raise RuntimeError(f"Unexpected /tokenize response: {result}")
    return len(tokens)


def make_evidence_blob(chars_per_chunk: int) -> str:
    chunks: list[str] = []
    for index in range(1, 11):
        template = PASSAGE_TEMPLATES[(index - 1) % len(PASSAGE_TEMPLATES)]
        source = f"policy-record-{index:03d}"
        section = f"section-{index:02d}"
        record_date = f"202{(index - 1) % 6}-0{(index - 1) % 9 + 1}-1{(index - 1) % 9}"
        seed = (
            f"{template} The evidence item is {source}, filed under {section}, "
            f"with record date {record_date}. The evidence reference is RAG-{index:05d}. "
            "The reviewer must distinguish the recorded requirement from an explanatory note, "
            "identify the relevant facts, and cite this source when relying on the statement. "
        )
        repeat_count = max(1, chars_per_chunk // len(seed) + 1)
        body = (seed * repeat_count)[:chars_per_chunk]
        chunks.append(f"[RAG CHUNK {index:02d}/10 | source={source} | {section} | date={record_date}]\n{body}")
    return "\n\n".join(chunks)


def build_prompt_for_target(base_url: str, target: int, timeout: float) -> tuple[str, int]:
    low, high = 0, max(5000, target * 3)
    best_prompt: str | None = None
    best_count: int | None = None
    best_delta = float("inf")
    while low <= high:
        chars = (low + high) // 2
        candidate = PROMPT_PREFIX + make_evidence_blob(chars) + PROMPT_SUFFIX
        count = tokenize(base_url, candidate, timeout=timeout)
        delta = abs(count - target)
        if delta < best_delta:
            best_prompt, best_count, best_delta = candidate, count, delta
        if count < target:
            low = chars + 1
        elif count > target:
            high = chars - 1
        else:
            break
    tolerance = max(20, int(target * 0.015))
    if best_prompt is None or best_count is None or best_delta > tolerance:
        raise RuntimeError(
            f"Could not construct a {target}-token prompt within ±{tolerance} tokens; "
            f"closest was {best_count}."
        )
    return best_prompt, best_count


def summarize(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"median": None, "mean": None, "p90": None, "min": None, "max": None}
    ordered = sorted(values)
    p90 = ordered[0] if len(ordered) == 1 else ordered[math.ceil(0.9 * (len(ordered) - 1))]
    return {
        "median": statistics.median(ordered),
        "mean": statistics.fmean(ordered),
        "p90": p90,
        "min": ordered[0],
        "max": ordered[-1],
    }


def values_for(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if as_number(row.get(key)) is not None]


def summarize_prompt_size(nonstream: list[dict[str, Any]], stream: list[dict[str, Any]]) -> dict[str, Any]:
    cache_values = values_for(nonstream, "server_cache_tokens")
    if len(cache_values) != len(nonstream):
        cache_status = "not_reported"
    elif any(value > 0 for value in cache_values):
        cache_status = "cache_reuse_detected"
    else:
        cache_status = "verified_zero"
    return {
        "prefill_latency_ms": summarize(values_for(nonstream, "server_prompt_ms")),
        "prefill_tokens_per_second": summarize(values_for(nonstream, "server_prompt_tps")),
        "decode_latency_ms": summarize(values_for(nonstream, "server_decode_ms")),
        "decode_tokens_per_second": summarize(values_for(nonstream, "server_decode_tps")),
        "ttft_ms": summarize(values_for(stream, "ttft_ms")),
        "nonstream_client_e2e_ms": summarize(values_for(nonstream, "client_e2e_ms")),
        "stream_client_e2e_ms": summarize(values_for(stream, "stream_e2e_ms")),
        "cache_tokens": summarize(cache_values),
        "cache_status": cache_status,
        "generated_tokens": summarize(values_for(nonstream, "server_generated_tokens")),
    }


def cold_completion(base_url: str, prompt: str, output_tokens: int, timeout: float) -> dict[str, Any]:
    payload = {
        "prompt": prompt,
        "n_predict": output_tokens,
        "temperature": 0.0,
        "top_k": 1,
        "ignore_eos": True,
        "cache_prompt": False,
        "stream": False,
        "timings_per_token": True,
    }
    started = time.perf_counter()
    response = http_json(f"{base_url}/completion", payload, timeout=timeout)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    timings = response.get("timings") if isinstance(response.get("timings"), dict) else {}

    prompt_tokens = first_number(timings.get("prompt_n"), response.get("tokens_evaluated"))
    cache_tokens = first_number(timings.get("cache_n"), response.get("tokens_cached"))
    generated_tokens = first_number(timings.get("predicted_n"), response.get("tokens_predicted"))
    prompt_ms = as_number(timings.get("prompt_ms"))
    decode_ms = as_number(timings.get("predicted_ms"))
    prompt_tps = as_number(timings.get("prompt_per_second"))
    decode_tps = as_number(timings.get("predicted_per_second"))
    if prompt_tps is None and prompt_ms and prompt_tokens is not None:
        prompt_tps = prompt_tokens * 1000.0 / prompt_ms
    if decode_tps is None and decode_ms and generated_tokens is not None:
        decode_tps = generated_tokens * 1000.0 / decode_ms

    if response.get("truncated") is True:
        raise RuntimeError("Server reports that the prompt or generation was truncated by the context window.")

    return {
        "client_e2e_ms": elapsed_ms,
        "server_prompt_tokens": prompt_tokens,
        "server_cache_tokens": cache_tokens,
        "server_generated_tokens": generated_tokens,
        "server_prompt_ms": prompt_ms,
        "server_prompt_tps": prompt_tps,
        "server_decode_ms": decode_ms,
        "server_decode_tps": decode_tps,
        "generation_target_met": generated_tokens == output_tokens if generated_tokens is not None else None,
        "truncated": response.get("truncated"),
        "stop_type": response.get("stop_type"),
        "response_chars": len(response.get("content", "") or ""),
        "raw_timings": timings,
    }


def stream_completion(base_url: str, prompt: str, output_tokens: int, timeout: float) -> dict[str, Any]:
    payload = {
        "prompt": prompt,
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
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    started = time.perf_counter()
    first_content_at: float | None = None
    chars = 0
    events = 0
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw in response:
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
                events += 1
                content = event.get("content")
                if isinstance(content, str) and content:
                    if first_content_at is None:
                        first_content_at = time.perf_counter()
                    chars += len(content)
                if event.get("stop") is True:
                    break
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Streaming HTTP {exc.code}: {detail[:2000]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Streaming request failed: {exc}") from exc

    ended = time.perf_counter()
    if first_content_at is None:
        raise RuntimeError("Streaming response ended without a non-empty content delta; TTFT unavailable.")
    return {
        "ttft_ms": (first_content_at - started) * 1000.0,
        "stream_e2e_ms": (ended - started) * 1000.0,
        "stream_content_chars": chars,
        "stream_events": events,
    }


def port_is_free(host: str, port: int) -> bool:
    """Attempt bind, which detects a listener more reliably than a connect probe."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
        return True
    except OSError:
        return False


def port_diagnostic(host: str, port: int) -> str:
    lines = [f"Port {port} on {host} is unavailable."]
    if shutil.which("ss"):
        try:
            result = subprocess.run(
                ["ss", "-ltnp", f"sport = :{port}"], capture_output=True, text=True, timeout=3
            )
            if result.stdout.strip():
                lines.append(result.stdout.strip())
        except Exception:
            pass
    lines.append("The benchmark will not kill a process it did not start. Stop the listener or choose another port.")
    return "\n".join(lines)


def process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


class ServerProcess:
    def __init__(self, name: str, binary: Path, model: Path, host: str, port: int,
                 threads: int, batch_threads: int, batch_size: int, ubatch_size: int,
                 ctx_size: int, log_path: Path, runtime_repack: bool = False,
                 parallel: int = 1, stop_timeout: float = 12.0):
        self.name = name
        self.binary = binary
        self.model = model
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.threads = threads
        self.batch_threads = batch_threads
        self.batch_size = batch_size
        self.ubatch_size = ubatch_size
        self.ctx_size = ctx_size
        self.runtime_repack = runtime_repack
        self.parallel = parallel
        self.log_path = log_path
        self.stop_timeout = stop_timeout
        self.command: list[str] = []
        self.process: subprocess.Popen[bytes] | None = None
        self.log_handle: Any = None
        self.startup_seconds: float | None = None

    def start(self, timeout: float) -> None:
        if not self.binary.is_file() or not os.access(self.binary, os.X_OK):
            raise FileNotFoundError(f"Server binary missing or not executable: {self.binary}")
        if not port_is_free(self.host, self.port):
            raise RuntimeError(port_diagnostic(self.host, self.port))

        self.command = [
            str(self.binary), "-m", str(self.model),
            "-c", str(self.ctx_size),
            "-t", str(self.threads),
            "-tb", str(self.batch_threads),
            "-b", str(self.batch_size),
            "-ub", str(self.ubatch_size),
            "-np", str(self.parallel),
            "--host", self.host,
            "--port", str(self.port),
        ]
        if self.runtime_repack:
            self.command.append("-rtr")

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_handle = self.log_path.open("wb")
        log(f"\nStarting {self.name} on {self.host}:{self.port}")
        log("Command: " + " ".join(self.command))
        start_time = time.perf_counter()
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

        deadline = time.monotonic() + timeout
        last_health_error: str | None = None
        while time.monotonic() < deadline:
            assert self.process is not None
            exit_code = self.process.poll()
            if exit_code is not None:
                raise RuntimeError(
                    f"{self.name} exited during startup with code {exit_code}.\n"
                    f"Log: {self.log_path}\n{tail_file(self.log_path)}"
                )
            try:
                health = http_json(f"{self.base_url}/health", timeout=2.0)
                if health.get("status") == "ok":
                    self.startup_seconds = time.perf_counter() - start_time
                    log(f"{self.name} ready in {self.startup_seconds:.2f}s")
                    return
                last_health_error = f"health returned {health}"
            except Exception as exc:
                last_health_error = str(exc)
            time.sleep(0.4)

        raise TimeoutError(
            f"Timed out waiting for {self.name} health on {self.base_url}. "
            f"Last health result: {last_health_error}\nLog: {self.log_path}\n{tail_file(self.log_path)}"
        )

    def stop(self) -> None:
        proc = self.process
        if proc is None:
            if self.log_handle is not None:
                self.log_handle.close()
                self.log_handle = None
            return

        pgid = proc.pid  # start_new_session=True makes the child PID its process-group ID.
        log(f"Stopping {self.name} process group {pgid}")
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass

        deadline = time.monotonic() + self.stop_timeout
        while time.monotonic() < deadline:
            proc.poll()  # reap the leader if it has exited
            if not process_group_exists(pgid):
                break
            time.sleep(0.15)

        if process_group_exists(pgid):
            log(f"{self.name} did not exit after SIGTERM; sending SIGKILL to process group {pgid}")
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            kill_deadline = time.monotonic() + 4.0
            while time.monotonic() < kill_deadline and process_group_exists(pgid):
                time.sleep(0.1)

        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            pass

        group_left = process_group_exists(pgid)
        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None
        self.process = None

        if group_left:
            raise RuntimeError(
                f"Could not fully terminate {self.name} process group {pgid}. "
                f"Inspect the process tree and log: {self.log_path}"
            )

        # Confirm the TCP listener is gone before considering the engine stopped.
        deadline = time.monotonic() + 4.0
        while time.monotonic() < deadline:
            if port_is_free(self.host, self.port):
                log(f"{self.name} stopped; port {self.port} is free")
                return
            time.sleep(0.1)
        raise RuntimeError(
            f"{self.name} process group exited but {self.host}:{self.port} is still occupied. "
            f"Another process may have claimed the port. See {self.log_path}."
        )

    def __enter__(self) -> "ServerProcess":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        try:
            self.stop()
        except Exception as cleanup_error:
            if exc is None:
                raise
            log(f"WARNING: cleanup also failed for {self.name}: {cleanup_error}")


def git_revision(binary: Path) -> str | None:
    if binary.parent.name != "bin" or len(binary.parents) < 3:
        return None
    repo = binary.parents[2]
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True, timeout=5,
        ).strip()
    except Exception:
        return None


def binary_version(binary: Path) -> str | None:
    try:
        result = subprocess.run([str(binary), "--version"], capture_output=True, text=True, timeout=10)
        value = (result.stdout or result.stderr).strip()
        return value[:1000] if value else None
    except Exception:
        return None


def cpu_metadata() -> dict[str, Any]:
    data: dict[str, Any] = {
        "name": platform.processor() or None,
        "logical_cpu_count": os.cpu_count(),
        "flags": [],
        "cgroup_cpu_max": None,
    }
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
        name = re.search(r"^model name\s*:\s*(.+)$", cpuinfo, flags=re.MULTILINE)
        flags_match = re.search(r"^flags\s*:\s*(.+)$", cpuinfo, flags=re.MULTILINE)
        if name:
            data["name"] = name.group(1).strip()
        if flags_match:
            flags = set(flags_match.group(1).split())
            data["flags"] = sorted(x for x in ("avx2", "avx_vnni", "avx512f", "avx512_vnni", "fma") if x in flags)
    except OSError:
        pass
    for cgroup_file in (Path("/sys/fs/cgroup/cpu.max"), Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")):
        try:
            data["cgroup_cpu_max"] = cgroup_file.read_text(encoding="utf-8").strip()
            break
        except OSError:
            continue
    return data


def validate_prompt_count(engine_name: str, expected: int, sample: dict[str, Any], target: int) -> None:
    processed = sample.get("server_prompt_tokens")
    cached = sample.get("server_cache_tokens")
    if processed is None:
        log(f"WARNING: {engine_name} did not report prompt token count for target {target}")
        return
    total = processed + (cached or 0)
    tolerance = max(24, int(expected * 0.02))
    if abs(total - expected) > tolerance:
        raise RuntimeError(
            f"{engine_name} evaluated/reused {total} prompt tokens, but /tokenize counted {expected} "
            f"(tolerance {tolerance}). This result is not a valid prompt-size comparison."
        )


def run_one_engine(engine: ServerProcess, args: argparse.Namespace, result_doc: dict[str, Any],
                   output_path: Path, prompt_records: list[dict[str, Any]] | None,
                   prompt_manifest_path: Path) -> list[dict[str, Any]]:
    record: dict[str, Any] = {
        "name": engine.name,
        "status": "starting",
        "binary": str(engine.binary),
        "git_revision": git_revision(engine.binary),
        "version": binary_version(engine.binary),
        "port": engine.port,
        "command": [],
        "log_file": str(engine.log_path),
        "startup_seconds": None,
        "runtime_repack": engine.runtime_repack,
        "prompt_sizes": [],
    }
    result_doc["engines"].append(record)
    atomic_write_json(output_path, result_doc)

    try:
        with engine:
            engine.start(args.startup_timeout)
            record["status"] = "running"
            record["command"] = engine.command
            record["startup_seconds"] = engine.startup_seconds
            atomic_write_json(output_path, result_doc)

            # Warm-up is deliberately not included in measurements.
            warmup_prompt = PROMPT_PREFIX + "[WARM-UP] One synthetic record is available.\n" + PROMPT_SUFFIX
            http_json(
                f"{engine.base_url}/completion",
                {"prompt": warmup_prompt, "n_predict": args.warmup_tokens, "temperature": 0.0,
                 "top_k": 1, "ignore_eos": True, "cache_prompt": False, "stream": False,
                 "timings_per_token": True},
                timeout=args.request_timeout,
            )

            if prompt_records is None:
                prompt_records = []
                for target in args.prompt_tokens:
                    prompt, count = build_prompt_for_target(engine.base_url, target, args.request_timeout)
                    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                    prompt_records.append({
                        "target_tokens": target,
                        "prompt_tokens_from_builder": count,
                        "prompt_sha256": digest,
                        "prompt": prompt,
                        "engine_token_counts": {},
                    })
                    log(f"Constructed target {target}: actual {count} tokens (sha256 {digest[:12]})")
                manifest = {
                    "created_at_utc": utc_now(),
                    "model": str(args.model),
                    "prompt_targets": args.prompt_tokens,
                    "prompts": [
                        {k: v for k, v in item.items() if k != "engine_token_counts"}
                        for item in prompt_records
                    ],
                }
                atomic_write_json(prompt_manifest_path, manifest)
                result_doc["prompt_manifest"] = str(prompt_manifest_path)
                atomic_write_json(output_path, result_doc)

            # Preflight token counts for this engine before timing any requests.
            for item in prompt_records:
                count = tokenize(engine.base_url, item["prompt"], timeout=args.request_timeout)
                item["engine_token_counts"][engine.name] = count
                expected = item["prompt_tokens_from_builder"]
                tolerance = max(8, int(expected * 0.005))
                if abs(count - expected) > tolerance:
                    raise RuntimeError(
                        f"Tokenizer mismatch for {engine.name}: generated prompt has {count} tokens here "
                        f"but {expected} on the prompt-building engine (tolerance {tolerance}). "
                        "Do not compare timings with different token counts."
                    )

            for item in prompt_records:
                target = item["target_tokens"]
                expected = item["engine_token_counts"][engine.name]
                log(f"\n{engine.name}: target {target} tokens; {args.repeats} repetitions")
                row: dict[str, Any] = {
                    "target_tokens": target,
                    "prompt_tokens": expected,
                    "prompt_sha256": item["prompt_sha256"],
                    "repeats": args.repeats,
                    "output_token_cap": args.output_tokens,
                    "nonstream_runs": [],
                    "stream_runs": [],
                    "summary": {},
                }
                record["prompt_sizes"].append(row)
                atomic_write_json(output_path, result_doc)

                for run_index in range(args.repeats):
                    # Alternate the measurement order to reduce systematic ordering bias.
                    if run_index % 2 == 0:
                        ns = cold_completion(engine.base_url, item["prompt"], args.output_tokens, args.request_timeout)
                        validate_prompt_count(engine.name, expected, ns, target)
                        st = stream_completion(engine.base_url, item["prompt"], args.output_tokens, args.request_timeout)
                    else:
                        st = stream_completion(engine.base_url, item["prompt"], args.output_tokens, args.request_timeout)
                        ns = cold_completion(engine.base_url, item["prompt"], args.output_tokens, args.request_timeout)
                        validate_prompt_count(engine.name, expected, ns, target)

                    row["nonstream_runs"].append(ns)
                    row["stream_runs"].append(st)
                    row["summary"] = summarize_prompt_size(row["nonstream_runs"], row["stream_runs"])
                    if ns.get("server_cache_tokens") is None:
                        cache_display = "cache=n/a"
                    else:
                        cache_display = f"cache={ns['server_cache_tokens']:.0f}"
                    if ns.get("generation_target_met") is False:
                        log(f"WARNING: {engine.name} generated {ns.get('server_generated_tokens')} tokens; requested {args.output_tokens}")
                    log(
                        f"  {run_index + 1}/{args.repeats}: "
                        f"prefill={ns.get('server_prompt_ms') if ns.get('server_prompt_ms') is not None else float('nan'):.1f} ms "
                        f"({ns.get('server_prompt_tps') if ns.get('server_prompt_tps') is not None else float('nan'):.1f} tok/s), "
                        f"TTFT={st['ttft_ms']:.1f} ms, stream E2E={st['stream_e2e_ms']:.1f} ms, {cache_display}"
                    )
                    atomic_write_json(output_path, result_doc)

                status = row["summary"]["cache_status"]
                if status != "verified_zero":
                    log(f"WARNING: {engine.name} cold-cache status for {target} is {status}")

            record["status"] = "complete"
            atomic_write_json(output_path, result_doc)
            return prompt_records
    except BaseException as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["log_tail"] = tail_file(engine.log_path)
        atomic_write_json(output_path, result_doc)
        raise


def compare_engines(result_doc: dict[str, Any]) -> dict[str, Any]:
    engines = {item["name"]: item for item in result_doc.get("engines", [])}
    if not all(name in engines for name in ("llama.cpp", "ik_llama.cpp")):
        return {}
    llama_rows = {row["target_tokens"]: row for row in engines["llama.cpp"].get("prompt_sizes", [])}
    ik_rows = {row["target_tokens"]: row for row in engines["ik_llama.cpp"].get("prompt_sizes", [])}
    metrics = (
        "prefill_latency_ms", "prefill_tokens_per_second", "ttft_ms",
        "decode_latency_ms", "decode_tokens_per_second",
        "nonstream_client_e2e_ms", "stream_client_e2e_ms",
    )
    output: dict[str, Any] = {}
    for target in sorted(set(llama_rows) & set(ik_rows)):
        per_target: dict[str, Any] = {}
        left = llama_rows[target]["summary"]
        right = ik_rows[target]["summary"]
        for metric in metrics:
            a = left[metric]["median"]
            b = right[metric]["median"]
            valid = as_number(a) is not None and as_number(b) is not None and float(a) > 0 and float(b) > 0
            throughput = metric.endswith("tokens_per_second")
            per_target[metric] = {
                "llama_cpp_median": a,
                "ik_llama_cpp_median": b,
                "ik_speedup_factor": ((float(b) / float(a)) if throughput else (float(a) / float(b))) if valid else None,
                "unit": "x; >1 means ik_llama.cpp is faster" if valid else None,
            }
        per_target["prompts_match"] = llama_rows[target].get("prompt_sha256") == ik_rows[target].get("prompt_sha256")
        output[str(target)] = per_target
    return output


def fmt_metric(summary: dict[str, Any], key: str) -> str:
    value = summary.get(key, {}).get("median")
    return f"{value:.1f}" if as_number(value) is not None else "n/a"


def print_summary(result_doc: dict[str, Any]) -> None:
    log("\nMEDIAN RESULTS")
    log("engine          tokens  prefill-ms  prefill-t/s  TTFT-ms  stream-E2E-ms  decode-t/s  cache")
    log("--------------  ------  ----------  -----------  -------  -------------  ----------  --------")
    for engine in result_doc.get("engines", []):
        for row in engine.get("prompt_sizes", []):
            s = row["summary"]
            log(
                f"{engine['name']:<14} {row['target_tokens']:>6}  "
                f"{fmt_metric(s, 'prefill_latency_ms'):>10}  "
                f"{fmt_metric(s, 'prefill_tokens_per_second'):>11}  "
                f"{fmt_metric(s, 'ttft_ms'):>7}  "
                f"{fmt_metric(s, 'stream_client_e2e_ms'):>13}  "
                f"{fmt_metric(s, 'decode_tokens_per_second'):>10}  {s['cache_status']}"
            )
    if result_doc.get("comparison"):
        log("\nIK / LLAMA SPEEDUP (greater than 1.0x means ik_llama.cpp is faster)")
        for target, metrics in result_doc["comparison"].items():
            prefill = metrics["prefill_tokens_per_second"]["ik_speedup_factor"]
            ttft = metrics["ttft_ms"]["ik_speedup_factor"]
            log(f"  {target} tokens: prefill {prefill:.2f}x, TTFT {ttft:.2f}x" if prefill and ttft else f"  {target} tokens: insufficient comparable metrics")


def parse_targets(value: str) -> list[int]:
    try:
        targets = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("prompt targets must be comma-separated integers") from exc
    if not targets or any(target <= 0 for target in targets) or len(set(targets)) != len(targets):
        raise argparse.ArgumentTypeError("prompt targets must be distinct positive integers")
    return targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare cold RAG prompt prefill and generation latency on llama.cpp and ik_llama.cpp."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--llama-server", type=Path, default=DEFAULT_LLAMA_SERVER)
    parser.add_argument("--ik-server", type=Path, default=DEFAULT_IK_SERVER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8097, help="llama.cpp port")
    parser.add_argument("--ik-port", type=int, default=8098, help="ik_llama.cpp port; keep distinct from --port")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch-threads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--ubatch-size", type=int, default=512)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--ctx-size", type=int, default=8192)
    parser.add_argument("--prompt-tokens", type=parse_targets, default=list(DEFAULT_TARGETS),
                        help="comma-separated prompt targets (default: 1024,2048,5120)")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--warmup-tokens", type=int, default=16)
    parser.add_argument("--startup-timeout", type=float, default=600.0)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--stop-timeout", type=float, default=12.0)
    parser.add_argument("--reverse-order", action="store_true")
    parser.add_argument("--ik-runtime-repack", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for name in ("threads", "batch_threads", "batch_size", "ubatch_size", "parallel", "ctx_size",
                 "repeats", "output_tokens", "warmup_tokens"):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if not (1 <= args.port <= 65535 and 1 <= args.ik_port <= 65535):
        raise SystemExit("Ports must be between 1 and 65535")
    if args.port == args.ik_port:
        raise SystemExit("Use two different ports: --port for llama.cpp and --ik-port for ik_llama.cpp")
    if args.startup_timeout <= 0 or args.request_timeout <= 0 or args.stop_timeout <= 0:
        raise SystemExit("Timeouts must be positive")
    if args.ctx_size < max(args.prompt_tokens) + args.output_tokens + 64:
        raise SystemExit("--ctx-size must exceed largest prompt target + output tokens + 64 tokens")

    args.model = args.model.expanduser().resolve()
    args.llama_server = args.llama_server.expanduser().resolve()
    args.ik_server = args.ik_server.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.log_dir = args.log_dir.expanduser().resolve()
    if not args.model.is_file():
        raise SystemExit(f"Model not found: {args.model}")
    for name, binary in (("llama.cpp", args.llama_server), ("ik_llama.cpp", args.ik_server)):
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise SystemExit(f"{name} server binary missing/not executable: {binary}")

    # Fail before doing any work if either listener port is already occupied.
    for port in (args.port, args.ik_port):
        if not port_is_free(args.host, port):
            raise SystemExit(port_diagnostic(args.host, port))

    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_dir = args.log_dir / run_id
    run_log_dir.mkdir(parents=True, exist_ok=True)
    prompt_manifest_path = args.output.with_name(args.output.stem + f"_{run_id}_prompts.json")

    model_stat = args.model.stat()
    result_doc: dict[str, Any] = {
        "schema_version": 2,
        "run_id": run_id,
        "created_at_utc": utc_now(),
        "status": "running",
        "hardware": {
            "platform": platform.platform(),
            "python": sys.version,
            "cpu": cpu_metadata(),
        },
        "model": {
            "path": str(args.model),
            "size_bytes": model_stat.st_size,
            "modified_utc": dt.datetime.fromtimestamp(model_stat.st_mtime, dt.timezone.utc).isoformat(),
        },
        "configuration": {
            "engine_order": ["ik_llama.cpp", "llama.cpp"] if args.reverse_order else ["llama.cpp", "ik_llama.cpp"],
            "ports": {"llama.cpp": args.port, "ik_llama.cpp": args.ik_port},
            "prompt_targets": args.prompt_tokens,
            "threads": args.threads,
            "batch_threads": args.batch_threads,
            "batch_size": args.batch_size,
            "ubatch_size": args.ubatch_size,
            "parallel": args.parallel,
            "context_size": args.ctx_size,
            "repeats": args.repeats,
            "output_token_cap": args.output_tokens,
            "warmup_tokens": args.warmup_tokens,
            "cache_prompt": False,
            "temperature": 0.0,
            "top_k": 1,
            "ignore_eos": True,
            "ik_runtime_repack": args.ik_runtime_repack,
        },
        "notes": [
            "The prompt is constructed once and the exact same UTF-8 text is sent to both engines.",
            "Prompt token counts are verified independently on each engine before measurements.",
            "cache_prompt=false is set for each measured request; cache counters are recorded when provided by the server.",
            "Prefill/decode timing fields come from the native /completion timings object. Missing fields remain null.",
            "TTFT is measured client-side from request start until the first non-empty streamed content delta.",
            "Each engine uses a separate TCP port to avoid listener-release and bind races.",
            "The process group started by the benchmark is terminated and checked before moving on.",
            "The second engine may benefit from the OS file page cache; use --reverse-order for a second run to assess order bias.",
            "Synthetic RAG passages measure engine latency, not retrieval quality or answer quality.",
        ],
        "prompt_manifest": str(prompt_manifest_path),
        "log_directory": str(run_log_dir),
        "engines": [],
        "comparison": {},
    }
    atomic_write_json(args.output, result_doc)

    engines = [
        ServerProcess(
            "llama.cpp", args.llama_server, args.model, args.host, args.port,
            args.threads, args.batch_threads, args.batch_size, args.ubatch_size,
            args.ctx_size, run_log_dir / "llama_cpp.log", parallel=args.parallel,
            stop_timeout=args.stop_timeout,
        ),
        ServerProcess(
            "ik_llama.cpp", args.ik_server, args.model, args.host, args.ik_port,
            args.threads, args.batch_threads, args.batch_size, args.ubatch_size,
            args.ctx_size, run_log_dir / "ik_llama_cpp.log", runtime_repack=args.ik_runtime_repack,
            parallel=args.parallel, stop_timeout=args.stop_timeout,
        ),
    ]
    if args.reverse_order:
        engines.reverse()

    prompts: list[dict[str, Any]] | None = None
    try:
        for engine in engines:
            prompts = run_one_engine(engine, args, result_doc, args.output, prompts, prompt_manifest_path)
        result_doc["comparison"] = compare_engines(result_doc)
        result_doc["status"] = "complete"
        result_doc["completed_at_utc"] = utc_now()
        atomic_write_json(args.output, result_doc)
        print_summary(result_doc)
        log(f"\nResults: {args.output}")
        log(f"Prompt manifest: {prompt_manifest_path}")
        log(f"Logs: {run_log_dir}")
        return 0
    except KeyboardInterrupt:
        result_doc["status"] = "interrupted"
        result_doc["updated_at_utc"] = utc_now()
        atomic_write_json(args.output, result_doc)
        log("\nInterrupted. Completed measurements and partial results have been saved.")
        return 130
    except Exception as exc:
        result_doc["status"] = "failed"
        result_doc["error"] = f"{type(exc).__name__}: {exc}"
        result_doc["updated_at_utc"] = utc_now()
        result_doc["comparison"] = compare_engines(result_doc)
        atomic_write_json(args.output, result_doc)
        log(f"\nERROR: {type(exc).__name__}: {exc}")
        log(f"Partial results saved to: {args.output}")
        log(f"Logs: {run_log_dir}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
