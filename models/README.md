# CPU RAG Engine Comparison: llama.cpp vs. ik_llama.cpp

## 1. Overview

This benchmark compares the CPU inference performance of `llama.cpp` and `ik_llama.cpp` using the same locally available Qwen3.5 GGUF model.

Both engines are tested under identical workload conditions to evaluate prompt-processing performance, time to first token (TTFT), generation speed, and prompt-cache behavior.

The benchmark uses existing compiled binaries and does not require rebuilding either engine.

## 2. Directory Structure

Place `compare_llm_engines.py` in `/workspace/models/`.

The expected directory structure is:

```text
/workspace/models/
├── compare_llm_engines.py
├── llm_engines_compare.json
├── llm/
│   └── Qwen3.5-0.8B-Q4_K_M.gguf
├── engines/
│   ├── llama.cpp/
│   │   └── build/bin/llama-server
│   └── ik_llama.cpp/
│       └── build/bin/llama-server
└── logs/
```

The benchmark creates the result file, timestamped prompt manifests, and per-run logs automatically.

`bootstrap_llm.sh` is provided for building both engines from source when necessary. Existing working binaries can be used directly without rebuilding.

## 3. Running the Benchmark

Execute:

```bash
python3 /workspace/models/compare_llm_engines.py --repeats 3
```

### Default benchmark configuration

| Parameter | Configuration |
|---|---|
| Model | Qwen3.5-0.8B-Q4_K_M.gguf |
| Context size | 8,192 tokens |
| Prompt windows | 1,024 / 2,048 / 5,120 tokens |
| Evidence chunks | 10 synthetic RAG chunks |
| Generation threads | 8 |
| Batch threads | 8 |
| Batch size | 1,024 |
| Micro-batch size | 512 |
| Parallel slots | 1 |
| Maximum generation | 128 tokens |
| EOS handling | EOS ignored |
| Prompt caching | Disabled (`cache_prompt=false`) |
| Repetitions | 3 |

The benchmark generates synthetic RAG prompts containing ten evidence chunks at each target prompt size.

Identical prompt text is submitted to both engines. Each engine's token count is independently verified.

### Server configuration

The engines use separate ports:

| Engine | Port |
|---|---:|
| llama.cpp | 8097 |
| ik_llama.cpp | 8098 |

Only one engine runs at a time.

After each measurement, the benchmark terminates the complete server process group and verifies that the listener has been released before starting the next engine.

It does not terminate unrelated processes when a port is occupied.

## 4. Results and Logging

The primary benchmark results are saved to:

```text
/workspace/models/llm_engines_compare.json
```

Timestamped execution logs are stored under:

```text
/workspace/models/logs/
```

The benchmark also saves the exact generated prompts in a timestamped `*_prompts.json` manifest alongside the results.

Results are written atomically after each measurement. If execution fails, previously completed measurements remain available.

### Reported metrics

| Metric | Description |
|---|---|
| `prefill_latency_ms` | Server-reported prompt-processing latency |
| `prefill_tokens_per_second` | Server-reported prompt-processing throughput |
| `ttft_ms` | Client-observed time to first streamed text token |
| `end_to_end_latency_ms` | Total client-observed generation latency |
| `decode_tokens_per_second` | Generation throughput |
| Cache status | Availability and verification of prompt-cache counters |

### Prompt-cache status

- `verified_zero`: The server reported zero cached prompt tokens for every applicable non-streaming measurement.
- `not_reported`: The backend did not expose a usable cache counter. This does not establish that the prompt cache was empty.

## 5. Reverse-Order Benchmarking

Sequential benchmarking can introduce measurement bias.

The second engine may benefit from operating-system file-page caching, while CPU temperature, power limits, and thermal throttling can affect later measurements.

To reduce ordering bias, perform a second benchmark with the execution order reversed:

```bash
python3 /workspace/models/compare_llm_engines.py \
    --repeats 3 \
    --reverse-order \
    --output /workspace/models/llm_engines_compare_reverse.json \
    --log-dir /workspace/models/logs_reverse
```

For consistent measurements:

- Keep the laptop connected to external power.
- Maintain the same operating-system power profile.
- Avoid running other CPU-intensive workloads.
- Allow the processor to return to a comparable thermal state between benchmark sessions.
- Compare results from both execution orders.

## 6. Additional Configuration Options

### Adjust CPU thread allocation

```bash
python3 /workspace/models/compare_llm_engines.py \
    --threads 6 \
    --batch-threads 8
```

### Disable ik_llama.cpp runtime repacking

```bash
python3 /workspace/models/compare_llm_engines.py \
    --no-ik-runtime-repack
```

### Change server ports

```bash
python3 /workspace/models/compare_llm_engines.py \
    --port 18097 \
    --ik-port 18098
```

### Run a quick benchmark

```bash
python3 /workspace/models/compare_llm_engines.py \
    --prompt-tokens 1024 \
    --repeats 1
```

## 7. CPU Resource Diagnostics

If measured performance is unexpectedly low, inspect the CPU resources visible to the benchmark environment:

```bash
lscpu
nproc
cat /sys/fs/cgroup/cpu.max 2>/dev/null || true
```

These commands help identify:

- Available physical and logical CPU cores.
- CPU architecture and instruction-set support.
- Container CPU quotas and throttling constraints.

## 8. Benchmark Methodology and Limitations

### Prompt processing

`prefill_latency_ms` and `prefill_tokens_per_second` are obtained from the server's native `/completion` timing information.

If the backend does not provide the required timing fields, the benchmark reports `n/a` rather than estimating or fabricating values.

### Time to first token

`ttft_ms` measures the time between the client's request initiation and receipt of the first non-empty streamed text delta.

This includes HTTP overhead and other client-observed request latency.

### Prompt caching

The benchmark explicitly sends `cache_prompt=false` for measured requests.

When a backend exposes cache counters, those counters are recorded and checked. Missing counters are reported separately rather than interpreted as evidence of a cold cache.

### Scope

This benchmark evaluates inference-engine performance using synthetic RAG workloads.

It does not measure:

- Answer accuracy or generation quality.
- Retrieval precision or recall.
- Performance against a production document corpus.
- End-to-end RAG pipeline latency.
- Concurrent multi-user throughput.

Results should therefore be interpreted as engine-level performance measurements rather than a complete evaluation of production RAG performance.

## 9. Building the Engines

`bootstrap_llm.sh` provides a repeatable process for compiling both `llama.cpp` and `ik_llama.cpp` from source.

It is intended for initial setup or rebuilding after source/configuration changes.

Once both binaries exist and run successfully, the benchmark can use them directly:

```text
/workspace/models/engines/llama.cpp/build/bin/llama-server
/workspace/models/engines/ik_llama.cpp/build/bin/llama-server
```

Rebuilding the engines is not required for routine benchmark execution.
