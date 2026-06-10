# Embedding backends: prefer-GPU with CPU standby

How the stack serves dense embeddings, how to choose CPU vs GPU, why the
GPU path requires the NVIDIA Container Toolkit, and what each switch does
(and does not) require. Updated 2026-06-10 alongside the
multilingual-e5-large migration.

## Architecture

The daemon's dense-embedding layer is provider-driven
(`embedding.provider` in `config.yaml`):

| Provider | What it is | When to use |
|---|---|---|
| `fastembed` | In-process ONNX, pinned to `all-MiniLM-L6-v2` (384d, English-only) | Zero-dependency local default; weakest retrieval quality |
| `openai_compatible` | HTTP client for any server speaking the OpenAI `/v1/embeddings` protocol | Real deployments — local TEI/Infinity containers, LM Studio, vLLM, or actual OpenAI |

With `openai_compatible`, two endpoints can be configured:

```yaml
embedding:
  provider: openai_compatible
  base_url: http://wqm-embeddings-gpu:7997      # PREFERRED endpoint (GPU)
  fallback_base_url: http://wqm-embeddings:80   # warm standby (CPU)
  model: intfloat/multilingual-e5-large
  output_dim: 1024
  remote_batch_size: 32
  api_key_env_var: OPENAI_API_KEY               # dummy value; see below
  document_prefix: "passage: "                  # REQUIRED by the e5 family
  query_prefix: "query: "                       # (trailing space included)
```

**Failover semantics** (`FailoverDenseProvider`): every call tries
`base_url` first. On any error the daemon logs a WARN, memoizes the outage
for ~60 s (so a dead primary is not re-dialed on every request), and serves
from `fallback_base_url`. After the memo expires the primary is retried
automatically — recovery needs no restart. The health probe reports healthy
if *either* endpoint responds.

Two invariants make this safe:

1. **Both endpoints must serve the same model.** Vectors are model-bound,
   not server-bound: `multilingual-e5-large` embeddings computed on CPU-TEI
   are bit-compatible with GPU-Infinity ones. Switching servers therefore
   **never requires a reembed**. Changing the *model* (or `output_dim`)
   always does — see [Changing the model](#changing-the-model-reembed).
2. **The API key env var must exist and be non-empty** even for local
   servers that ignore auth (TEI/Infinity). The compose stack passes
   `OPENAI_API_KEY` through; `docker/.env` sets a dummy
   (`OPENAI_API_KEY=local-dev-no-auth`).

> **Gotcha — `--config` is mandatory.** The memexd entrypoint *validates*
> `/etc/wqm/config.yaml` but the daemon only *reads* it when started with
> `--config /etc/wqm/config.yaml`. The reference compose passes it in the
> service `command:`. Without it the daemon silently runs on built-in
> defaults + env overrides — `model`, `output_dim`, the prefixes and
> `fallback_base_url` are config-file-only and would all be ignored.

## The two compose backends

Both serve `intfloat/multilingual-e5-large` (1024d) and are gated behind
[compose profiles](https://docs.docker.com/compose/profiles/), so which
ones run is a single line in `docker/.env`:

```bash
COMPOSE_PROFILES=embeddings-cpu                  # CPU only
COMPOSE_PROFILES=embeddings-cpu,embeddings-gpu   # prefer GPU, CPU standby (recommended)
```

| Service | Container | Image | Endpoint (in-network) | Profile |
|---|---|---|---|---|
| `embeddings` | `wqm-embeddings` | `text-embeddings-inference:cpu-1.7` (TEI) | `http://wqm-embeddings:80` | `embeddings-cpu` |
| `embeddings-gpu` | `wqm-embeddings-gpu` | `michaelf34/infinity:latest` | `http://wqm-embeddings-gpu:7997` | `embeddings-gpu` |

Notes:

- The CPU service runs TEI with `--auto-truncate` — chunk texts longer than
  the model's 512-token window are truncated instead of erroring (semantic
  chunks can exceed 512 tokens by design).
- The GPU service runs Infinity with `--url-prefix /v1` so the
  OpenAI-compatible route is `/v1/embeddings`, matching what the daemon's
  provider calls.
- Neither service publishes a host port; they are reachable only on the
  compose network (`workspace-network`). The daemon resolves them by
  container name.
- Model weights are cached in named volumes (`tei_data`, `infinity_data`),
  so restarts do not re-download (~2.2 GB per backend on first start).

### Measured throughput (RTX 5070 Ti vs CPU, 2026-06-10)

| Backend | Throughput (batch 32) | Full ~14k-file reembed |
|---|---|---|
| TEI CPU | ~4 embeddings/s at 512 tokens | many hours |
| Infinity GPU | 216 emb/s (512 tokens) – 1 812 emb/s (short) | minutes |

Interactive search queries are fine on either (one short text per query);
the GPU matters for ingestion/reembed throughput.

## Cross-encoder reranker (2nd-stage search)

The GPU Infinity service also serves a **second** model — the multilingual
cross-encoder `BAAI/bge-reranker-v2-m3` (second `--model-id` in the compose
`command:`) — for the search tool's optional rerank stage, exposed at
`POST /v1/rerank`. The daemon targets it via env (compose passthrough from
`docker/.env`; there are no `config.yaml` keys for these):

```bash
WQM_RERANK_BASE_URL=http://wqm-embeddings-gpu:7997  # empty = in-process fastembed (jina-turbo, EN-only)
WQM_RERANK_MODEL=BAAI/bge-reranker-v2-m3            # default when unset
```

The rerank stage itself stays **opt-in on the MCP side**: per-call
`rerank: true` (plus `rerankWeight`, 0–1) or deployment-wide
`WQM_SEARCH_RERANK=1` / `WQM_SEARCH_RERANK_WEIGHT`. The final pool order
blends both signals — `(1-w)·norm(rrf_boosted) + w·norm(rerank)` — instead
of fully replacing the bi-encoder order; `w=1` reproduces the legacy
pure-reranker order.

Weight sweep on the 44-query benchmark (2026-06-10, semantic mode —
top1 / top3 / top10 / recall@10 / MRR / avg ms):

| w | top1 | top3 | top10 | rec@10 | MRR | ms |
|---|---|---|---|---|---|---|
| 0 (baseline) | 31.8 | 56.8 | 68.2 | 60.2 | 0.45 | 43 |
| **0.25 (default)** | **34.1** | **59.1** | **72.7** | **62.5** | **0.47** | 67 |
| 0.5 | 25.0 | 50.0 | 72.7 | 61.4 | 0.40 | 84 |
| 1.0 (pure reranker) | 6.8 | 29.5 | 59.1 | 48.9 | 0.21 | 135 |

The cross-encoder helps only as a **weak nudge**: at w=0.25 every semantic
aggregate beats the no-rerank baseline (hybrid top3 50→56.8 too), while
giving it full authority (w=1) is strictly worse even multilingual — same
pathology previously measured with the English jina-turbo model.

There is deliberately **no failover** for reranking: the CPU TEI standby
serves only the e5 embedder, and silently swapping to the (worse,
English-only) local fastembed model would change scoring semantics
mid-flight. On any rerank failure the search layer fails open to the
pre-rerank order.

Measured on the RTX 5070 Ti (2026-06-10): ~330 ms warm for a worst-case
pool (30 docs × 4 000 chars ≈ 30k tokens); typical semantic chunks are far
shorter. VRAM cost next to e5 fp16: ~2.1 GB.

## Why the NVIDIA Container Toolkit is required

`nvidia-smi` working inside WSL only proves the **host/WSL VM** sees the
GPU. Containers are isolated from host devices by default: the Docker
Engine knows nothing about GPUs, and `docker run --gpus all` fails with
`failed to discover GPU vendor from CDI: no known GPU vendor found` until
something teaches it how to inject the NVIDIA devices and driver libraries
into a container.

That something is the [NVIDIA Container
Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/):
it registers an `nvidia` runtime (and/or generates a CDI spec) that, at
container start, mounts the driver's user-space libraries and exposes
`/dev/nvidia*` inside the container. Without it the compose
`embeddings-gpu` service cannot start, regardless of how healthy the GPU
looks in `nvidia-smi`.

Install on the WSL distro that runs the Docker Engine (one-time, needs
sudo):

```bash
# NVIDIA apt repo + package (Ubuntu 24.04)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit

# Register the runtime with Docker and restart the engine
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

> **Side effect:** restarting the Docker engine restarts the whole wqm
> stack. All services have `restart: always` and come back on their own,
> but in-flight queue items are re-leased and the MCP connection drops
> briefly. The unified queue is restart-safe — nothing is lost.

Verify:

```bash
docker info | grep -i runtimes          # must list: nvidia
docker compose --env-file docker/.env up -d embeddings-gpu
docker logs wqm-embeddings-gpu | grep -i "embeddings/sec"   # warm-up benchmark on the GPU
```

### Why Infinity (not TEI) on the GPU

TEI's CUDA images are compiled per GPU *compute capability* and, as of
`:1.7`, top out before Blackwell: an RTX 5070 Ti reports `sm_120` and TEI
aborts with `Runtime compute cap 120 is not compatible with compile time
compute cap 80`. Infinity is PyTorch-based; PyTorch ≥ 2.7 with CUDA 12.8
supports `sm_120`, so it runs on Blackwell consumer cards today. (On older
cards — Ampere/Ada — TEI's CUDA images work fine and would also do.)

## Operations

### Enable GPU later (no reembed)

1. Install the toolkit (above) — once.
2. `docker/.env`: `COMPOSE_PROFILES=embeddings-cpu,embeddings-gpu`
3. `docker compose --env-file docker/.env -f docker-compose.yml up -d`
4. Done. `config.yaml` already prefers the GPU endpoint; the daemon picks
   it up within ~60 s (failover memo expiry) or immediately after a
   restart. The WARN→recovered transition is visible in `docker logs
   wqm-memexd`.

### Run CPU-only (e.g. GPU busy with something else)

`COMPOSE_PROFILES=embeddings-cpu`, `up -d --remove-orphans` (stops the GPU
service). The daemon fails over to the CPU endpoint within one memo window;
no other change needed.

### Changing the model (= reembed)

`model`, `output_dim` and the prefixes are model-bound. To change them:

1. Edit `state/memexd/config.yaml` (and the backend services' `--model-id`
   in `docker-compose.yml`).
2. Start memexd once with `--bootstrap-reembed` appended to the compose
   command (suppresses the startup dim-mismatch guard).
3. `docker exec wqm-memexd wqm admin reembed --confirm` — **destructive**:
   drops and recreates the 4 canonical collections at the new dimension and
   re-enqueues every source. Search is degraded until the queue drains.
4. Remove the flag, `up -d` normally.

### Which endpoint is serving right now?

```bash
docker logs wqm-memexd --since 10m 2>&1 | grep -iE "fallback|recovered"
docker logs wqm-embeddings-gpu --since 5m 2>&1 | grep -c "POST /v1/embeddings"
```

No `fallback` WARNs + request lines on the GPU container = primary (GPU) is
serving.

## Configuration reference

| `config.yaml` key | Env override | Notes |
|---|---|---|
| `embedding.provider` | `WQM_EMBEDDING_PROVIDER` | `fastembed` \| `openai_compatible` |
| `embedding.base_url` | `WQM_EMBEDDING_BASE_URL` | preferred endpoint |
| `embedding.fallback_base_url` | `WQM_EMBEDDING_FALLBACK_BASE_URL` | warm standby; empty = none |
| `embedding.model` | — (config-file only) | must match what the backends serve |
| `embedding.output_dim` | — (config-file only) | drives the dim guard + reembed recreation |
| `embedding.document_prefix` | `WQM_EMBEDDING_DOCUMENT_PREFIX` | e5: `"passage: "` |
| `embedding.query_prefix` | `WQM_EMBEDDING_QUERY_PREFIX` | e5: `"query: "` |
| `embedding.api_key_env_var` | `WQM_EMBEDDING_API_KEY_ENV_VAR` | named var must exist and be non-empty |
| `embedding.remote_batch_size` | — | ≤ the backend's max client batch (TEI default 32) |

`fastembed` pinning: when `provider=fastembed`, model/dim/prefixes/URLs are
forced to the MiniLM values regardless of config — the table above only
applies to `openai_compatible`.
