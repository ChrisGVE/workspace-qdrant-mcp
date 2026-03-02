# Configuration Reference

All three components — daemon (`memexd`), CLI (`wqm`), and MCP Server — share a single configuration file. This prevents configuration drift and ensures consistent behavior across the system.

Configuration is entirely optional. All defaults are embedded in the binaries. A config file is only required to override default values.

---

## Configuration File Locations

### Search order (first match wins, identical across all components)

| Priority | Path | Notes |
|----------|------|-------|
| 1 | Path specified in `WQM_CONFIG_PATH` | Explicit override |
| 2 | `~/.workspace-qdrant/config.yaml` | Primary location (macOS/Linux) |
| 2 | `~/.workspace-qdrant/config.yml` | Alternate extension |
| 3 | `$XDG_CONFIG_HOME/workspace-qdrant/config.yaml` | XDG (Linux; defaults to `~/.config`) |
| 4 | `~/Library/Application Support/workspace-qdrant/config.yaml` | macOS secondary |
| 5 | `%APPDATA%\workspace-qdrant\config.yaml` | Windows |

No project-local configuration file is searched. All components use the same search cascade.

### Generate the default configuration file

```bash
wqm config init     # Write embedded defaults to ~/.workspace-qdrant/config.yaml
wqm config show     # Print active configuration (merged defaults + overrides)
wqm config edit     # Open config in $EDITOR
```

---

## Data Directories

| Path | Purpose | XDG Equivalent |
|------|---------|----------------|
| `~/.workspace-qdrant/config.yaml` | Configuration | `$XDG_CONFIG_HOME/workspace-qdrant/` |
| `~/.workspace-qdrant/state.db` | SQLite state database | `$XDG_DATA_HOME/workspace-qdrant/` |
| `~/.workspace-qdrant/search.db` | FTS5 full-text search database | `$XDG_DATA_HOME/workspace-qdrant/` |
| `~/.workspace-qdrant/grammars/` | Tree-sitter grammar cache | — |

### Log directories (OS-canonical, separate from config)

| OS | Log Directory | Override |
|----|---------------|----------|
| Linux | `~/.local/state/workspace-qdrant/logs/` | `WQM_LOG_DIR` |
| macOS | `~/Library/Logs/workspace-qdrant/` | `WQM_LOG_DIR` |
| Windows | `%LOCALAPPDATA%\workspace-qdrant\logs\` | `WQM_LOG_DIR` |

---

## Environment Variables

Environment variables override the equivalent values in the configuration file.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `WQM_CONFIG_PATH` | string | — | Absolute path to config file (bypasses search order) |
| `WQM_DATABASE_PATH` | string | `~/.workspace-qdrant/state.db` | Override SQLite database path |
| `QDRANT_URL` | string | `http://localhost:6333` | Qdrant server URL |
| `QDRANT_API_KEY` | string | — | Qdrant API key (required for secured instances) |
| `FASTEMBED_MODEL` | string | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model identifier |
| `WQM_DAEMON_PORT` | integer | `50051` | Daemon gRPC port |
| `WQM_LOG_LEVEL` | string | `info` | Minimum log level (`trace`, `debug`, `info`, `warn`, `error`) |
| `WQM_LOG_DIR` | string | (OS-canonical) | Override log directory for all components |
| `WQM_LOG_JSON` | bool | `true` | Enable JSON-formatted log output (daemon) |
| `WQM_LOG_CONSOLE` | bool | `false` (service) / `true` (foreground) | Console log output (daemon) |
| `WQM_STDIO_MODE` | bool | `false` | Force stdio transport mode |
| `WQM_CLI_MODE` | bool | `false` | Force CLI mode |
| `RUST_LOG` | string | — | Fine-grained Rust module log filtering (e.g. `memexd=debug,hyper=warn`) |

Queue processor overrides:

| Variable | Description |
|----------|-------------|
| `WQM_QUEUE_BATCH_SIZE` | Items dequeued per batch |
| `WQM_QUEUE_POLL_INTERVAL_MS` | Milliseconds between polls |
| `WQM_QUEUE_MAX_RETRIES` | Max retry attempts per item |
| `WQM_QUEUE_TARGET_THROUGHPUT` | Target docs/min for monitoring |
| `WQM_QUEUE_ENABLE_METRICS` | Enable queue performance metrics |

Other subsystem overrides:

| Variable | Description |
|----------|-------------|
| `WQM_GIT_ENABLE_BRANCH_DETECTION` | Enable Git branch tracking |
| `WQM_GIT_CACHE_TTL_SECONDS` | Branch info cache TTL |
| `WQM_EMBEDDING_CACHE_MAX_ENTRIES` | Max cached embedding results |
| `WQM_EMBEDDING_MODEL_CACHE_DIR` | Override model download directory |
| `WQM_MONITOR_ENABLE` | Enable tool monitoring |
| `WQM_MONITOR_CHECK_ON_STARTUP` | Run monitor check at startup |
| `WQM_MONITOR_CHECK_INTERVAL_HOURS` | Hours between monitor checks |
| `WQM_MCP_LOG_LEVEL` | Override log level for MCP Server only |

---

## Configuration Sections

The complete configuration structure with all parameters and their defaults.

### `deployment`

Controls asset file resolution and deployment mode.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `develop` | bool | `false` | When `true`, assets load from project-relative `assets/` directory; when `false`, from system paths |
| `base_path` | string\|null | `null` | Override base path for asset resolution; `null` uses automatic detection |

### `server`

HTTP server settings (used when MCP Server runs in HTTP mode).

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `host` | string | `"127.0.0.1"` | Network interface to bind |
| `port` | integer | `8000` | TCP port |
| `debug` | bool | `false` | Enable debug mode and enhanced logging |

### `qdrant`

Connection and behavior settings for the Qdrant vector database.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `url` | string | `"http://localhost:6333"` | Qdrant server URL; supports `http://`, `https://`, `grpc://` |
| `api_key` | string\|null | `null` | Authentication key; required for Qdrant Cloud and secured self-hosted instances |
| `timeout` | duration | `30s` | Request timeout |
| `prefer_grpc` | bool | `true` | Prefer gRPC binary protocol over HTTP REST |
| `transport` | string | `"grpc"` | Primary protocol: `"grpc"` or `"http"` |
| `tls` | bool | `false` | Enable TLS for Qdrant connection |
| `check_compatibility` | bool | `true` | Verify Qdrant version compatibility on startup |

**Collection defaults** (`qdrant.default_collection`):

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `vector_size` | integer | `384` | Dense vector dimensions; must match embedding model output |
| `distance_metric` | string | `"Cosine"` | Similarity metric: `"Cosine"`, `"Euclidean"`, or `"Dot"` |
| `enable_indexing` | bool | `true` | Build HNSW index for fast approximate search |
| `replication_factor` | integer | `1` | Number of data replicas (single-node: 1) |
| `shard_number` | integer | `1` | Number of collection shards |
| `on_disk_vectors` | bool | `false` | Store vectors on disk instead of in memory |

**HNSW index tuning** (`qdrant.default_collection.hnsw`):

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `m` | integer | `16` | Bidirectional links per node; range 4–64; higher = better recall, more memory |
| `ef_construct` | integer | `100` | Candidate list size during index build; higher = better quality, slower build |
| `ef` | integer | `64` | Candidate list during search; higher = better recall, slower search |
| `full_scan_threshold` | integer | `10000` | Collection size below which exact search is used instead of HNSW |

**Connection pool** (`qdrant.pool`):

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `max_connections` | integer | `10` | Maximum concurrent Qdrant connections |
| `min_idle_connections` | integer | `2` | Connections kept warm in idle state |
| `max_idle_time` | duration | `5m` | Maximum time to retain unused connections |
| `max_connection_lifetime` | duration | `1h` | Maximum total lifetime of any connection |
| `acquisition_timeout` | duration | `30s` | Maximum wait for a pool connection |

**Circuit breaker** (`qdrant.circuit_breaker`):

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | `true` | Enable circuit breaker pattern for Qdrant operations |
| `failure_threshold` | integer | `5` | Consecutive failures required to open the circuit |
| `success_threshold` | integer | `3` | Consecutive successes required to close the circuit |
| `timeout` | duration | `60s` | Time the circuit stays open before testing recovery |
| `half_open_timeout` | duration | `30s` | Timeout for recovery probe requests |

### `daemon`

Top-level daemon settings.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `log_file` | string\|null | `null` | Override log file path; `null` uses OS-canonical location |
| `log_level` | string | `"info"` | Minimum log level for daemon output |
| `max_concurrent_tasks` | integer | `4` | Maximum parallel processing tasks |
| `default_timeout_ms` | integer | `30000` | Task timeout in milliseconds |
| `enable_preemption` | bool | `true` | Allow higher-priority tasks to preempt lower-priority ones |
| `chunk_size` | integer | `1000` | Default batch processing unit size |
| `grpc_port` | integer | `50051` | gRPC server port |

**Resource limits** (`daemon.resource_limits`):

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `nice_level` | integer | `10` | OS process priority (-20 = highest, 19 = lowest) |
| `inter_item_delay_ms` | integer | `50` | Delay between queue items (0–5000 ms); prevents CPU saturation |
| `max_concurrent_embeddings` | integer | `2` | Concurrent ONNX embedding operations (1–8) |
| `max_memory_percent` | integer | `70` | Pause processing when system memory exceeds this percentage (20–95) |

### `queue_processor`

Unified queue settings.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `batch_size` | integer | `10` | Items dequeued per processing cycle |
| `poll_interval_ms` | integer | `500` | Milliseconds between dequeue polls |
| `max_retries` | integer | `5` | Maximum retry attempts before marking an item failed |
| `retry_delays_seconds` | array | `[60, 300, 900, 3600]` | Backoff schedule (seconds) for successive retries |
| `target_throughput` | integer | `1000` | Target documents/minute for monitoring alerts |
| `enable_metrics` | bool | `true` | Enable queue performance metrics collection |

### `embedding`

Text embedding pipeline settings.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `model` | string | `"sentence-transformers/all-MiniLM-L6-v2"` | FastEmbed model identifier; 384-dimensional, 256-token max |
| `enable_sparse_vectors` | bool | `true` | Enable BM25 sparse vectors for hybrid search |
| `sparse_vector_mode` | string | `"bm25"` | Sparse embedding algorithm: `"bm25"` or `"splade"` |
| `chunk_size` | integer | `384` | Characters per chunk (~82 prose tokens, ~110 code tokens) |
| `chunk_overlap` | integer | `58` | Character overlap between consecutive chunks (15% of chunk_size) |
| `batch_size` | integer | `50` | Chunks processed per embedding batch |
| `cache_enabled` | bool | `true` | Cache embedding results to avoid recomputation |
| `cache_max_entries` | integer | `1000` | Maximum cached embedding results (LRU eviction) |
| `model_cache_dir` | string\|null | `null` | Override model download directory (`null` uses `~/.cache/fastembed/`) |

**Supported embedding models:**

| Model | Dimensions | Notes |
|-------|-----------|-------|
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | Default; fast, good quality |
| `sentence-transformers/all-mpnet-base-v2` | 768 | Higher quality, slower |
| `BAAI/bge-small-en-v1.5` | 384 | English-optimized |
| `intfloat/e5-small-v2` | 384 | Multilingual support |

When changing models, `vector_size` in `qdrant.default_collection` must match the model's output dimension, and all existing collections must be re-indexed.

### `lsp`

Language Server Protocol integration settings.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `user_path` | string\|null | `null` | Additional PATH entries for finding language server binaries |
| `max_servers_per_project` | integer | `3` | Maximum concurrent LSP servers per active project |
| `auto_start_on_activation` | bool | `true` | Start LSP servers when a project becomes active |
| `deactivation_delay_secs` | integer | `60` | Seconds to wait before stopping servers after project deactivation |
| `enable_enrichment_cache` | bool | `true` | Cache LSP symbol resolution results |
| `cache_ttl_secs` | integer | `300` | LSP enrichment cache TTL |
| `startup_timeout_secs` | integer | `30` | Timeout for LSP server initialization |
| `request_timeout_secs` | integer | `10` | Timeout per LSP request |
| `health_check_interval_secs` | integer | `60` | Interval between LSP server health checks |
| `max_restart_attempts` | integer | `3` | Maximum restart attempts for a crashed LSP server |
| `restart_backoff_multiplier` | float | `2.0` | Exponential backoff multiplier between restarts |
| `enable_auto_restart` | bool | `true` | Automatically restart crashed LSP servers |
| `stability_reset_secs` | integer | `3600` | Seconds of uptime before restart counter resets |

### `grammars`

Tree-sitter grammar cache settings.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `cache_dir` | string | `~/.workspace-qdrant/grammars` | Local directory for compiled grammar files |
| `required` | array | `[]` | Grammars to pre-download on startup (empty = on-demand via auto_download) |
| `auto_download` | bool | `true` | Automatically download missing grammars |
| `tree_sitter_version` | string | `"0.24"` | Tree-sitter ABI version to target |
| `verify_checksums` | bool | `true` | Verify grammar file integrity |
| `lazy_loading` | bool | `true` | Load grammars on first use instead of at startup |
| `check_interval_hours` | integer | `168` | Hours between grammar update checks (168 = weekly) |
| `idle_update_check_enabled` | bool | `true` | Check for grammar updates when queue is idle |
| `idle_update_check_delay_secs` | integer | `300` | Seconds of idle time before triggering update check |

### `git`

Git integration settings.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enable_branch_detection` | bool | `true` | Track branch changes for multi-branch indexing |
| `cache_ttl_seconds` | integer | `60` | Branch info cache TTL |

### `updates`

Daemon self-update settings.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `auto_check` | bool | `true` | Check for available updates periodically |
| `channel` | string | `"stable"` | Update channel: `"stable"`, `"beta"`, or `"dev"` |
| `notify_only` | bool | `true` | Announce updates without auto-installing |
| `check_interval_hours` | integer | `24` | Hours between update checks |

### `monitoring`

Tool monitoring settings.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enable_monitoring` | bool | `true` | Enable tool version monitoring |
| `check_on_startup` | bool | `true` | Run tool check at daemon startup |
| `check_interval_hours` | integer | `24` | Hours between monitoring checks |

### `observability`

Metrics and telemetry settings.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `collection_interval` | integer | `60` | Seconds between metric snapshots |
| `metrics.enabled` | bool | `false` | Enable Prometheus-compatible metrics endpoint |
| `telemetry.enabled` | bool | `false` | Enable internal telemetry collection |
| `telemetry.history_retention` | integer | `120` | Minutes of telemetry history to retain |
| `telemetry.cpu_usage` | bool | `true` | Collect CPU usage metrics |
| `telemetry.memory_usage` | bool | `true` | Collect memory usage metrics |
| `telemetry.latency` | bool | `true` | Collect operation latency metrics |
| `telemetry.queue_depth` | bool | `true` | Collect queue depth metrics |
| `telemetry.throughput` | bool | `true` | Collect processing throughput metrics |

### `logging`

Structured log content settings.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `info_includes_connection_events` | bool | `true` | Include connection open/close events in INFO logs |
| `info_includes_transport_details` | bool | `true` | Include transport protocol details in INFO logs |
| `info_includes_retry_attempts` | bool | `true` | Include retry attempt counts in INFO logs |
| `info_includes_fallback_behavior` | bool | `true` | Include fallback activation events in INFO logs |
| `error_includes_stack_trace` | bool | `true` | Include stack traces in ERROR logs |
| `error_includes_connection_state` | bool | `true` | Include connection state in ERROR logs |

### `watching`

File watching and ingestion filter settings.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `allowed_extensions` | array | 400+ entries | File extensions eligible for ingestion (e.g. `".rs"`, `".py"`, `".md"`) |
| `allowed_filenames` | array | 30+ entries | Exact filenames without extension (e.g. `"Makefile"`, `"Dockerfile"`) |
| `exclude_directories` | array | 40+ entries | Directory names always excluded (e.g. `"node_modules"`, `"target"`, `".git"`) |
| `exclude_patterns` | array | | Additional file patterns to exclude (e.g. `"*.pyc"`, `"*.class"`) |
| `size_restricted_extensions` | array | `.csv`, `.json`, `.sql`, `.log`, etc. | Extensions allowed but with stricter size limit |
| `size_restricted_max_mb` | float | `1` | Maximum file size (MB) for size-restricted extensions |

The complete default extension and directory lists are embedded in the binaries from `assets/default_configuration.yaml`. User configuration entries are merged with the defaults; they do not replace them.

### `auto_ingestion`

Automatic ingestion pipeline settings.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | `true` | Enable automatic file ingestion |
| `auto_create_watches` | bool | `true` | Automatically register detected projects for watching |
| `include_common_files` | bool | `true` | Include common non-source files (config, docs) |
| `include_source_files` | bool | `true` | Include source code files |
| `max_files_per_batch` | integer | `5` | Maximum files processed per ingestion batch |
| `batch_delay_seconds` | float | `2.0` | Delay between ingestion batches |
| `max_file_size_mb` | integer | `50` | Maximum file size eligible for ingestion |
| `recursive_depth` | integer | `5` | Maximum directory depth for recursive scanning |
| `debounce_seconds` | integer | `10` | File event debounce window |

### `collections`

Collection naming settings.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `rules_collection_name` | string | `"rules"` | Collection name for behavioral rules |

### `grpc`

Daemon gRPC server settings (for communication between MCP Server and daemon).

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | `true` | Enable the gRPC server in the daemon |
| `host` | string | `"127.0.0.1"` | gRPC server bind address |
| `port` | integer | `50051` | gRPC server port |
| `fallback_to_direct` | bool | `true` | Fall back to direct Qdrant access if gRPC is unavailable |

---

## Example Configurations

### Local development (minimal)

```yaml
qdrant:
  url: "http://localhost:6333"
```

No other settings are needed for a local Qdrant instance at the default address.

### Qdrant Cloud

```yaml
qdrant:
  url: "https://your-cluster.qdrant.io:6333"
  api_key: "your-api-key-here"
  tls: true
  transport: "grpc"
```

Store the API key in the `QDRANT_API_KEY` environment variable instead of the config file when possible:

```bash
export QDRANT_URL=https://your-cluster.qdrant.io:6333
export QDRANT_API_KEY=your-api-key-here
```

### Custom installation paths

```yaml
database:
  path: /mnt/fast-ssd/workspace-qdrant/state.db

embedding:
  model_cache_dir: /mnt/fast-ssd/fastembed-models
```

### Reduced resource usage (low-memory system)

```yaml
daemon:
  max_concurrent_tasks: 2
  resource_limits:
    nice_level: 15
    inter_item_delay_ms: 200
    max_concurrent_embeddings: 1
    max_memory_percent: 50

queue_processor:
  batch_size: 5

embedding:
  batch_size: 10
  cache_max_entries: 200
```

### Higher throughput (dedicated server)

```yaml
daemon:
  max_concurrent_tasks: 8
  resource_limits:
    nice_level: 0
    inter_item_delay_ms: 0
    max_concurrent_embeddings: 4
    max_memory_percent: 85

queue_processor:
  batch_size: 25
  poll_interval_ms: 100

qdrant:
  pool:
    max_connections: 20
    min_idle_connections: 5
```

### Debug logging

```yaml
daemon:
  log_level: "debug"

logging:
  info_includes_connection_events: true
  info_includes_transport_details: true
  error_includes_stack_trace: true
```

Or use the environment variable without changing the config file:

```bash
WQM_LOG_LEVEL=debug memexd --foreground
```

---

## Qdrant Dashboard Visualization

When using the Qdrant web UI to visualize collections, specify the named vector to use.

| Collection | Named Vectors | Recommended for visualization |
|------------|---------------|-------------------------------|
| `projects` | `dense` (384-dim), `sparse` (BM25) | Use `dense` |
| `libraries` | `dense` (384-dim), `sparse` (BM25) | Use `dense` |
| `rules` | `dense` (384-dim) | Use `dense` |

Select `dense` in the dashboard's Visualize tab `using` parameter to produce meaningful spatial clusters. Do not select `sparse` — BM25 sparse vectors are not suitable for 2D/3D projection.
