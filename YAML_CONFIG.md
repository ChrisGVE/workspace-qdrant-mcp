# YAML Configuration Support

The workspace-qdrant-mcp system supports YAML configuration files for project-specific configuration management.

## Configuration Systems

This project has two main components with separate configuration systems:

1. **MCP Server/Python Components** - Uses the configuration structure documented below
2. **Rust Daemon** - Uses a separate configuration structure (see daemon logs for current config path)

The configuration precedence and structure documented below applies to the MCP server and Python components.

## Usage

Start the server with a YAML configuration file:

```bash
workspace-qdrant-mcp --config=project-config.yaml
```

## Configuration Precedence

The configuration system follows this precedence order (highest to lowest):

1. **Command line arguments** (e.g., `--host`, `--port`)
2. **Explicit YAML configuration file** (when `--config` is specified)
3. **Auto-discovered configuration files** (searched in this order):
   - Project-specific: `.workspace-qdrant.yaml` or `.workspace-qdrant.yml` in current directory
   - Project-specific: `workspace_qdrant_config.yaml` or `workspace_qdrant_config.yml` in current directory
   - User XDG config: `~/.config/workspace-qdrant/config.yaml` or `config.yml`
   - User config: `~/.config/workspace-qdrant/workspace_qdrant_config.yaml` or `workspace_qdrant_config.yml`
   - System config: `/etc/workspace-qdrant/config.yaml` or `config.yml` (Unix-like systems)
4. **Environment variables** (`WORKSPACE_QDRANT_*`)
5. **Default values**

**Supported file extensions**: `.yaml`, `.yml` (when both exist, `.yaml` takes precedence)

## YAML Configuration Structure

```yaml
# Qdrant database configuration
qdrant:
  url: "http://localhost:6333"
  api_key: null  # Set to your API key for Qdrant Cloud
  timeout_seconds: 30
  retry_count: 3
  use_https: false
  verify_ssl: true
  prefer_grpc: true

# Daemon configuration
daemon:
  database_path: "~/.config/workspace-qdrant/state.db"
  max_concurrent_jobs: 4
  job_timeout_seconds: 300
  max_memory_mb: 1024
  max_cpu_percent: 80

  priority_levels:
    mcp_server: "high"
    cli_commands: "medium"
    background_watching: "low"

  grpc:
    host: "127.0.0.1"
    port: 50051
    max_message_size_mb: 100

# Embedding configuration
embedding:
  provider: "fastembed"  # Options: fastembed, openai, huggingface
  dense_model: "sentence-transformers/all-MiniLM-L6-v2"
  sparse_model: "prithivida/Splade_PP_en_v1"

  openai:
    api_key: null
    model: "text-embedding-3-small"

  huggingface:
    api_key: null
    model: "sentence-transformers/all-MiniLM-L6-v2"

# Collection management
collections:
  auto_create: false
  default_global: ["scratchbook"]
  reserved_prefixes: ["_", "system_", "temp_"]

  settings:
    scratchbook:
      description: "Quick notes and temporary documents"
      auto_ingest: true

# File watching configuration
watching:
  auto_watch_project: true
  debounce_seconds: 5
  max_file_size_mb: 50
  recursive: true
  max_depth: 5
  follow_symlinks: false

  include_patterns:
    - "*.txt"
    - "*.md"
    - "*.pdf"
    - "*.epub"
    - "*.docx"
    - "*.py"
    - "*.js"
    - "*.ts"
    - "*.html"
    - "*.css"
    - "*.yaml"
    - "*.yml"
    - "*.json"
    - "*.toml"
    - "Dockerfile"
    - "*.dockerfile"
    - "Makefile"
    - "*.mk"
    - ".github/**/*.yml"
    - ".github/**/*.yaml"

  ignore_patterns:
    - "*.tmp"
    - "*.log"
    - "*.cache"
    - ".git/*"
    - "node_modules/*"
    - "__pycache__/*"
    - "*.pyc"
    - ".DS_Store"

  # Additional patterns beyond defaults
  custom_include_patterns: []
  custom_exclude_patterns: []

# Document processing configuration
processing:
  chunk_size: 1000
  chunk_overlap: 200
  min_chunk_size: 100

  pdf:
    extract_images: false
    extract_tables: true

  docx:
    extract_images: false
    extract_styles: false

  code:
    include_comments: true
    language_detection: true

# Web UI configuration
web_ui:
  enabled: true
  host: "127.0.0.1"
  port: 3000
  auto_launch: false
  cors_origins: ["http://localhost:3000"]
  auth_required: false

# Logging configuration
logging:
  level: "INFO"  # Options: DEBUG, INFO, WARN, ERROR
  format: "json"  # Options: json, text
  file_path: "~/.config/workspace-qdrant/logs/daemon.log"
  max_file_size_mb: 10
  backup_count: 5

  components:
    qdrant_client: "WARN"
    embedding: "INFO"
    file_watcher: "INFO"
    grpc_server: "WARN"

# Performance monitoring
monitoring:
  enabled: true
  metrics_port: 9090
  health_check_interval_seconds: 60

  thresholds:
    max_response_time_ms: 1000
    max_memory_usage_mb: 512
    max_cpu_usage_percent: 70

# Development settings
development:
  debug_mode: false
  profile_performance: false
  mock_embedding: false
  test_data_path: "tests/data"
```

## Environment Variable Substitution

YAML configuration files support environment variable substitution using `${VAR_NAME}` or `${VAR_NAME:default_value}` syntax:

```yaml
qdrant:
  url: "${QDRANT_URL:http://localhost:6333}"
  api_key: "${QDRANT_API_KEY}"

embedding:
  openai:
    api_key: "${OPENAI_API_KEY}"
```

## Configuration Examples

### Development Configuration

```yaml
# dev-config.yaml
qdrant:
  url: "http://localhost:6333"

embedding:
  provider: "fastembed"
  dense_model: "sentence-transformers/all-MiniLM-L6-v2"

daemon:
  max_concurrent_jobs: 2

collections:
  auto_create: true

watching:
  debounce_seconds: 2

logging:
  level: "DEBUG"
  format: "text"

development:
  debug_mode: true
  profile_performance: true
```

### Production Configuration

```yaml
# prod-config.yaml
qdrant:
  url: "https://your-qdrant-cluster.example.com"
  api_key: "${QDRANT_API_KEY}"
  timeout_seconds: 60
  use_https: true
  prefer_grpc: true

embedding:
  provider: "fastembed"
  dense_model: "sentence-transformers/all-MiniLM-L6-v2"
  sparse_model: "prithivida/Splade_PP_en_v1"

daemon:
  max_concurrent_jobs: 8
  max_memory_mb: 2048

collections:
  auto_create: false
  default_global: ["docs", "standards", "references"]

monitoring:
  enabled: true
  metrics_port: 9090

logging:
  level: "INFO"
  format: "json"
```

### Qdrant Cloud Configuration

```yaml
# cloud-config.yaml
qdrant:
  url: "https://xyz-abc-def.us-east-1-0.aws.cloud.qdrant.io"
  api_key: "${QDRANT_CLOUD_API_KEY}"
  timeout_seconds: 30
  use_https: true
  prefer_grpc: true

embedding:
  provider: "openai"
  openai:
    api_key: "${OPENAI_API_KEY}"
    model: "text-embedding-3-small"

daemon:
  grpc:
    host: "0.0.0.0"
    port: 50051
```

## Configuration Validation

The server validates your YAML configuration on startup and provides detailed error messages:

```bash
workspace-qdrant-mcp --config=invalid-config.yaml
# Error: Configuration validation failed: Chunk overlap must be less than chunk size
```

Common validation errors:
- Invalid Qdrant URL format
- Chunk overlap larger than chunk size
- Invalid log levels or formats
- Missing required API keys for selected providers
- Invalid priority levels (must be: low, medium, high)