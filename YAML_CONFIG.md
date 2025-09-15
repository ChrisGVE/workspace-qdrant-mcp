# YAML Configuration Support

The workspace-qdrant-mcp server now supports YAML configuration files for easier project-specific configuration management.

## Usage

Start the server with a YAML configuration file:

```bash
workspace-qdrant-mcp --config=project-config.yaml
```

## Configuration Precedence

The configuration system follows this precedence order (highest to lowest):

1. **Command line arguments** (e.g., `--host`, `--port`)
2. **YAML configuration file** (when `--config` is specified)
3. **Environment variables** (`WORKSPACE_QDRANT_*` or legacy variables)
4. **Default values**

## YAML Configuration Structure

The YAML configuration file supports all the same options as environment variables, but with a more readable structure:

```yaml
# Server configuration
host: "127.0.0.1"
port: 8000
debug: false

# Qdrant database configuration
qdrant:
  url: "http://localhost:6333"
  api_key: null  # Set to your API key for Qdrant Cloud
  timeout: 30
  prefer_grpc: false

# Embedding configuration
embedding:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  enable_sparse_vectors: true
  chunk_size: 800
  chunk_overlap: 120
  batch_size: 50

# Workspace configuration
workspace:
  collection_types: ["project"]
  global_collections: ["docs", "references", "standards"]
  github_user: null  # Set to your GitHub username
  auto_create_collections: false
  memory_collection_name: "__memory"
  code_collection_name: "__code"
```

## Environment Variable Equivalents

For reference, the YAML configuration maps to these environment variables:

| YAML Path | Environment Variable |
|-----------|---------------------|
| `host` | `WORKSPACE_QDRANT_HOST` |
| `port` | `WORKSPACE_QDRANT_PORT` |
| `debug` | `WORKSPACE_QDRANT_DEBUG` |
| `qdrant.url` | `WORKSPACE_QDRANT_QDRANT__URL` or `QDRANT_URL` (legacy) |
| `qdrant.api_key` | `WORKSPACE_QDRANT_QDRANT__API_KEY` or `QDRANT_API_KEY` (legacy) |
| `embedding.model` | `WORKSPACE_QDRANT_EMBEDDING__MODEL` or `FASTEMBED_MODEL` (legacy) |
| `workspace.collection_types` | `WORKSPACE_QDRANT_WORKSPACE__COLLECTION_TYPES` (comma-separated) |
| `workspace.global_collections` | `WORKSPACE_QDRANT_WORKSPACE__GLOBAL_COLLECTIONS` (comma-separated) |
| `workspace.github_user` | `WORKSPACE_QDRANT_WORKSPACE__GITHUB_USER` |

## Examples

### Development Configuration

```yaml
# dev-config.yaml
debug: true
qdrant:
  url: "http://localhost:6333"
embedding:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  batch_size: 10  # Smaller batch for development
workspace:
  collection_types: ["dev", "test"]
  global_collections: ["docs"]
```

### Production Configuration

```yaml
# prod-config.yaml
host: "0.0.0.0"
port: 8000
debug: false
qdrant:
  url: "https://your-qdrant-cluster.example.com"
  api_key: "your-production-api-key"
  timeout: 60
  prefer_grpc: true
embedding:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  enable_sparse_vectors: true
  batch_size: 100
workspace:
  collection_types: ["project", "docs", "tests"]
  global_collections: ["standards", "references", "shared"]
  github_user: "your-username"
  auto_create_collections: true
```

### Cloud Configuration

```yaml
# cloud-config.yaml
qdrant:
  url: "https://xyz-abc-def.us-east-1-0.aws.cloud.qdrant.io"
  api_key: "your-qdrant-cloud-api-key"
  timeout: 30
  prefer_grpc: false
embedding:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  enable_sparse_vectors: true
workspace:
  collection_types: ["project"]
  global_collections: ["docs", "references"]
  github_user: "your-github-username"
```

## Benefits of YAML Configuration

1. **Project-specific configs**: Each project can have its own configuration file
2. **Version control**: YAML configs can be committed to your repository
3. **Readability**: Much easier to read and edit than environment variables
4. **Validation**: Built-in validation with helpful error messages
5. **Documentation**: Self-documenting with comments and structure
6. **Sharing**: Easy to share configurations between team members

## Validation

The server will validate your YAML configuration on startup and provide helpful error messages if there are issues:

```bash
workspace-qdrant-mcp --config=invalid-config.yaml
# Error: Configuration validation failed: Chunk overlap must be less than chunk size
```

## Backward Compatibility

All existing environment variable configurations continue to work as before. The YAML configuration system is additive and doesn't break existing setups.