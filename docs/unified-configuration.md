# Unified Configuration System

The workspace-qdrant-mcp project implements a unified configuration system that supports both the Rust daemon (`memexd`) and Python MCP server components. This system allows seamless configuration management across different components using multiple file formats.

## Overview

The unified configuration system provides:

- **Multi-format support**: TOML, YAML, and JSON configuration files
- **Automatic format detection**: Based on file extension
- **Environment variable overrides**: Consistent across both components
- **Format conversion**: Bidirectional conversion between formats
- **Configuration validation**: Shared validation rules
- **Hot-reload capability**: File watching with automatic reloading
- **Backwards compatibility**: Existing configurations continue to work

## Supported Formats

### TOML (Preferred by Rust Daemon)
```toml
# Rust-friendly TOML configuration
log_level = "info"
chunk_size = 1000

[qdrant]
url = "http://localhost:6333"
timeout_ms = 30000

[auto_ingestion]
enabled = true
target_collection_suffix = "scratchbook"
```

### YAML (Preferred by Python MCP Server)
```yaml
# Python-friendly YAML configuration
host: "127.0.0.1"
port: 8000
debug: false

qdrant:
  url: "http://localhost:6333"
  timeout: 30

embedding:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  chunk_size: 800

auto_ingestion:
  enabled: true
  target_collection_suffix: "scratchbook"
```

### JSON (Universal Support)
```json
{
  "host": "127.0.0.1",
  "port": 8000,
  "qdrant": {
    "url": "http://localhost:6333",
    "timeout": 30
  },
  "auto_ingestion": {
    "enabled": true,
    "target_collection_suffix": "scratchbook"
  }
}
```

## Configuration File Discovery

The system searches for configuration files in this order:

1. `workspace_qdrant_config.toml`
2. `.workspace-qdrant.toml`
3. `config.toml`
4. `workspace_qdrant_config.yaml`
5. `workspace_qdrant_config.yml`
6. `.workspace-qdrant.yaml`
7. `.workspace-qdrant.yml`
8. `config.yaml`
9. `config.yml`

## Environment Variable Overrides

All configuration values can be overridden using environment variables with the `WORKSPACE_QDRANT_` prefix:

### Server Configuration
```bash
export WORKSPACE_QDRANT_HOST=0.0.0.0
export WORKSPACE_QDRANT_PORT=8080
export WORKSPACE_QDRANT_DEBUG=true
```

### Nested Configuration (Double Underscore Syntax)
```bash
export WORKSPACE_QDRANT_QDRANT__URL=http://remote:6333
export WORKSPACE_QDRANT_QDRANT__API_KEY=secret-key
export WORKSPACE_QDRANT_EMBEDDING__MODEL=sentence-transformers/all-mpnet-base-v2
export WORKSPACE_QDRANT_WORKSPACE__COLLECTION_SUFFIXES=docs,code,notes
export WORKSPACE_QDRANT_AUTO_INGESTION__ENABLED=false
```

## Python Usage

### Using the Unified Configuration Manager

```python
from workspace_qdrant_mcp.core.unified_config import UnifiedConfigManager, ConfigFormat

# Auto-discover and load configuration
config_manager = UnifiedConfigManager()
config = config_manager.load_config()

# Load specific configuration file
config = config_manager.load_config(config_file="my_config.yaml")

# Prefer specific format during auto-discovery
config = config_manager.load_config(prefer_format=ConfigFormat.TOML)
```

### Configuration Information
```python
# Get information about discovered config sources
info = config_manager.get_config_info()
print(f"Config directory: {info['config_dir']}")
print(f"Found {len(info['sources'])} config files")

# List all sources
for source in info['sources']:
    if source['exists']:
        print(f"  - {source['file_path']} ({source['format']})")
```

### Format Conversion
```python
# Convert between formats
config_manager.convert_config(
    source_file="config.toml",
    target_file="config.yaml", 
    target_format=ConfigFormat.YAML
)

# Save configuration in specific format
config_manager.save_config(config, "output.json", ConfigFormat.JSON)
```

### Configuration Validation
```python
# Validate configuration file
issues = config_manager.validate_config_file("config.yaml")
if issues:
    print("Configuration issues:")
    for issue in issues:
        print(f"  - {issue}")

# Validate loaded configuration
issues = config.validate_config()
```

### Configuration Watching
```python
def on_config_change(new_config):
    print("Configuration changed!")
    # Handle configuration update

config_manager.watch_config(on_config_change)
# ... do other work ...
config_manager.stop_watching()
```

### Creating Default Configurations
```python
# Create default config files in multiple formats
created_files = config_manager.create_default_configs([
    ConfigFormat.TOML,
    ConfigFormat.YAML
])

for format_type, file_path in created_files.items():
    print(f"Created {format_type.value} config: {file_path}")
```

## Rust Usage

### Using the Unified Configuration Manager

```rust
use workspace_qdrant_core::unified_config::{UnifiedConfigManager, ConfigFormat};

// Auto-discover and load configuration
let config_manager = UnifiedConfigManager::new(None);
let daemon_config = config_manager.load_config(None)?;

// Load specific configuration file
let config_file = Path::new("config.toml");
let daemon_config = config_manager.load_config(Some(config_file))?;

// Use with ProcessingEngine
let engine = ProcessingEngine::with_unified_config(None, None)?;
// or
let engine = ProcessingEngine::from_unified_config()?;
```

### Configuration Information
```rust
// Get configuration information
let info = config_manager.get_config_info();
println!("Config directory: {}", info["config_dir"]);

// Discover configuration sources
let sources = config_manager.discover_config_sources();
for (path, format, exists) in sources {
    if exists {
        println!("Found config: {} ({:?})", path.display(), format);
    }
}
```

### Format Conversion
```rust
// Convert between formats
config_manager.convert_config(
    Path::new("config.yaml"),
    Path::new("config.toml"),
    Some(ConfigFormat::Toml)
)?;

// Save configuration
config_manager.save_config(&daemon_config, Path::new("output.toml"), ConfigFormat::Toml)?;
```

## CLI Usage

The unified configuration system is integrated into the CLI with comprehensive commands:

### Configuration Information
```bash
# Show configuration sources and status
wqm config info

# Show configuration with format preference
wqm config info --format toml
```

### Configuration Display
```bash
# Show current configuration
wqm config show

# Show in specific format
wqm config show --format json

# Show specific section
wqm config show --section qdrant
```

### Configuration Validation
```bash
# Validate current configuration
wqm config validate

# Validate specific file
wqm config validate --file my_config.yaml

# Verbose validation output
wqm config validate --verbose
```

### Format Conversion
```bash
# Convert configuration formats
wqm config convert config.toml config.yaml

# Specify target format explicitly
wqm config convert config.yaml config.json --target-format json

# Validate before conversion
wqm config convert source.toml target.yaml --validate
```

### Creating Default Configurations
```bash
# Initialize default configuration files
wqm config init-unified

# Create specific formats
wqm config init-unified --format toml --format yaml

# Overwrite existing files
wqm config init-unified --force
```

### Configuration Watching
```bash
# Watch configuration file for changes
wqm config watch

# Watch specific file
wqm config watch --file my_config.yaml

# Custom check interval
wqm config watch --interval 5
```

### Environment Variables Help
```bash
# Show all available environment variable overrides
wqm config env-vars
```

## Configuration Schema

The unified configuration supports all settings used by both components:

### Core Settings
- `host`: Server host address (Python MCP server)
- `port`: Server port (Python MCP server)
- `debug`: Debug mode flag
- `log_level`: Logging level
- `log_file`: Log file path (Rust daemon)

### Qdrant Configuration
- `qdrant.url`: Qdrant server URL
- `qdrant.api_key`: API key for authentication
- `qdrant.timeout`: Connection timeout (seconds for Python, milliseconds for Rust)
- `qdrant.prefer_grpc`: Use gRPC protocol (Python)
- `qdrant.transport`: Transport mode "Http"/"Grpc" (Rust)

### Embedding Configuration
- `embedding.model`: Embedding model name
- `embedding.enable_sparse_vectors`: Enable sparse vector generation
- `embedding.chunk_size`: Text chunk size
- `embedding.chunk_overlap`: Chunk overlap size
- `embedding.batch_size`: Processing batch size

### Workspace Configuration
- `workspace.collection_suffixes`: Collection name suffixes
- `workspace.global_collections`: Cross-project collections
- `workspace.github_user`: GitHub username
- `workspace.collection_prefix`: Collection name prefix
- `workspace.max_collections`: Maximum collections limit
- `workspace.auto_create_collections`: Auto-create collections flag

### Auto-Ingestion Configuration
- `auto_ingestion.enabled`: Enable auto-ingestion
- `auto_ingestion.auto_create_watches`: Auto-create file watches
- `auto_ingestion.include_common_files`: Include common document types
- `auto_ingestion.include_source_files`: Include source code files
- `auto_ingestion.target_collection_suffix`: Target collection suffix
- `auto_ingestion.max_files_per_batch`: Maximum files per batch
- `auto_ingestion.batch_delay_seconds`: Delay between batches
- `auto_ingestion.max_file_size_mb`: Maximum file size limit
- `auto_ingestion.recursive_depth`: Directory recursion depth
- `auto_ingestion.debounce_seconds`: File change debounce time

### gRPC Configuration
- `grpc.enabled`: Enable gRPC communication
- `grpc.host`: gRPC server host
- `grpc.port`: gRPC server port
- `grpc.fallback_to_direct`: Fallback to direct mode
- `grpc.connection_timeout`: Connection timeout
- `grpc.max_retries`: Maximum retry attempts

## Best Practices

### Configuration File Organization
1. **Use descriptive file names**: `workspace_qdrant_config.toml` or `workspace_qdrant_config.yaml`
2. **Prefer TOML for Rust-heavy deployments**: Better performance and native support
3. **Prefer YAML for Python-heavy deployments**: More expressive and readable
4. **Use JSON for programmatic configuration**: Universal parsing support

### Environment Variable Usage
1. **Use environment variables for secrets**: API keys, passwords
2. **Use environment variables for deployment-specific settings**: URLs, ports
3. **Use configuration files for stable settings**: Models, chunk sizes, features
4. **Follow the double-underscore convention**: For nested configuration

### Validation and Testing
1. **Always validate configuration**: Before deploying to production
2. **Test format conversions**: Ensure data integrity
3. **Use the CLI validation commands**: `wqm config validate`
4. **Monitor configuration changes**: Use file watching for critical deployments

### Migration Strategies
1. **Start with existing format**: Convert gradually
2. **Test both components**: Ensure compatibility
3. **Use validation scripts**: Verify functionality
4. **Backup original configurations**: Before conversion

## Troubleshooting

### Common Issues

#### Configuration File Not Found
```bash
# Check discovered sources
wqm config info

# Create default configuration
wqm config init-unified
```

#### Validation Errors
```bash
# Get detailed validation information
wqm config validate --verbose

# Check specific configuration file
wqm config validate --file problematic_config.yaml
```

#### Environment Override Not Working
```bash
# Verify environment variables are set
env | grep WORKSPACE_QDRANT

# Check variable format (use double underscore for nested)
export WORKSPACE_QDRANT_QDRANT__URL=http://localhost:6333
```

#### Format Conversion Issues
```bash
# Validate source before conversion
wqm config validate --file source.toml

# Convert with explicit format
wqm config convert source.toml target.yaml --target-format yaml
```

### Debug Mode
Enable debug logging to troubleshoot configuration issues:

```bash
# Python component
export WORKSPACE_QDRANT_DEBUG=true

# Rust component  
export WORKSPACE_QDRANT_LOG_LEVEL=debug
```

## Migration Guide

### From Separate Configurations

If you currently have separate configuration files for the Rust daemon and Python MCP server:

1. **Identify common settings**: Compare both configurations
2. **Create unified configuration**: Merge compatible settings
3. **Test both components**: Ensure both can load the unified config
4. **Validate functionality**: Use validation scripts
5. **Deploy gradually**: Test in development first

### Example Migration

**Before** - Separate files:
```toml
# rust_config.toml
[qdrant]
url = "http://localhost:6333"

[auto_ingestion]
enabled = true
```

```yaml
# python_config.yaml
host: "127.0.0.1" 
port: 8000
qdrant:
  url: "http://localhost:6333"
```

**After** - Unified configuration:
```yaml
# unified_config.yaml
host: "127.0.0.1"
port: 8000

qdrant:
  url: "http://localhost:6333"
  timeout: 30

auto_ingestion:
  enabled: true
  target_collection_suffix: "scratchbook"
```

## Testing and Validation

### Validation Scripts

The project includes validation scripts to test the unified configuration system:

```bash
# Basic functionality test (no external dependencies)
python3 scripts/simple_config_test.py

# Comprehensive validation (requires all dependencies)
python3 scripts/validate_unified_config.py
```

### Unit Tests

Run the comprehensive test suite:

```bash
# Run unified configuration tests
pytest tests/test_unified_config.py -v

# Run with coverage
pytest tests/test_unified_config.py --cov=workspace_qdrant_mcp.core.unified_config
```

### Integration Testing

Test with both components:

```bash
# Test Python MCP server with unified config
wqm config validate

# Test Rust daemon configuration (when available)
cargo test unified_config
```

## Performance Considerations

### File Format Performance
- **TOML**: Fastest parsing for Rust components
- **YAML**: Good balance of readability and performance
- **JSON**: Fastest parsing for Python components, least readable

### Configuration Loading
- **Auto-discovery**: Minimal overhead, searches in order
- **Explicit file**: Fastest, no search overhead
- **Environment overrides**: Processed after file loading
- **Validation**: Optional, can be skipped for performance

### Memory Usage
- **Rust**: Minimal memory overhead with TOML
- **Python**: Moderate memory usage with any format
- **Watching**: Additional memory for file system monitoring

## Security Considerations

### Sensitive Data
- **Never commit secrets**: Use environment variables
- **Use restricted file permissions**: 600 for config files with secrets
- **Rotate API keys regularly**: Especially for cloud deployments

### File System Security
- **Validate file paths**: Prevent path traversal attacks
- **Use absolute paths**: For production deployments
- **Monitor file changes**: Log configuration modifications

### Environment Variables
- **Sanitize environment**: In shared environments
- **Use prefixes**: To avoid variable name collisions
- **Document required variables**: For deployment

## Contributing

When adding new configuration options:

1. **Update both Python and Rust**: Maintain compatibility
2. **Add environment variable support**: Follow naming conventions
3. **Update validation**: Add appropriate validation rules
4. **Update documentation**: Include new options in this guide
5. **Add tests**: Cover new functionality
6. **Update templates**: Include in default configurations

### Configuration Schema Updates

When modifying the configuration schema:

1. **Maintain backwards compatibility**: When possible
2. **Add migration logic**: For breaking changes
3. **Update validation**: Include new validation rules
4. **Update conversion logic**: Ensure format conversion works
5. **Test thoroughly**: With both components

## Conclusion

The unified configuration system provides a robust, flexible foundation for managing configuration across the workspace-qdrant-mcp project. It supports multiple formats, provides comprehensive validation, and maintains backwards compatibility while enabling advanced features like hot-reloading and format conversion.

For questions or issues, please refer to the troubleshooting section or consult the validation scripts for examples of proper usage.