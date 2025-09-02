# Configuration Management Guide

This guide covers the comprehensive configuration management system for workspace-qdrant-mcp, including environment-based configuration, YAML files, security features, and deployment profiles.

## Overview

The workspace-qdrant-mcp configuration system supports:

- **Multiple Environments**: development, staging, production
- **Configuration Sources**: YAML files, environment variables, .env files
- **Hot-Reload**: Development-time configuration changes
- **Validation**: Comprehensive validation with error reporting
- **Security**: Sensitive data masking and secure defaults
- **Profiles**: Pre-built configuration templates

## Configuration Hierarchy

Configuration is loaded in the following order of precedence (highest to lowest):

1. **Environment Variables** (highest priority)
2. **Local Configuration** (`local.yaml`)
3. **Environment-Specific Files** (`development.yaml`, `staging.yaml`, `production.yaml`)
4. **.env Files** in current directory
5. **Default Values** (lowest priority)

## Quick Start

### Basic Usage

```python
from workspace_qdrant_mcp.core.enhanced_config import EnhancedConfig

# Load configuration for development
config = EnhancedConfig(environment="development")

# Check if configuration is valid
if not config.is_valid:
    print("Configuration errors:")
    for error in config.validation_errors:
        print(f"  - {error}")
    exit(1)

# Access configuration
print(f"Server: {config.host}:{config.port}")
print(f"Qdrant: {config.qdrant.url}")
print(f"Environment: {config.environment}")
```

### Environment Detection

The environment is automatically detected from the `APP_ENV` environment variable:

```bash
export APP_ENV=production
python -m workspace_qdrant_mcp
```

## Configuration Files

### Environment-Specific Configuration

Create YAML files in `src/workspace_qdrant_mcp/config/`:

- `development.yaml` - Local development settings
- `staging.yaml` - Staging environment settings  
- `production.yaml` - Production deployment settings

### Local Overrides

Create `local.yaml` for personal development customizations:

```yaml
# src/workspace_qdrant_mcp/config/local.yaml
server:
  port: 8001  # Use different port
  debug: true

qdrant:
  url: "http://localhost:6334"  # Custom Qdrant port

development:
  hot_reload: true
  performance_metrics: false
```

**Note**: `local.yaml` is git-ignored and won't be committed.

### Configuration Schema

```yaml
# Server settings
server:
  host: "127.0.0.1"
  port: 8000
  debug: false
  log_level: "INFO"
  reload: false

# Qdrant database connection
qdrant:
  url: "http://localhost:6333"
  api_key: null
  timeout: 30
  prefer_grpc: false
  health_check_interval: 60
  retry_attempts: 3
  connection_pool_size: 5

# Embedding configuration
embedding:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  enable_sparse_vectors: true
  chunk_size: 800
  chunk_overlap: 120
  batch_size: 50
  cache_embeddings: false
  embedding_timeout: 30
  max_concurrent_requests: 10

# Workspace management
workspace:
  collections: ["project"]
  global_collections: ["docs", "references", "standards"]
  github_user: null
  collection_prefix: ""
  max_collections: 100
  auto_create_collections: true
  cleanup_on_exit: false

# Security settings
security:
  mask_sensitive_logs: true
  validate_ssl: true
  allow_http: false
  cors_enabled: false
  cors_origins: []
  rate_limiting: false
  max_requests_per_minute: 1000

# Monitoring and observability
monitoring:
  metrics_enabled: false
  tracing_enabled: false
  health_endpoint: "/health"
  metrics_endpoint: "/metrics"
  log_structured: false
  retention_days: 7
  alerts_enabled: false

# Performance optimization
performance:
  worker_processes: 1
  max_connections: 100
  keepalive_timeout: 65
  request_timeout: 30
  memory_limit: "512MB"
  cpu_limit: 1.0

# Development features
development:
  hot_reload: false
  config_watch: false
  detailed_logging: false
  performance_metrics: false
  mock_external_services: false
```

## Environment Variables

### Primary Environment Variables

Use the `WORKSPACE_QDRANT_` prefix for primary configuration:

```bash
# Server configuration
export WORKSPACE_QDRANT_HOST=0.0.0.0
export WORKSPACE_QDRANT_PORT=8080
export WORKSPACE_QDRANT_DEBUG=true

# Component configuration (nested)
export WORKSPACE_QDRANT_QDRANT__URL=https://my-qdrant.example.com
export WORKSPACE_QDRANT_QDRANT__API_KEY=your-api-key
export WORKSPACE_QDRANT_EMBEDDING__MODEL=sentence-transformers/all-MiniLM-L6-v2
export WORKSPACE_QDRANT_WORKSPACE__GITHUB_USER=myusername
```

### Legacy Environment Variables

Backward compatibility with existing variables:

```bash
# Legacy Qdrant configuration
export QDRANT_URL=http://localhost:6333
export QDRANT_API_KEY=your-api-key

# Legacy embedding configuration
export FASTEMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
export ENABLE_SPARSE_VECTORS=true
export CHUNK_SIZE=800
export CHUNK_OVERLAP=120
export BATCH_SIZE=50

# Legacy workspace configuration
export GITHUB_USER=myusername
export COLLECTIONS=project,docs
export GLOBAL_COLLECTIONS=references,standards
```

### Environment Variable Substitution

YAML files support environment variable substitution:

```yaml
qdrant:
  url: "${QDRANT_URL:-http://localhost:6333}"
  api_key: "${QDRANT_API_KEY}"

server:
  host: "${SERVER_HOST:-127.0.0.1}"
  port: ${SERVER_PORT:-8000}

workspace:
  github_user: "${GITHUB_USER}"
  collection_prefix: "${TENANT_ID}_"
```

## Configuration Profiles

### Using Profiles

Configuration profiles provide ready-to-use templates:

```bash
# List available profiles
ls src/workspace_qdrant_mcp/config/profiles/

# Copy a profile to your environment
cp src/workspace_qdrant_mcp/config/profiles/local-development.yaml \
   src/workspace_qdrant_mcp/config/development.yaml
```

### Available Profiles

- **`local-development.yaml`**: Single-developer local setup
- **`kubernetes-deployment.yaml`**: Kubernetes cluster deployment
- **`enterprise-deployment.yaml`**: Enterprise with enhanced security

### Creating Custom Profiles

Create profiles for your deployment scenarios:

```yaml
# profiles/my-custom-profile.yaml
server:
  host: "0.0.0.0"
  port: 8000

qdrant:
  url: "${CUSTOM_QDRANT_URL}"
  api_key: "${CUSTOM_API_KEY}"

# Custom sections
custom:
  feature_flags: true
  custom_setting: "value"
```

## Development Features

### Hot-Reload Configuration

Enable configuration hot-reload for development:

```yaml
# development.yaml
development:
  hot_reload: true
  config_watch: true
```

The system will automatically reload when configuration files change.

### Configuration Validation

Validate configuration at startup:

```python
config = EnhancedConfig()

if not config.is_valid:
    print("Configuration validation failed:")
    for error in config.validation_errors:
        print(f"  ❌ {error}")
    exit(1)

print("✅ Configuration is valid")
```

### Configuration Summary

Get configuration summary for debugging:

```python
config = EnhancedConfig()
summary = config.get_config_summary()

print(f"Environment: {summary['environment']}")
print(f"Config files loaded: {summary['config_files_loaded']}")
print(f"Validation status: {summary['validation_status']}")
```

## Security

### Sensitive Data Masking

Sensitive values are automatically masked in logs:

```python
config = EnhancedConfig()
config.security.mask_sensitive_logs = True

# API keys will be masked in logs
masked = config.mask_sensitive_value("secret-api-key-123")
# Returns: "se****23"
```

### Production Security

Production environments enforce security policies:

```yaml
# production.yaml
security:
  mask_sensitive_logs: true
  validate_ssl: true
  allow_http: false
  cors_enabled: true
  cors_origins: ["https://myapp.com"]
  rate_limiting: true
  authentication_required: true
```

### Environment-Specific Validation

Different environments have different validation rules:

- **Development**: Relaxed validation, warnings only
- **Staging**: Production-like validation with some flexibility
- **Production**: Strict validation, security-focused

## Deployment

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.11-slim

# Copy configuration files
COPY src/workspace_qdrant_mcp/config/ /app/config/

# Set environment
ENV APP_ENV=production
ENV WORKSPACE_QDRANT_QDRANT__URL=https://qdrant-prod:6333

CMD ["python", "-m", "workspace_qdrant_mcp"]
```

### Kubernetes Deployment

```yaml
# kubernetes-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: workspace-qdrant-config
data:
  production.yaml: |
    server:
      host: "0.0.0.0"
      port: 8000
    qdrant:
      url: "${QDRANT_SERVICE_URL}"
    security:
      mask_sensitive_logs: true

---
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      containers:
      - name: workspace-qdrant-mcp
        env:
        - name: APP_ENV
          value: "production"
        - name: QDRANT_SERVICE_URL
          value: "http://qdrant-service:6333"
        volumeMounts:
        - name: config-volume
          mountPath: /app/config
      volumes:
      - name: config-volume
        configMap:
          name: workspace-qdrant-config
```

### Environment Management

```bash
# Development
export APP_ENV=development
python -m workspace_qdrant_mcp

# Staging
export APP_ENV=staging
export QDRANT_URL=https://qdrant-staging.example.com
python -m workspace_qdrant_mcp

# Production
export APP_ENV=production
export QDRANT_URL=https://qdrant-prod.example.com
export QDRANT_API_KEY=prod-api-key
python -m workspace_qdrant_mcp
```

## Monitoring and Observability

### Health Checks

Configuration includes health check endpoints:

```yaml
monitoring:
  health_endpoint: "/health"
  metrics_endpoint: "/metrics"
```

### Structured Logging

Enable structured logging for production:

```yaml
monitoring:
  log_structured: true
```

### Metrics Collection

Enable metrics for monitoring:

```yaml
monitoring:
  metrics_enabled: true
  tracing_enabled: true
```

## Troubleshooting

### Common Issues

1. **Configuration not loading**
   - Check file permissions
   - Verify YAML syntax
   - Check environment variable names

2. **Validation errors**
   - Review validation messages
   - Check required fields
   - Verify value ranges

3. **Environment variables not working**
   - Check variable names and prefixes
   - Verify export in current shell
   - Check for typos in nested variables

### Debug Configuration

```python
from workspace_qdrant_mcp.core.enhanced_config import EnhancedConfig
import json

config = EnhancedConfig()
summary = config.get_config_summary()
print(json.dumps(summary, indent=2))
```

### Validation Details

```python
config = EnhancedConfig()
errors = config.validate_config()

if errors:
    print("Configuration issues found:")
    for i, error in enumerate(errors, 1):
        print(f"{i}. {error}")
else:
    print("Configuration is valid!")
```

## Best Practices

### Development

1. Use `local.yaml` for personal settings
2. Enable hot-reload for faster iteration
3. Use debug logging for troubleshooting
4. Validate configuration early

### Staging

1. Mirror production settings where possible
2. Enable monitoring and logging
3. Use environment-specific secrets
4. Test configuration changes

### Production

1. Use HTTPS everywhere
2. Enable security features
3. Monitor configuration changes
4. Regular security audits
5. Backup configuration files

### Security

1. Never commit sensitive values to git
2. Use environment variables for secrets
3. Enable log masking in production
4. Regularly rotate API keys
5. Audit configuration access

## Migration Guide

### From Legacy Configuration

Migrating from the original config system:

1. **Keep existing environment variables** - Legacy variables are still supported
2. **Create environment-specific files** - Add YAML files for each environment
3. **Update application code** - Switch to `EnhancedConfig` class
4. **Test configuration loading** - Verify all settings load correctly
5. **Enable new features** - Add monitoring, security features as needed

### Example Migration

```python
# Before (legacy)
from workspace_qdrant_mcp.core.config import Config
config = Config()

# After (enhanced)
from workspace_qdrant_mcp.core.enhanced_config import EnhancedConfig
config = EnhancedConfig(environment="development")

# All existing config attributes still work
print(config.qdrant.url)
print(config.embedding.model)
print(config.workspace.collections)
```

## API Reference

See the [API documentation](../API.md) for detailed class and method documentation.