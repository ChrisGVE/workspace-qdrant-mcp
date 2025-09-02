# Configuration Profiles

This directory contains predefined configuration profiles for common deployment scenarios.

## Available Profiles

### Development Profiles
- `local-development.yaml` - Standard local development setup
- `docker-development.yaml` - Development with Docker Compose
- `team-development.yaml` - Shared development environment

### Deployment Profiles  
- `cloud-deployment.yaml` - Generic cloud deployment
- `kubernetes-deployment.yaml` - Kubernetes-specific configuration
- `enterprise-deployment.yaml` - Enterprise setup with enhanced security

### Testing Profiles
- `unit-testing.yaml` - Configuration for unit tests
- `integration-testing.yaml` - Integration testing setup
- `performance-testing.yaml` - Performance benchmarking

## Using Profiles

Copy a profile to your environment-specific configuration:

```bash
# Use local development profile
cp profiles/local-development.yaml development.yaml

# Use cloud deployment profile for production
cp profiles/cloud-deployment.yaml production.yaml
```

Then customize the copied file for your specific needs.

## Creating Custom Profiles

Profiles are standard YAML configuration files that follow the same structure as environment configurations. They can include:

- Complete configuration sets
- Partial configurations that extend base settings
- Environment variable placeholders
- Comments and documentation

## Profile Validation

All profiles should be validated before use:

```python
from workspace_qdrant_mcp.core.enhanced_config import EnhancedConfig

config = EnhancedConfig(environment="development")
if not config.is_valid:
    print("Configuration errors:", config.validation_errors)
```