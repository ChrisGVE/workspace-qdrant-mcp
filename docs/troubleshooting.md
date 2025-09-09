# Troubleshooting Guide

## Workspace Qdrant MCP Server Troubleshooting

This guide helps diagnose and resolve common issues with the workspace-qdrant-mcp server.

## Quick Diagnostics

### Health Check

First, verify the server is running and responsive:

```bash
# Check if server is responding
curl http://localhost:8000/health

# Expected response:
# {"status": "healthy", "version": "0.2.1", "uptime": "00:05:23"}
```

### Service Status

Check if the service is running:

```bash
# For systemd (Linux)
sudo systemctl status workspace-qdrant-mcp

# For launchd (macOS)  
launchctl list | grep workspace-qdrant-mcp

# For Docker
docker ps | grep workspace-qdrant-mcp
```

### Log Inspection

Check recent logs for errors:

```bash
# System logs
journalctl -u workspace-qdrant-mcp -n 50

# Application logs
tail -f ~/.workspace-qdrant-mcp/logs/workspace-qdrant-mcp.log

# Docker logs
docker logs workspace-qdrant-mcp
```

## Common Issues and Solutions

### 1. Server Won't Start

#### Symptom
```
Error: Failed to start workspace-qdrant-mcp server
```

#### Possible Causes & Solutions

**Port already in use:**
```bash
# Check what's using port 8000
lsof -i :8000
netstat -tulpn | grep 8000

# Solution: Change port in configuration
# config/default.yaml
port: 8001
```

**Missing dependencies:**
```bash
# Reinstall with all dependencies
pip install --force-reinstall workspace-qdrant-mcp

# Or with uv
uv tool install --force workspace-qdrant-mcp
```

**Permission issues:**
```bash
# Check file permissions
ls -la ~/.workspace-qdrant-mcp/

# Fix permissions
chmod -R 755 ~/.workspace-qdrant-mcp/
chmod 600 ~/.workspace-qdrant-mcp/config/*.yaml
```

### 2. Qdrant Connection Issues

#### Symptom
```
ERROR: Failed to connect to Qdrant at http://localhost:6333
ConnectionError: Connection refused
```

#### Solutions

**Start Qdrant service:**
```bash
# Using Docker
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant

# Using systemd (if installed)
sudo systemctl start qdrant

# Verify Qdrant is running
curl http://localhost:6333/health
```

**Check Qdrant configuration:**
```bash
# Test connection
curl -v http://localhost:6333/collections

# Check Qdrant logs
docker logs qdrant
```

**Network connectivity issues:**
```bash
# Test network connectivity
telnet localhost 6333
ping localhost

# Check firewall
sudo ufw status
sudo iptables -L
```

### 3. Embedding Model Issues

#### Symptom
```
ERROR: Failed to load embedding model 'sentence-transformers/all-MiniLM-L6-v2'
```

#### Solutions

**Download model manually:**
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
```

**Check internet connectivity:**
```bash
curl -I https://huggingface.co
ping huggingface.co
```

**Use offline model:**
```yaml
# config/default.yaml
embedding:
  model: "/path/to/local/model"
  # or use a different model
  model: "sentence-transformers/paraphrase-MiniLM-L3-v2"
```

**Clear model cache:**
```bash
rm -rf ~/.cache/huggingface/transformers/
rm -rf ~/.cache/torch/sentence_transformers/
```

### 4. File Ingestion Problems

#### Symptom
```
ERROR: Failed to ingest file: Permission denied
```

#### Solutions

**Check file permissions:**
```bash
# Verify file is readable
ls -la /path/to/file
cat /path/to/file | head -n 5

# Fix permissions
chmod 644 /path/to/file
```

**Path resolution issues:**
```bash
# Use absolute paths
realpath /path/to/file

# Check current working directory
pwd
```

**File format issues:**
```bash
# Check file type
file /path/to/document.pdf
head /path/to/document.txt

# Verify encoding
chardet /path/to/document.txt
```

### 5. Search Performance Issues

#### Symptom
```
Searches are very slow or timing out
```

#### Solutions

**Check collection size:**
```python
# Get collection info
curl "http://localhost:6333/collections/your-collection"
```

**Optimize search parameters:**
```yaml
# config/default.yaml
performance:
  max_concurrent_requests: 5
  request_timeout_seconds: 60

embedding:
  batch_size: 25  # Reduce if memory issues
```

**Enable result caching:**
```yaml
performance:
  enable_result_cache: true
  cache_size: 1000
  cache_ttl_seconds: 3600
```

### 6. Memory Issues

#### Symptom
```
OutOfMemoryError: Killed
ERROR: Worker process exceeded memory limit
```

#### Solutions

**Monitor memory usage:**
```bash
# System memory
free -h
top -p $(pgrep -f workspace-qdrant-mcp)

# Docker memory
docker stats workspace-qdrant-mcp
```

**Reduce memory usage:**
```yaml
# config/default.yaml
performance:
  max_memory_mb: 1024  # Reduce limit
  cleanup_threshold: 0.7  # Earlier cleanup

embedding:
  batch_size: 10  # Smaller batches
  
auto_ingestion:
  max_files_per_batch: 2
```

**Increase system memory:**
```bash
# For Docker
docker run --memory=4g workspace-qdrant-mcp

# For systemd service
# Edit /etc/systemd/system/workspace-qdrant-mcp.service
[Service]
MemoryLimit=4G
```

### 7. Collection Management Issues

#### Symptom
```
ERROR: Collection 'project-main' not found
ERROR: Failed to create collection
```

#### Solutions

**List existing collections:**
```bash
curl http://localhost:6333/collections
```

**Recreate collections:**
```python
from workspace_qdrant_mcp.core.client import QdrantClientManager
client = QdrantClientManager()
client.create_collection("project-main")
```

**Check collection naming:**
```bash
# Collection names must follow pattern: {project}-{suffix}
# Valid: myproject-main, docs-project
# Invalid: myproject_main, MYPROJECT-main
```

### 8. Auto-ingestion Issues

#### Symptom
```
Files are not being automatically ingested
File watcher not responding to changes
```

#### Solutions

**Check auto-ingestion status:**
```bash
# View configuration
cat ~/.workspace-qdrant-mcp/config/default.yaml | grep -A 20 auto_ingestion
```

**Verify file patterns:**
```yaml
auto_ingestion:
  enabled: true
  include_patterns:
    - "*.md"
    - "*.py"
    - "*.txt"
  exclude_patterns:
    - ".git/*"
    - "__pycache__/*"
```

**Check file watcher limits:**
```bash
# Increase inotify limits (Linux)
echo fs.inotify.max_user_watches=524288 | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

**Test manual ingestion:**
```bash
# Try ingesting a file manually
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{"file_paths": ["README.md"]}'
```

### 9. Authentication/Permission Errors

#### Symptom
```
ERROR: Unauthorized access
ERROR: API key required
```

#### Solutions

**Check API keys:**
```bash
# Verify environment variables
echo $OPENAI_API_KEY
echo $QDRANT_API_KEY

# Check configuration file
grep -A 5 -B 5 api_key ~/.workspace-qdrant-mcp/config/default.yaml
```

**File system permissions:**
```bash
# Check project directory permissions
ls -la /path/to/project/
stat /path/to/project/

# Fix ownership
chown -R $USER:$USER /path/to/project/
```

### 10. Configuration Issues

#### Symptom
```
ERROR: Invalid configuration
YAML parsing error
```

#### Solutions

**Validate YAML syntax:**
```bash
# Test YAML parsing
python -c "import yaml; yaml.safe_load(open('config/default.yaml'))"

# Or use yamllint
yamllint config/default.yaml
```

**Check configuration paths:**
```bash
# Verify config file exists
ls -la ~/.workspace-qdrant-mcp/config/default.yaml

# Check environment variable
echo $WORKSPACE_QDRANT_CONFIG
```

**Reset to defaults:**
```bash
# Backup current config
cp ~/.workspace-qdrant-mcp/config/default.yaml{,.backup}

# Copy default configuration
cp default-config.yaml ~/.workspace-qdrant-mcp/config/default.yaml
```

## Advanced Debugging

### Enable Debug Logging

```yaml
# config/default.yaml
logging:
  level: "DEBUG"
  loggers:
    "workspace_qdrant_mcp": "DEBUG"
    "qdrant_client": "DEBUG"
```

### Performance Profiling

```yaml
development:
  enable_profiling: true
  log_request_details: true
```

### Network Debugging

```bash
# Capture network traffic
tcpdump -i lo -A -s 0 'port 6333 or port 8000'

# Test with verbose curl
curl -v -H "Content-Type: application/json" http://localhost:8000/health
```

### Memory Profiling

```bash
# Install memory profiler
pip install memory-profiler

# Profile memory usage
python -m memory_profiler workspace_qdrant_mcp/server.py
```

## Diagnostic Commands

### System Information

```bash
#!/bin/bash
echo "=== System Information ==="
uname -a
python --version
pip list | grep -E "(workspace-qdrant|qdrant|sentence-transformers)"

echo -e "\n=== Service Status ==="
systemctl is-active workspace-qdrant-mcp
docker ps | grep -E "(qdrant|workspace)"

echo -e "\n=== Network Connectivity ==="
curl -s http://localhost:6333/health | jq .
curl -s http://localhost:8000/health | jq .

echo -e "\n=== Resource Usage ==="
free -h
df -h ~/.workspace-qdrant-mcp/
ps aux | grep -E "(workspace-qdrant|qdrant)"

echo -e "\n=== Recent Errors ==="
journalctl -u workspace-qdrant-mcp --since "1 hour ago" | grep ERROR | tail -10
```

### Configuration Validation

```python
#!/usr/bin/env python3
"""Configuration validation script"""
import yaml
import os
from pathlib import Path

def validate_config():
    config_path = Path.home() / ".workspace-qdrant-mcp" / "config" / "default.yaml"
    
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return False
    
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        print("✅ YAML syntax is valid")
        
        # Check required sections
        required_sections = ['qdrant', 'embedding', 'workspace', 'auto_ingestion']
        for section in required_sections:
            if section in config:
                print(f"✅ {section} section present")
            else:
                print(f"❌ {section} section missing")
        
        # Check Qdrant URL
        if 'url' in config.get('qdrant', {}):
            print(f"✅ Qdrant URL: {config['qdrant']['url']}")
        
        return True
        
    except yaml.YAMLError as e:
        print(f"❌ YAML parsing error: {e}")
        return False
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

if __name__ == "__main__":
    validate_config()
```

## Getting Help

### Log Collection

When seeking help, collect these logs:

```bash
#!/bin/bash
# Log collection script
mkdir -p /tmp/workspace-qdrant-debug

# System information
uname -a > /tmp/workspace-qdrant-debug/system-info.txt
python --version >> /tmp/workspace-qdrant-debug/system-info.txt

# Service logs
journalctl -u workspace-qdrant-mcp --since "1 hour ago" > /tmp/workspace-qdrant-debug/service.log

# Application logs
cp ~/.workspace-qdrant-mcp/logs/workspace-qdrant-mcp.log /tmp/workspace-qdrant-debug/

# Configuration (sanitized)
cp ~/.workspace-qdrant-mcp/config/default.yaml /tmp/workspace-qdrant-debug/config.yaml
sed -i 's/api_key:.*/api_key: [REDACTED]/g' /tmp/workspace-qdrant-debug/config.yaml

# Create archive
tar -czf workspace-qdrant-debug-$(date +%Y%m%d_%H%M%S).tar.gz -C /tmp workspace-qdrant-debug/
echo "Debug info collected in: workspace-qdrant-debug-$(date +%Y%m%d_%H%M%S).tar.gz"
```

### Support Channels

1. **GitHub Issues**: [Create an issue](https://github.com/ChrisGVE/workspace-qdrant-mcp/issues/new) with:
   - Error messages and logs
   - System information
   - Steps to reproduce
   - Configuration details (sanitized)

2. **GitHub Discussions**: [Join discussions](https://github.com/ChrisGVE/workspace-qdrant-mcp/discussions) for:
   - General questions
   - Feature requests
   - Community support

3. **Documentation**: Check existing documentation:
   - [Installation Guide](../INSTALLATION.md)
   - [Configuration Guide](configuration.md)
   - [Deployment Guide](deployment.md)
   - [API Documentation](API.md)

### Issue Template

When reporting issues, include:

```
## Environment
- OS: [e.g., Ubuntu 22.04, macOS 14.0]
- Python version: [e.g., 3.11.5]
- Package version: [e.g., 0.2.1]
- Installation method: [pip, uv, Docker, etc.]

## Problem Description
[Clear description of the issue]

## Steps to Reproduce
1. [First step]
2. [Second step]
3. [And so on...]

## Expected Behavior
[What you expected to happen]

## Actual Behavior
[What actually happened]

## Error Messages
```
[Paste error messages and relevant logs here]
```

## Configuration
```yaml
[Paste relevant configuration sections, remove sensitive data]
```

## Additional Context
[Any other relevant information]
```

## Prevention

### Regular Maintenance

```bash
# Update to latest version
uv tool upgrade workspace-qdrant-mcp

# Clean up old logs
find ~/.workspace-qdrant-mcp/logs -name "*.log.*" -mtime +30 -delete

# Backup configuration
cp ~/.workspace-qdrant-mcp/config/default.yaml ~/.workspace-qdrant-mcp/config/default.yaml.$(date +%Y%m%d)

# Check disk space
df -h ~/.workspace-qdrant-mcp/
```

### Monitoring Setup

Consider setting up monitoring for:
- Service health checks
- Resource usage (CPU, memory, disk)
- Error rates in logs
- Qdrant connectivity
- Search performance metrics

This helps detect issues before they become critical problems.