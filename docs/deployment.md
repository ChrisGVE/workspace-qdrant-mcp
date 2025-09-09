# Deployment Guide

## Workspace Qdrant MCP Server Deployment

This guide covers various deployment scenarios for the workspace-qdrant-mcp server.

## Prerequisites

### System Requirements

- Python 3.10 or higher
- 4GB+ RAM recommended
- 10GB+ free disk space
- Network access to Qdrant instance (local or cloud)

### Dependencies

- Qdrant vector database (local or cloud instance)
- FastEmbed for embeddings
- Required Python packages (installed automatically)

## Deployment Options

### 1. Local Development Setup

For development and testing purposes:

```bash
# Clone the repository
git clone https://github.com/ChrisGVE/workspace-qdrant-mcp.git
cd workspace-qdrant-mcp

# Install with uv (recommended)
uv pip install -e .

# Or with pip
pip install -e .

# Start local Qdrant instance (Docker)
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

# Configure environment
export QDRANT_URL=http://localhost:6333
export OPENAI_API_KEY=your_api_key_here

# Run the server
python -m workspace_qdrant_mcp.server
```

### 2. Production Installation

For production use with global installation:

```bash
# Use the installation script
curl -sSL https://raw.githubusercontent.com/ChrisGVE/workspace-qdrant-mcp/main/install.sh | bash

# Or manual installation with uv
uv tool install workspace-qdrant-mcp

# Create configuration directory
mkdir -p ~/.workspace-qdrant-mcp/config

# Copy default configuration
cp default-config.yaml ~/.workspace-qdrant-mcp/config/default.yaml

# Edit configuration as needed
$EDITOR ~/.workspace-qdrant-mcp/config/default.yaml

# Run the server
workspace-qdrant-mcp --config-file ~/.workspace-qdrant-mcp/config/default.yaml
```

### 3. Docker Deployment

#### Using Docker Compose (Recommended)

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_storage:/qdrant/storage
    environment:
      - QDRANT__SERVICE__HTTP_PORT=6333
      - QDRANT__SERVICE__GRPC_PORT=6334
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  workspace-qdrant-mcp:
    image: workspace-qdrant-mcp:latest
    ports:
      - "8000:8000"
    depends_on:
      qdrant:
        condition: service_healthy
    environment:
      - QDRANT_URL=http://qdrant:6333
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - PYTHONPATH=/app/src
    volumes:
      - ./config:/app/config
      - ./data:/app/data
    command: ["python", "-m", "workspace_qdrant_mcp.server", "--config-file", "/app/config/default.yaml"]

volumes:
  qdrant_storage:
```

Run with:
```bash
# Set environment variables
echo "OPENAI_API_KEY=your_api_key" > .env

# Start services
docker-compose up -d

# Check status
docker-compose ps
docker-compose logs workspace-qdrant-mcp
```

#### Building Docker Image

```bash
# Build the image
docker build -t workspace-qdrant-mcp:latest .

# Run standalone container
docker run -d \
  --name workspace-qdrant-mcp \
  -p 8000:8000 \
  -e QDRANT_URL=http://your-qdrant-instance:6333 \
  -e OPENAI_API_KEY=your_api_key \
  -v $(pwd)/config:/app/config \
  workspace-qdrant-mcp:latest
```

### 4. Kubernetes Deployment

See [KUBERNETES_DEPLOYMENT.md](KUBERNETES_DEPLOYMENT.md) for detailed Kubernetes deployment instructions.

### 5. Cloud Deployment

#### Qdrant Cloud Integration

```yaml
# config/production.yaml
qdrant:
  url: "https://your-cluster.qdrant.cloud"
  api_key: "your-qdrant-cloud-api-key"
  timeout: 60
  prefer_grpc: true
  
# Environment variables
export QDRANT_URL="https://your-cluster.qdrant.cloud"
export QDRANT_API_KEY="your-qdrant-cloud-api-key"
```

#### AWS Deployment

Using AWS ECS or EC2:

```bash
# Install on EC2 instance
sudo yum update -y
sudo yum install -y python3 python3-pip docker

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install workspace-qdrant-mcp
uv tool install workspace-qdrant-mcp

# Configure systemd service (see systemd section below)
```

### 6. System Service Setup

#### systemd (Linux)

Create `/etc/systemd/system/workspace-qdrant-mcp.service`:

```ini
[Unit]
Description=Workspace Qdrant MCP Server
After=network.target qdrant.service
Wants=qdrant.service

[Service]
Type=simple
User=qdrant-mcp
Group=qdrant-mcp
WorkingDirectory=/opt/workspace-qdrant-mcp
Environment=PYTHONPATH=/opt/workspace-qdrant-mcp/src
Environment=QDRANT_URL=http://localhost:6333
EnvironmentFile=/opt/workspace-qdrant-mcp/.env
ExecStart=/usr/local/bin/workspace-qdrant-mcp --config-file /opt/workspace-qdrant-mcp/config/production.yaml
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=workspace-qdrant-mcp

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable workspace-qdrant-mcp
sudo systemctl start workspace-qdrant-mcp
sudo systemctl status workspace-qdrant-mcp
```

#### launchd (macOS)

Create `~/Library/LaunchAgents/com.workspace-qdrant-mcp.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.workspace-qdrant-mcp</string>
    <key>ProgramArguments</key>
    <array>
        <string>/opt/homebrew/bin/workspace-qdrant-mcp</string>
        <string>--config-file</string>
        <string>/Users/yourusername/.workspace-qdrant-mcp/config/default.yaml</string>
    </array>
    <key>EnvironmentVariables</key>
    <dict>
        <key>QDRANT_URL</key>
        <string>http://localhost:6333</string>
        <key>OPENAI_API_KEY</key>
        <string>your_api_key_here</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/Users/yourusername/.workspace-qdrant-mcp/logs/stdout.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/yourusername/.workspace-qdrant-mcp/logs/stderr.log</string>
</dict>
</plist>
```

Load the service:
```bash
launchctl load ~/Library/LaunchAgents/com.workspace-qdrant-mcp.plist
launchctl start com.workspace-qdrant-mcp
```

## Configuration Management

### Environment-Specific Configurations

Create different configuration files for each environment:

```
config/
├── development.yaml
├── staging.yaml
├── production.yaml
└── local.yaml
```

Example production configuration:
```yaml
# config/production.yaml
host: "0.0.0.0"
port: 8000
debug: false

qdrant:
  url: "https://your-qdrant-cloud.qdrant.cloud"
  api_key: "${QDRANT_API_KEY}"
  timeout: 60
  prefer_grpc: true

logging:
  level: "INFO"
  file: "/var/log/workspace-qdrant-mcp/server.log"
  
security:
  rate_limit_requests_per_minute: 100
  max_request_size_mb: 50
  
performance:
  max_memory_mb: 4096
  max_concurrent_requests: 20
```

### Secrets Management

Use environment variables for sensitive data:

```bash
# .env file (never commit to git)
QDRANT_API_KEY=your_qdrant_cloud_api_key
OPENAI_API_KEY=your_openai_api_key
GITHUB_API_KEY=your_github_token
```

Or use a secrets management service:
- AWS Secrets Manager
- Azure Key Vault  
- HashiCorp Vault
- Kubernetes Secrets

## Monitoring and Logging

### Health Checks

The server provides health check endpoints:

```bash
# Basic health check
curl http://localhost:8000/health

# Detailed health info
curl http://localhost:8000/health/detailed
```

### Logging Configuration

Configure structured logging:

```yaml
logging:
  level: "INFO"
  format: "json"
  handlers:
    - type: "file"
      filename: "/var/log/workspace-qdrant-mcp/server.log"
      max_size: "100MB"
      backup_count: 5
    - type: "syslog"
      facility: "daemon"
    - type: "stdout"
      level: "WARNING"
```

### Metrics and Monitoring

Integration with monitoring systems:

```python
# Enable Prometheus metrics
from prometheus_client import start_http_server, Counter, Histogram

# Custom metrics
search_requests = Counter('search_requests_total', 'Total search requests')
search_duration = Histogram('search_duration_seconds', 'Search request duration')
```

## Security Considerations

### Network Security

- Use HTTPS in production
- Configure firewall rules
- Implement rate limiting
- Use VPC/private networks where possible

### Authentication

Configure API key authentication:

```yaml
security:
  require_api_key: true
  api_keys:
    - name: "claude_client"
      key: "${CLAUDE_API_KEY}"
      permissions: ["search", "ingest"]
    - name: "admin_client"  
      key: "${ADMIN_API_KEY}"
      permissions: ["*"]
```

### File Access Control

Restrict file system access:

```yaml
workspace:
  allowed_paths:
    - "/home/users/*/projects"
    - "/opt/shared/docs"
  denied_patterns:
    - "*.key"
    - "*.pem" 
    - "/etc/*"
    - "/var/log/*"
```

## Backup and Recovery

### Data Backup

Backup Qdrant data regularly:

```bash
# Create backup
qdrant-backup create --collection-name project-main --output backup-$(date +%Y%m%d).tar.gz

# Restore from backup
qdrant-backup restore --input backup-20241201.tar.gz --collection-name project-main
```

### Configuration Backup

Version control your configurations:

```bash
git init config-repo
git add config/
git commit -m "Initial configuration"
git tag v1.0.0
```

## Scaling Considerations

### Horizontal Scaling

Deploy multiple instances behind a load balancer:

```yaml
# docker-compose.yml with scaling
version: '3.8'
services:
  workspace-qdrant-mcp:
    image: workspace-qdrant-mcp:latest
    deploy:
      replicas: 3
  
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
```

### Performance Tuning

Optimize for your workload:

```yaml
performance:
  # Adjust based on available resources
  max_memory_mb: 8192
  max_concurrent_requests: 50
  
  # Enable caching
  enable_result_cache: true
  cache_size: 10000
  cache_ttl_seconds: 1800
  
embedding:
  # Use larger batch sizes for better throughput
  batch_size: 100
  
  # Consider GPU acceleration for large deployments
  device: "cuda"  # if available
```

## Troubleshooting

### Common Issues

1. **Connection refused to Qdrant**
   - Check Qdrant service is running
   - Verify network connectivity
   - Check firewall settings

2. **Out of memory errors**
   - Reduce batch sizes
   - Increase system memory
   - Enable memory management features

3. **Slow search performance**
   - Check Qdrant index status
   - Optimize collection configuration
   - Enable result caching

### Diagnostic Commands

```bash
# Check service status
systemctl status workspace-qdrant-mcp

# View recent logs
journalctl -u workspace-qdrant-mcp -n 100

# Test API connectivity
curl -v http://localhost:8000/health

# Check Qdrant status
curl http://localhost:6333/collections
```

### Log Analysis

Monitor key log patterns:

```bash
# Search for errors
grep "ERROR" /var/log/workspace-qdrant-mcp/server.log

# Monitor performance
grep "search_duration" /var/log/workspace-qdrant-mcp/server.log | tail -100

# Check connection issues
grep "ConnectionError" /var/log/workspace-qdrant-mcp/server.log
```

For additional troubleshooting information, see [CONTAINER_TROUBLESHOOTING.md](CONTAINER_TROUBLESHOOTING.md).

## Support

For deployment support:
- Check existing [GitHub Issues](https://github.com/ChrisGVE/workspace-qdrant-mcp/issues)
- Create a new issue with deployment details
- Join [GitHub Discussions](https://github.com/ChrisGVE/workspace-qdrant-mcp/discussions)
- Review [troubleshooting documentation](CONTAINER_TROUBLESHOOTING.md)