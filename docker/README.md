# Docker and Kubernetes Deployment

This directory contains complete Docker and Kubernetes deployment infrastructure for Workspace Qdrant MCP with enterprise-grade security, monitoring, and automation.

## 📁 Directory Structure

```
docker/
├── Dockerfile                    # Multi-stage Docker image with security hardening
├── entrypoint.sh                 # Container entrypoint with health checks
├── docker-compose.yml            # Production Docker Compose configuration
├── docker-compose.dev.yml        # Development Docker Compose with hot reload
├── docker-compose.prod.yml       # Production overrides and optimization
├── build-and-push.sh            # Multi-architecture build and registry publishing
├── deploy.sh                    # Automated deployment script
├── .env.example                 # Comprehensive environment configuration
├── nginx/                       # Nginx reverse proxy configuration
│   ├── nginx.conf              # Main Nginx configuration
│   └── conf.d/                 # Additional server configurations
├── k8s/                         # Kubernetes manifests
│   ├── namespace.yaml          # Namespace with RBAC and network policies
│   ├── configmap.yaml          # Application configuration
│   ├── secrets.yaml            # Secrets management (base64 encoded)
│   ├── persistentvolume.yaml   # Storage configuration for all components
│   ├── rbac.yaml              # Complete RBAC with service accounts
│   ├── deployment.yaml         # Application and database deployments
│   ├── service.yaml            # Services for internal communication
│   ├── ingress.yaml            # Ingress with TLS termination
│   └── hpa.yaml               # Horizontal Pod Autoscaler
└── README.md                   # This file
```

## 🚀 Quick Start

### Development Environment

```bash
# 1. Set up environment
cp docker/.env.example docker/.env
# Edit .env with your configuration

# 2. Deploy with automation script
./docker/deploy.sh --type docker-compose --env development

# 3. Access application
curl http://localhost:8000/health
```

### Production Environment

```bash
# 1. Configure production environment
cp docker/.env.example docker/.env
# Set production values in .env

# 2. Deploy to Kubernetes
./docker/deploy.sh --type kubernetes --env production

# 3. Verify deployment
kubectl get pods -n workspace-qdrant-mcp
```

## 🐳 Docker Features

### Multi-Stage Dockerfile
- **Development Stage**: Hot reload, debugging tools, dev dependencies
- **Testing Stage**: Test runners, coverage tools, security scanning
- **Production Stage**: Minimal attack surface, non-root user, read-only filesystem
- **Security Scan Stage**: Vulnerability scanning with Grype/Trivy

### Security Hardening
- Non-root user (UID 65534)
- Read-only root filesystem where possible
- Minimal base image with only required dependencies
- Dropped capabilities and security contexts
- Multi-architecture builds (amd64, arm64)

### Build and Registry Features
- Automated semantic versioning
- Multi-architecture builds
- Container vulnerability scanning
- SBOM (Software Bill of Materials) generation
- Provenance attestations
- Layer caching optimization

## ☸️ Kubernetes Features

### Production-Ready Manifests
- Complete RBAC with principle of least privilege
- Network policies for microsegmentation
- Pod security contexts and security standards
- Resource quotas and limits
- Storage classes for different performance tiers

### High Availability
- Multi-replica deployments with pod anti-affinity
- Horizontal Pod Autoscaler (HPA) with custom metrics
- Pod Disruption Budgets (PDB)
- Rolling updates with zero downtime
- Health checks and readiness probes

### Storage Management
- Persistent Volume Claims for data persistence
- Storage classes for different performance needs
- Backup and recovery procedures
- Volume expansion capabilities

### Observability
- Prometheus metrics scraping
- Grafana dashboard provisioning
- Distributed tracing with Jaeger
- Centralized logging with Loki
- Service mesh ready (Istio compatible)

## 📊 Monitoring Stack

### Included Components
- **Prometheus**: Metrics collection and alerting
- **Grafana**: Visualization and dashboards
- **Jaeger**: Distributed tracing
- **Loki**: Log aggregation
- **AlertManager**: Alert routing and management
- **Node Exporter**: System metrics
- **cAdvisor**: Container metrics

### Custom Metrics
- Application-specific business metrics
- Request rate and latency percentiles
- Database connection pool status
- Vector search performance metrics
- Custom alerting rules

## 🔐 Security Features

### Authentication and Authorization
- API key management for Qdrant
- Redis password authentication
- TLS termination at ingress
- Service-to-service authentication

### Network Security
- Network policies for traffic segmentation
- Ingress with TLS termination
- Internal service discovery
- Rate limiting and DDoS protection

### Secrets Management
- Kubernetes secrets for sensitive data
- External secrets operator support
- Vault integration ready
- Secret rotation procedures

### Container Security
- Non-root containers
- Read-only filesystems
- Capability dropping
- Security context constraints
- Regular vulnerability scanning

## 🔧 Configuration Management

### Environment Variables
The `.env.example` file contains comprehensive configuration options:

- **Application Settings**: Host, port, logging, debug mode
- **Database Configuration**: Qdrant and Redis connection settings
- **Security Settings**: API keys, passwords, TLS configuration
- **Monitoring Settings**: Prometheus, Grafana, Jaeger configuration
- **Storage Settings**: Volume paths and retention policies
- **Performance Tuning**: Resource limits, connection pooling
- **Feature Flags**: Enable/disable specific features

### Multi-Environment Support
- **Development**: Debug tools, hot reload, relaxed security
- **Staging**: Production-like with testing tools
- **Production**: Hardened security, performance optimization

## 🛠️ Deployment Automation

### Deployment Script Features
- **Multi-Platform**: Docker Compose and Kubernetes support
- **Environment Detection**: Automatic environment-specific configuration
- **Pre-deployment Backup**: Automatic backup before updates
- **Health Monitoring**: Post-deployment health verification
- **Rollback Capability**: Automatic rollback on failure
- **Dry Run Mode**: Preview changes before deployment

### CI/CD Integration
- **GitHub Actions**: Automated testing and deployment
- **GitLab CI**: Pipeline integration
- **Jenkins**: Pipeline support
- **ArgoCD**: GitOps deployment
- **Tekton**: Cloud-native CI/CD

## 🔄 Operations

### Scaling Operations
```bash
# Docker Compose scaling
docker-compose up -d --scale workspace-qdrant-mcp=3

# Kubernetes scaling
kubectl scale deployment workspace-qdrant-mcp --replicas=5 -n workspace-qdrant-mcp

# Auto-scaling with HPA
kubectl get hpa -n workspace-qdrant-mcp
```

### Backup and Recovery
```bash
# Automated backup
./docker/backup.sh --type full --retention 7d

# Point-in-time recovery
./docker/restore.sh --backup-id 20231201-120000 --point-in-time

# Disaster recovery
kubectl apply -f docker/k8s/disaster-recovery.yaml
```

### Maintenance Operations
```bash
# Rolling updates
kubectl set image deployment/workspace-qdrant-mcp workspace-qdrant-mcp=workspace-qdrant-mcp:v0.3.0 -n workspace-qdrant-mcp

# Certificate renewal
./docker/scripts/renew-certificates.sh

# Database maintenance
kubectl exec -it deployment/qdrant -n workspace-qdrant-mcp -- qdrant-cli optimize
```

## 📈 Performance Tuning

### Resource Optimization
- **CPU**: Optimized for vector operations
- **Memory**: Tuned for large vector datasets  
- **Storage**: SSD storage classes for performance
- **Network**: Optimized service mesh configuration

### Database Tuning
- **Qdrant**: Optimized for vector search performance
- **Redis**: Configured for caching efficiency
- **Connection Pooling**: Optimized pool sizes
- **Query Optimization**: Indexed search patterns

## 🐛 Troubleshooting

### Common Issues

1. **Pod Startup Issues**
   ```bash
   kubectl describe pod <pod-name> -n workspace-qdrant-mcp
   kubectl logs <pod-name> -n workspace-qdrant-mcp --previous
   ```

2. **Service Discovery Problems**
   ```bash
   kubectl get endpoints -n workspace-qdrant-mcp
   kubectl exec -it deployment/workspace-qdrant-mcp -n workspace-qdrant-mcp -- nslookup qdrant-service
   ```

3. **Storage Issues**
   ```bash
   kubectl get pvc -n workspace-qdrant-mcp
   kubectl describe storageclass
   ```

4. **Network Connectivity**
   ```bash
   kubectl get networkpolicy -n workspace-qdrant-mcp
   kubectl port-forward service/workspace-qdrant-mcp-service 8000:8000 -n workspace-qdrant-mcp
   ```

### Debug Tools
- **Debug Pod**: Run debug container in cluster
- **Port Forwarding**: Access services locally
- **Log Aggregation**: Centralized logging with Loki
- **Metrics Dashboard**: Real-time monitoring with Grafana

## 📚 Documentation

### Comprehensive Guides
- [Docker Deployment Guide](../docs/DOCKER_DEPLOYMENT.md)
- [Kubernetes Deployment Guide](../docs/KUBERNETES_DEPLOYMENT.md)  
- [Container Troubleshooting Guide](../docs/CONTAINER_TROUBLESHOOTING.md)

### API Documentation
- Application API: `http://localhost:8000/docs`
- Prometheus Metrics: `http://localhost:9090`
- Grafana Dashboards: `http://localhost:3000`

## 🤝 Contributing

### Development Workflow
1. Copy and modify `.env.example`
2. Use development Docker Compose for local testing
3. Test Kubernetes manifests in development cluster
4. Submit PR with deployment changes

### Security Requirements
- All secrets must be properly configured
- Container images must pass security scanning
- Network policies must be tested
- RBAC permissions must follow least privilege

## 📄 License

This deployment infrastructure is part of the Workspace Qdrant MCP project and is licensed under the MIT License.

---

For detailed deployment instructions and troubleshooting, see the complete documentation in the `docs/` directory.