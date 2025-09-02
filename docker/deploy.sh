#!/bin/bash
# Deployment Automation Script for Workspace Qdrant MCP
# Supports Docker Compose and Kubernetes deployments with full automation

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Default configuration
DEPLOYMENT_TYPE="docker-compose"
ENVIRONMENT="development"
NAMESPACE="workspace-qdrant-mcp"
CONFIG_FILE=""
DRY_RUN="false"
SKIP_BUILD="false"
SKIP_HEALTH_CHECK="false"
FORCE_RECREATE="false"
BACKUP_BEFORE_DEPLOY="true"
ROLLBACK_ON_FAILURE="true"
TIMEOUT="600"

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" >&2
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" >&2
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" >&2
}

# Error handling
error_exit() {
    log_error "$1"
    exit 1
}

# Help function
show_help() {
    cat << EOF
Workspace Qdrant MCP Deployment Automation

USAGE:
    $(basename "$0") [OPTIONS]

OPTIONS:
    -h, --help                  Show this help message
    -t, --type TYPE             Deployment type: docker-compose, kubernetes (default: docker-compose)
    -e, --env ENVIRONMENT       Environment: development, staging, production (default: development)
    -n, --namespace NAMESPACE   Kubernetes namespace (default: workspace-qdrant-mcp)
    -c, --config CONFIG_FILE    Path to additional configuration file
    -d, --dry-run              Show what would be deployed without executing
    --skip-build               Skip building images
    --skip-health-check        Skip health checks after deployment
    --force-recreate           Force recreation of all resources
    --no-backup                Skip backup before deployment
    --no-rollback             Don't rollback on failure
    --timeout SECONDS          Deployment timeout in seconds (default: 600)

DEPLOYMENT TYPES:
    docker-compose             Deploy using Docker Compose (local/single-host)
    kubernetes                 Deploy to Kubernetes cluster

ENVIRONMENTS:
    development                Development environment with debug tools
    staging                    Staging environment for testing
    production                 Production environment with security hardening

EXAMPLES:
    # Development deployment with Docker Compose
    $(basename "$0") --type docker-compose --env development

    # Production deployment to Kubernetes
    $(basename "$0") --type kubernetes --env production

    # Dry run to see what would be deployed
    $(basename "$0") --type kubernetes --env staging --dry-run

    # Force recreation of all resources
    $(basename "$0") --type docker-compose --env development --force-recreate

    # Skip backup and health checks for fast deployment
    $(basename "$0") --type docker-compose --env development --no-backup --skip-health-check

PREREQUISITES:
    Docker Compose:
    - Docker Engine 20.10+
    - Docker Compose 2.0+
    - Valid .env file

    Kubernetes:
    - kubectl configured for target cluster
    - Sufficient cluster resources
    - Storage classes configured
    - Secrets configured (if not using default)

CONFIGURATION:
    Create .env file from .env.example and customize for your environment.
    Ensure all secrets are properly configured before deployment.
EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -t|--type)
                DEPLOYMENT_TYPE="$2"
                shift 2
                ;;
            -e|--env)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -n|--namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            -c|--config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            -d|--dry-run)
                DRY_RUN="true"
                shift
                ;;
            --skip-build)
                SKIP_BUILD="true"
                shift
                ;;
            --skip-health-check)
                SKIP_HEALTH_CHECK="true"
                shift
                ;;
            --force-recreate)
                FORCE_RECREATE="true"
                shift
                ;;
            --no-backup)
                BACKUP_BEFORE_DEPLOY="false"
                shift
                ;;
            --no-rollback)
                ROLLBACK_ON_FAILURE="false"
                shift
                ;;
            --timeout)
                TIMEOUT="$2"
                shift 2
                ;;
            *)
                error_exit "Unknown option: $1"
                ;;
        esac
    done
}

# Validate environment and dependencies
validate_environment() {
    log_info "Validating deployment environment..."
    
    # Validate deployment type
    if [[ ! "$DEPLOYMENT_TYPE" =~ ^(docker-compose|kubernetes)$ ]]; then
        error_exit "Invalid deployment type: $DEPLOYMENT_TYPE"
    fi
    
    # Validate environment
    if [[ ! "$ENVIRONMENT" =~ ^(development|staging|production)$ ]]; then
        error_exit "Invalid environment: $ENVIRONMENT"
    fi
    
    # Check dependencies based on deployment type
    case "$DEPLOYMENT_TYPE" in
        docker-compose)
            validate_docker_compose
            ;;
        kubernetes)
            validate_kubernetes
            ;;
    esac
    
    log_success "Environment validation completed"
}

# Validate Docker Compose environment
validate_docker_compose() {
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error_exit "Docker is not installed or not in PATH"
    fi
    
    # Check Docker Compose
    if ! docker compose version &> /dev/null && ! docker-compose --version &> /dev/null; then
        error_exit "Docker Compose is not installed or not in PATH"
    fi
    
    # Check if using legacy docker-compose
    if command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_warn "Using legacy docker-compose. Consider upgrading to Docker Compose V2"
        COMPOSE_CMD="docker-compose"
    else
        COMPOSE_CMD="docker compose"
    fi
    
    # Check .env file
    if [[ ! -f "${SCRIPT_DIR}/.env" ]]; then
        log_warn ".env file not found. Using default configuration."
        if [[ -f "${SCRIPT_DIR}/.env.example" ]]; then
            log_info "Example .env file available at ${SCRIPT_DIR}/.env.example"
        fi
    fi
    
    # Check compose files exist
    local compose_files=("docker-compose.yml")
    case "$ENVIRONMENT" in
        development)
            compose_files+=("docker-compose.dev.yml")
            ;;
        production)
            compose_files+=("docker-compose.prod.yml")
            ;;
    esac
    
    for file in "${compose_files[@]}"; do
        if [[ ! -f "${SCRIPT_DIR}/${file}" ]]; then
            error_exit "Compose file not found: ${SCRIPT_DIR}/${file}"
        fi
    done
}

# Validate Kubernetes environment
validate_kubernetes() {
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        error_exit "kubectl is not installed or not in PATH"
    fi
    
    # Check cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        error_exit "Cannot connect to Kubernetes cluster"
    fi
    
    # Check namespace (create if doesn't exist)
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_warn "Namespace $NAMESPACE does not exist"
        if [[ "$DRY_RUN" != "true" ]]; then
            log_info "Creating namespace $NAMESPACE"
            kubectl create namespace "$NAMESPACE"
        fi
    fi
    
    # Check required manifests
    local manifest_files=(
        "k8s/namespace.yaml"
        "k8s/configmap.yaml"
        "k8s/secrets.yaml"
        "k8s/deployment.yaml"
        "k8s/service.yaml"
    )
    
    for file in "${manifest_files[@]}"; do
        if [[ ! -f "${SCRIPT_DIR}/${file}" ]]; then
            error_exit "Kubernetes manifest not found: ${SCRIPT_DIR}/${file}"
        fi
    done
    
    # Check storage classes
    if ! kubectl get storageclass &> /dev/null; then
        log_warn "No storage classes found. PVC creation may fail."
    fi
}

# Load configuration
load_configuration() {
    log_info "Loading configuration..."
    
    # Load .env file if it exists
    if [[ -f "${SCRIPT_DIR}/.env" ]]; then
        log_info "Loading environment variables from .env"
        set -a
        source "${SCRIPT_DIR}/.env"
        set +a
    fi
    
    # Load additional config file if specified
    if [[ -n "$CONFIG_FILE" ]]; then
        if [[ -f "$CONFIG_FILE" ]]; then
            log_info "Loading additional configuration from $CONFIG_FILE"
            set -a
            source "$CONFIG_FILE"
            set +a
        else
            error_exit "Configuration file not found: $CONFIG_FILE"
        fi
    fi
    
    # Set environment-specific defaults
    case "$ENVIRONMENT" in
        development)
            export LOG_LEVEL="${LOG_LEVEL:-DEBUG}"
            export REPLICAS="${REPLICAS:-1}"
            ;;
        staging)
            export LOG_LEVEL="${LOG_LEVEL:-INFO}"
            export REPLICAS="${REPLICAS:-2}"
            ;;
        production)
            export LOG_LEVEL="${LOG_LEVEL:-WARN}"
            export REPLICAS="${REPLICAS:-3}"
            ;;
    esac
    
    log_success "Configuration loaded successfully"
}

# Create backup before deployment
create_backup() {
    if [[ "$BACKUP_BEFORE_DEPLOY" != "true" ]]; then
        log_info "Backup skipped as requested"
        return 0
    fi
    
    log_info "Creating backup before deployment..."
    
    local backup_dir="backup/$(date +%Y%m%d-%H%M%S)-pre-deploy"
    mkdir -p "$backup_dir"
    
    case "$DEPLOYMENT_TYPE" in
        docker-compose)
            create_docker_backup "$backup_dir"
            ;;
        kubernetes)
            create_kubernetes_backup "$backup_dir"
            ;;
    esac
    
    log_success "Backup created at $backup_dir"
}

# Create Docker Compose backup
create_docker_backup() {
    local backup_dir="$1"
    
    # Backup volumes
    if docker compose -f "${SCRIPT_DIR}/docker-compose.yml" ps -q | grep -q .; then
        log_info "Backing up Docker volumes..."
        
        # Backup Qdrant data
        docker compose -f "${SCRIPT_DIR}/docker-compose.yml" exec -T qdrant tar czf - /qdrant/storage > "$backup_dir/qdrant-storage.tar.gz" 2>/dev/null || true
        
        # Backup Redis data
        docker compose -f "${SCRIPT_DIR}/docker-compose.yml" exec -T redis redis-cli --rdb - > "$backup_dir/redis.rdb" 2>/dev/null || true
        
        # Backup application data
        docker compose -f "${SCRIPT_DIR}/docker-compose.yml" exec -T workspace-qdrant-mcp tar czf - /app/data > "$backup_dir/app-data.tar.gz" 2>/dev/null || true
    fi
    
    # Backup configuration
    cp -r "${SCRIPT_DIR}"/*.yml "$backup_dir/" 2>/dev/null || true
    cp "${SCRIPT_DIR}/.env" "$backup_dir/" 2>/dev/null || true
}

# Create Kubernetes backup
create_kubernetes_backup() {
    local backup_dir="$1"
    
    log_info "Backing up Kubernetes resources..."
    
    # Backup all resources in namespace
    kubectl get all,configmap,secret,pvc,ingress -n "$NAMESPACE" -o yaml > "$backup_dir/resources.yaml" 2>/dev/null || true
    
    # Backup PVC data (if possible)
    local pvcs=$(kubectl get pvc -n "$NAMESPACE" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || echo "")
    for pvc in $pvcs; do
        log_info "Attempting to backup PVC: $pvc"
        kubectl run backup-pod-$pvc --image=busybox --rm --restart=Never \
            --overrides="$(cat <<EOF
{
  "spec": {
    "containers": [
      {
        "name": "backup",
        "image": "busybox",
        "command": ["tar", "czf", "-", "/data"],
        "volumeMounts": [
          {
            "name": "data",
            "mountPath": "/data"
          }
        ]
      }
    ],
    "volumes": [
      {
        "name": "data",
        "persistentVolumeClaim": {
          "claimName": "$pvc"
        }
      }
    ]
  }
}
EOF
)" -n "$NAMESPACE" > "$backup_dir/$pvc.tar.gz" 2>/dev/null || true
    done
}

# Deploy with Docker Compose
deploy_docker_compose() {
    log_info "Deploying with Docker Compose (Environment: $ENVIRONMENT)..."
    
    # Determine compose files
    local compose_files=("-f" "${SCRIPT_DIR}/docker-compose.yml")
    
    case "$ENVIRONMENT" in
        development)
            compose_files+=("-f" "${SCRIPT_DIR}/docker-compose.dev.yml")
            ;;
        production)
            compose_files+=("-f" "${SCRIPT_DIR}/docker-compose.prod.yml")
            ;;
    esac
    
    # Build images if not skipping
    if [[ "$SKIP_BUILD" != "true" ]]; then
        log_info "Building images..."
        if [[ "$DRY_RUN" == "true" ]]; then
            log_info "DRY RUN: Would execute: $COMPOSE_CMD ${compose_files[*]} build"
        else
            $COMPOSE_CMD "${compose_files[@]}" build
        fi
    fi
    
    # Deploy services
    local deploy_args=("${compose_files[@]}" "up" "-d")
    
    if [[ "$FORCE_RECREATE" == "true" ]]; then
        deploy_args+=("--force-recreate")
    fi
    
    log_info "Starting services..."
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would execute: $COMPOSE_CMD ${deploy_args[*]}"
    else
        $COMPOSE_CMD "${deploy_args[@]}"
    fi
    
    # Wait for services to be ready
    if [[ "$DRY_RUN" != "true" && "$SKIP_HEALTH_CHECK" != "true" ]]; then
        wait_for_docker_services "${compose_files[@]}"
    fi
}

# Deploy to Kubernetes
deploy_kubernetes() {
    log_info "Deploying to Kubernetes (Environment: $ENVIRONMENT, Namespace: $NAMESPACE)..."
    
    # Apply manifests in order
    local manifests=(
        "k8s/namespace.yaml"
        "k8s/configmap.yaml"
        "k8s/secrets.yaml"
        "k8s/persistentvolume.yaml"
        "k8s/rbac.yaml"
        "k8s/deployment.yaml"
        "k8s/service.yaml"
        "k8s/ingress.yaml"
        "k8s/hpa.yaml"
    )
    
    for manifest in "${manifests[@]}"; do
        local manifest_path="${SCRIPT_DIR}/${manifest}"
        if [[ -f "$manifest_path" ]]; then
            log_info "Applying $manifest..."
            if [[ "$DRY_RUN" == "true" ]]; then
                log_info "DRY RUN: Would apply $manifest_path"
                kubectl apply --dry-run=client -f "$manifest_path" -n "$NAMESPACE" || true
            else
                kubectl apply -f "$manifest_path" -n "$NAMESPACE"
            fi
        else
            log_warn "Manifest not found: $manifest_path"
        fi
    done
    
    # Wait for deployment to be ready
    if [[ "$DRY_RUN" != "true" && "$SKIP_HEALTH_CHECK" != "true" ]]; then
        wait_for_kubernetes_deployment
    fi
}

# Wait for Docker services to be healthy
wait_for_docker_services() {
    local compose_files=("$@")
    log_info "Waiting for services to be healthy..."
    
    local max_attempts=$((TIMEOUT / 10))
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        log_info "Health check attempt $attempt/$max_attempts"
        
        # Check main application
        if curl -f -s http://localhost:8000/health > /dev/null 2>&1; then
            log_success "Application is healthy"
            return 0
        fi
        
        # Check service status
        $COMPOSE_CMD "${compose_files[@]}" ps --services --filter "status=running" | wc -l > /tmp/running_count
        local running_count=$(cat /tmp/running_count)
        local total_count=$($COMPOSE_CMD "${compose_files[@]}" config --services | wc -l)
        
        log_info "Services running: $running_count/$total_count"
        
        if [[ $running_count -eq $total_count ]]; then
            sleep 10  # Give services time to initialize
            continue
        fi
        
        sleep 10
        ((attempt++))
    done
    
    error_exit "Services failed to become healthy within $TIMEOUT seconds"
}

# Wait for Kubernetes deployment to be ready
wait_for_kubernetes_deployment() {
    log_info "Waiting for Kubernetes deployment to be ready..."
    
    # Wait for pods to be ready
    if ! kubectl wait --for=condition=ready pod -l app=workspace-qdrant-mcp -n "$NAMESPACE" --timeout="${TIMEOUT}s"; then
        error_exit "Deployment failed to become ready within $TIMEOUT seconds"
    fi
    
    # Check application health
    local max_attempts=30
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        log_info "Application health check attempt $attempt/$max_attempts"
        
        # Port forward and test
        if kubectl port-forward -n "$NAMESPACE" service/workspace-qdrant-mcp-service 8080:8000 --address=127.0.0.1 &
        then
            local port_forward_pid=$!
            sleep 2
            
            if curl -f -s http://localhost:8080/health > /dev/null 2>&1; then
                kill $port_forward_pid 2>/dev/null || true
                log_success "Application is healthy"
                return 0
            fi
            
            kill $port_forward_pid 2>/dev/null || true
        fi
        
        sleep 5
        ((attempt++))
    done
    
    error_exit "Application failed health checks"
}

# Rollback deployment
rollback_deployment() {
    if [[ "$ROLLBACK_ON_FAILURE" != "true" ]]; then
        log_info "Rollback disabled, skipping..."
        return 0
    fi
    
    log_warn "Rolling back deployment..."
    
    case "$DEPLOYMENT_TYPE" in
        docker-compose)
            rollback_docker_compose
            ;;
        kubernetes)
            rollback_kubernetes
            ;;
    esac
    
    log_success "Rollback completed"
}

# Rollback Docker Compose deployment
rollback_docker_compose() {
    # Find latest backup
    local latest_backup=$(find backup -maxdepth 1 -name "*-pre-deploy" -type d | sort -r | head -n1)
    
    if [[ -n "$latest_backup" ]]; then
        log_info "Rolling back to backup: $latest_backup"
        
        # Stop current services
        $COMPOSE_CMD -f "${SCRIPT_DIR}/docker-compose.yml" down
        
        # Restore configuration
        if [[ -f "$latest_backup/.env" ]]; then
            cp "$latest_backup/.env" "${SCRIPT_DIR}/"
        fi
        
        # Start services with previous configuration
        $COMPOSE_CMD -f "${SCRIPT_DIR}/docker-compose.yml" up -d
    else
        log_warn "No backup found for rollback"
    fi
}

# Rollback Kubernetes deployment
rollback_kubernetes() {
    # Get previous revision
    local current_revision=$(kubectl rollout history deployment/workspace-qdrant-mcp -n "$NAMESPACE" | tail -n1 | awk '{print $1}')
    local previous_revision=$((current_revision - 1))
    
    if [[ $previous_revision -gt 0 ]]; then
        log_info "Rolling back to revision $previous_revision"
        kubectl rollout undo deployment/workspace-qdrant-mcp -n "$NAMESPACE" --to-revision="$previous_revision"
        kubectl rollout status deployment/workspace-qdrant-mcp -n "$NAMESPACE" --timeout="${TIMEOUT}s"
    else
        log_warn "No previous revision found for rollback"
    fi
}

# Post-deployment tasks
post_deployment() {
    log_info "Running post-deployment tasks..."
    
    # Display deployment status
    case "$DEPLOYMENT_TYPE" in
        docker-compose)
            log_info "Docker Compose Services:"
            $COMPOSE_CMD -f "${SCRIPT_DIR}/docker-compose.yml" ps
            ;;
        kubernetes)
            log_info "Kubernetes Resources:"
            kubectl get all -n "$NAMESPACE"
            ;;
    esac
    
    # Display access information
    display_access_info
    
    log_success "Post-deployment tasks completed"
}

# Display access information
display_access_info() {
    log_info "Access Information:"
    echo "=================================="
    
    case "$DEPLOYMENT_TYPE" in
        docker-compose)
            echo "Application: http://localhost:8000"
            echo "Health Check: http://localhost:8000/health"
            echo "API Documentation: http://localhost:8000/docs"
            echo "Qdrant Dashboard: http://localhost:6333/dashboard"
            if [[ "$ENVIRONMENT" == "development" ]]; then
                echo "Grafana: http://localhost:3000 (admin/admin)"
                echo "Prometheus: http://localhost:9090"
                echo "Jaeger: http://localhost:16686"
            fi
            ;;
        kubernetes)
            echo "Use kubectl port-forward to access services:"
            echo "kubectl port-forward -n $NAMESPACE service/workspace-qdrant-mcp-service 8000:8000"
            echo "kubectl port-forward -n $NAMESPACE service/grafana-service 3000:3000"
            echo "kubectl port-forward -n $NAMESPACE service/prometheus-service 9090:9090"
            ;;
    esac
    
    echo "=================================="
}

# Cleanup function
cleanup() {
    log_info "Cleaning up temporary files..."
    rm -f /tmp/running_count
}

# Main execution function
main() {
    trap cleanup EXIT
    
    log_info "Starting Workspace Qdrant MCP Deployment"
    log_info "Script: $(realpath "$0")"
    log_info "Working Directory: $(pwd)"
    log_info "Deployment Type: $DEPLOYMENT_TYPE"
    log_info "Environment: $ENVIRONMENT"
    log_info "Namespace: $NAMESPACE"
    log_info "Dry Run: $DRY_RUN"
    
    # Validate and prepare
    parse_args "$@"
    validate_environment
    load_configuration
    
    # Create backup if not dry run
    if [[ "$DRY_RUN" != "true" ]]; then
        create_backup
    fi
    
    # Deploy based on type
    case "$DEPLOYMENT_TYPE" in
        docker-compose)
            deploy_docker_compose
            ;;
        kubernetes)
            deploy_kubernetes
            ;;
    esac
    
    # Post-deployment tasks
    if [[ "$DRY_RUN" != "true" ]]; then
        post_deployment
    fi
    
    log_success "Deployment completed successfully!"
    
} 2>&1 | tee "deployment-$(date +%Y%m%d-%H%M%S).log"

# Error handling for main execution
set +e
main "$@"
exit_code=$?
set -e

if [[ $exit_code -ne 0 ]]; then
    log_error "Deployment failed with exit code $exit_code"
    if [[ "$DRY_RUN" != "true" ]]; then
        rollback_deployment
    fi
    exit $exit_code
fi