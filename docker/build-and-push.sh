#!/bin/bash
# Container Registry Publishing Pipeline
# Multi-architecture builds with security scanning and automated tagging

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DOCKER_DIR="${SCRIPT_DIR}"

# Default values
IMAGE_NAME="${IMAGE_NAME:-workspace-qdrant-mcp}"
REGISTRY="${REGISTRY:-docker.io}"
NAMESPACE="${NAMESPACE:-chrisgve}"
VERSION="${VERSION:-$(git describe --tags --always --dirty 2>/dev/null || echo 'dev')}"
PLATFORMS="${PLATFORMS:-linux/amd64,linux/arm64}"
BUILD_CONTEXT="${BUILD_CONTEXT:-${PROJECT_ROOT}}"
DOCKERFILE="${DOCKERFILE:-${DOCKER_DIR}/Dockerfile}"

# Build configuration
CACHE_FROM="${CACHE_FROM:-}"
CACHE_TO="${CACHE_TO:-}"
PUSH="${PUSH:-false}"
LOAD="${LOAD:-false}"
SCAN="${SCAN:-true}"
SBOM="${SBOM:-true}"
PROVENANCE="${PROVENANCE:-true}"

# Registry configuration
FULL_IMAGE_NAME="${REGISTRY}/${NAMESPACE}/${IMAGE_NAME}"
BUILD_TAGS=""

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

# Error handler
error_exit() {
    log_error "$1"
    exit 1
}

# Help function
show_help() {
    cat << EOF
Container Registry Publishing Pipeline

Usage: $(basename "$0") [OPTIONS]

OPTIONS:
    -h, --help              Show this help message
    -n, --name NAME         Image name (default: workspace-qdrant-mcp)
    -r, --registry REG      Container registry (default: docker.io)
    -s, --namespace NS      Registry namespace (default: chrisgve)
    -v, --version VER       Image version/tag (default: auto-detected from git)
    -p, --platforms PLAT    Target platforms (default: linux/amd64,linux/arm64)
    -c, --context PATH      Build context path (default: project root)
    -f, --dockerfile PATH   Dockerfile path (default: docker/Dockerfile)
    --push                  Push images to registry
    --load                  Load single-platform image to local Docker
    --no-scan               Skip security scanning
    --no-sbom               Skip SBOM generation
    --no-provenance         Skip provenance attestation
    --cache-from CACHE      Import build cache from
    --cache-to CACHE        Export build cache to

ENVIRONMENT VARIABLES:
    REGISTRY_USERNAME       Registry username for authentication
    REGISTRY_PASSWORD       Registry password/token for authentication
    DOCKER_CLI_EXPERIMENTAL Enable experimental Docker features
    BUILDX_PLATFORMS        Override target platforms

EXAMPLES:
    # Build and load locally
    $(basename "$0") --load

    # Build and push to registry
    $(basename "$0") --push

    # Build specific version with custom registry
    $(basename "$0") --version v1.2.3 --registry ghcr.io --namespace myorg --push

    # Build with cache
    $(basename "$0") --cache-from type=registry,ref=myregistry/buildcache \\
                     --cache-to type=registry,ref=myregistry/buildcache,mode=max \\
                     --push
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
            -n|--name)
                IMAGE_NAME="$2"
                shift 2
                ;;
            -r|--registry)
                REGISTRY="$2"
                shift 2
                ;;
            -s|--namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            -v|--version)
                VERSION="$2"
                shift 2
                ;;
            -p|--platforms)
                PLATFORMS="$2"
                shift 2
                ;;
            -c|--context)
                BUILD_CONTEXT="$2"
                shift 2
                ;;
            -f|--dockerfile)
                DOCKERFILE="$2"
                shift 2
                ;;
            --push)
                PUSH="true"
                shift
                ;;
            --load)
                LOAD="true"
                shift
                ;;
            --no-scan)
                SCAN="false"
                shift
                ;;
            --no-sbom)
                SBOM="false"
                shift
                ;;
            --no-provenance)
                PROVENANCE="false"
                shift
                ;;
            --cache-from)
                CACHE_FROM="$2"
                shift 2
                ;;
            --cache-to)
                CACHE_TO="$2"
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
    log_info "Validating environment..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error_exit "Docker is not installed or not in PATH"
    fi
    
    # Check buildx
    if ! docker buildx version &> /dev/null; then
        error_exit "Docker Buildx is not available"
    fi
    
    # Check build context
    if [[ ! -d "$BUILD_CONTEXT" ]]; then
        error_exit "Build context directory does not exist: $BUILD_CONTEXT"
    fi
    
    # Check Dockerfile
    if [[ ! -f "$DOCKERFILE" ]]; then
        error_exit "Dockerfile does not exist: $DOCKERFILE"
    fi
    
    # Validate platforms
    if [[ "$LOAD" == "true" ]] && [[ "$PLATFORMS" == *","* ]]; then
        log_warn "Cannot load multi-platform build, using single platform"
        PLATFORMS="linux/amd64"
    fi
    
    # Registry authentication
    if [[ "$PUSH" == "true" ]]; then
        if [[ -n "${REGISTRY_USERNAME:-}" && -n "${REGISTRY_PASSWORD:-}" ]]; then
            log_info "Authenticating to registry..."
            echo "$REGISTRY_PASSWORD" | docker login "$REGISTRY" --username "$REGISTRY_USERNAME" --password-stdin
        else
            log_warn "No registry credentials provided (REGISTRY_USERNAME/REGISTRY_PASSWORD)"
        fi
    fi
    
    log_success "Environment validation completed"
}

# Setup build environment
setup_buildx() {
    log_info "Setting up Docker Buildx..."
    
    # Create or use existing builder
    local builder_name="workspace-qdrant-builder"
    
    if ! docker buildx inspect "$builder_name" &> /dev/null; then
        log_info "Creating new buildx builder: $builder_name"
        docker buildx create \
            --name "$builder_name" \
            --driver docker-container \
            --bootstrap \
            --use
    else
        log_info "Using existing buildx builder: $builder_name"
        docker buildx use "$builder_name"
    fi
    
    # Inspect builder
    docker buildx inspect --bootstrap
    
    log_success "Buildx setup completed"
}

# Generate build tags
generate_tags() {
    log_info "Generating build tags..."
    
    local tags=()
    
    # Always add version tag
    tags+=("${FULL_IMAGE_NAME}:${VERSION}")
    
    # Add semantic version tags if version follows semver pattern
    if [[ "$VERSION" =~ ^v?[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        local version_clean="${VERSION#v}"
        local major="${version_clean%%.*}"
        local minor="${version_clean%.*}"
        
        tags+=("${FULL_IMAGE_NAME}:${major}")
        tags+=("${FULL_IMAGE_NAME}:${minor}")
        tags+=("${FULL_IMAGE_NAME}:latest")
    fi
    
    # Add latest tag for main/master branch
    local current_branch="${CI_COMMIT_REF_NAME:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '')}"
    if [[ "$current_branch" == "main" || "$current_branch" == "master" ]]; then
        if [[ ! " ${tags[@]} " =~ " ${FULL_IMAGE_NAME}:latest " ]]; then
            tags+=("${FULL_IMAGE_NAME}:latest")
        fi
    fi
    
    # Add branch tag for non-main branches
    if [[ "$current_branch" != "main" && "$current_branch" != "master" && -n "$current_branch" ]]; then
        local branch_tag="${current_branch//\//-}"  # Replace / with -
        tags+=("${FULL_IMAGE_NAME}:${branch_tag}")
    fi
    
    # Add commit SHA tag
    local commit_sha="${CI_COMMIT_SHA:-$(git rev-parse --short HEAD 2>/dev/null || echo '')}"
    if [[ -n "$commit_sha" ]]; then
        tags+=("${FULL_IMAGE_NAME}:sha-${commit_sha}")
    fi
    
    # Build tags string for docker buildx
    BUILD_TAGS=""
    for tag in "${tags[@]}"; do
        BUILD_TAGS="${BUILD_TAGS} --tag ${tag}"
    done
    
    log_info "Generated tags: ${tags[*]}"
    log_success "Tag generation completed"
}

# Build image
build_image() {
    log_info "Building container image..."
    
    local build_args=(
        "buildx" "build"
        "--platform" "$PLATFORMS"
        "--file" "$DOCKERFILE"
    )
    
    # Add tags
    if [[ -n "$BUILD_TAGS" ]]; then
        build_args+=($BUILD_TAGS)
    fi
    
    # Add cache configuration
    if [[ -n "$CACHE_FROM" ]]; then
        build_args+=("--cache-from" "$CACHE_FROM")
    fi
    
    if [[ -n "$CACHE_TO" ]]; then
        build_args+=("--cache-to" "$CACHE_TO")
    fi
    
    # Add build metadata
    build_args+=(
        "--label" "org.opencontainers.image.title=${IMAGE_NAME}"
        "--label" "org.opencontainers.image.description=Workspace Qdrant MCP Server"
        "--label" "org.opencontainers.image.version=${VERSION}"
        "--label" "org.opencontainers.image.created=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        "--label" "org.opencontainers.image.revision=$(git rev-parse HEAD 2>/dev/null || echo 'unknown')"
        "--label" "org.opencontainers.image.source=https://github.com/ChrisGVE/workspace-qdrant-mcp"
        "--label" "org.opencontainers.image.url=https://github.com/ChrisGVE/workspace-qdrant-mcp"
        "--label" "org.opencontainers.image.vendor=Christian C. Berclaz"
        "--label" "org.opencontainers.image.licenses=MIT"
    )
    
    # Add attestations
    if [[ "$SBOM" == "true" ]]; then
        build_args+=("--sbom=true")
    fi
    
    if [[ "$PROVENANCE" == "true" ]]; then
        build_args+=("--provenance=true")
    fi
    
    # Add output options
    if [[ "$PUSH" == "true" ]]; then
        build_args+=("--push")
    elif [[ "$LOAD" == "true" ]]; then
        build_args+=("--load")
    else
        build_args+=("--output" "type=oci,dest=/tmp/image.tar")
    fi
    
    # Add build context
    build_args+=("$BUILD_CONTEXT")
    
    log_info "Running: docker ${build_args[*]}"
    
    # Execute build
    if ! docker "${build_args[@]}"; then
        error_exit "Image build failed"
    fi
    
    log_success "Image build completed"
}

# Security scanning
scan_image() {
    if [[ "$SCAN" != "true" ]]; then
        log_info "Security scanning disabled, skipping..."
        return 0
    fi
    
    log_info "Running security scan..."
    
    local scan_image="${FULL_IMAGE_NAME}:${VERSION}"
    
    # Try different scanners
    if command -v grype &> /dev/null; then
        log_info "Using Grype for vulnerability scanning..."
        grype "$scan_image" --output json --file "/tmp/grype-report.json" || log_warn "Grype scan failed"
        grype "$scan_image" --output table || log_warn "Grype scan display failed"
    elif command -v trivy &> /dev/null; then
        log_info "Using Trivy for vulnerability scanning..."
        trivy image "$scan_image" --format json --output "/tmp/trivy-report.json" || log_warn "Trivy scan failed"
        trivy image "$scan_image" || log_warn "Trivy scan display failed"
    else
        log_warn "No vulnerability scanner found (grype or trivy), skipping security scan"
    fi
    
    log_success "Security scanning completed"
}

# Generate and display build summary
build_summary() {
    log_info "Build Summary:"
    echo "================================="
    echo "Image Name:       $FULL_IMAGE_NAME"
    echo "Version:          $VERSION"
    echo "Platforms:        $PLATFORMS"
    echo "Build Context:    $BUILD_CONTEXT"
    echo "Dockerfile:       $DOCKERFILE"
    echo "Push to Registry: $PUSH"
    echo "Load Locally:     $LOAD"
    echo "Security Scan:    $SCAN"
    echo "SBOM Generation:  $SBOM"
    echo "Provenance:       $PROVENANCE"
    echo "================================="
    
    if [[ "$PUSH" == "true" ]]; then
        echo
        echo "Published Images:"
        for tag in ${BUILD_TAGS}; do
            if [[ "$tag" != "--tag" ]]; then
                echo "  $tag"
            fi
        done
    fi
}

# Cleanup function
cleanup() {
    log_info "Cleaning up..."
    
    # Remove temporary files
    rm -f /tmp/image.tar /tmp/grype-report.json /tmp/trivy-report.json
    
    # Logout from registry if we logged in
    if [[ "$PUSH" == "true" && -n "${REGISTRY_USERNAME:-}" ]]; then
        docker logout "$REGISTRY" || true
    fi
}

# Main execution
main() {
    trap cleanup EXIT
    
    log_info "Starting Container Registry Publishing Pipeline"
    log_info "Script: $(realpath "$0")"
    log_info "Working Directory: $(pwd)"
    
    parse_args "$@"
    validate_environment
    setup_buildx
    generate_tags
    build_image
    scan_image
    build_summary
    
    log_success "Container Registry Publishing Pipeline completed successfully"
}

# Execute main function
main "$@"