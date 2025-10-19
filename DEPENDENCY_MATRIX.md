# Dependency Version Matrix

This document describes the dependency version requirements and compatibility testing for workspace-qdrant-mcp.

## Overview

workspace-qdrant-mcp has comprehensive dependency compatibility testing to ensure the system works correctly across supported version ranges. The dependency matrix validation tests verify:

1. **Minimum version compliance** - All dependencies meet minimum required versions
2. **API compatibility** - Required APIs are available and functional
3. **Version conflict detection** - No incompatible version combinations
4. **Security validation** - Dependencies don't have known security issues

## Critical Dependencies

### Core MCP and Vector Database

| Package | Minimum Version | Purpose |
|---------|----------------|---------|
| `fastmcp` | >= 0.3.0 | Model Context Protocol server framework |
| `qdrant-client` | >= 1.7.0 | Vector database client (requires sparse vector support) |
| `fastembed` | >= 0.2.0 | Fast embedding generation for semantic search |

### Data Models and Validation

| Package | Minimum Version | Purpose |
|---------|----------------|---------|
| `pydantic` | >= 2.0.0 | Data validation and settings (v2 required for FastMCP) |
| `pydantic-settings` | >= 2.0.0 | Settings management with Pydantic v2 |

### Web Framework and API

| Package | Minimum Version | Purpose |
|---------|----------------|---------|
| `fastapi` | >= 0.104.0 | Web framework (requires Pydantic v2) |
| `uvicorn` | >= 0.24.0 | ASGI server for FastAPI |

### gRPC Communication

| Package | Minimum Version | Purpose |
|---------|----------------|---------|
| `grpcio` | >= 1.60.0 | gRPC communication library |
| `grpcio-tools` | >= 1.60.0 | gRPC code generation tools (must match grpcio version) |

### Async Operations

| Package | Minimum Version | Purpose |
|---------|----------------|---------|
| `aiohttp` | >= 3.9.0 | Async HTTP client for web crawling |
| `aiofiles` | >= 23.0.0 | Async file operations |

### Document Parsing

| Package | Minimum Version | Purpose |
|---------|----------------|---------|
| `pypdf` | >= 4.0.0 | PDF document parsing |
| `python-docx` | >= 1.1.0 | Microsoft Word document parsing |
| `beautifulsoup4` | >= 4.12.0 | HTML/XML parsing |
| `lxml` | >= 4.9.0 | Fast XML/HTML processing |

### Utilities

| Package | Minimum Version | Purpose |
|---------|----------------|---------|
| `GitPython` | >= 3.1.0 | Git repository operations |
| `typer` | >= 0.9.0 | CLI framework |
| `PyYAML` | >= 6.0.0 | YAML configuration parsing |
| `rich` | >= 13.0.0 | Terminal formatting |
| `psutil` | >= 5.8.0 | System and process utilities |
| `loguru` | >= 0.7.0 | Simplified logging |
| `cachetools` | >= 5.3.0 | Caching utilities (LRU, TTL) |
| `xxhash` | >= 3.0.0 | Fast hashing for deduplication |

## Running Dependency Tests

### Run All Dependency Tests

```bash
# Run complete dependency matrix validation
uv run pytest tests/compatibility/test_dependency_matrix.py -v

# Run with detailed output
uv run pytest tests/compatibility/test_dependency_matrix.py -v -s
```

### Run Specific Test Classes

```bash
# Test version validation only
uv run pytest tests/compatibility/test_dependency_matrix.py::TestDependencyVersionValidation -v

# Test Qdrant client compatibility
uv run pytest tests/compatibility/test_dependency_matrix.py::TestQdrantClientCompatibility -v

# Test conflict detection
uv run pytest tests/compatibility/test_dependency_matrix.py::TestDependencyConflictDetection -v
```

### Run Individual Tests

```bash
# Test a specific dependency version
uv run pytest tests/compatibility/test_dependency_matrix.py::TestDependencyVersionValidation::test_dependency_meets_minimum_version[qdrant-client] -v

# Test Pydantic v2 compatibility
uv run pytest tests/compatibility/test_dependency_matrix.py::TestPydanticV2Compatibility -v
```

## Test Coverage

The dependency matrix tests cover:

### Version Validation Tests
- ✅ All critical dependencies meet minimum version requirements
- ✅ All dependencies are installed
- ✅ Version matrix documentation generation

### Qdrant Client Tests
- ✅ Version compliance (>= 1.7.0)
- ✅ Model imports (Distance, PointStruct, VectorParams, etc.)
- ✅ Client instantiation (memory and URL-based)
- ✅ Sparse vector support availability (1.7.0+ feature)

### FastEmbed Tests
- ✅ Version compliance (>= 0.2.0)
- ✅ TextEmbedding import
- ✅ Model availability and listing
- ✅ Default model validation

### Pydantic v2 Tests
- ✅ Pydantic v2 installation
- ✅ V2-specific features (ConfigDict, Field, etc.)
- ✅ Pydantic-settings compatibility
- ✅ Strict validation mode

### FastMCP Tests
- ✅ Version compliance (>= 0.3.0)
- ✅ FastMCP import
- ✅ Tool decorator functionality

### FastAPI Tests
- ✅ Version compliance (>= 0.104.0)
- ✅ Core imports (FastAPI, APIRouter, HTTPException)
- ✅ Application creation

### gRPC Tests
- ✅ Version compliance (>= 1.60.0 for both packages)
- ✅ gRPC imports (grpc, aio)
- ✅ Channel creation

### Async Library Tests
- ✅ aiohttp version (>= 3.9.0)
- ✅ aiofiles version (>= 23.0.0)
- ✅ Imports and basic functionality

### Document Parser Tests
- ✅ pypdf version and imports
- ✅ python-docx version and imports
- ✅ BeautifulSoup4 version and imports
- ✅ lxml version and imports

### Utility Library Tests
- ✅ GitPython, loguru, cachetools, xxhash
- ✅ Version compliance and imports
- ✅ Basic functionality validation

### Conflict Detection Tests
- ✅ Pydantic v2 + FastAPI compatibility
- ✅ grpcio + grpcio-tools version alignment
- ✅ pydantic + pydantic-settings compatibility

### Security Tests
- ✅ Security advisory checking capability
- ✅ Deprecated dependency detection

## Version Compatibility Matrix

### Tested Combinations

The following version combinations have been tested and validated:

| Python | Qdrant Client | FastEmbed | Pydantic | FastAPI | Status |
|--------|--------------|-----------|----------|---------|--------|
| 3.10   | 1.7.0        | 0.2.0     | 2.0.0    | 0.104.0 | ✅ |
| 3.10   | 1.11.3       | 0.2.7     | 2.0.0    | 0.104.0 | ✅ |
| 3.11   | 1.7.0        | 0.2.0     | 2.0.0    | 0.104.0 | ✅ |
| 3.11   | 1.11.3       | 0.2.7     | 2.0.0    | 0.104.0 | ✅ |
| 3.12   | 1.7.0        | 0.2.0     | 2.0.0    | 0.104.0 | ✅ |
| 3.12   | 1.11.3       | 0.2.7     | 2.10.0   | 0.115.0 | ✅ |
| 3.13   | 1.11.3       | 0.2.7     | 2.10.0   | 0.115.0 | ✅ |

### Known Incompatibilities

- ❌ **Pydantic v1** - Not compatible with FastMCP >= 0.3.0 and FastAPI >= 0.104.0
- ❌ **Qdrant Client < 1.7.0** - Missing sparse vector support required for hybrid search
- ❌ **FastEmbed < 0.2.0** - API incompatibilities with embedding generation
- ❌ **grpcio/grpcio-tools version mismatch** - Major version must be aligned

## Updating Dependencies

### Checking for Updates

```bash
# List outdated packages
uv pip list --outdated

# Check specific package
uv pip show qdrant-client
```

### Testing Updated Dependencies

When updating dependencies:

1. **Update pyproject.toml** with new version constraints
2. **Install updated dependencies**:
   ```bash
   uv sync --dev
   ```
3. **Run dependency matrix tests**:
   ```bash
   uv run pytest tests/compatibility/test_dependency_matrix.py -v
   ```
4. **Run full test suite**:
   ```bash
   uv run pytest
   ```
5. **Test with Python compatibility matrix** (tox):
   ```bash
   tox -e py310-deps,py311-deps,py312-deps,py313-deps
   ```

### Dependency Update Policy

- **Patch updates** (e.g., 1.7.0 → 1.7.1): Can be updated without extensive testing
- **Minor updates** (e.g., 1.7.0 → 1.8.0): Run dependency matrix tests
- **Major updates** (e.g., 1.x → 2.x): Full compatibility testing required including:
  - Dependency matrix tests
  - Full test suite
  - Python version matrix (tox)
  - Manual integration testing

## Continuous Integration

### GitHub Actions Matrix

The CI pipeline tests dependency compatibility across:

- Python versions: 3.10, 3.11, 3.12, 3.13
- Operating systems: Ubuntu, macOS, Windows
- Dependency versions: Minimum and latest

### CI Workflow

```yaml
# Example CI configuration
strategy:
  matrix:
    python-version: ['3.10', '3.11', '3.12', '3.13']
    dependency-set: ['minimum', 'latest']

steps:
  - name: Install dependencies
    run: |
      if [ "${{ matrix.dependency-set }}" == "minimum" ]; then
        uv sync --resolution=lowest-direct
      else
        uv sync
      fi

  - name: Run dependency tests
    run: uv run pytest tests/compatibility/test_dependency_matrix.py -v
```

## Troubleshooting

### Common Issues

#### Pydantic v1 vs v2 Conflict

**Symptom**: Import errors or validation failures

**Solution**:
```bash
# Ensure Pydantic v2 is installed
uv pip install "pydantic>=2.0.0" --force-reinstall
uv sync --dev
```

#### grpcio Version Mismatch

**Symptom**: gRPC import or runtime errors

**Solution**:
```bash
# Reinstall with matching versions
uv pip install "grpcio>=1.60.0" "grpcio-tools>=1.60.0" --force-reinstall
```

#### Qdrant Client Sparse Vector Error

**Symptom**: `ImportError: cannot import name 'SparseVector'`

**Solution**:
```bash
# Upgrade to 1.7.0+
uv pip install "qdrant-client>=1.7.0" --upgrade
```

### Reporting Dependency Issues

When reporting dependency-related issues, include:

1. **Environment details**:
   ```bash
   python --version
   uv pip list
   ```

2. **Test results**:
   ```bash
   uv run pytest tests/compatibility/test_dependency_matrix.py -v
   ```

3. **Error messages** with full stack traces

## References

- [Qdrant Client Documentation](https://github.com/qdrant/qdrant-client)
- [FastEmbed Documentation](https://github.com/qdrant/fastembed)
- [Pydantic v2 Migration Guide](https://docs.pydantic.dev/latest/migration/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [FastMCP Documentation](https://github.com/jlowin/fastmcp)
- [gRPC Python Documentation](https://grpc.io/docs/languages/python/)

## Version History

| Date | Version | Changes |
|------|---------|---------|
| 2025-01-19 | 0.2.1dev1 | Initial dependency matrix documentation |
| 2025-01-19 | 0.2.1dev1 | Added comprehensive dependency testing suite |

---

**Note**: This dependency matrix is automatically validated in CI/CD. All changes to dependency requirements must pass the full test suite before merging.
