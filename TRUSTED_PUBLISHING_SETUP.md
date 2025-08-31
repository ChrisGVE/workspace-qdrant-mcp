# Trusted Publishing Setup for PyPI

This document explains how to configure trusted publishing for this repository on PyPI and TestPyPI.

## Current Issue

The CI workflow is failing because the repository is not configured as a trusted publisher on PyPI/TestPyPI. The error message indicates:

```
* `invalid-publisher`: valid token, but no corresponding publisher (Publisher with matching claims was not found)
```

## Required Setup Steps

### 1. TestPyPI Trusted Publishing Configuration

1. Go to https://test.pypi.org/manage/account/publishing/
2. Click "Add a new pending publisher" 
3. Configure with these exact values:
   - **PyPI Project Name**: `workspace-qdrant-mcp`
   - **Owner**: `ChrisGVE` 
   - **Repository name**: `workspace-qdrant-mcp`
   - **Workflow filename**: `publish-to-pypi.yml`
   - **Environment name**: `testpypi`

### 2. PyPI Trusted Publishing Configuration  

1. Go to https://pypi.org/manage/account/publishing/
2. Click "Add a new pending publisher"
3. Configure with these exact values:
   - **PyPI Project Name**: `workspace-qdrant-mcp`
   - **Owner**: `ChrisGVE`
   - **Repository name**: `workspace-qdrant-mcp` 
   - **Workflow filename**: `publish-to-pypi.yml`
   - **Environment name**: `pypi`

## GitHub Environment Setup

The workflow also requires GitHub environments to be configured:

1. Go to repository Settings > Environments
2. Create environment `testpypi` (for TestPyPI publishing)
3. Create environment `pypi` (for PyPI publishing) 
4. Configure protection rules as needed

## Workflow Claims

The workflow provides these claims for trusted publishing:

- **sub**: `repo:ChrisGVE/workspace-qdrant-mcp:environment:testpypi`
- **repository**: `ChrisGVE/workspace-qdrant-mcp`
- **repository_owner**: `ChrisGVE`
- **repository_owner_id**: `214433`
- **workflow_ref**: `ChrisGVE/workspace-qdrant-mcp/.github/workflows/publish-to-pypi.yml@refs/heads/main`

These values must match exactly in the PyPI trusted publisher configuration.

## Testing the Setup

1. Once trusted publishers are configured on both PyPI services
2. Push a commit to the main branch 
3. The workflow should succeed in uploading to TestPyPI
4. Tag a version (e.g., `v0.2.1`) to trigger PyPI upload

## Alternative: Manual Upload

If trusted publishing setup is not possible immediately, packages can be manually uploaded:

```bash
# Build packages locally (requires Rust and maturin)
python -m pip install maturin
python -m maturin build --release --out dist
python -m maturin sdist --out dist

# Upload to TestPyPI
python -m pip install twine
python -m twine upload --repository testpypi dist/*

# Upload to PyPI (after testing)
python -m twine upload dist/*
```

Note: Manual uploads require API tokens to be configured in repository secrets.