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

## Workflow Fallback Options

The enhanced workflow supports multiple publishing strategies:

### Option 1: Trusted Publishing (Recommended)
Complete the setup steps above for both TestPyPI and PyPI.

### Option 2: Token Fallback (Immediate Solution)
If trusted publishing setup is not possible immediately:

1. Go to repository Settings > Variables and secrets > Variables
2. Add repository variable: `USE_TOKEN_FALLBACK` = `true`  
3. Go to repository Settings > Variables and secrets > Secrets
4. Add these secrets:
   - `TEST_PYPI_API_TOKEN` - Token from https://test.pypi.org/manage/account/token/
   - `PYPI_API_TOKEN` - Token from https://pypi.org/manage/account/token/

### Option 3: Hybrid Approach
Configure both trusted publishing AND token fallback. The workflow will:
1. Try trusted publishing first (preferred)
2. Fall back to tokens if trusted publishing fails
3. Provide clear status reporting

## Testing the Setup

1. Push a commit to the main branch to test TestPyPI upload
2. Tag a version (e.g., `v0.2.1`) to test PyPI upload 
3. Check workflow logs for publishing status messages
4. Verify packages appear on PyPI/TestPyPI

## Status Messages

The workflow provides clear feedback:

- ✅ **Success**: "Successfully published to [TestPyPI/PyPI] using trusted publishing"
- ⚠️ **Fallback**: "Trusted publishing failed, attempted token fallback"
- ❌ **Failed**: "Trusted publishing failed" with setup instructions

## Manual Upload (Development/Testing)

For local testing or one-off uploads:

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

## Troubleshooting

- **Build failures**: Ensure Rust toolchain is available
- **Publishing failures**: Check trusted publisher configuration matches exactly
- **Token issues**: Verify tokens have correct scope and are not expired
- **Environment issues**: Ensure GitHub environments exist and are accessible