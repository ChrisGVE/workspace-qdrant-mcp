# PyPI Trusted Publishing Setup Guide

This guide walks through setting up PyPI Trusted Publishing for secure, token-free automated releases.

## 🎯 Overview

Trusted Publishing uses OpenID Connect (OIDC) to securely authenticate GitHub Actions workflows with PyPI without requiring API tokens. This is more secure and easier to maintain than traditional API token-based publishing.

## 📋 Prerequisites

- PyPI account with maintainer access to `workspace-qdrant-mcp` project
- GitHub repository with admin access
- Project already exists on PyPI (for first-time setup, use manual upload)

## 🔧 Step-by-Step Setup

### 1. Configure PyPI Trusted Publishing

1. **Log into PyPI** at https://pypi.org
2. **Navigate to your project**: https://pypi.org/project/workspace-qdrant-mcp/
3. **Go to Settings** → **Publishing** 
4. **Click "Add a new trusted publisher"**

### 2. Fill in Publisher Details

Configure **two publishers** for production and test environments:

#### Production Publisher (PyPI)
```
Publisher: GitHub
Owner: ChrisGVE
Repository name: workspace-qdrant-mcp
Workflow filename: semantic-release.yml
Environment name: pypi
```

#### Test Publisher (TestPyPI) - Optional but Recommended
```
Publisher: GitHub  
Owner: ChrisGVE
Repository name: workspace-qdrant-mcp
Workflow filename: semantic-release.yml
Environment name: testpypi
```

### 3. TestPyPI Setup (Recommended)

For testing releases, also configure TestPyPI:

1. **Create TestPyPI account** at https://test.pypi.org
2. **Create the same project** on TestPyPI
3. **Configure trusted publisher** with same details as above
4. **Use `testpypi` environment name**

### 4. GitHub Repository Configuration  

#### Create Environments

1. **Go to repository Settings** → **Environments**
2. **Create `pypi` environment**:
   - Add environment protection rules (optional)
   - Required reviewers for production releases (optional)
3. **Create `testpypi` environment** (if using TestPyPI)

#### Configure Branch Protection (Recommended)

1. **Go to Settings** → **Branches**  
2. **Add rule for `main` branch**:
   - Require status checks to pass
   - Require branches to be up to date
   - Restrict pushes to main (require PRs)

## ✅ Verification

### Test the Setup

1. **Create a test commit** with conventional format:
   ```bash
   git commit -m "docs: test trusted publishing setup"
   git push origin main
   ```

2. **Monitor the workflow**:
   - Go to **Actions** tab in GitHub
   - Watch **Semantic Release and PyPI Publishing** workflow
   - Should complete without requiring API tokens

3. **Check PyPI**:
   - Verify new version appears on PyPI
   - Test installation: `pip install workspace-qdrant-mcp==<new-version>`

### Troubleshooting Common Issues

#### "Invalid token" error
- **Cause**: Trusted publishing not properly configured
- **Solution**: Double-check repository name, workflow filename, and environment name match exactly

#### "Workflow not found" error  
- **Cause**: Workflow filename doesn't match PyPI configuration
- **Solution**: Ensure workflow is named `semantic-release.yml` exactly

#### "Environment not found" error
- **Cause**: GitHub environment doesn't exist
- **Solution**: Create environments in repository settings

#### "No permission to publish" error
- **Cause**: User configuring trusted publishing lacks maintainer access
- **Solution**: Ensure PyPI account has maintainer or owner access to project

## 🔒 Security Considerations

### Benefits of Trusted Publishing

- **No long-lived tokens**: Eliminates token rotation and compromise risks
- **Scoped access**: Limited to specific repository and workflow
- **Audit trail**: All actions tracked through GitHub Actions logs
- **Automatic revocation**: Access automatically expires with workflow completion

### Additional Security Measures

1. **Environment protection**: Require manual approval for production releases
2. **Branch protection**: Prevent direct pushes to main branch  
3. **Required reviews**: Require PR reviews before merging
4. **Status checks**: Require CI to pass before merging
5. **Audit logs**: Regularly review GitHub Actions and PyPI logs

## 🔄 Migration from API Tokens

If migrating from API token-based publishing:

1. **Set up trusted publishing** (above steps)
2. **Test with non-production release** first
3. **Remove old API token secrets** from GitHub repository
4. **Update any documentation** referencing token-based setup
5. **Notify team members** of the change

### Remove Old Token Secrets

1. **Go to repository Settings** → **Secrets and variables** → **Actions**
2. **Delete these secrets** (if they exist):
   - `PYPI_API_TOKEN`
   - `TEST_PYPI_API_TOKEN`  
   - `PYPI_TOKEN`
   - Any other PyPI-related tokens

## 📚 Additional Resources

- [PyPI Trusted Publishing Documentation](https://docs.pypi.org/trusted-publishers/)
- [GitHub OIDC Documentation](https://docs.github.com/en/actions/deployment/security-hardening-your-deployments/about-security-hardening-with-openid-connect)
- [PyPA gh-action-pypi-publish](https://github.com/pypa/gh-action-pypi-publish)

## 🆘 Support

If you encounter issues:

1. **Check workflow logs** for detailed error messages
2. **Verify all configuration** matches this guide exactly  
3. **Test with simple commit** to isolate issues
4. **Contact PyPI support** for trusted publishing issues
5. **Open GitHub issue** for project-specific problems

---

*This setup enables secure, automated PyPI publishing without managing API tokens. Once configured, releases happen automatically based on conventional commits to the main branch.*