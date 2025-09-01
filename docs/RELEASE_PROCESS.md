# Release Process Documentation

This document describes the automated release process for workspace-qdrant-mcp, including semantic versioning, automated PyPI publishing, and emergency rollback procedures.

## 📋 Overview

Our release system is fully automated using semantic versioning with conventional commits. Every commit to the `main` branch is analyzed for release necessity, and appropriate versions are automatically published to PyPI.

### Key Components

- **Semantic Release Pipeline** (`.github/workflows/semantic-release.yml`)
- **Emergency Rollback System** (`.github/workflows/release-rollback.yml`) 
- **Release Verification & Monitoring** (`.github/workflows/release-verification.yml`)
- **Legacy Manual Publishing** (`.github/workflows/publish-to-pypi.yml`) - Deprecated

## 🚀 Normal Release Process

### 1. Development Workflow

All releases are triggered by commits to the `main` branch using conventional commit messages:

```bash
# Feature releases (minor version bump)
git commit -m "feat: add new search algorithm"
git commit -m "feat(api): add new MCP tool for bulk operations"

# Bug fixes (patch version bump)
git commit -m "fix: resolve memory leak in document ingestion"
git commit -m "fix(client): handle connection timeout gracefully"

# Breaking changes (major version bump)
git commit -m "feat!: redesign API interface

BREAKING CHANGE: API endpoints have changed. See migration guide."

# Non-release commits (no version bump)
git commit -m "docs: update README examples"
git commit -m "test: add integration tests"
git commit -m "chore: update dependencies"
```

### 2. Automated Pipeline Stages

When you push to `main`, the semantic release pipeline automatically:

#### Stage 1: Pre-flight Checks ✈️
- Analyzes commits for release necessity
- Runs quality checks (Ruff, Black, MyPy)
- Determines new version number
- Exits early if no release needed

#### Stage 2: Semantic Release 📝
- Calculates version based on conventional commits
- Updates version in `pyproject.toml` and `__init__.py`  
- Generates changelog from commit messages
- Creates Git tag and GitHub release

#### Stage 3: Distribution Building 📦
- Builds source distribution (sdist)
- Builds cross-platform wheels (Linux, macOS, Windows)
- Runs comprehensive verification tests
- Uploads artifacts for publishing

#### Stage 4: Testing & Validation 🧪
- Tests installation from built distributions
- Runs integration tests with Qdrant
- Verifies CLI commands work correctly
- Validates metadata and dependencies

#### Stage 5: PyPI Publishing 🚀
- Publishes to TestPyPI first for validation
- Tests installation from TestPyPI
- Publishes to production PyPI
- Verifies availability and functionality

#### Stage 6: Post-Release Verification ✅
- Waits for PyPI propagation
- Tests production installation
- Updates GitHub release with status
- Creates issues if problems detected

### 3. Version Calculation Rules

| Commit Type | Release Type | Example |
|-------------|-------------|---------|
| `feat:` | Minor | 1.0.0 → 1.1.0 |
| `fix:`, `perf:` | Patch | 1.0.0 → 1.0.1 |
| `feat!:` or `BREAKING CHANGE:` | Major | 1.0.0 → 2.0.0 |
| `docs:`, `test:`, `chore:` | None | No release |

## 🚨 Emergency Rollback Process

If a release introduces critical issues, use the emergency rollback workflow:

### 1. Trigger Rollback

Go to **Actions** → **Release Rollback and Recovery** → **Run workflow**

**Required inputs:**
- **Rollback Version**: The last known good version (e.g., `0.1.9`)
- **Rollback Reason**: Clear description of the issue
- **PyPI Action**: Choose from:
  - `yank`: Mark problematic version as yanked (recommended)
  - `leave-as-is`: Keep problematic version available  
  - `delete-if-possible`: Attempt deletion (rarely works on PyPI)

### 2. Rollback Process

The system automatically:

1. **Validates** rollback version exists in Git history
2. **Creates** emergency hotfix branch from rollback version
3. **Builds** new emergency release with incremented version
4. **Tests** emergency release thoroughly
5. **Handles** PyPI actions (yanking/leaving problematic version)
6. **Publishes** emergency release to PyPI
7. **Creates** GitHub issue tracking the rollback
8. **Merges** hotfix back to main branch

### 3. Post-Rollback Actions

After successful rollback:
- Monitor emergency release stability
- Investigate root cause of original issue
- Develop proper forward-fix
- Plan new stable release
- Update documentation and notify users

## 📊 Release Verification & Monitoring

### Automatic Verification

Every published release is automatically verified by:

- **Cross-platform testing** on Linux, macOS, Windows
- **Multi-Python version** compatibility (3.10, 3.11, 3.12)
- **Integration testing** with live Qdrant instance  
- **Performance benchmarking** for regressions
- **Security scanning** of dependencies
- **CLI functionality** verification

### Monitoring Schedule

- **On Release**: Immediate verification after PyPI publishing
- **Daily**: Scheduled verification of latest release at 6 AM UTC
- **On Demand**: Manual verification of specific versions

### Failure Handling

If verification fails:
- Automatic GitHub issue creation
- Detailed failure report generation
- Notification to maintainers
- Recommendations for remediation

## 🛠️ Manual Operations

### Emergency Manual Release

If the automated system fails, use the deprecated manual workflow:

1. Go to **Actions** → **[DEPRECATED] Manual PyPI Publishing**
2. Enter `MANUAL_PUBLISH` in the confirmation field  
3. Specify the Git tag to publish (e.g., `v1.2.3`)
4. Monitor the workflow progress

⚠️ **Warning**: Manual releases bypass semantic versioning and safety checks.

### Dry Run Testing

Test semantic release without publishing:

1. Go to **Actions** → **Semantic Release and PyPI Publishing**
2. Check **Run semantic-release in dry-run mode**
3. Review what would be released without actual publication

## 📚 Configuration Files

### Semantic Release Config (`.releaserc.json`)

```json
{
  "branches": ["main"],
  "plugins": [
    "@semantic-release/commit-analyzer",
    "@semantic-release/release-notes-generator", 
    "@semantic-release/changelog",
    "@semantic-release/exec",  // Python version management
    "@semantic-release/github",
    "@semantic-release/git"
  ]
}
```

### Package Dependencies (`package.json`)

Node.js dependencies for semantic-release automation:

```json
{
  "devDependencies": {
    "@semantic-release/changelog": "^6.0.3",
    "@semantic-release/commit-analyzer": "^11.0.0",
    "@semantic-release/exec": "^6.0.3",
    "@semantic-release/git": "^10.0.1",
    "@semantic-release/github": "^9.2.0",
    "@semantic-release/release-notes-generator": "^12.0.0",
    "semantic-release": "^22.0.0"
  }
}
```

## 🔧 Environment Setup

### GitHub Repository Settings

**Required secrets:**
- PyPI trusted publishing configured for repository
- No manual API tokens needed (uses OIDC)

**Required environments:**
- `pypi`: Production PyPI publishing
- `testpypi`: TestPyPI publishing for validation

**Branch protection:**
- Require status checks on `main` branch
- Require up-to-date branches before merging
- Restrict pushes to `main` (use PR workflow)

### PyPI Trusted Publishing Setup

1. Go to [PyPI Trusted Publishing](https://pypi.org/manage/account/publishing/)
2. Add publisher for `ChrisGVE/workspace-qdrant-mcp`
3. Set workflow: `semantic-release.yml`
4. Set environment: `pypi`

## 🐛 Troubleshooting

### Common Issues

#### "No release necessary"
- Check commit message follows conventional format
- Ensure commits contain features/fixes since last release
- Verify previous release was successful

#### "PyPI publishing failed"  
- Check trusted publishing configuration
- Verify repository and workflow names match PyPI settings
- Review PyPI project permissions

#### "Cross-platform tests failing"
- Check for platform-specific dependencies
- Verify Rust toolchain compatibility
- Review wheel building configuration

#### "Emergency rollback failed"
- Ensure rollback version exists as Git tag
- Check rollback version is older than current
- Verify repository write permissions

### Getting Help

1. **Check workflow logs** in GitHub Actions for detailed error messages
2. **Review issues** labeled `release-failure` or `rollback` 
3. **Manual intervention** may be required for complex failures
4. **Contact maintainers** for persistent issues

## 📈 Best Practices

### For Developers

1. **Use conventional commits** consistently
2. **Test changes thoroughly** before merging to main
3. **Monitor release notifications** in GitHub
4. **Report issues quickly** if problems detected post-release

### For Maintainers

1. **Review automated releases** regularly
2. **Monitor verification reports** for trends
3. **Update dependencies** proactively 
4. **Practice rollback procedures** periodically
5. **Keep documentation current** with system changes

## 📋 Release Checklist

Before merging to main:

- [ ] Commit messages follow conventional format
- [ ] Changes are tested locally
- [ ] CI checks are passing
- [ ] Breaking changes are documented
- [ ] Version bump expectations are clear

After release:

- [ ] Verify release published successfully
- [ ] Check cross-platform installation works
- [ ] Monitor for user reports of issues
- [ ] Update downstream dependencies if needed

---

*This document is automatically updated with each release. For questions or improvements, please open an issue.*