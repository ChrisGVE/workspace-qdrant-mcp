# CI/CD Pipeline Documentation

This directory contains the complete GitHub Actions CI/CD pipeline for the workspace-qdrant-mcp project, featuring evidence-based quality gates and comprehensive testing.

## 🏗️ Pipeline Overview

Our CI/CD pipeline implements a comprehensive testing strategy with evidence-based quality gates derived from our 21,930-query benchmark analysis:

- **Symbol Search**: ≥90% precision/recall (measured: 100%, n=1,930)
- **Exact Search**: ≥90% precision/recall (measured: 100%, n=10,000)  
- **Semantic Search**: ≥84% precision, ≥70% recall (measured: 94.2%/78.3%, n=10,000)

## 📁 Workflow Files

### Core CI/CD Workflows

#### `ci.yml` - Main CI/CD Pipeline
**Triggers**: Push to main/develop, Pull requests, Releases
- Multi-Python version testing (3.8-3.12)
- Evidence-based performance validation
- Security scanning (bandit, safety)
- Code quality checks (ruff, mypy)
- Test coverage reporting
- Package building and validation
- Automatic PyPI publishing on releases

#### `quality.yml` - Quality Assurance
**Triggers**: Push, PR, Daily at 2 AM UTC
- Comprehensive code quality analysis
- Dependency security auditing
- Performance monitoring
- Documentation quality checks
- PR comments with quality reports

#### `benchmark.yml` - Performance Benchmarks  
**Triggers**: Push to main, PR, Daily at 3 AM UTC, Manual
- Simple benchmark validation
- Comprehensive benchmark with Qdrant
- Large-scale benchmarks (scheduled/manual only)
- Performance threshold validation
- Benchmark result comparisons

#### `release.yml` - Release Automation
**Triggers**: Version tags (v*.*.*), Manual dispatch
- Release readiness validation
- Multi-version testing
- Performance validation
- Package building and testing  
- GitHub release creation
- PyPI/Test PyPI publishing
- Post-release version bumping

#### `security.yml` - Security Scanning
**Triggers**: Push, PR, Daily at 1 AM UTC, Manual
- Static security analysis (bandit, semgrep)
- Dependency vulnerability scanning (safety, pip-audit)
- Secrets detection (TruffleHog, gitleaks)
- Code quality security checks
- Security summary reporting

### Configuration Files

#### `dependabot.yml` - Automated Dependency Updates
- Daily Python dependency updates
- Weekly GitHub Actions updates
- Grouped updates by dependency type
- Auto-merge for security and patch updates
- Security-focused update prioritization

## 🎯 Quality Gates

### Evidence-Based Thresholds

Our quality gates are based on comprehensive benchmark analysis of 21,930 queries:

| Search Type | Precision Threshold | Recall Threshold | Measured Performance |
|-------------|--------------------|--------------------|---------------------|
| Symbol Search | ≥90% | ≥90% | 100% (n=1,930) |
| Exact Search | ≥90% | ≥90% | 100% (n=10,000) |
| Semantic Search | ≥84% | ≥70% | 94.2%/78.3% (n=10,000) |

### Quality Requirements

- **Test Coverage**: ≥80%
- **Python Versions**: 3.8, 3.9, 3.10, 3.11, 3.12
- **Security**: No high-severity vulnerabilities
- **Code Quality**: Passing ruff and mypy checks
- **Performance**: Meeting evidence-based benchmarks

## 🚀 Workflow Execution

### On Push/PR
1. **Parallel execution** of quality checks and tests
2. **Multi-version testing** across Python 3.8-3.12
3. **Security scanning** for vulnerabilities
4. **Performance validation** against benchmarks
5. **Package building** and console script testing

### On Release
1. **Release validation** (version format, changelog)
2. **Full test suite** across all Python versions
3. **Performance benchmarking** with evidence-based validation
4. **Package building** and installation testing
5. **GitHub release creation** with automated notes
6. **PyPI publishing** (stable) or Test PyPI (prerelease)
7. **Post-release cleanup** and version bumping

### Scheduled Jobs
- **1 AM UTC**: Security scanning
- **2 AM UTC**: Quality assurance analysis
- **3 AM UTC**: Performance benchmarking
- **4 AM UTC**: Dependency updates (Dependabot)

## 🔧 Environment Setup

### Required Secrets

```bash
# PyPI Publishing
PYPI_API_TOKEN=pypi-...           # For stable releases
TEST_PYPI_API_TOKEN=pypi-...      # For prereleases

# Security Scanning (optional)
GITLEAKS_LICENSE=...              # For gitleaks pro features
```

### Required Environments

Create these environments in GitHub repository settings:

- **`pypi`**: For stable release publishing to PyPI
- **`test-pypi`**: For prerelease publishing to Test PyPI

## 📊 Artifacts and Reports

### Generated Artifacts

Each workflow produces comprehensive artifacts:

#### CI Pipeline Artifacts
- `test-results-python-X.X` - Test results per Python version
- `build-artifacts` - Wheel and sdist packages
- `benchmark-results` - Performance benchmark outputs

#### Quality Assurance Artifacts  
- `quality-reports` - Code quality analysis
- `dependency-reports` - Dependency security analysis
- `performance-reports` - Performance monitoring data
- `documentation-reports` - Documentation quality checks

#### Security Artifacts
- `security-reports` - Security scan results
- `dependency-security-reports` - Dependency vulnerabilities
- `security-summary` - Consolidated security status

#### Benchmark Artifacts
- `simple-benchmark-results` - Basic performance tests
- `comprehensive-benchmark-results` - Full Qdrant integration tests
- `large-scale-benchmark-results` - Large dataset benchmarks
- `benchmark-comparison-report` - Cross-benchmark analysis

### Report Features

- **PR Comments**: Automatic quality and benchmark summaries
- **GitHub Step Summaries**: Rich workflow result displays
- **Artifact Storage**: 90-day retention of all reports
- **Trend Analysis**: Performance monitoring over time

## 🔒 Security Features

### Multi-Layer Security
1. **Static Analysis**: Bandit, Semgrep
2. **Dependency Scanning**: Safety, pip-audit
3. **Secrets Detection**: TruffleHog, Gitleaks  
4. **Code Quality**: Security-focused linting rules
5. **Automated Updates**: Dependabot with security prioritization

### Security-First Design
- High-severity vulnerabilities block releases
- Automated security patches via Dependabot
- Daily vulnerability scans
- Secrets scanning on every commit
- File permission validation

## 🎛️ Monitoring and Alerting

### Performance Monitoring
- Daily benchmark execution
- Trend analysis for performance regression
- Evidence-based threshold enforcement
- Performance validation on every release

### Quality Monitoring
- Daily code quality analysis
- Dependency freshness tracking
- Security posture monitoring
- Documentation quality assessment

## 🚨 Troubleshooting

### Common Issues

#### Build Failures
```bash
# Check Python version compatibility
python --version

# Verify dependencies install cleanly
uv pip install -e ".[dev]"

# Run tests locally
pytest --cov=src/workspace_qdrant_mcp
```

#### Performance Test Failures
```bash
# Run simple benchmark locally
python simple_benchmark.py

# Check Qdrant connectivity
curl -f http://localhost:6333/health
```

#### Security Scan Failures
```bash
# Run security checks locally
bandit -r src/workspace_qdrant_mcp/
safety check
```

### Workflow Debugging

1. **Check workflow logs**: Go to Actions tab → Select failing workflow
2. **Review artifacts**: Download relevant report artifacts
3. **Local reproduction**: Run same commands locally
4. **Environment issues**: Check secret configuration
5. **Timing issues**: Consider Qdrant startup time

## 📈 Performance Optimization

### Pipeline Efficiency
- **Parallel execution**: Independent jobs run concurrently
- **Caching**: uv dependencies cached by Python version
- **Conditional execution**: Heavy benchmarks only on schedule/main
- **Artifact reuse**: Build once, test everywhere

### Resource Management
- **Service containers**: Qdrant only when needed
- **Timeout controls**: Prevent runaway processes
- **Memory profiling**: Track resource usage
- **Execution time limits**: 15-30 minute maximum per job

## 🔄 Maintenance

### Regular Tasks
- **Review benchmark results**: Ensure thresholds remain appropriate
- **Update dependencies**: Monitor Dependabot PRs
- **Security review**: Check daily security scan results
- **Performance analysis**: Review trend data monthly

### Workflow Updates
- **Version bumps**: Update action versions regularly
- **Security patches**: Apply promptly via Dependabot
- **Feature additions**: Follow atomic commit pattern
- **Threshold adjustments**: Based on evidence, not guesswork

## 📚 References

- **GitHub Actions Documentation**: https://docs.github.com/actions
- **uv Package Manager**: https://github.com/astral-sh/uv
- **Qdrant Vector Database**: https://qdrant.tech/
- **Evidence-Based Testing**: See `simple_benchmark.py` for methodology

---

**Note**: This pipeline implements evidence-based quality gates derived from comprehensive benchmark analysis. All thresholds are based on measured performance data, not arbitrary values.