## Description

**What does this PR do?**
<!-- Provide a clear, concise description of the changes -->

**Why is this needed?**
<!-- Explain the motivation, problem being solved, or requirement addressed -->

**How does it work?**
<!-- Brief technical overview of the approach taken -->

## Type of Change

<!-- Please check the one that applies to this PR using "x" -->

- [ ] 🐛 Bug fix (non-breaking change which fixes an issue)
- [ ] ✨ New feature (non-breaking change which adds functionality)
- [ ] 💥 Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] 📚 Documentation update
- [ ] 🧹 Code refactoring (no functional changes)
- [ ] ⚡ Performance improvement
- [ ] 🔧 Build/CI changes
- [ ] 🧪 Test changes
- [ ] 🔒 Security improvement

## Related Issue

<!-- If this PR addresses an issue, link it here -->
Fixes #(issue)

## Changes Made

<!-- List the specific technical changes made in this PR -->

### Core Changes
- 
- 
- 

### Supporting Changes
- 
- 

### Files Modified
- 
- 

## Testing Done

<!-- Describe the tests you ran to verify your changes -->

### Automated Testing
- [ ] Unit tests pass (`pytest tests/unit/`)
- [ ] Integration tests pass (`pytest tests/integration/`)
- [ ] End-to-end tests pass (`pytest tests/e2e/`)
- [ ] Performance benchmarks meet requirements (`pytest tests/benchmarks/`)
- [ ] All CI workflows pass (see checks above)

### Manual Testing
- [ ] Manual testing completed
- [ ] Edge cases tested
- [ ] Error conditions verified

### Test Coverage
- [ ] New code has ≥90% test coverage
- [ ] Critical paths have 100% coverage
- [ ] Regression tests added for bug fixes

### Test Evidence

<!-- Include test results, screenshots, or logs -->

## Performance Impact

<!-- Check all that apply and provide details below -->

- [ ] No performance impact expected
- [ ] Performance improvement expected
- [ ] Performance degradation possible (please explain below)
- [ ] Performance impact unknown (requires benchmarking)
- [ ] Benchmarks run and within acceptable thresholds

**Performance Requirements:**
- Search operations: <100ms average response time
- Memory usage: <200MB RSS during normal operations  
- Precision/Recall: Maintain or improve current metrics

<!-- If performance impact expected, provide benchmark results -->

**Benchmark Results:**
```
# Include relevant benchmark output or "N/A" if no performance impact
```

## Breaking Changes

<!-- If this is a breaking change, describe what breaks and how to migrate -->

## Documentation Updated

<!-- Check what documentation was updated -->

- [ ] README.md updated (if user-facing changes)
- [ ] API documentation updated (if API changes)
- [ ] CONTRIBUTING.md updated (if development process changes)
- [ ] Docstrings added/updated for new/modified functions
- [ ] Type hints added for all new functions
- [ ] Examples updated (if behavior changes)
- [ ] Migration guide provided (for breaking changes)

## Checklist

<!-- Please check all that apply using "x" -->

### Development Guidelines
- [ ] I have read and followed [CONTRIBUTING.md](CONTRIBUTING.md)
- [ ] Code follows [PEP 8](https://pep8.org/) style guidelines (Black formatting)
- [ ] All public functions have type hints and docstrings
- [ ] Self-review of code completed
- [ ] Comments added for complex logic
- [ ] No debugging code or print statements left in
- [ ] Imports are properly organized

### Quality Gates
- [ ] **Code Formatting**: `black src/ tests/` passes
- [ ] **Linting**: `ruff check src/ tests/` passes  
- [ ] **Type Checking**: `mypy src/` passes
- [ ] **Security Scan**: No new vulnerabilities introduced
- [ ] **Dependency Check**: New dependencies justified and documented

### Testing Requirements
- [ ] **Unit Tests**: New tests added for new functionality
- [ ] **Integration Tests**: Component interactions tested
- [ ] **Test Coverage**: ≥80% coverage for all new code
- [ ] **Performance Tests**: Benchmarks run and within thresholds
- [ ] **Edge Cases**: Error conditions and boundary values tested
- [ ] **Backwards Compatibility**: Existing functionality not broken

### CI/CD Integration
- [ ] **Quality Workflow**: `.github/workflows/quality.yml` passes
- [ ] **CI Workflow**: `.github/workflows/ci.yml` passes
- [ ] **Security Workflow**: `.github/workflows/security.yml` passes
- [ ] **Benchmark Workflow**: `.github/workflows/benchmark.yml` passes (if performance-related)
- [ ] **Branch Status**: Up to date with `main` branch
- [ ] **Conventional Commits**: All commit messages follow [conventional format](https://www.conventionalcommits.org/)

### Security Considerations
- [ ] Security implications considered and documented
- [ ] No sensitive data (keys, tokens, passwords) exposed
- [ ] Input validation implemented for user-facing functionality
- [ ] Dependencies reviewed for known vulnerabilities
- [ ] Authentication/authorization not bypassed
- [ ] Data access patterns follow principle of least privilege

## Breaking Changes

<!-- If this PR introduces breaking changes, describe them here -->

**⚠️ Breaking Changes (if any):**
- 
- 

**Migration Steps:**
<!-- Provide clear steps for users to migrate their code -->
1. 
2. 

## Additional Context

<!-- Any additional information that reviewers should know -->

### Related Issues
<!-- Link related issues -->
- Fixes #(issue)
- Related to #(issue)
- Addresses #(issue)

### Dependencies
<!-- Any dependencies on other PRs or external changes -->
- Depends on PR #(number)
- Requires external change: 

### Deployment Notes
<!-- Special deployment considerations -->
- [ ] Database migration required
- [ ] Configuration changes needed
- [ ] Service restart required
- [ ] Rollback plan documented

## Screenshots/Evidence

<!-- Add screenshots, logs, or GIFs to help explain your changes -->

---

## Reviewer Guidelines

**Before approving, please verify:**
- [ ] All checklist items are completed
- [ ] All CI workflows pass (Quality, CI, Security, Benchmarks)
- [ ] Code follows project conventions and [CONTRIBUTING.md](CONTRIBUTING.md)
- [ ] Test the changes locally if possible
- [ ] Security implications have been considered
- [ ] Performance impact is acceptable (check benchmarks)
- [ ] Documentation is comprehensive and accurate
- [ ] Breaking changes are justified and properly documented
- [ ] Commit messages follow [conventional commits](https://www.conventionalcommits.org/)

**Testing Checklist for Reviewers:**
- [ ] Manual testing of main functionality
- [ ] Edge cases and error conditions
- [ ] Performance under expected load
- [ ] Integration with existing components
- [ ] Backwards compatibility (unless breaking change)

**Review Focus Areas:**
- Code quality and maintainability
- Test coverage and test quality
- Security considerations
- Performance implications
- Documentation completeness
- Adherence to project standards