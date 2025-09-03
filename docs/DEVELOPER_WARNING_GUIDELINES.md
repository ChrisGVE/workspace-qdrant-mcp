# Developer Guidelines for Warning-Free Code

## Overview

This document establishes standards and practices for maintaining warning-free code across the workspace-qdrant-mcp project. These guidelines help prevent warning accumulation and ensure consistent code quality.

## Python Development Standards

### Type System Requirements

#### Mandatory Type Annotations
```python
# ✅ Correct: Complete function signatures
def process_documents(docs: list[dict[str, Any]], config: ProcessConfig) -> ProcessResult:
    """Process documents with full type safety."""
    return ProcessResult(processed=len(docs))

# ❌ Avoid: Missing type annotations  
def process_documents(docs, config):
    return ProcessResult(processed=len(docs))
```

#### Modern Type Syntax (Python 3.10+)
```python
# ✅ Correct: Modern union syntax
def get_user(user_id: str) -> User | None:
    """Retrieve user by ID."""
    return user_store.get(user_id)

# ✅ Correct: Built-in generics
def process_items(items: list[dict[str, Any]]) -> dict[str, list[int]]:
    """Process items with modern type annotations."""
    return {"processed": [len(item) for item in items]}

# ❌ Avoid: Legacy typing imports
from typing import List, Dict, Optional, Union
def get_user(user_id: str) -> Optional[User]:  # Use User | None
    return user_store.get(user_id)
```

#### Exception Handling Patterns
```python
# ✅ Correct: Proper exception chaining
try:
    result = risky_operation()
except ValueError as e:
    raise ProcessingError("Failed to process data") from e

# ✅ Correct: Suppress chaining when appropriate  
try:
    result = risky_operation()
except ValueError:
    raise ProcessingError("Failed to process data") from None

# ❌ Avoid: No exception chaining
try:
    result = risky_operation()
except ValueError as e:
    raise ProcessingError("Failed to process data")  # Missing 'from e'
```

### Code Organization Standards

#### Import Organization
```python
# Standard library imports
import asyncio
import json
from pathlib import Path

# Third-party imports  
import typer
from pydantic import BaseModel

# Local imports
from workspace_qdrant_mcp.core import Client
from workspace_qdrant_mcp.memory.types import MemoryRule
```

#### Variable Naming and Usage
```python
# ✅ Correct: Use descriptive names, avoid unused variables
for document in documents:
    process_document(document)

# ✅ Correct: Use underscore for unused variables
for document, _metadata in document_pairs:
    process_document(document)

# ❌ Avoid: Unused variables without underscore prefix
for document, metadata in document_pairs:  # metadata unused
    process_document(document)
```

### Pre-commit Requirements

Set up `.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.12.11
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy  
    rev: v1.17.1
    hooks:
      - id: mypy
        args: [--strict, --show-error-codes]
        additional_dependencies: [types-all]
```

## Rust Development Standards

### Zero Warnings Policy

All Rust code must compile without warnings:

```rust
// ✅ Correct: Clean compilation
#[derive(Debug, Clone)]
pub struct ProcessConfig {
    pub batch_size: usize,
    pub timeout_ms: u64,
}

impl ProcessConfig {
    pub fn new(batch_size: usize) -> Self {
        Self {
            batch_size,
            timeout_ms: 5000,
        }
    }
}

// ❌ Avoid: Code that generates warnings
#[derive(Debug, Clone)]  
pub struct ProcessConfig {
    pub batch_size: usize,
    pub timeout_ms: u64,
    unused_field: String,  // Dead code warning
}
```

### Trait Implementation Completeness

```rust
// ✅ Correct: Complete trait implementation
impl IngestService for IngestionService {
    type ProcessFolderStream = ResponseStream<ProcessFolderProgress>;
    
    async fn process_folder(
        &self,
        request: Request<ProcessFolderRequest>,
    ) -> Result<Response<Self::ProcessFolderStream>, Status> {
        // Complete implementation
        unimplemented!("TODO: Implement process_folder")
    }
    
    // ... all other required methods
}

// ❌ Avoid: Incomplete trait implementations
impl IngestService for IngestionService {
    // Missing required methods will cause compilation errors
}
```

### Error Handling Patterns

```rust
// ✅ Correct: Proper Result handling
pub async fn load_config(path: &Path) -> Result<Config, ConfigError> {
    let content = tokio::fs::read_to_string(path)
        .await
        .map_err(|e| ConfigError::FileRead(e.to_string()))?;
        
    serde_yaml::from_str(&content)
        .map_err(|e| ConfigError::Parse(e.to_string()))
}

// ✅ Correct: Clone for thread-safe sharing
let stats = {
    let guard = self.stats.read().await;
    (*guard).clone()  // Explicit dereference and clone
};
```

### Clippy Configuration

Add to `Cargo.toml`:
```toml
[workspace.lints.clippy]
# Deny common issues
unused_imports = "deny"
dead_code = "deny"
unused_variables = "deny"

# Warn on performance issues
redundant_clone = "warn"
inefficient_to_string = "warn"
large_enum_variant = "warn"

# Allow certain patterns where needed
too_many_arguments = "allow"
```

## IDE Integration

### VS Code Configuration

`.vscode/settings.json`:
```json
{
  "python.defaultInterpreter": "./venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.mypyEnabled": true,
  "python.linting.mypyArgs": ["--strict"],
  "python.formatting.provider": "none",
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.fixAll.ruff": true,
      "source.organizeImports.ruff": true
    }
  },
  "rust-analyzer.checkOnSave.command": "clippy",
  "rust-analyzer.checkOnSave.allTargets": true
}
```

## Code Review Checklist

### Python Code Review

#### Type Safety
- [ ] All functions have complete type annotations
- [ ] Uses modern type syntax (`dict`, `list`, `X | None`)
- [ ] No `Any` types without justification
- [ ] Generic types properly parameterized

#### Code Quality
- [ ] No unused imports or variables
- [ ] Exception handling uses proper chaining (`from e` or `from None`)
- [ ] Follows PEP 8 naming conventions
- [ ] Docstrings present for public APIs

#### Testing
- [ ] mypy --strict passes without errors
- [ ] ruff check passes with auto-fixes applied
- [ ] bandit security scan clean

### Rust Code Review

#### Compilation
- [ ] Code compiles without warnings
- [ ] All trait implementations complete
- [ ] No dead code or unused imports

#### Performance
- [ ] Clippy suggestions applied
- [ ] Appropriate use of references vs. owned values
- [ ] Async patterns used correctly for I/O

#### Safety
- [ ] No unsafe code without justification
- [ ] Proper error handling with Result types
- [ ] Thread safety considerations addressed

## Warning Budget System

### Tracking Metrics
Monitor warning counts in CI/CD:

```yaml
# GitHub Actions example
- name: Check Warning Budget
  run: |
    MYPY_ERRORS=$(mypy --strict src/ | grep "error:" | wc -l)
    RUFF_WARNINGS=$(ruff check src/ | grep "warning\|error" | wc -l)
    
    echo "Current mypy errors: $MYPY_ERRORS (budget: 1000)"
    echo "Current ruff warnings: $RUFF_WARNINGS (budget: 800)"
    
    if [ $MYPY_ERRORS -gt 1000 ]; then
      echo "❌ mypy error budget exceeded"
      exit 1
    fi
    
    if [ $RUFF_WARNINGS -gt 800 ]; then  
      echo "❌ ruff warning budget exceeded"
      exit 1
    fi
```

### Monthly Improvement Targets
- Reduce mypy errors by 100/month
- Reduce ruff warnings by 200/month  
- Maintain zero Rust compilation warnings

## Common Warning Patterns and Solutions

### Python Patterns

#### Missing Return Type
```python
# ❌ Warning: Function is missing a return type annotation
def process_data(data):
    return {"processed": True}

# ✅ Fixed: Add return type annotation
def process_data(data: dict[str, Any]) -> dict[str, bool]:
    return {"processed": True}
```

#### Legacy Type Imports
```python
# ❌ Warning: UP035 `typing.List` is deprecated
from typing import List, Dict, Optional

def process_items(items: Optional[List[Dict[str, Any]]]) -> None:
    pass

# ✅ Fixed: Use modern syntax
def process_items(items: list[dict[str, Any]] | None) -> None:
    pass
```

### Rust Patterns

#### Unused Variables
```rust
// ❌ Warning: unused variable 'config'
fn process_data(data: &[u8], config: &Config) {
    println!("Processing {} bytes", data.len());
}

// ✅ Fixed: Use underscore prefix
fn process_data(data: &[u8], _config: &Config) {
    println!("Processing {} bytes", data.len());
}
```

#### Clone Issues
```rust
// ❌ Error: no method named `clone` found for RwLockReadGuard
let stats = self.stats.read().await.clone();

// ✅ Fixed: Explicit dereference and clone
let stats = {
    let guard = self.stats.read().await;
    (*guard).clone()
};
```

## Emergency Procedures

### Critical Warning Accumulation
If warnings exceed emergency thresholds:

1. **Immediate Actions:**
   - Stop feature development
   - Focus team on warning elimination
   - Review recent commits for warning introduction

2. **Recovery Plan:**
   - Identify warning sources with detailed analysis
   - Create focused sprint for warning resolution
   - Implement stricter CI/CD gates

3. **Prevention:**
   - Review and strengthen development guidelines
   - Improve team training on warning prevention
   - Enhance tooling automation

### Build Failures
For Rust compilation failures:

1. **Immediate Response:**
   - Revert problematic commits if recent
   - Create hotfix branch for compilation issues
   - Implement minimal stub methods for missing traits

2. **Long-term Resolution:**
   - Complete trait implementations properly
   - Add integration tests to prevent regression
   - Review architecture for trait design issues

## Conclusion

These guidelines establish a framework for maintaining high code quality and preventing warning accumulation. Regular adherence to these standards ensures:

- Consistent code quality across the project
- Early detection of potential issues
- Improved maintainability and readability
- Better development team productivity

**Remember:** Warnings are indicators of potential issues. Address them promptly rather than allowing accumulation.

---
*Guidelines established as part of Task 74.3 - Warning Documentation and Resolution Validation*
*Last updated: 2025-09-03*