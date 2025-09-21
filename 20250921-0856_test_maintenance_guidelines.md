# Test Maintenance Guidelines - Coverage Achievement Strategy

**Document:** Test Structure and Maintenance Procedures
**Date:** 2025-09-21 08:56
**Purpose:** Guidelines for maintaining and improving test coverage toward 100%

## Test Architecture Overview

### Current Test Structure
```
tests/
├── unit/                     # Component-level tests (single modules)
│   ├── test_memory_system.py    # ✅ 24/29 passing
│   ├── test_hybrid_search.py    # ⚠️ API compatibility issues
│   └── test_mcp_server_tools.py # Import structure issues
├── integration/              # Cross-component tests
│   ├── test_mcp_server_comprehensive.py  # ❌ Import errors
│   └── test_document_ingestion_pipeline.py  # ❌ Package structure
├── memory/                   # Memory system tests
│   ├── test_memory_manager.py    # ✅ 97/179 passing (54%)
│   ├── test_claude_integration.py # ⚠️ API changes
│   └── test_conflict_detector.py  # ❌ Missing dependencies
├── e2e/                     # End-to-end workflow tests
├── performance/             # Performance and benchmark tests
└── cli/                     # CLI component tests
    ├── test_comprehensive_cli.py  # ❌ Import structure mismatch
    └── parsers/            # Document parser tests
```

### Package Structure Alignment Issue

#### Current Problem
Tests expect imports like:
```python
from workspace_qdrant_mcp.cli.main import app, cli  # ❌ Not found
from workspace_qdrant_mcp.tools import memory_tools  # ❌ Wrong path
```

#### Actual Structure
```python
from wqm_cli.cli.main import app, cli  # ✅ Correct
from workspace_qdrant_mcp.tools import memory_tools  # ✅ Correct
from common.core.memory import MemoryManager  # ✅ Correct
```

## Coverage Improvement Workflow

### Phase 1: Foundation (Current: 3.83% → Target: 15%)

#### Step 1: Fix Import Structure
```bash
# 1. Update CLI test imports
find tests/ -name "*.py" -exec sed -i '' 's/from workspace_qdrant_mcp\.cli/from wqm_cli.cli/g' {} +

# 2. Update common module imports
find tests/ -name "*.py" -exec sed -i '' 's/from workspace_qdrant_mcp\.core/from common.core/g' {} +

# 3. Verify imports work
uv run python -c "from wqm_cli.cli.main import app; print('✅ CLI imports work')"
```

#### Step 2: Install Missing Dependencies
```bash
# Add missing test dependencies
uv add tiktoken anthropic
uv add --group dev pytest-playwright playwright

# Verify installations
uv run python -c "import tiktoken; print('✅ tiktoken available')"
uv run python -c "import anthropic; print('✅ anthropic available')"
```

#### Step 3: API Compatibility Updates
```python
# Fix RRFFusionRanker usage
# Old API:
ranker.explain_fusion(dense_weight=0.7)  # ❌ Unexpected argument

# New API:
ranker.explain_fusion(results_dense, results_sparse)  # ✅ Correct signature

# Fix KeywordIndexParams
# Old API:
params = KeywordIndexParams({})  # ❌ Missing required 'type' field

# New API:
params = KeywordIndexParams(type="keyword")  # ✅ Required field provided
```

### Phase 2: Component Coverage (Target: 15% → 50%)

#### Memory System Tests (Priority 1)
```bash
# Current status: 97/179 passing (54%)
# Target: 100% passing, expand coverage

# Fix memory collection naming
sed -i '' 's/memory_rules/memory/g' tests/memory/test_memory_schema.py

# Update MemoryRule initialization
# Add missing parameters: context, scope, priority
```

#### CLI Component Tests (Priority 2)
```bash
# Fix 51 failing test modules with import errors
# Update to use wqm_cli structure

# Test areas to cover:
# - Command parsing and validation
# - Configuration loading and validation
# - Error handling and user feedback
# - Help system and documentation
```

#### MCP Server Tools (Priority 3)
```bash
# Core tools that need comprehensive testing:
# - workspace_status: Basic health checks
# - search_workspace: Hybrid search functionality
# - get_server_info: Server capabilities and configuration
# - echo_test: Protocol validation
```

### Phase 3: Integration Coverage (Target: 50% → 80%)

#### Cross-Component Integration
```bash
# Test workflows that span multiple components:
# 1. Document ingestion → Vector storage → Search retrieval
# 2. CLI commands → MCP server → Qdrant operations
# 3. Configuration loading → Service initialization → Error handling
# 4. Project detection → Collection creation → Memory management
```

#### Error Handling and Edge Cases
```bash
# Comprehensive error scenario testing:
# - Network failures and retries
# - Invalid input validation
# - Resource exhaustion handling
# - Concurrent access scenarios
# - Configuration edge cases
```

## Test Execution Strategies

### Development Testing
```bash
# Quick unit test feedback (< 30 seconds)
uv run pytest tests/unit/ -v --tb=short

# Component-specific testing
uv run pytest tests/memory/ -v --cov=src/python/common/memory

# Integration testing with coverage
uv run pytest tests/integration/ -v --cov=src/python --cov-report=term-missing
```

### Coverage Measurement
```bash
# Full coverage analysis
uv run pytest --cov=src/python --cov-report=html --cov-report=xml --cov-branch

# Coverage of specific modules
uv run pytest tests/unit/test_memory_system.py --cov=src/python/common/memory --cov-report=term-missing

# Branch coverage analysis
uv run pytest --cov=src/python --cov-branch --cov-report=term-missing | grep -E "TOTAL|Missing"
```

### Performance Testing
```bash
# Performance regression testing
uv run pytest tests/performance/ --benchmark-only

# Memory usage validation
uv run pytest tests/performance/ -k "memory" --tb=short
```

## Coverage Quality Guidelines

### Line Coverage Standards
- **Critical Paths:** 100% line coverage required
- **Error Handling:** All exception paths must be tested
- **Configuration:** All config options and edge cases
- **API Endpoints:** Complete input/output validation

### Branch Coverage Standards
- **Conditional Logic:** All if/else branches tested
- **Loop Variations:** Empty, single, multiple iterations
- **Exception Handling:** Both success and failure paths
- **State Transitions:** All valid state changes covered

### Test Quality Metrics
```python
# Example of comprehensive test structure
class TestMemoryManager:
    """Memory manager component tests."""

    def test_initialization_success(self):
        """Test successful initialization."""
        # ✅ Happy path coverage
        pass

    def test_initialization_invalid_config(self):
        """Test initialization with invalid config."""
        # ✅ Error handling coverage
        pass

    def test_add_memory_rule_success(self):
        """Test successful rule addition."""
        # ✅ Core functionality coverage
        pass

    def test_add_memory_rule_duplicate(self):
        """Test duplicate rule handling."""
        # ✅ Edge case coverage
        pass

    def test_add_memory_rule_invalid_data(self):
        """Test invalid rule data handling."""
        # ✅ Input validation coverage
        pass

    @pytest.mark.parametrize("rule_type", ["note", "preference", "behavior"])
    def test_rule_type_variations(self, rule_type):
        """Test all supported rule types."""
        # ✅ Parametric coverage for variations
        pass
```

## CI/CD Integration

### Automated Coverage Enforcement
```yaml
# .github/workflows/ci.yml coverage configuration
- name: Run comprehensive test suite with coverage
  run: |
    pytest --cov=src/python --cov-report=xml --cov-report=html --cov-fail-under=10 --junitxml=pytest-results.xml

# Progressive threshold increases
# Week 1: --cov-fail-under=10  (current baseline)
# Week 2: --cov-fail-under=15  (import fixes)
# Week 3: --cov-fail-under=30  (component tests)
# Week 4: --cov-fail-under=50  (integration tests)
```

### Coverage Regression Prevention
```bash
# Pre-commit hook for coverage validation
#!/bin/bash
# .git/hooks/pre-commit
current_coverage=$(uv run coverage report | grep TOTAL | awk '{print $4}' | sed 's/%//')
if [ $current_coverage -lt 10 ]; then
    echo "❌ Coverage regression detected: ${current_coverage}% < 10%"
    exit 1
fi
echo "✅ Coverage check passed: ${current_coverage}%"
```

## Testing Best Practices

### Test Independence
```python
# ✅ Good: Each test is independent
def test_memory_rule_creation():
    manager = MemoryManager()  # Fresh instance
    rule = manager.add_rule("test content")
    assert rule.id is not None

# ❌ Bad: Tests depend on each other
rule_id = None  # Shared state
def test_create_rule():
    global rule_id
    rule_id = manager.add_rule("test")  # Modifies global state

def test_update_rule():
    manager.update_rule(rule_id, "new content")  # Depends on previous test
```

### Mock Usage Guidelines
```python
# ✅ Good: Mock external dependencies
@patch('qdrant_client.QdrantClient')
def test_memory_search(mock_qdrant):
    mock_qdrant.search.return_value = []
    manager = MemoryManager()
    results = manager.search("query")
    assert results == []

# ✅ Good: Mock unstable external services
@patch('anthropic.Anthropic')
def test_ai_analysis(mock_anthropic):
    mock_anthropic.completions.create.return_value = {"analysis": "test"}
    # Test logic without depending on external AI service
```

### Error Testing Patterns
```python
# ✅ Comprehensive error testing
def test_memory_search_qdrant_unavailable():
    with patch('qdrant_client.QdrantClient') as mock:
        mock.side_effect = ConnectionError("Qdrant unavailable")
        manager = MemoryManager()

        # Should handle gracefully, not crash
        results = manager.search("query")
        assert results == []  # Graceful degradation

def test_memory_search_invalid_query():
    manager = MemoryManager()

    # Test various invalid inputs
    assert manager.search("") == []        # Empty query
    assert manager.search(None) == []      # None query
    assert manager.search("x" * 10000) == []  # Oversized query
```

## Coverage Reporting and Analysis

### HTML Coverage Reports
```bash
# Generate interactive coverage browser
uv run pytest --cov=src/python --cov-report=html

# View detailed coverage analysis
open htmlcov/index.html

# Key metrics to monitor:
# - Line coverage percentage
# - Branch coverage percentage
# - Missing line numbers
# - Partially covered branches
```

### Coverage Trend Analysis
```python
# coverage_tracker.py - Track coverage over time
import json
from datetime import datetime

def record_coverage(coverage_pct):
    """Record coverage measurement with timestamp."""
    data = {
        "timestamp": datetime.now().isoformat(),
        "coverage": coverage_pct,
        "commit": os.environ.get("GITHUB_SHA", "unknown")
    }

    with open("coverage_history.json", "a") as f:
        f.write(json.dumps(data) + "\n")

# Usage in CI:
# coverage_pct=$(coverage report | grep TOTAL | awk '{print $4}' | sed 's/%//')
# python coverage_tracker.py $coverage_pct
```

## Maintenance Schedule

### Weekly Coverage Review
- **Monday:** Analyze coverage reports from weekend CI runs
- **Wednesday:** Review new feature test coverage requirements
- **Friday:** Update coverage thresholds if targets are consistently met

### Monthly Coverage Audit
- **Week 1:** Review test execution time and optimize slow tests
- **Week 2:** Analyze coverage gaps and plan improvement strategies
- **Week 3:** Update test infrastructure and dependencies
- **Week 4:** Performance testing and coverage validation

### Quarterly Coverage Assessment
- **Review:** Overall test architecture and coverage strategy
- **Update:** Coverage targets and quality standards
- **Plan:** Next quarter's coverage improvement initiatives

---
**Guidelines Updated:** 2025-09-21 08:56
**Next Review:** 2025-10-21
**Coverage Target:** 100% by 2025-12-21