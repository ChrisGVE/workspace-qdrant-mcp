# Comprehensive Test Coverage Plan for workspace-qdrant-mcp

## Current Situation Analysis

- **Total Python source files**: 372
- **Current unit test files**: 221
- **Current coverage**: 2.06%
- **Root issue**: Import path problems preventing tests from running properly

## Critical Issues Identified

1. **Import Problems**: Tests fail due to relative import issues in proxy modules
2. **Path Configuration**: Tests not using correct Python path setup
3. **Mock Strategy**: Need comprehensive mocking for external dependencies
4. **Coverage Gaps**: Large modules with minimal actual test execution

## Phase 1: Core Module Testing Strategy

### Priority 1: Critical Core Modules
1. `common/core/client.py` - Main Qdrant workspace client
2. `common/core/hybrid_search.py` - Hybrid search engine with fusion algorithms
3. `common/core/memory.py` - Document memory management
4. `workspace_qdrant_mcp/server.py` - FastMCP server implementation

### Priority 2: Core Supporting Modules
5. `common/core/embeddings.py` - Embedding service integration
6. `common/core/collections.py` - Collection management
7. `common/core/config.py` - Configuration management
8. `common/core/metadata_schema.py` - Metadata handling

### Priority 3: Security and Utilities
9. `common/security/` modules - Access control, encryption, etc.
10. `common/utils/` modules - Utilities and validation
11. `wqm_cli/` modules - CLI interface components

## Testing Strategy

### Import Strategy
- Use direct imports from `common.core.*` modules
- Add `src/python` to Python path in tests
- Avoid importing through proxy modules in `workspace_qdrant_mcp`

### Mocking Strategy
- Mock external services: Qdrant, FastEmbed, Git operations
- Mock file system operations and network calls
- Mock async operations properly with AsyncMock
- Create comprehensive fixture sets

### Coverage Goals
- **Phase 1**: Achieve 60%+ coverage on core modules
- **Phase 2**: Achieve 80%+ coverage on supporting modules
- **Phase 3**: Achieve 90%+ overall coverage

## Implementation Plan

### Step 1: Fix Existing Tests
- Update import paths in existing test files
- Fix broken test configurations
- Ensure all existing tests can run

### Step 2: Create Comprehensive Core Tests
- client.py: Test initialization, project detection, collection operations
- hybrid_search.py: Test fusion algorithms, search engine, performance monitoring
- memory.py: Test document lifecycle, storage operations, retrieval
- server.py: Test MCP protocol, tool implementations, error handling

### Step 3: Expand Supporting Module Coverage
- Create tests for remaining core modules
- Add edge case and error condition testing
- Implement performance and load testing scenarios

### Step 4: Quality Assurance
- Validate actual code execution with coverage reports
- Ensure meaningful assertions, not just import tests
- Add integration test scenarios for critical workflows

## Test Structure Organization

```
tests/
├── unit/
│   ├── common/
│   │   ├── core/
│   │   │   ├── test_client.py
│   │   │   ├── test_hybrid_search.py
│   │   │   ├── test_memory.py
│   │   │   ├── test_embeddings.py
│   │   │   └── test_collections.py
│   │   ├── security/
│   │   └── utils/
│   └── workspace_qdrant_mcp/
│       ├── test_server.py
│       └── tools/
├── fixtures/
│   ├── mock_configs.py
│   ├── mock_qdrant.py
│   └── test_data.py
└── integration/
    └── test_end_to_end_workflows.py
```

## Success Metrics

1. **Coverage Metrics**: >90% line coverage on critical modules
2. **Test Quality**: All tests have meaningful assertions
3. **Error Coverage**: Both happy path and error scenarios tested
4. **Performance**: Tests run efficiently with proper mocking
5. **Maintainability**: Clear test structure and documentation