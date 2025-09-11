# Comprehensive MCP Testing Campaign Plan

## Current Status
- ✅ Async bugs fixed - no more await expression errors
- ✅ Collection configuration partially working  
- ❌ Still some collection creation issues identified in server logs
- ❌ MCP tools not available in current Claude session (connection issue)

## Identified Issues to Fix During Testing
1. **Wrong collection name**: Server creating `references` instead of `reference`
2. **Missing collection**: `workspace-qdrant-mcp-repo` not being created  
3. **Collection info warnings**: `"'dict' object has no attribute 'distance'"` errors
4. **Auto-ingestion error**: `"'PosixPath' object has no attribute 'is_readable'"`

## Testing Strategy: Test → Fix → Test Protocol
1. Test each tool systematically
2. When bugs are found, deploy specialized agents to fix them
3. Re-test to verify fixes work
4. Continue with next tools
5. Create comprehensive test reports

## Phase 2: Workspace Management Testing (8 tools)

### Test 1: workspace_status
**Purpose**: Get system health and connection status
**Command**: `workspace_qdrant_mcp__workspace_status`
**Expected**: System status, collections info, server health
**Test Steps**:
```bash
# Via MCP
workspace_status

# Via HTTP API
curl -X POST http://127.0.0.1:8000/api/tools \
  -H "Content-Type: application/json" \
  -d '{"name": "workspace_status"}'

# Via CLI
python -m workspace_qdrant_mcp.cli workspace-status --config workspace_qdrant_config.yaml
```
**Success Criteria**: Returns valid status with no errors

### Test 2: list_workspace_collections  
**Purpose**: Verify collection detection and naming
**Command**: `workspace_qdrant_mcp__list_workspace_collections`
**Expected**: List of all collections with correct names
**Test Steps**:
```bash
# Via MCP
list_workspace_collections

# Via HTTP API
curl -X POST http://127.0.0.1:8000/api/tools \
  -H "Content-Type: application/json" \
  -d '{"name": "list_workspace_collections"}'
```
**Success Criteria**: 
- Shows `reference` collection (not `references`)
- Shows `workspace-qdrant-mcp-repo` collection
- No distance attribute errors

### Test 3: search_workspace_tool
**Purpose**: Basic search functionality across collections
**Command**: `workspace_qdrant_mcp__search_workspace_tool`
**Parameters**: `{"query": "test query", "limit": 5}`
**Test Steps**:
```bash
# Via MCP
search_workspace_tool(query="test query", limit=5)

# Via HTTP API
curl -X POST http://127.0.0.1:8000/api/tools \
  -H "Content-Type: application/json" \
  -d '{"name": "search_workspace_tool", "arguments": {"query": "test query", "limit": 5}}'
```
**Success Criteria**: Returns search results without errors

### Test 4: add_document_tool
**Purpose**: Document addition functionality
**Command**: `workspace_qdrant_mcp__add_document_tool`
**Parameters**: 
```json
{
  "content": "Test document for Phase 2 testing",
  "metadata": {
    "test_phase": "phase_2",
    "test_type": "add_document",
    "timestamp": "2025-01-04T12:00:00Z"
  },
  "collection": "reference"
}
```
**Test Steps**:
```bash
# Via MCP
add_document_tool(content="Test document", metadata={...}, collection="reference")

# Via HTTP API
curl -X POST http://127.0.0.1:8000/api/tools \
  -H "Content-Type: application/json" \
  -d '{"name": "add_document_tool", "arguments": {...}}'
```
**Success Criteria**: Document added successfully, returns document ID

### Test 5: get_document_tool
**Purpose**: Document retrieval functionality  
**Command**: `workspace_qdrant_mcp__get_document_tool`
**Parameters**: `{"document_id": "<id_from_test_4>", "collection": "reference"}`
**Test Steps**:
```bash
# Use document ID from Test 4
get_document_tool(document_id="...", collection="reference")
```
**Success Criteria**: Retrieves exact document added in Test 4

### Test 6: search_by_metadata_tool
**Purpose**: Metadata-based search
**Command**: `workspace_qdrant_mcp__search_by_metadata_tool`
**Parameters**: 
```json
{
  "metadata_filter": {"test_phase": "phase_2"},
  "collection": "reference",
  "limit": 10
}
```
**Test Steps**:
```bash
search_by_metadata_tool(metadata_filter={"test_phase": "phase_2"}, collection="reference")
```
**Success Criteria**: Returns test document from Test 4

### Test 7: research_workspace
**Purpose**: Advanced research functionality
**Command**: `workspace_qdrant_mcp__research_workspace`
**Parameters**: 
```json
{
  "research_question": "What test documents are available in phase 2 testing?",
  "max_results": 5
}
```
**Test Steps**:
```bash
research_workspace(research_question="...", max_results=5)
```
**Success Criteria**: Provides coherent research results

### Test 8: hybrid_search_advanced_tool
**Purpose**: Advanced hybrid search capabilities
**Command**: `workspace_qdrant_mcp__hybrid_search_advanced_tool`
**Parameters**:
```json
{
  "query": "test document phase 2",
  "collections": ["reference"],
  "limit": 5,
  "use_reranking": true,
  "metadata_filter": {"test_phase": "phase_2"}
}
```
**Test Steps**:
```bash
hybrid_search_advanced_tool(query="test document phase 2", collections=["reference"], ...)
```
**Success Criteria**: Returns relevant results with reranking applied

## Bug Fixing Protocol

### When Tests Fail:
1. **Capture Error Details**: Log exact error message, stack trace, parameters used
2. **Deploy Specialized Agent**: Use appropriate agent (backend-developer, test-automator, etc.)
3. **Apply Targeted Fix**: Fix specific issue identified  
4. **Verify Fix**: Re-run failing test to confirm resolution
5. **Regression Test**: Run previous successful tests to ensure no regression
6. **Document Fix**: Update test report with fix details

### Agent Deployment Strategy:
- **Collection Issues**: → backend-developer (database/collection management)
- **API Errors**: → api-developer (endpoint and request handling)
- **Search Problems**: → backend-developer (search and indexing)
- **Config Issues**: → devops-engineer (configuration management)
- **Test Framework**: → test-automator (test infrastructure)

## Phase 2 Test Report Template

```markdown
# Phase 2 Test Results - Workspace Management

## Test Summary
- **Tests Executed**: 8/8
- **Tests Passed**: X/8  
- **Tests Failed**: Y/8
- **Critical Issues Found**: Z
- **Bugs Fixed**: A
- **Regression Issues**: B

## Individual Test Results

### Test 1: workspace_status
- **Status**: ✅ PASS / ❌ FAIL
- **Error Details**: [if failed]
- **Fix Applied**: [if fixed]
- **Notes**: [observations]

[... repeat for all 8 tests ...]

## Issues Identified and Fixed
1. **Issue**: [description]
   - **Agent Used**: [agent-type]
   - **Fix Applied**: [solution]
   - **Status**: ✅ Fixed / ❌ Pending

## Recommendations for Phase 3
- [Based on findings from Phase 2]
- [Performance observations]
- [Architecture improvements needed]
```

## Next Steps After Phase 2
1. Complete Phase 2 testing with full bug fixing
2. Generate comprehensive Phase 2 report
3. Apply any architectural improvements identified  
4. Proceed to Phase 3: Document Management (12 tools)
5. Continue systematic testing through all 6 phases

## Tools Required for Testing
- MCP connection to workspace-qdrant-mcp server
- HTTP API access (fallback method)
- CLI access for direct testing
- Bash/curl for HTTP testing
- jq for JSON parsing
- Test result logging capability

## Success Metrics
- All 8 Phase 2 tools working without errors
- Collection naming issues resolved
- Auto-ingestion path issues fixed  
- Clean server logs with no warnings
- Consistent results across MCP, HTTP, and CLI interfaces