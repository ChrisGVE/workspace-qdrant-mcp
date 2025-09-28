# Daemon Diagnosis & Fix Plan - September 27, 2025

## 🔴 CRITICAL ISSUES IDENTIFIED

### Issue 1: Architectural Violation (BLOCKER)
- **Problem**: Static `project_path` in config violates PRD design
- **PRD Intent**: "MCP Server startup triggers daemon initialization" with dynamic project detection
- **Current**: Daemon has hardcoded path, no MCP server communication
- **Impact**: Breaks fundamental project-aware architecture

### Issue 2: Daemon Non-Functional (BLOCKER)
- **Problem**: Daemon running but not ingesting files (1/27,596 files processed)
- **Fixed**: Config format (YAML→TOML), service configuration
- **Still Broken**: File processing pipeline itself

### Issue 3: Missing MCP Server Integration
- **Problem**: No communication channel between MCP server and daemon
- **Impact**: Project detection happens in isolation

## 📋 SYSTEMATIC EXECUTION PLAN

### Phase 1: Deep Daemon Debugging (30 min)
**Objective**: Determine why daemon doesn't process files despite correct config

1. **Database Analysis**
   - Check daemon state database for watch configurations
   - Verify Qdrant connection status
   - Test manual file processing

2. **Connectivity Testing**
   - Verify daemon→Qdrant communication
   - Test gRPC/HTTP transport issues
   - Check collection creation permissions

3. **Manual Trigger Testing**
   - Force daemon to process a single file
   - Test if auto-ingestion logic is working
   - Identify specific failure point

### Phase 2: Architecture Fix (45 min)
**Objective**: Implement proper MCP server → daemon communication

1. **Remove Static Project Path**
   - Modify daemon to start without project_path
   - Implement "waiting for project assignment" state

2. **MCP Server Project Detection**
   - Add project detection on MCP server startup
   - Implement IPC call to daemon: `set_current_project(path)`

3. **Test Dynamic Project Switching**
   - Verify MCP server can control daemon project context
   - Test multi-project scenarios

### Phase 3: Integration Testing (15 min)
**Objective**: Verify end-to-end functionality

1. **Real Ingestion Test**
   - Start MCP server in project directory
   - Verify daemon receives project context
   - Confirm file ingestion works

2. **Collection Validation**
   - Check project-specific collections created
   - Verify search functionality works
   - Test file updates and monitoring

## 🎯 AGENT DEPLOYMENT STRATEGY

### Agent 1: Daemon Database Analyst
**Task**: Analyze daemon state and connectivity
**Deliverables**:
- Database state summary
- Connectivity test results
- Specific failure point identification

### Agent 2: Configuration Debugger
**Task**: Test manual daemon operations
**Deliverables**:
- Manual file processing test results
- Auto-ingestion pipeline analysis
- Error identification and fixes

### Agent 3: Architecture Implementer
**Task**: Implement MCP server → daemon communication
**Deliverables**:
- Remove static project_path from config
- Add project detection to MCP server
- Implement IPC communication

### Agent 4: Integration Validator
**Task**: End-to-end testing and validation
**Deliverables**:
- Full workflow test results
- Performance validation
- System readiness assessment

## ⚠️ AGENT INSTRUCTION GUIDELINES

1. **Be Specific**: Each agent gets exact scope and deliverables
2. **Test Incrementally**: Verify each change before proceeding
3. **Document Findings**: Clear evidence of what works/doesn't work
4. **Stop on Blockers**: Don't proceed if dependencies aren't met
5. **Report Back**: Provide concrete results, not assumptions

## 🎖️ SUCCESS CRITERIA

- [ ] Daemon processes files from current directory automatically
- [ ] MCP server can dynamically set daemon project context
- [ ] File ingestion shows significant progress (>100 files)
- [ ] Collections created properly for current project
- [ ] Search functionality works with ingested content
- [ ] System architecture matches PRD specifications

## 📊 CURRENT STATUS
- **Files Ingested**: 1/27,596 (0.004%)
- **Daemon State**: Running with TOML config
- **Config Issues**: Fixed (YAML→TOML, service plist)
- **Architecture**: Violates PRD (static vs dynamic project detection)
- **Next Action**: Deploy Agent 1 for database analysis