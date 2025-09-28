# CRITICAL Configuration Analysis - PRDv3 Compliance Issues

**Date**: 2025-09-27 21:16
**Status**: CRITICAL - Configuration format completely wrong

## Core Problem Identified

The current configuration template does NOT match PRDv3.txt specifications. The agent created a configuration based on assumptions rather than the actual PRDv3 requirements.

## Critical Issues to Address

### 1. **CPU Threshold Ambiguity**
- **Problem**: `cpu_threshold_percent: 80.0` - ambiguous in multi-core world
- **Questions**:
  - Per-core threshold?
  - Total system CPU usage?
  - Average across all cores?
- **Required**: Clear definition and documentation

### 2. **Disk Threshold Ambiguity**
- **Problem**: `disk_threshold_percent: 90.0` - which disk?
- **Questions**:
  - Project files disk?
  - State database disk?
  - Qdrant data disk?
  - All disks?
- **Required**: Specify which disk(s) and purpose

### 3. **Hardcoded Path Issues**
- **Problem**: `./security_audit.log` - appears in project root
- **Problem**: `./workspace_daemon.db` - appears in project root
- **Required**: Use XDG/OS standard locations
- **Required**: Create `workspace_qdrant_mcp` folders automatically

### 4. **Unit Parsing Requirements**
- **Size Units**: Must support B/KB/MB/GB/TB AND B/K/M/G/T
- **Time Units**: Must support m/s/ms
- **Option Names**: Remove unit suffixes from option names
- **Example**: `max_file_size_mb` → `max_file_size: "100MB"`

### 5. **Hardcoded Elements in Config**
- **File Extensions**: Must be removed from config → assets
- **LSP Configs**: Must be removed from config → assets
- **Based on**: Extensive language research already conducted

### 6. **LSP Always Enabled**
- **LSP**: Always enabled, never configurable
- **Missing LSP**: Record in status table
- **Example**: "Missing `ruff-lsp` or not available in the path"

### 7. **Ambiguous Units**
- **Problem**: `default_chunk_size: 1000` - bytes? characters? tokens?
- **Required**: Explicit units for all size measurements

### 8. **Hardcoded File Names**
- **Log Files**: Should be hardcoded constants shared between Rust/Python
- **State DB**: Should be `state.db` (hardcoded)
- **Location**: Store constants in code assets

## Action Plan

### Phase 1: Read PRDv3.txt Correctly
1. Extract EXACT configuration format from PRDv3.txt
2. Identify ALL user-configurable vs embedded options
3. Map XDG directory requirements
4. Identify hardcoded constants

### Phase 2: Fix Configuration Template
1. Rebuild templates/default_config.yaml to match PRDv3.txt EXACTLY
2. Remove all hardcoded elements (file extensions, LSP configs)
3. Add proper unit parsing conventions
4. Use XDG standard locations
5. Add clear documentation for ambiguous settings

### Phase 3: Update Rust Configuration
1. Update config.rs to match new format
2. Implement unit parsing (B/KB/MB/GB/TB, m/s/ms)
3. Implement XDG directory resolution
4. Add status table for missing LSP servers

### Phase 4: Create Shared Constants
1. Create assets/constants.rs for hardcoded values
2. Share constants between Rust and Python
3. Define standard file names (state.db, log file names)

## Immediate Next Step

**STOP ALL WORK** and properly read PRDv3.txt to understand the ACTUAL configuration requirements, not assumptions.