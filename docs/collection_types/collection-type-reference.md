[38;2;127;132;156m   1[0m [38;2;205;214;244m# Collection Type Reference[0m
[38;2;127;132;156m   2[0m 
[38;2;127;132;156m   3[0m [38;2;205;214;244mDetailed reference for all collection types with behaviors, requirements, and examples.[0m
[38;2;127;132;156m   4[0m 
[38;2;127;132;156m   5[0m [38;2;205;214;244m## Overview[0m
[38;2;127;132;156m   6[0m 
[38;2;127;132;156m   7[0m [38;2;205;214;244mFour collection types provide different behaviors optimized for specific use cases:[0m
[38;2;127;132;156m   8[0m 
[38;2;127;132;156m   9[0m [38;2;205;214;244m| Type | Pattern | Deletion | Primary Use |[0m
[38;2;127;132;156m  10[0m [38;2;205;214;244m|------|---------|----------|-------------|[0m
[38;2;127;132;156m  11[0m [38;2;205;214;244m| SYSTEM | `__name` | Cumulative | System configuration |[0m
[38;2;127;132;156m  12[0m [38;2;205;214;244m| LIBRARY | `_name` | Cumulative | Language documentation |[0m
[38;2;127;132;156m  13[0m [38;2;205;214;244m| PROJECT | `{id}-{suffix}` | Dynamic | Project content |[0m
[38;2;127;132;156m  14[0m [38;2;205;214;244m| GLOBAL | Fixed | Dynamic | Cross-project resources |[0m
[38;2;127;132;156m  15[0m 
[38;2;127;132;156m  16[0m [38;2;205;214;244m## SYSTEM Collections[0m
[38;2;127;132;156m  17[0m 
[38;2;127;132;156m  18[0m [38;2;205;214;244m### Naming Convention[0m
[38;2;127;132;156m  19[0m 
[38;2;127;132;156m  20[0m [38;2;205;214;244m**Pattern**: `__[a-zA-Z0-9_-]+`[0m
[38;2;127;132;156m  21[0m 
[38;2;127;132;156m  22[0m [38;2;205;214;244m**Examples**:[0m
[38;2;127;132;156m  23[0m [38;2;205;214;244m- `__system_docs`[0m
[38;2;127;132;156m  24[0m [38;2;205;214;244m- `__cli_config`[0m
[38;2;127;132;156m  25[0m [38;2;205;214;244m- `__admin_state`[0m
[38;2;127;132;156m  26[0m [38;2;205;214;244m- `__error_tracking`[0m
[38;2;127;132;156m  27[0m 
[38;2;127;132;156m  28[0m [38;2;205;214;244m**Invalid**:[0m
[38;2;127;132;156m  29[0m [38;2;205;214;244m- `_system` (single underscore)[0m
[38;2;127;132;156m  30[0m [38;2;205;214;244m- `system__docs` (not at start)[0m
[38;2;127;132;156m  31[0m [38;2;205;214;244m- `__a` (too short, min 3 chars)[0m
[38;2;127;132;156m  32[0m 
[38;2;127;132;156m  33[0m [38;2;205;214;244m### Deletion Handling[0m
[38;2;127;132;156m  34[0m 
[38;2;127;132;156m  35[0m [38;2;205;214;244m**Mode**: CUMULATIVE[0m
[38;2;127;132;156m  36[0m 
[38;2;127;132;156m  37[0m [38;2;205;214;244mFiles are marked as deleted and cleaned up in batches:[0m
[38;2;127;132;156m  38[0m [38;2;205;214;244m1. File deleted → Marked in `cumulative_deletions_queue`[0m
[38;2;127;132;156m  39[0m [38;2;205;214;244m2. Batch cleanup runs every 24h or when 1000 items queued[0m
[38;2;127;132;156m  40[0m [38;2;205;214;244m3. Actual deletion from Qdrant in batch[0m
[38;2;127;132;156m  41[0m [38;2;205;214;244m4. Queue entry marked as deleted[0m
[38;2;127;132;156m  42[0m 
[38;2;127;132;156m  43[0m [38;2;205;214;244m**Benefits**:[0m
[38;2;127;132;156m  44[0m [38;2;205;214;244m- Better performance (batch operations)[0m
[38;2;127;132;156m  45[0m [38;2;205;214;244m- Consistency (atomic batch cleanup)[0m
[38;2;127;132;156m  46[0m [38;2;205;214;244m- Rollback possible (deletions tracked)[0m
[38;2;127;132;156m  47[0m 
[38;2;127;132;156m  48[0m [38;2;205;214;244m**Trade-offs**:[0m
[38;2;127;132;156m  49[0m [38;2;205;214;244m- Deletion not immediate (up to 24h delay)[0m
[38;2;127;132;156m  50[0m [38;2;205;214;244m- Storage overhead for deletion queue[0m
[38;2;127;132;156m  51[0m 
[38;2;127;132;156m  52[0m [38;2;205;214;244m### Required Metadata[0m
[38;2;127;132;156m  53[0m 
[38;2;127;132;156m  54[0m [38;2;205;214;244m```yaml[0m
[38;2;127;132;156m  55[0m [38;2;205;214;244mcollection_name: "__system_docs"  # Pattern: ^__[a-zA-Z0-9_-]+$[0m
[38;2;127;132;156m  56[0m [38;2;205;214;244mcreated_at: "2025-01-03T10:00:00Z"[0m
[38;2;127;132;156m  57[0m [38;2;205;214;244mcollection_category: "system"[0m
[38;2;127;132;156m  58[0m [38;2;205;214;244m```[0m
[38;2;127;132;156m  59[0m 
[38;2;127;132;156m  60[0m [38;2;205;214;244m### Optional Metadata[0m
[38;2;127;132;156m  61[0m 
[38;2;127;132;156m  62[0m [38;2;205;214;244m```yaml[0m
[38;2;127;132;156m  63[0m [38;2;205;214;244mupdated_at: "2025-01-03T12:00:00Z"[0m
[38;2;127;132;156m  64[0m [38;2;205;214;244mdescription: "System documentation"[0m
[38;2;127;132;156m  65[0m [38;2;205;214;244mcli_writable: true  # Default: true[0m
[38;2;127;132;156m  66[0m [38;2;205;214;244m```[0m
[38;2;127;132;156m  67[0m 
[38;2;127;132;156m  68[0m [38;2;205;214;244m### Access Control[0m
[38;2;127;132;156m  69[0m 
[38;2;127;132;156m  70[0m [38;2;205;214;244m- **CLI**: Write access (create, update, delete)[0m
[38;2;127;132;156m  71[0m [38;2;205;214;244m- **MCP/LLM**: Read-only access[0m
[38;2;127;132;156m  72[0m [38;2;205;214;244m- **Global Search**: Not included[0m
[38;2;127;132;156m  73[0m [38;2;205;214;244m- **Tenant Isolation**: Enabled[0m
[38;2;127;132;156m  74[0m 
[38;2;127;132;156m  75[0m [38;2;205;214;244m### Performance Settings[0m
[38;2;127;132;156m  76[0m 
[38;2;127;132;156m  77[0m [38;2;205;214;244m- Batch size: 50 (smaller for reliability)[0m
[38;2;127;132;156m  78[0m [38;2;205;214;244m- Max concurrent: 3 (avoid contention)[0m
[38;2;127;132;156m  79[0m [38;2;205;214;244m- Priority: 4/5 (high priority)[0m
[38;2;127;132;156m  80[0m [38;2;205;214;244m- Cache TTL: 600s (10 minutes)[0m
[38;2;127;132;156m  81[0m 
[38;2;127;132;156m  82[0m [38;2;205;214;244m### Use Cases[0m
[38;2;127;132;156m  83[0m 
[38;2;127;132;156m  84[0m [38;2;205;214;244m1. **CLI Configuration**: Settings for CLI commands[0m
[38;2;127;132;156m  85[0m [38;2;205;214;244m2. **Admin State**: Administrative tracking[0m
[38;2;127;132;156m  86[0m [38;2;205;214;244m3. **System Logs**: Internal logging (rotate regularly)[0m
[38;2;127;132;156m  87[0m [38;2;205;214;244m4. **Error Messages**: Error tracking and reporting[0m
[38;2;127;132;156m  88[0m 
[38;2;127;132;156m  89[0m [38;2;205;214;244m### Example[0m
[38;2;127;132;156m  90[0m 
[38;2;127;132;156m  91[0m [38;2;205;214;244mSee [system-collection-example.yaml](examples/system-collection-example.yaml)[0m
[38;2;127;132;156m  92[0m 
[38;2;127;132;156m  93[0m [38;2;205;214;244m---[0m
[38;2;127;132;156m  94[0m 
[38;2;127;132;156m  95[0m [38;2;205;214;244m## LIBRARY Collections[0m
[38;2;127;132;156m  96[0m 
[38;2;127;132;156m  97[0m [38;2;205;214;244m### Naming Convention[0m
[38;2;127;132;156m  98[0m 
[38;2;127;132;156m  99[0m [38;2;205;214;244m**Pattern**: `_[a-zA-Z0-9_-]+` (single underscore, NOT double)[0m
[38;2;127;132;156m 100[0m 
[38;2;127;132;156m 101[0m [38;2;205;214;244m**Examples**:[0m
[38;2;127;132;156m 102[0m [38;2;205;214;244m- `_python_stdlib`[0m
[38;2;127;132;156m 103[0m [38;2;205;214;244m- `_react_docs`[0m
[38;2;127;132;156m 104[0m [38;2;205;214;244m- `_rust_std`[0m
[38;2;127;132;156m 105[0m [38;2;205;214;244m- `_node_api`[0m
[38;2;127;132;156m 106[0m 
[38;2;127;132;156m 107[0m [38;2;205;214;244m**Invalid**:[0m
[38;2;127;132;156m 108[0m [38;2;205;214;244m- `__python` (double underscore - that's SYSTEM)[0m
[38;2;127;132;156m 109[0m [38;2;205;214;244m- `python_stdlib` (no underscore prefix)[0m
[38;2;127;132;156m 110[0m [38;2;205;214;244m- `_py` (too short, min 2 chars after _)[0m
[38;2;127;132;156m 111[0m 
[38;2;127;132;156m 112[0m [38;2;205;214;244m### Deletion Handling[0m
[38;2;127;132;156m 113[0m 
[38;2;127;132;156m 114[0m [38;2;205;214;244m**Mode**: CUMULATIVE[0m
[38;2;127;132;156m 115[0m 
[38;2;127;132;156m 116[0m [38;2;205;214;244mSame as SYSTEM collections - marked then batch cleanup.[0m
[38;2;127;132;156m 117[0m 
[38;2;127;132;156m 118[0m [38;2;205;214;244m**Rationale**: Library docs rarely deleted, but when they are:[0m
[38;2;127;132;156m 119[0m [38;2;205;214;244m- Batch processing more efficient[0m
[38;2;127;132;156m 120[0m [38;2;205;214;244m- Allows validation before permanent deletion[0m
[38;2;127;132;156m 121[0m [38;2;205;214;244m- Can rollback if library update fails[0m
[38;2;127;132;156m 122[0m 
[38;2;127;132;156m 123[0m [38;2;205;214;244m### Required Metadata[0m
[38;2;127;132;156m 124[0m 
[38;2;127;132;156m 125[0m [38;2;205;214;244m```yaml[0m
[38;2;127;132;156m 126[0m [38;2;205;214;244mcollection_name: "_python_stdlib"  # Pattern: ^_[a-zA-Z0-9_-]+$[0m
[38;2;127;132;156m 127[0m [38;2;205;214;244mcreated_at: "2025-01-03T10:00:00Z"[0m
[38;2;127;132;156m 128[0m [38;2;205;214;244mcollection_category: "library"[0m
[38;2;127;132;156m 129[0m [38;2;205;214;244mlanguage: "python"  # REQUIRED for LIBRARY[0m
[38;2;127;132;156m 130[0m [38;2;205;214;244m```[0m
[38;2;127;132;156m 131[0m 
[38;2;127;132;156m 132[0m [38;2;205;214;244m### Optional Metadata[0m
[38;2;127;132;156m 133[0m 
[38;2;127;132;156m 134[0m [38;2;205;214;244m```yaml[0m
[38;2;127;132;156m 135[0m [38;2;205;214;244mupdated_at: "2025-01-03T12:00:00Z"[0m
[38;2;127;132;156m 136[0m [38;2;205;214;244mdescription: "Python standard library"[0m
[38;2;127;132;156m 137[0m [38;2;205;214;244msymbols: ["os.path", "sys.argv"]  # Extracted via LSP[0m
[38;2;127;132;156m 138[0m [38;2;205;214;244mdependencies: ["setuptools>=40.0"][0m
[38;2;127;132;156m 139[0m [38;2;205;214;244mversion: "3.12.0"[0m
[38;2;127;132;156m 140[0m [38;2;205;214;244mmcp_readonly: true  # Default: true[0m
[38;2;127;132;156m 141[0m [38;2;205;214;244m```[0m
[38;2;127;132;156m 142[0m 
[38;2;127;132;156m 143[0m [38;2;205;214;244m### LSP Integration[0m
[38;2;127;132;156m 144[0m 
[38;2;127;132;156m 145[0m [38;2;205;214;244m**Language Detection**:[0m
[38;2;127;132;156m 146[0m [38;2;205;214;244m- From file extension: `.py` → `python`[0m
[38;2;127;132;156m 147[0m [38;2;205;214;244m- From metadata: `language` field[0m
[38;2;127;132;156m 148[0m [38;2;205;214;244m- From LSP server detection[0m
[38;2;127;132;156m 149[0m 
[38;2;127;132;156m 150[0m [38;2;205;214;244m**Metadata Extraction**:[0m
[38;2;127;132;156m 151[0m [38;2;205;214;244mWhen LSP available for language:[0m
[38;2;127;132;156m 152[0m [38;2;205;214;244m1. Extract symbols (functions, classes)[0m
[38;2;127;132;156m 153[0m [38;2;205;214;244m2. Extract dependencies (imports)[0m
[38;2;127;132;156m 154[0m [38;2;205;214;244m3. Extract documentation (docstrings)[0m
[38;2;127;132;156m 155[0m [38;2;205;214;244m4. Extract type information[0m
[38;2;127;132;156m 156[0m 
[38;2;127;132;156m 157[0m [38;2;205;214;244mWhen LSP unavailable:[0m
[38;2;127;132;156m 158[0m [38;2;205;214;244m- File queued to `missing_metadata_queue`[0m
[38;2;127;132;156m 159[0m [38;2;205;214;244m- Processed when LSP becomes available[0m
[38;2;127;132;156m 160[0m [38;2;205;214;244m- Falls back to Tree-sitter if configured[0m
[38;2;127;132;156m 161[0m 
[38;2;127;132;156m 162[0m [38;2;205;214;244m### Access Control[0m
[38;2;127;132;156m 163[0m 
[38;2;127;132;156m 164[0m [38;2;205;214;244m- **CLI**: Write access[0m
[38;2;127;132;156m 165[0m [38;2;205;214;244m- **MCP/LLM**: Read-only[0m
[38;2;127;132;156m 166[0m [38;2;205;214;244m- **Global Search**: Included[0m
[38;2;127;132;156m 167[0m [38;2;205;214;244m- **Tenant Isolation**: Not required (global libraries)[0m
[38;2;127;132;156m 168[0m 
[38;2;127;132;156m 169[0m [38;2;205;214;244m### Performance Settings[0m
[38;2;127;132;156m 170[0m 
[38;2;127;132;156m 171[0m [38;2;205;214;244m- Batch size: 100 (larger for many docs)[0m
[38;2;127;132;156m 172[0m [38;2;205;214;244m- Max concurrent: 5[0m
[38;2;127;132;156m 173[0m [38;2;205;214;244m- Priority: 3/5 (medium)[0m
[38;2;127;132;156m 174[0m [38;2;205;214;244m- Cache TTL: 900s (15 minutes)[0m
[38;2;127;132;156m 175[0m 
[38;2;127;132;156m 176[0m [38;2;205;214;244m### Use Cases[0m
[38;2;127;132;156m 177[0m 
[38;2;127;132;156m 178[0m [38;2;205;214;244m1. **Standard Libraries**: Language stdlib documentation[0m
[38;2;127;132;156m 179[0m [38;2;205;214;244m2. **Third-Party Libraries**: npm, PyPI, crates.io packages[0m
[38;2;127;132;156m 180[0m [38;2;205;214;244m3. **API Documentation**: Framework/SDK references[0m
[38;2;127;132;156m 181[0m [38;2;205;214;244m4. **Code Examples**: Library usage examples[0m
[38;2;127;132;156m 182[0m 
[38;2;127;132;156m 183[0m [38;2;205;214;244m### Example[0m
[38;2;127;132;156m 184[0m 
[38;2;127;132;156m 185[0m [38;2;205;214;244mSee [library-collection-example.yaml](examples/library-collection-example.yaml)[0m
[38;2;127;132;156m 186[0m 
[38;2;127;132;156m 187[0m [38;2;205;214;244m---[0m
[38;2;127;132;156m 188[0m 
[38;2;127;132;156m 189[0m [38;2;205;214;244m## PROJECT Collections[0m
[38;2;127;132;156m 190[0m 
[38;2;127;132;156m 191[0m [38;2;205;214;244m### Naming Convention[0m
[38;2;127;132;156m 192[0m 
[38;2;127;132;156m 193[0m [38;2;205;214;244m**Pattern**: `{project_id}-{suffix}`[0m
[38;2;127;132;156m 194[0m 
[38;2;127;132;156m 195[0m [38;2;205;214;244mWhere:[0m
[38;2;127;132;156m 196[0m [38;2;205;214;244m- `project_id`: 12-character hash (a-zA-Z0-9)[0m
[38;2;127;132;156m 197[0m [38;2;205;214;244m- `suffix`: Descriptive type (docs, memory, notes, etc.)[0m
[38;2;127;132;156m 198[0m 
[38;2;127;132;156m 199[0m [38;2;205;214;244m**Examples**:[0m
[38;2;127;132;156m 200[0m [38;2;205;214;244m- `a1b2c3d4e5f6-docs`[0m
[38;2;127;132;156m 201[0m [38;2;205;214;244m- `abc123def456-memory`[0m
[38;2;127;132;156m 202[0m [38;2;205;214;244m- `xyz789uvw012-notes`[0m
[38;2;127;132;156m 203[0m 
[38;2;127;132;156m 204[0m [38;2;205;214;244m**Invalid**:[0m
[38;2;127;132;156m 205[0m [38;2;205;214;244m- `myproject-docs` (project_id not 12 chars)[0m
[38;2;127;132;156m 206[0m [38;2;205;214;244m- `a1b2c3d4e5f6` (no suffix)[0m
[38;2;127;132;156m 207[0m [38;2;205;214;244m- `_a1b2-docs` (has underscore prefix)[0m
[38;2;127;132;156m 208[0m 
[38;2;127;132;156m 209[0m [38;2;205;214;244m### Deletion Handling[0m
[38;2;127;132;156m 210[0m 
[38;2;127;132;156m 211[0m [38;2;205;214;244m**Mode**: DYNAMIC (immediate)[0m
[38;2;127;132;156m 212[0m 
[38;2;127;132;156m 213[0m [38;2;205;214;244mFiles deleted from filesystem are immediately removed from Qdrant:[0m
[38;2;127;132;156m 214[0m [38;2;205;214;244m1. File deleted from disk[0m
[38;2;127;132;156m 215[0m [38;2;205;214;244m2. File watcher detects deletion[0m
[38;2;127;132;156m 216[0m [38;2;205;214;244m3. Deletion queued for processing[0m
[38;2;127;132;156m 217[0m [38;2;205;214;244m4. Point removed from Qdrant immediately[0m
[38;2;127;132;156m 218[0m [38;2;205;214;244m5. Collection reflects file system state[0m
[38;2;127;132;156m 219[0m 
[38;2;127;132;156m 220[0m [38;2;205;214;244m**Benefits**:[0m
[38;2;127;132;156m 221[0m [38;2;205;214;244m- Real-time synchronization[0m
[38;2;127;132;156m 222[0m [38;2;205;214;244m- Collection always matches filesystem[0m
[38;2;127;132;156m 223[0m [38;2;205;214;244m- No cleanup lag[0m
[38;2;127;132;156m 224[0m 
[38;2;127;132;156m 225[0m [38;2;205;214;244m**Trade-offs**:[0m
[38;2;127;132;156m 226[0m [38;2;205;214;244m- Higher deletion rate[0m
[38;2;127;132;156m 227[0m [38;2;205;214;244m- No batch optimization[0m
[38;2;127;132;156m 228[0m [38;2;205;214;244m- Cannot rollback easily[0m
[38;2;127;132;156m 229[0m 
[38;2;127;132;156m 230[0m [38;2;205;214;244m### Required Metadata[0m
[38;2;127;132;156m 231[0m 
[38;2;127;132;156m 232[0m [38;2;205;214;244m```yaml[0m
[38;2;127;132;156m 233[0m [38;2;205;214;244mproject_name: "workspace-qdrant-mcp"[0m
[38;2;127;132;156m 234[0m [38;2;205;214;244mproject_id: "a1b2c3d4e5f6"  # Exactly 12 chars[0m
[38;2;127;132;156m 235[0m [38;2;205;214;244mcollection_type: "docs"[0m
[38;2;127;132;156m 236[0m [38;2;205;214;244mcreated_at: "2025-01-03T10:00:00Z"[0m
[38;2;127;132;156m 237[0m [38;2;205;214;244m```[0m
[38;2;127;132;156m 238[0m 
[38;2;127;132;156m 239[0m [38;2;205;214;244m### Optional Metadata[0m
[38;2;127;132;156m 240[0m 
[38;2;127;132;156m 241[0m [38;2;205;214;244m```yaml[0m
[38;2;127;132;156m 242[0m [38;2;205;214;244mupdated_at: "2025-01-03T12:00:00Z"[0m
[38;2;127;132;156m 243[0m [38;2;205;214;244mdescription: "Project documentation"[0m
[38;2;127;132;156m 244[0m [38;2;205;214;244mtenant_namespace: "user_chris"[0m
[38;2;127;132;156m 245[0m [38;2;205;214;244mtags: ["mcp", "vector-database"][0m
[38;2;127;132;156m 246[0m [38;2;205;214;244mpriority: 3  # 1-5[0m
[38;2;127;132;156m 247[0m [38;2;205;214;244mbranch: "main"  # From git[0m
[38;2;127;132;156m 248[0m [38;2;205;214;244m```[0m
[38;2;127;132;156m 249[0m 
[38;2;127;132;156m 250[0m [38;2;205;214;244m### Git Integration[0m
[38;2;127;132;156m 251[0m 
[38;2;127;132;156m 252[0m [38;2;205;214;244m**Branch Detection**:[0m
[38;2;127;132;156m 253[0m [38;2;205;214;244m- Automatic from `git branch --show-current`[0m
[38;2;127;132;156m 254[0m [38;2;205;214;244m- Stored in metadata[0m
[38;2;127;132;156m 255[0m [38;2;205;214;244m- Used for branch-specific isolation[0m
[38;2;127;132;156m 256[0m 
[38;2;127;132;156m 257[0m [38;2;205;214;244m**Branch-Specific Collections** (optional):[0m
[38;2;127;132;156m 258[0m [38;2;205;214;244m- Pattern: `{project_id}-{type}-{branch}`[0m
[38;2;127;132;156m 259[0m [38;2;205;214;244m- Example: `a1b2c3d4e5f6-docs-main`[0m
[38;2;127;132;156m 260[0m [38;2;205;214;244m- Cleanup when branch deleted[0m
[38;2;127;132;156m 261[0m 
[38;2;127;132;156m 262[0m [38;2;205;214;244m### Access Control[0m
[38;2;127;132;156m 263[0m 
[38;2;127;132;156m 264[0m [38;2;205;214;244m- **CLI**: Write access (file watcher)[0m
[38;2;127;132;156m 265[0m [38;2;205;214;244m- **MCP/LLM**: Write access (for Claude Code)[0m
[38;2;127;132;156m 266[0m [38;2;205;214;244m- **Global Search**: Not included[0m
[38;2;127;132;156m 267[0m [38;2;205;214;244m- **Tenant Isolation**: Enabled[0m
[38;2;127;132;156m 268[0m [38;2;205;214;244m- **Branch Isolation**: Optional[0m
[38;2;127;132;156m 269[0m 
[38;2;127;132;156m 270[0m [38;2;205;214;244m### Performance Settings[0m
[38;2;127;132;156m 271[0m 
[38;2;127;132;156m 272[0m [38;2;205;214;244m- Batch size: 150 (largest)[0m
[38;2;127;132;156m 273[0m [38;2;205;214;244m- Max concurrent: 10 (highest)[0m
[38;2;127;132;156m 274[0m [38;2;205;214;244m- Priority: 2/5 (user-specific)[0m
[38;2;127;132;156m 275[0m [38;2;205;214;244m- Cache TTL: 300s (5 minutes)[0m
[38;2;127;132;156m 276[0m 
[38;2;127;132;156m 277[0m [38;2;205;214;244m### Use Cases[0m
[38;2;127;132;156m 278[0m 
[38;2;127;132;156m 279[0m [38;2;205;214;244m1. **Documentation**: Project README, guides[0m
[38;2;127;132;156m 280[0m [38;2;205;214;244m2. **Memory**: Claude Code memory[0m
[38;2;127;132;156m 281[0m [38;2;205;214;244m3. **Notes**: Development notes[0m
[38;2;127;132;156m 282[0m [38;2;205;214;244m4. **Context**: Project-specific context[0m
[38;2;127;132;156m 283[0m [38;2;205;214;244m5. **Code**: Code snippets[0m
[38;2;127;132;156m 284[0m 
[38;2;127;132;156m 285[0m [38;2;205;214;244m### Example[0m
[38;2;127;132;156m 286[0m 
[38;2;127;132;156m 287[0m [38;2;205;214;244mSee [project-collection-example.yaml](examples/project-collection-example.yaml)[0m
[38;2;127;132;156m 288[0m 
[38;2;127;132;156m 289[0m [38;2;205;214;244m---[0m
[38;2;127;132;156m 290[0m 
[38;2;127;132;156m 291[0m [38;2;205;214;244m## GLOBAL Collections[0m
[38;2;127;132;156m 292[0m 
[38;2;127;132;156m 293[0m [38;2;205;214;244m### Naming Convention[0m
[38;2;127;132;156m 294[0m 
[38;2;127;132;156m 295[0m [38;2;205;214;244m**Fixed Predefined Names**:[0m
[38;2;127;132;156m 296[0m [38;2;205;214;244m- `algorithms`: Algorithm implementations[0m
[38;2;127;132;156m 297[0m [38;2;205;214;244m- `codebase`: Cross-project code knowledge[0m
[38;2;127;132;156m 298[0m [38;2;205;214;244m- `context`: Global context/memories[0m
[38;2;127;132;156m 299[0m [38;2;205;214;244m- `documents`: System-wide docs[0m
[38;2;127;132;156m 300[0m [38;2;205;214;244m- `knowledge`: Knowledge base/FAQs[0m
[38;2;127;132;156m 301[0m [38;2;205;214;244m- `memory`: Long-term system memory[0m
[38;2;127;132;156m 302[0m [38;2;205;214;244m- `projects`: Project metadata[0m
[38;2;127;132;156m 303[0m [38;2;205;214;244m- `workspace`: Workspace configuration[0m
[38;2;127;132;156m 304[0m 
[38;2;127;132;156m 305[0m [38;2;205;214;244m**No Other Names Allowed**[0m
[38;2;127;132;156m 306[0m 
[38;2;127;132;156m 307[0m [38;2;205;214;244m### Deletion Handling[0m
[38;2;127;132;156m 308[0m 
[38;2;127;132;156m 309[0m [38;2;205;214;244m**Mode**: DYNAMIC (immediate)[0m
[38;2;127;132;156m 310[0m 
[38;2;127;132;156m 311[0m [38;2;205;214;244mSame as PROJECT collections - immediate deletion for real-time consistency.[0m
[38;2;127;132;156m 312[0m 
[38;2;127;132;156m 313[0m [38;2;205;214;244m### Required Metadata[0m
[38;2;127;132;156m 314[0m 
[38;2;127;132;156m 315[0m [38;2;205;214;244m```yaml[0m
[38;2;127;132;156m 316[0m [38;2;205;214;244mcollection_name: "knowledge"  # Must be from predefined list[0m
[38;2;127;132;156m 317[0m [38;2;205;214;244mcreated_at: "2025-01-03T10:00:00Z"[0m
[38;2;127;132;156m 318[0m [38;2;205;214;244m```[0m
[38;2;127;132;156m 319[0m 
[38;2;127;132;156m 320[0m [38;2;205;214;244m### Optional Metadata[0m
[38;2;127;132;156m 321[0m 
[38;2;127;132;156m 322[0m [38;2;205;214;244m```yaml[0m
[38;2;127;132;156m 323[0m [38;2;205;214;244mupdated_at: "2025-01-03T12:00:00Z"[0m
[38;2;127;132;156m 324[0m [38;2;205;214;244mdescription: "Global knowledge base"[0m
[38;2;127;132;156m 325[0m [38;2;205;214;244mworkspace_scope: "global"  # Default[0m
[38;2;127;132;156m 326[0m [38;2;205;214;244m```[0m
[38;2;127;132;156m 327[0m 
[38;2;127;132;156m 328[0m [38;2;205;214;244m### Access Control[0m
[38;2;127;132;156m 329[0m 
[38;2;127;132;156m 330[0m [38;2;205;214;244m- **CLI**: Write access[0m
[38;2;127;132;156m 331[0m [38;2;205;214;244m- **MCP/LLM**: Write access[0m
[38;2;127;132;156m 332[0m [38;2;205;214;244m- **Global Search**: Included[0m
[38;2;127;132;156m 333[0m [38;2;205;214;244m- **Tenant Isolation**: Disabled (system-wide)[0m
[38;2;127;132;156m 334[0m [38;2;205;214;244m- **Cross-Project**: Enabled[0m
[38;2;127;132;156m 335[0m 
[38;2;127;132;156m 336[0m [38;2;205;214;244m### Performance Settings[0m
[38;2;127;132;156m 337[0m 
[38;2;127;132;156m 338[0m [38;2;205;214;244m- Batch size: 200 (largest)[0m
[38;2;127;132;156m 339[0m [38;2;205;214;244m- Max concurrent: 8[0m
[38;2;127;132;156m 340[0m [38;2;205;214;244m- Priority: 5/5 (highest)[0m
[38;2;127;132;156m 341[0m [38;2;205;214;244m- Cache TTL: 1800s (30 minutes)[0m
[38;2;127;132;156m 342[0m 
[38;2;127;132;156m 343[0m [38;2;205;214;244m### Collection-Specific Schemas[0m
[38;2;127;132;156m 344[0m 
[38;2;127;132;156m 345[0m [38;2;205;214;244m#### algorithms[0m
[38;2;127;132;156m 346[0m [38;2;205;214;244m- Algorithm name, complexity, language[0m
[38;2;127;132;156m 347[0m [38;2;205;214;244m- Implementations and explanations[0m
[38;2;127;132;156m 348[0m 
[38;2;127;132;156m 349[0m [38;2;205;214;244m#### codebase[0m
[38;2;127;132;156m 350[0m [38;2;205;214;244m- Code snippets, patterns, anti-patterns[0m
[38;2;127;132;156m 351[0m [38;2;205;214;244m- Cross-project code knowledge[0m
[38;2;127;132;156m 352[0m 
[38;2;127;132;156m 353[0m [38;2;205;214;244m#### context[0m
[38;2;127;132;156m 354[0m [38;2;205;214;244m- System-wide context[0m
[38;2;127;132;156m 355[0m [38;2;205;214;244m- User preferences[0m
[38;2;127;132;156m 356[0m [38;2;205;214;244m- Conversation history[0m
[38;2;127;132;156m 357[0m 
[38;2;127;132;156m 358[0m [38;2;205;214;244m#### documents[0m
[38;2;127;132;156m 359[0m [38;2;205;214;244m- User guides, API docs[0m
[38;2;127;132;156m 360[0m [38;2;205;214;244m- Troubleshooting, release notes[0m
[38;2;127;132;156m 361[0m 
[38;2;127;132;156m 362[0m [38;2;205;214;244m#### knowledge[0m
[38;2;127;132;156m 363[0m [38;2;205;214;244m- FAQs, best practices[0m
[38;2;127;132;156m 364[0m [38;2;205;214;244m- Tutorials, glossary[0m
[38;2;127;132;156m 365[0m 
[38;2;127;132;156m 366[0m [38;2;205;214;244m#### memory[0m
[38;2;127;132;156m 367[0m [38;2;205;214;244m- Long-term system state[0m
[38;2;127;132;156m 368[0m [38;2;205;214;244m- Learned patterns[0m
[38;2;127;132;156m 369[0m [38;2;205;214;244m- Important decisions[0m
[38;2;127;132;156m 370[0m 
[38;2;127;132;156m 371[0m [38;2;205;214;244m#### projects[0m
[38;2;127;132;156m 372[0m [38;2;205;214;244m- Project metadata[0m
[38;2;127;132;156m 373[0m [38;2;205;214;244m- Relationships, history[0m
[38;2;127;132;156m 374[0m [38;2;205;214;244m- Dependencies[0m
[38;2;127;132;156m 375[0m 
[38;2;127;132;156m 376[0m [38;2;205;214;244m#### workspace[0m
[38;2;127;132;156m 377[0m [38;2;205;214;244m- Workspace settings[0m
[38;2;127;132;156m 378[0m [38;2;205;214;244m- Global configuration[0m
[38;2;127;132;156m 379[0m [38;2;205;214;244m- Tool integrations[0m
[38;2;127;132;156m 380[0m 
[38;2;127;132;156m 381[0m [38;2;205;214;244m### Example[0m
[38;2;127;132;156m 382[0m 
[38;2;127;132;156m 383[0m [38;2;205;214;244mSee [global-collection-example.yaml](examples/global-collection-example.yaml)[0m
[38;2;127;132;156m 384[0m 
[38;2;127;132;156m 385[0m [38;2;205;214;244m---[0m
[38;2;127;132;156m 386[0m 
[38;2;127;132;156m 387[0m [38;2;205;214;244m## Deletion Mode Comparison[0m
[38;2;127;132;156m 388[0m 
[38;2;127;132;156m 389[0m [38;2;205;214;244m### Cumulative Deletion (SYSTEM, LIBRARY)[0m
[38;2;127;132;156m 390[0m 
[38;2;127;132;156m 391[0m [38;2;205;214;244m```mermaid[0m
[38;2;127;132;156m 392[0m [38;2;205;214;244msequenceDiagram[0m
[38;2;127;132;156m 393[0m [38;2;205;214;244m    participant F as File Deleted[0m
[38;2;127;132;156m 394[0m [38;2;205;214;244m    participant Q as Deletion Queue[0m
[38;2;127;132;156m 395[0m [38;2;205;214;244m    participant B as Batch Cleanup[0m
[38;2;127;132;156m 396[0m [38;2;205;214;244m    participant V as Qdrant[0m
[38;2;127;132;156m 397[0m 
[38;2;127;132;156m 398[0m [38;2;205;214;244m    F->>Q: Mark as deleted[0m
[38;2;127;132;156m 399[0m [38;2;205;214;244m    Note over Q: Wait for batch<br/>(24h or 1000 items)[0m
[38;2;127;132;156m 400[0m [38;2;205;214;244m    B->>Q: Get pending deletions[0m
[38;2;127;132;156m 401[0m [38;2;205;214;244m    Q->>B: Return batch[0m
[38;2;127;132;156m 402[0m [38;2;205;214;244m    B->>V: Delete points[0m
[38;2;127;132;156m 403[0m [38;2;205;214;244m    B->>Q: Mark as processed[0m
[38;2;127;132;156m 404[0m [38;2;205;214;244m```[0m
[38;2;127;132;156m 405[0m 
[38;2;127;132;156m 406[0m [38;2;205;214;244m**When to Use**:[0m
[38;2;127;132;156m 407[0m [38;2;205;214;244m- Collections with infrequent deletions[0m
[38;2;127;132;156m 408[0m [38;2;205;214;244m- When batch performance matters[0m
[38;2;127;132;156m 409[0m [38;2;205;214;244m- When rollback capability needed[0m
[38;2;127;132;156m 410[0m [38;2;205;214;244m- For system stability[0m
[38;2;127;132;156m 411[0m 
[38;2;127;132;156m 412[0m [38;2;205;214;244m### Dynamic Deletion (PROJECT, GLOBAL)[0m
[38;2;127;132;156m 413[0m 
[38;2;127;132;156m 414[0m [38;2;205;214;244m```mermaid[0m
[38;2;127;132;156m 415[0m [38;2;205;214;244msequenceDiagram[0m
[38;2;127;132;156m 416[0m [38;2;205;214;244m    participant F as File Deleted[0m
[38;2;127;132;156m 417[0m [38;2;205;214;244m    participant Q as Deletion Queue[0m
[38;2;127;132;156m 418[0m [38;2;205;214;244m    participant P as Processor[0m
[38;2;127;132;156m 419[0m [38;2;205;214;244m    participant V as Qdrant[0m
[38;2;127;132;156m 420[0m 
[38;2;127;132;156m 421[0m [38;2;205;214;244m    F->>Q: Delete queued[0m
[38;2;127;132;156m 422[0m [38;2;205;214;244m    Q->>P: Process immediately[0m
[38;2;127;132;156m 423[0m [38;2;205;214;244m    P->>V: Delete point[0m
[38;2;127;132;156m 424[0m [38;2;205;214;244m    V->>P: Confirm[0m
[38;2;127;132;156m 425[0m [38;2;205;214;244m    P->>Q: Mark complete[0m
[38;2;127;132;156m 426[0m [38;2;205;214;244m```[0m
[38;2;127;132;156m 427[0m 
[38;2;127;132;156m 428[0m [38;2;205;214;244m**When to Use**:[0m
[38;2;127;132;156m 429[0m [38;2;205;214;244m- Collections with frequent changes[0m
[38;2;127;132;156m 430[0m [38;2;205;214;244m- When real-time sync required[0m
[38;2;127;132;156m 431[0m [38;2;205;214;244m- For user-facing content[0m
[38;2;127;132;156m 432[0m [38;2;205;214;244m- When immediate consistency needed[0m
[38;2;127;132;156m 433[0m 
[38;2;127;132;156m 434[0m [38;2;205;214;244m---[0m
[38;2;127;132;156m 435[0m 
[38;2;127;132;156m 436[0m [38;2;205;214;244m## Metadata Field Types[0m
[38;2;127;132;156m 437[0m 
[38;2;127;132;156m 438[0m [38;2;205;214;244m### Common Fields[0m
[38;2;127;132;156m 439[0m 
[38;2;127;132;156m 440[0m [38;2;205;214;244mAll collections should have:[0m
[38;2;127;132;156m 441[0m [38;2;205;214;244m```yaml[0m
[38;2;127;132;156m 442[0m [38;2;205;214;244mcreated_at: "ISO 8601 timestamp"[0m
[38;2;127;132;156m 443[0m [38;2;205;214;244mupdated_at: "ISO 8601 timestamp"[0m
[38;2;127;132;156m 444[0m [38;2;205;214;244mdescription: "Human-readable description"[0m
[38;2;127;132;156m 445[0m [38;2;205;214;244m```[0m
[38;2;127;132;156m 446[0m 
[38;2;127;132;156m 447[0m [38;2;205;214;244m### Type-Specific Fields[0m
[38;2;127;132;156m 448[0m 
[38;2;127;132;156m 449[0m [38;2;205;214;244m#### SYSTEM[0m
[38;2;127;132;156m 450[0m [38;2;205;214;244m```yaml[0m
[38;2;127;132;156m 451[0m [38;2;205;214;244mcollection_category: "system"[0m
[38;2;127;132;156m 452[0m [38;2;205;214;244mcli_writable: boolean[0m
[38;2;127;132;156m 453[0m [38;2;205;214;244m```[0m
[38;2;127;132;156m 454[0m 
[38;2;127;132;156m 455[0m [38;2;205;214;244m#### LIBRARY[0m
[38;2;127;132;156m 456[0m [38;2;205;214;244m```yaml[0m
[38;2;127;132;156m 457[0m [38;2;205;214;244mcollection_category: "library"[0m
[38;2;127;132;156m 458[0m [38;2;205;214;244mlanguage: "programming language"[0m
[38;2;127;132;156m 459[0m [38;2;205;214;244msymbols: ["list", "of", "symbols"][0m
[38;2;127;132;156m 460[0m [38;2;205;214;244mdependencies: ["list", "of", "deps"][0m
[38;2;127;132;156m 461[0m [38;2;205;214;244mversion: "semantic version"[0m
[38;2;127;132;156m 462[0m [38;2;205;214;244m```[0m
[38;2;127;132;156m 463[0m 
[38;2;127;132;156m 464[0m [38;2;205;214;244m#### PROJECT[0m
[38;2;127;132;156m 465[0m [38;2;205;214;244m```yaml[0m
[38;2;127;132;156m 466[0m [38;2;205;214;244mproject_name: "string"[0m
[38;2;127;132;156m 467[0m [38;2;205;214;244mproject_id: "12-char hash"[0m
[38;2;127;132;156m 468[0m [38;2;205;214;244mcollection_type: "docs|memory|notes|etc"[0m
[38;2;127;132;156m 469[0m [38;2;205;214;244mtenant_namespace: "string"[0m
[38;2;127;132;156m 470[0m [38;2;205;214;244mtags: ["list"][0m
[38;2;127;132;156m 471[0m [38;2;205;214;244mpriority: 1-5[0m
[38;2;127;132;156m 472[0m [38;2;205;214;244mbranch: "git branch"[0m
[38;2;127;132;156m 473[0m [38;2;205;214;244m```[0m
[38;2;127;132;156m 474[0m 
[38;2;127;132;156m 475[0m [38;2;205;214;244m#### GLOBAL[0m
[38;2;127;132;156m 476[0m [38;2;205;214;244m```yaml[0m
[38;2;127;132;156m 477[0m [38;2;205;214;244mworkspace_scope: "global"[0m
[38;2;127;132;156m 478[0m [38;2;205;214;244m```[0m
[38;2;127;132;156m 479[0m 
[38;2;127;132;156m 480[0m [38;2;205;214;244m### Validation[0m
[38;2;127;132;156m 481[0m 
[38;2;127;132;156m 482[0m [38;2;205;214;244mMetadata validated against type configuration:[0m
[38;2;127;132;156m 483[0m [38;2;205;214;244m- Required fields must be present[0m
[38;2;127;132;156m 484[0m [38;2;205;214;244m- Field types must match (string, int, list, etc.)[0m
[38;2;127;132;156m 485[0m [38;2;205;214;244m- Constraints checked (length, range, pattern)[0m
[38;2;127;132;156m 486[0m [38;2;205;214;244m- Custom validation rules applied[0m
[38;2;127;132;156m 487[0m 
[38;2;127;132;156m 488[0m [38;2;205;214;244m---[0m
[38;2;127;132;156m 489[0m 
[38;2;127;132;156m 490[0m [38;2;205;214;244m## Migration Between Types[0m
[38;2;127;132;156m 491[0m 
[38;2;127;132;156m 492[0m [38;2;205;214;244m### Allowed Migrations[0m
[38;2;127;132;156m 493[0m 
[38;2;127;132;156m 494[0m [38;2;205;214;244m| From | To | Difficulty |[0m
[38;2;127;132;156m 495[0m [38;2;205;214;244m|------|--------|-----------|[0m
[38;2;127;132;156m 496[0m [38;2;205;214;244m| UNKNOWN | Any | Easy |[0m
[38;2;127;132;156m 497[0m [38;2;205;214;244m| SYSTEM | LIBRARY | Medium |[0m
[38;2;127;132;156m 498[0m [38;2;205;214;244m| LIBRARY | SYSTEM | Medium |[0m
[38;2;127;132;156m 499[0m [38;2;205;214;244m| PROJECT | GLOBAL | Hard |[0m
[38;2;127;132;156m 500[0m [38;2;205;214;244m| GLOBAL | PROJECT | Hard |[0m
[38;2;127;132;156m 501[0m 
[38;2;127;132;156m 502[0m [38;2;205;214;244m### Migration Impact[0m
[38;2;127;132;156m 503[0m 
[38;2;127;132;156m 504[0m [38;2;205;214;244m**Name Changes Required**:[0m
[38;2;127;132;156m 505[0m [38;2;205;214;244m- SYSTEM ↔ LIBRARY: Prefix change (__ ↔ _)[0m
[38;2;127;132;156m 506[0m [38;2;205;214;244m- PROJECT → GLOBAL: Rename to predefined name[0m
[38;2;127;132;156m 507[0m [38;2;205;214;244m- GLOBAL → PROJECT: Add project_id prefix[0m
[38;2;127;132;156m 508[0m 
[38;2;127;132;156m 509[0m [38;2;205;214;244m**Metadata Changes**:[0m
[38;2;127;132;156m 510[0m [38;2;205;214;244m- Add/remove type-specific required fields[0m
[38;2;127;132;156m 511[0m [38;2;205;214;244m- Update collection_category[0m
[38;2;127;132;156m 512[0m [38;2;205;214;244m- Adjust access control metadata[0m
[38;2;127;132;156m 513[0m 
[38;2;127;132;156m 514[0m [38;2;205;214;244m**Deletion Mode Changes**:[0m
[38;2;127;132;156m 515[0m [38;2;205;214;244m- Cumulative → Dynamic: Move queued deletions[0m
[38;2;127;132;156m 516[0m [38;2;205;214;244m- Dynamic → Cumulative: Create deletion queue[0m
[38;2;127;132;156m 517[0m 
[38;2;127;132;156m 518[0m [38;2;205;214;244mSee [Migration Guide](migration-guide.md) for procedures.[0m
[38;2;127;132;156m 519[0m 
[38;2;127;132;156m 520[0m [38;2;205;214;244m---[0m
[38;2;127;132;156m 521[0m 
[38;2;127;132;156m 522[0m [38;2;205;214;244m## Performance Characteristics[0m
[38;2;127;132;156m 523[0m 
[38;2;127;132;156m 524[0m [38;2;205;214;244m### Throughput Comparison[0m
[38;2;127;132;156m 525[0m 
[38;2;127;132;156m 526[0m [38;2;205;214;244m| Type | Batch Size | Concurrency | Expected Throughput |[0m
[38;2;127;132;156m 527[0m [38;2;205;214;244m|------|------------|-------------|---------------------|[0m
[38;2;127;132;156m 528[0m [38;2;205;214;244m| SYSTEM | 50 | 3 | ~150 docs/sec |[0m
[38;2;127;132;156m 529[0m [38;2;205;214;244m| LIBRARY | 100 | 5 | ~500 docs/sec |[0m
[38;2;127;132;156m 530[0m [38;2;205;214;244m| PROJECT | 150 | 10 | ~1500 docs/sec |[0m
[38;2;127;132;156m 531[0m [38;2;205;214;244m| GLOBAL | 200 | 8 | ~1600 docs/sec |[0m
[38;2;127;132;156m 532[0m 
[38;2;127;132;156m 533[0m [38;2;205;214;244m*Throughput varies based on document size and system resources*[0m
[38;2;127;132;156m 534[0m 
[38;2;127;132;156m 535[0m [38;2;205;214;244m### Resource Usage[0m
[38;2;127;132;156m 536[0m 
[38;2;127;132;156m 537[0m [38;2;205;214;244m| Type | Memory/Doc | CPU Usage | Cache Size |[0m
[38;2;127;132;156m 538[0m [38;2;205;214;244m|------|------------|-----------|------------|[0m
[38;2;127;132;156m 539[0m [38;2;205;214;244m| SYSTEM | Low | Low | Small |[0m
[38;2;127;132;156m 540[0m [38;2;205;214;244m| LIBRARY | Medium | Medium | Large |[0m
[38;2;127;132;156m 541[0m [38;2;205;214;244m| PROJECT | Medium | High | Small |[0m
[38;2;127;132;156m 542[0m [38;2;205;214;244m| GLOBAL | Low | Medium | Large |[0m
[38;2;127;132;156m 543[0m 
[38;2;127;132;156m 544[0m [38;2;205;214;244m---[0m
[38;2;127;132;156m 545[0m 
[38;2;127;132;156m 546[0m [38;2;205;214;244m## Best Practices[0m
[38;2;127;132;156m 547[0m 
[38;2;127;132;156m 548[0m [38;2;205;214;244m### Choosing a Type[0m
[38;2;127;132;156m 549[0m 
[38;2;127;132;156m 550[0m [38;2;205;214;244m**Use SYSTEM when**:[0m
[38;2;127;132;156m 551[0m [38;2;205;214;244m- Managed exclusively by CLI[0m
[38;2;127;132;156m 552[0m [38;2;205;214;244m- System configuration/state[0m
[38;2;127;132;156m 553[0m [38;2;205;214;244m- Internal use only[0m
[38;2;127;132;156m 554[0m 
[38;2;127;132;156m 555[0m [38;2;205;214;244m**Use LIBRARY when**:[0m
[38;2;127;132;156m 556[0m [38;2;205;214;244m- Language documentation[0m
[38;2;127;132;156m 557[0m [38;2;205;214;244m- Globally searchable needed[0m
[38;2;127;132;156m 558[0m [38;2;205;214;244m- LSP/Tree-sitter integration useful[0m
[38;2;127;132;156m 559[0m 
[38;2;127;132;156m 560[0m [38;2;205;214;244m**Use PROJECT when**:[0m
[38;2;127;132;156m 561[0m [38;2;205;214;244m- Project-specific content[0m
[38;2;127;132;156m 562[0m [38;2;205;214;244m- File system synchronization needed[0m
[38;2;127;132;156m 563[0m [38;2;205;214;244m- User/LLM editable[0m
[38;2;127;132;156m 564[0m 
[38;2;127;132;156m 565[0m [38;2;205;214;244m**Use GLOBAL when**:[0m
[38;2;127;132;156m 566[0m [38;2;205;214;244m- Cross-project knowledge[0m
[38;2;127;132;156m 567[0m [38;2;205;214;244m- System-wide availability needed[0m
[38;2;127;132;156m 568[0m [38;2;205;214;244m- Predefined schema fits[0m
[38;2;127;132;156m 569[0m 
[38;2;127;132;156m 570[0m [38;2;205;214;244m### Naming Best Practices[0m
[38;2;127;132;156m 571[0m 
[38;2;127;132;156m 572[0m [38;2;205;214;244m1. **Be Descriptive**: `_python_stdlib` not `_pystd`[0m
[38;2;127;132;156m 573[0m [38;2;205;214;244m2. **Be Consistent**: Use same suffixes for PROJECT collections[0m
[38;2;127;132;156m 574[0m [38;2;205;214;244m3. **Follow Patterns**: Respect prefix requirements[0m
[38;2;127;132;156m 575[0m [38;2;205;214;244m4. **Check Conflicts**: Avoid name collisions[0m
[38;2;127;132;156m 576[0m 
[38;2;127;132;156m 577[0m [38;2;205;214;244m### Metadata Best Practices[0m
[38;2;127;132;156m 578[0m 
[38;2;127;132;156m 579[0m [38;2;205;214;244m1. **Required Fields**: Always provide all required fields[0m
[38;2;127;132;156m 580[0m [38;2;205;214;244m2. **Timestamps**: Use ISO 8601 format[0m
[38;2;127;132;156m 581[0m [38;2;205;214;244m3. **Descriptions**: Write clear, helpful descriptions[0m
[38;2;127;132;156m 582[0m [38;2;205;214;244m4. **Tags**: Use for organization[0m
[38;2;127;132;156m 583[0m [38;2;205;214;244m5. **Validation**: Validate before ingestion[0m
[38;2;127;132;156m 584[0m 
[38;2;127;132;156m 585[0m [38;2;205;214;244m---[0m
[38;2;127;132;156m 586[0m 
[38;2;127;132;156m 587[0m [38;2;205;214;244m## Next Steps[0m
[38;2;127;132;156m 588[0m 
[38;2;127;132;156m 589[0m [38;2;205;214;244m- **Migrate Collections**: [Migration Guide](migration-guide.md)[0m
[38;2;127;132;156m 590[0m [38;2;205;214;244m- **Optimize Performance**: [Performance Tuning](performance-tuning.md)[0m
[38;2;127;132;156m 591[0m [38;2;205;214;244m- **Troubleshoot Issues**: [Troubleshooting](troubleshooting.md)[0m
[38;2;127;132;156m 592[0m [38;2;205;214;244m- **Use API**: [API Reference](api-reference.md)[0m
