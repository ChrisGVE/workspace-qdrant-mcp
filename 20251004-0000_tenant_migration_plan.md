[38;2;127;132;156m   1[0m [38;2;205;214;244m# Tenant ID Migration System - Implementation Plan[0m
[38;2;127;132;156m   2[0m 
[38;2;127;132;156m   3[0m [38;2;205;214;244m## Context[0m
[38;2;127;132;156m   4[0m [38;2;205;214;244mWorkspace-qdrant-mcp uses tenant IDs to organize data by project. A tenant ID can be:[0m
[38;2;127;132;156m   5[0m [38;2;205;214;244m1. **Local path hash** (e.g., `path_abc123def456789a`) - for projects without Git remotes[0m
[38;2;127;132;156m   6[0m [38;2;205;214;244m2. **Normalized Git remote URL** (e.g., `github_com_user_repo`) - for projects with remotes[0m
[38;2;127;132;156m   7[0m 
[38;2;127;132;156m   8[0m [38;2;205;214;244mWhen a project gains/loses a Git remote, its tenant ID changes, requiring data migration.[0m
[38;2;127;132;156m   9[0m 
[38;2;127;132;156m  10[0m [38;2;205;214;244m## Current State Analysis[0m
[38;2;127;132;156m  11[0m 
[38;2;127;132;156m  12[0m [38;2;205;214;244m### Tenant ID Calculation (project_detection.py)[0m
[38;2;127;132;156m  13[0m [38;2;205;214;244m- `calculate_tenant_id(project_root: Path) -> str`[0m
[38;2;127;132;156m  14[0m [38;2;205;214;244m- Tries Git remote first (origin → upstream → any remote)[0m
[38;2;127;132;156m  15[0m [38;2;205;214;244m- Sanitizes remote URL to create tenant ID[0m
[38;2;127;132;156m  16[0m [38;2;205;214;244m- Falls back to SHA256 hash of path if no remote[0m
[38;2;127;132;156m  17[0m 
[38;2;127;132;156m  18[0m [38;2;205;214;244m### Data Structures Using Tenant ID[0m
[38;2;127;132;156m  19[0m [38;2;205;214;244m1. **ingestion_queue table** (SQLite):[0m
[38;2;127;132;156m  20[0m [38;2;205;214;244m   - `tenant_id TEXT DEFAULT 'default'`[0m
[38;2;127;132;156m  21[0m [38;2;205;214;244m   - Indexed with `(collection_name, tenant_id, branch)`[0m
[38;2;127;132;156m  22[0m 
[38;2;127;132;156m  23[0m [38;2;205;214;244m2. **Metadata tracking** (file_processing table):[0m
[38;2;127;132;156m  24[0m [38;2;205;214;244m   - No direct tenant_id field currently[0m
[38;2;127;132;156m  25[0m [38;2;205;214;244m   - Uses `collection` field which may encode tenant info[0m
[38;2;127;132;156m  26[0m 
[38;2;127;132;156m  27[0m [38;2;205;214;244m3. **Qdrant collections**:[0m
[38;2;127;132;156m  28[0m [38;2;205;214;244m   - Project collections: `_{project_id}` (12-char hex hash)[0m
[38;2;127;132;156m  29[0m [38;2;205;214;244m   - Metadata filtering uses `project_id` field[0m
[38;2;127;132;156m  30[0m [38;2;205;214;244m   - Need to update filters when tenant ID changes[0m
[38;2;127;132;156m  31[0m 
[38;2;127;132;156m  32[0m [38;2;205;214;244m## Implementation Steps[0m
[38;2;127;132;156m  33[0m 
[38;2;127;132;156m  34[0m [38;2;205;214;244m### Step 1: Detection System[0m
[38;2;127;132;156m  35[0m [38;2;205;214;244mCreate `TenantChangeDetector` class to:[0m
[38;2;127;132;156m  36[0m [38;2;205;214;244m- Monitor Git repository for remote changes[0m
[38;2;127;132;156m  37[0m [38;2;205;214;244m- Compare old/new tenant IDs[0m
[38;2;127;132;156m  38[0m [38;2;205;214;244m- Detect when migration is needed[0m
[38;2;127;132;156m  39[0m 
[38;2;127;132;156m  40[0m [38;2;205;214;244m### Step 2: Migration Core[0m
[38;2;127;132;156m  41[0m [38;2;205;214;244mCreate `TenantMigrationManager` class with:[0m
[38;2;127;132;156m  42[0m [38;2;205;214;244m- `detect_tenant_change(project_root)` - Check if tenant ID changed[0m
[38;2;127;132;156m  43[0m [38;2;205;214;244m- `plan_migration(old_tenant, new_tenant)` - Create migration plan[0m
[38;2;127;132;156m  44[0m [38;2;205;214;244m- `execute_migration(plan)` - Perform atomic migration[0m
[38;2;127;132;156m  45[0m [38;2;205;214;244m- `rollback_migration(plan)` - Revert on failure[0m
[38;2;127;132;156m  46[0m [38;2;205;214;244m- `validate_migration(plan)` - Verify consistency[0m
[38;2;127;132;156m  47[0m 
[38;2;127;132;156m  48[0m [38;2;205;214;244m### Step 3: Queue Migration[0m
[38;2;127;132;156m  49[0m [38;2;205;214;244mMigrate ingestion_queue entries:[0m
[38;2;127;132;156m  50[0m [38;2;205;214;244m```sql[0m
[38;2;127;132;156m  51[0m [38;2;205;214;244mBEGIN TRANSACTION;[0m
[38;2;127;132;156m  52[0m [38;2;205;214;244mUPDATE ingestion_queue [0m
[38;2;127;132;156m  53[0m [38;2;205;214;244mSET tenant_id = ? [0m
[38;2;127;132;156m  54[0m [38;2;205;214;244mWHERE tenant_id = ? AND collection_name = ?;[0m
[38;2;127;132;156m  55[0m [38;2;205;214;244mCOMMIT;[0m
[38;2;127;132;156m  56[0m [38;2;205;214;244m```[0m
[38;2;127;132;156m  57[0m 
[38;2;127;132;156m  58[0m [38;2;205;214;244m### Step 4: Metadata Migration  [0m
[38;2;127;132;156m  59[0m [38;2;205;214;244mUpdate file_processing metadata:[0m
[38;2;127;132;156m  60[0m [38;2;205;214;244m```sql[0m
[38;2;127;132;156m  61[0m [38;2;205;214;244mBEGIN TRANSACTION;[0m
[38;2;127;132;156m  62[0m [38;2;205;214;244mUPDATE file_processing[0m
[38;2;127;132;156m  63[0m [38;2;205;214;244mSET metadata = json_set(metadata, '$.tenant_id', ?)[0m
[38;2;127;132;156m  64[0m [38;2;205;214;244mWHERE json_extract(metadata, '$.tenant_id') = ?;[0m
[38;2;127;132;156m  65[0m [38;2;205;214;244mCOMMIT;[0m
[38;2;127;132;156m  66[0m [38;2;205;214;244m```[0m
[38;2;127;132;156m  67[0m 
[38;2;127;132;156m  68[0m [38;2;205;214;244m### Step 5: Qdrant Filter Updates[0m
[38;2;127;132;156m  69[0m [38;2;205;214;244mFor each affected collection:[0m
[38;2;127;132;156m  70[0m [38;2;205;214;244m1. Query points with old tenant filter[0m
[38;2;127;132;156m  71[0m [38;2;205;214;244m2. Update metadata with new tenant ID[0m
[38;2;127;132;156m  72[0m [38;2;205;214;244m3. Reindex if necessary[0m
[38;2;127;132;156m  73[0m 
[38;2;127;132;156m  74[0m [38;2;205;214;244m### Step 6: Progress Tracking[0m
[38;2;127;132;156m  75[0m [38;2;205;214;244m- Track migration progress in SQLite[0m
[38;2;127;132;156m  76[0m [38;2;205;214;244m- Support resumption after interruption[0m
[38;2;127;132;156m  77[0m [38;2;205;214;244m- Report statistics (entries migrated, errors, etc.)[0m
[38;2;127;132;156m  78[0m 
[38;2;127;132;156m  79[0m [38;2;205;214;244m### Step 7: Integration Points[0m
[38;2;127;132;156m  80[0m [38;2;205;214;244m- Hook into Git monitoring (if exists)[0m
[38;2;127;132;156m  81[0m [38;2;205;214;244m- Provide CLI command for manual migration[0m
[38;2;127;132;156m  82[0m [38;2;205;214;244m- Automatic detection on project initialization[0m
[38;2;127;132;156m  83[0m 
[38;2;127;132;156m  84[0m [38;2;205;214;244m### Step 8: Validation & Audit[0m
[38;2;127;132;156m  85[0m [38;2;205;214;244m- Log all migrations with timestamps[0m
[38;2;127;132;156m  86[0m [38;2;205;214;244m- Validate data consistency before/after[0m
[38;2;127;132;156m  87[0m [38;2;205;214;244m- Create audit trail in SQLite[0m
[38;2;127;132;156m  88[0m 
[38;2;127;132;156m  89[0m [38;2;205;214;244m## Database Schema Additions[0m
[38;2;127;132;156m  90[0m 
[38;2;127;132;156m  91[0m [38;2;205;214;244m```sql[0m
[38;2;127;132;156m  92[0m [38;2;205;214;244mCREATE TABLE tenant_migrations ([0m
[38;2;127;132;156m  93[0m [38;2;205;214;244m    id INTEGER PRIMARY KEY AUTOINCREMENT,[0m
[38;2;127;132;156m  94[0m [38;2;205;214;244m    project_root TEXT NOT NULL,[0m
[38;2;127;132;156m  95[0m [38;2;205;214;244m    old_tenant_id TEXT NOT NULL,[0m
[38;2;127;132;156m  96[0m [38;2;205;214;244m    new_tenant_id TEXT NOT NULL,[0m
[38;2;127;132;156m  97[0m [38;2;205;214;244m    status TEXT NOT NULL,  -- planning, executing, completed, failed, rolled_back[0m
[38;2;127;132;156m  98[0m [38;2;205;214;244m    started_at TIMESTAMP NOT NULL,[0m
[38;2;127;132;156m  99[0m [38;2;205;214;244m    completed_at TIMESTAMP,[0m
[38;2;127;132;156m 100[0m [38;2;205;214;244m    entries_migrated INTEGER DEFAULT 0,[0m
[38;2;127;132;156m 101[0m [38;2;205;214;244m    entries_total INTEGER DEFAULT 0,[0m
[38;2;127;132;156m 102[0m [38;2;205;214;244m    error_message TEXT,[0m
[38;2;127;132;156m 103[0m [38;2;205;214;244m    rollback_info TEXT,  -- JSON with rollback data[0m
[38;2;127;132;156m 104[0m [38;2;205;214;244m    audit_log TEXT  -- JSON with detailed audit trail[0m
[38;2;127;132;156m 105[0m [38;2;205;214;244m);[0m
[38;2;127;132;156m 106[0m 
[38;2;127;132;156m 107[0m [38;2;205;214;244mCREATE INDEX idx_tenant_migrations_status ON tenant_migrations(status);[0m
[38;2;127;132;156m 108[0m [38;2;205;214;244mCREATE INDEX idx_tenant_migrations_project_root ON tenant_migrations(project_root);[0m
[38;2;127;132;156m 109[0m [38;2;205;214;244m```[0m
[38;2;127;132;156m 110[0m 
[38;2;127;132;156m 111[0m [38;2;205;214;244m## Key Design Decisions[0m
[38;2;127;132;156m 112[0m 
[38;2;127;132;156m 113[0m [38;2;205;214;244m1. **Atomic Transactions**: Use SQLite transactions with savepoints[0m
[38;2;127;132;156m 114[0m [38;2;205;214;244m2. **Rollback Support**: Store rollback information before migration[0m
[38;2;127;132;156m 115[0m [38;2;205;214;244m3. **Progress Tracking**: Update migration record after each batch[0m
[38;2;127;132;156m 116[0m [38;2;205;214;244m4. **Batch Processing**: Migrate in batches of 1000 entries for large datasets[0m
[38;2;127;132;156m 117[0m [38;2;205;214;244m5. **Validation**: Check consistency before committing[0m
[38;2;127;132;156m 118[0m [38;2;205;214;244m6. **Idempotency**: Safe to re-run if interrupted[0m
[38;2;127;132;156m 119[0m 
[38;2;127;132;156m 120[0m [38;2;205;214;244m## Testing Strategy[0m
[38;2;127;132;156m 121[0m 
[38;2;127;132;156m 122[0m [38;2;205;214;244m1. Unit tests for tenant ID calculation changes[0m
[38;2;127;132;156m 123[0m [38;2;205;214;244m2. Integration tests for queue migration[0m
[38;2;127;132;156m 124[0m [38;2;205;214;244m3. End-to-end tests with real Git repos[0m
[38;2;127;132;156m 125[0m [38;2;205;214;244m4. Rollback scenario testing[0m
[38;2;127;132;156m 126[0m [38;2;205;214;244m5. Large-scale migration performance testing[0m
[38;2;127;132;156m 127[0m 
[38;2;127;132;156m 128[0m [38;2;205;214;244m## Deliverables[0m
[38;2;127;132;156m 129[0m 
[38;2;127;132;156m 130[0m [38;2;205;214;244m1. `src/python/common/core/tenant_migration.py` - Main module[0m
[38;2;127;132;156m 131[0m [38;2;205;214;244m2. `tests/unit/test_tenant_migration.py` - Unit tests[0m
[38;2;127;132;156m 132[0m [38;2;205;214;244m3. `tests/integration/test_tenant_migration_integration.py` - Integration tests[0m
[38;2;127;132;156m 133[0m [38;2;205;214;244m4. CLI command (optional): `wqm migrate tenant`[0m
[38;2;127;132;156m 134[0m [38;2;205;214;244m5. Documentation in module docstrings[0m
