# Path Abstraction Audit (A-1)

**Status:** Complete  
**Produced by:** Audit task A-1 (task-master tag: path-abstraction, task ID: 4)  
**Date:** 2026-05-14  
**Spec reference:** `docs/specs/16-path-abstraction.md` §6.1, §6.1.1

---

## Summary

| Class | Count |
|---|---|
| canonical | 26 |
| relative | 7 |
| process-local | 16 |
| disambiguation-suffix | 2 |
| non-path | 10 |
| **total** | **61** |

Classification definitions (from spec §6.1.1):

- **canonical** — host-absolute, must become `CanonicalPath` (or producer-validated for nested proto types)
- **relative** — anchored to a watch_folder root or project root; already portable
- **process-local** — SQLite DB file path, config path, log path, etc.; never serialized over gRPC or stored in Qdrant
- **disambiguation-suffix** — path suffix for clone disambiguation; relative semantically; see spec §6.1
- **non-path** — field name matches pattern but is not a filesystem path (e.g. traversal description string, URL, filter pattern)

---

## SQLite Columns

### Canonical (must become `CanonicalPath`)

| Site | Table | Column | Type | Class | Notes |
|---|---|---|---|---|---|
| `schema/watch_folders_schema.sql:14` | `watch_folders` | `path` | `TEXT NOT NULL UNIQUE` | canonical | Project/library root; absolute. Hot-path: read by queue priority JOIN and path-validator loop. |
| `tracked_files_schema/schema.rs:12` | `tracked_files` | `file_path` | `TEXT NOT NULL` | canonical | Absolute file path; in UNIQUE constraint `(watch_folder_id, file_path, branch)`. Hot-path: every watcher event writes here. |
| `code_lines_schema.rs:158` | `file_metadata` (search.db) | `file_path` | `TEXT NOT NULL` | canonical | Denormalized absolute path copied from `tracked_files.file_path` for FTS5 scoping. |
| `graph/schema.rs:186` | `graph_nodes` | `file_path` | `TEXT NOT NULL` | canonical | Absolute path to source file containing the symbol. |
| `graph/schema.rs:216` | `graph_edges` | `source_file` | `TEXT NOT NULL` | canonical | Absolute path to file containing the source node. |
| `unified_queue_schema/sql.rs:42` | `unified_queue` | `file_path` | `TEXT` (nullable) | canonical | Absolute file path for per-file dedup; NULL for non-file item types. Hot-path: on every enqueue. |
| `schema_version/v34.rs:24` | `ignore_file_mtimes` | `project_root` | `TEXT NOT NULL` (PK part) | canonical | Absolute path to project root; part of composite PK. |
| `schema_version/v34.rs:25` | `ignore_file_mtimes` | `file_path` | `TEXT NOT NULL` (PK part) | canonical | Absolute path to `.gitignore`/`.wqmignore` file; part of composite PK. |

### Relative (no type change — already portable)

| Site | Table | Column | Anchor | Class | Notes |
|---|---|---|---|---|---|
| `schema/watch_folders_schema.sql:21` | `watch_folders` | `submodule_path` | Relative to parent `watch_folders.path` | relative | NULL for top-level watches. |
| `tracked_files_schema/schema.rs:29` | `tracked_files` | `relative_path` | Relative to `watch_folders.path` | relative | Added in migration v19. Used by Qdrant payload alignment. |
| `code_lines_schema.rs:168` | `file_metadata` (search.db) | `relative_path` | Relative to `watch_folders.path` | relative | Added in search.db migration v5. |

### Disambiguation-suffix

| Site | Table | Column | Class | Notes |
|---|---|---|---|---|
| `schema/watch_folders_schema.sql:26` | `watch_folders` | `disambiguation_path` | disambiguation-suffix | Path suffix for clone disambiguation; not a full path; see spec §6.1. Treat as `String` unless a `DisambiguationSuffix` newtype is introduced. |

---

## Qdrant Payloads (Rust serde structs — `common/src/payloads/`)

### Canonical (must become `CanonicalPath`)

| Site | Struct | Field | Rust type | Class | Notes |
|---|---|---|---|---|---|
| `common/src/payloads/filesystem.rs:17` | `FilePayload` | `file_path` | `String` | canonical | Absolute file path; serialized into Qdrant point payload JSON. |
| `common/src/payloads/filesystem.rs:29` | `FilePayload` | `old_path` | `Option<String>` | canonical | Previous absolute path before rename (op=Rename). Absent on non-rename ops. |
| `common/src/payloads/filesystem.rs:36` | `FolderPayload` | `folder_path` | `String` | canonical | Absolute folder path. |
| `common/src/payloads/filesystem.rs:51` | `FolderPayload` | `old_path` | `Option<String>` | canonical | Previous absolute path before rename. |
| `common/src/payloads/library.rs:78` | `LibraryDocumentPayload` | `document_path` | `String` | canonical | Absolute path to library document on disk (PDF, EPUB, etc.). |
| `daemon/core/src/image_search.rs:79` | `ImageSearchResult` | `file_path` | `String` | canonical | Absolute path to source document; decoded from Qdrant `images` collection payload. |

### Relative (no canonical wrapping)

| Site | Struct | Field | Anchor | Class | Notes |
|---|---|---|---|---|---|
| `common/src/payloads/library.rs:93` | `LibraryDocumentPayload` | `library_path` | Relative within library hierarchy | relative | e.g. `"cs/design_patterns"`. Empty string for root-level docs. Not a filesystem path to an existing file — a logical hierarchy label. |

### Serde non-path fields in payloads (false positives)

| Site | Struct | Field | Class | Notes |
|---|---|---|---|---|
| `common/src/payloads/library.rs:35` | `LibraryPayload` | `source_url` | non-path | URL string; not a filesystem path. |

---

## Proto-defined Messages (`.proto` source — prost-generated in Rust)

### Canonical — handler-validated at entry (request messages)

| Proto line | Message | Field | Class | Notes |
|---|---|---|---|---|
| `proto:429` | `RegisterProjectRequest` | `path` | canonical | Absolute path to project root; feeds `watch_folders.path`. Primary site: `project_service/registration.rs`. |
| `proto:450` | `DeprioritizeProjectRequest` | `watch_path` | canonical | Optional; absolute path to specific watch root for multi-clone disambiguation. |
| `proto:1083` | `SetIncrementalRequest` | `file_paths` | canonical | `repeated string`; each element is an absolute file path. See spec §7.4 item 2 — requires `extract_canonical_paths!` macro. |
| `proto:648` | `ImpactAnalysisRequest` | `file_path` | canonical | Optional; narrows graph query to specific file. |

### Canonical — producer-validated (response messages emitted by daemon handlers)

| Proto line | Message | Field | Class | Notes |
|---|---|---|---|---|
| `proto:444` | `RegisterProjectResponse` | `watch_path` | canonical | Confirmed registered watch path; built by registration handler. |
| `proto:468` | `GetProjectStatusResponse` | `project_root` | canonical | Absolute project root from `watch_folders.path`. |
| `proto:475` | `GetProjectStatusResponse` | `main_worktree_path` | canonical | Absolute path to main working tree when project is a worktree. |
| `proto:492` | `ProjectInfo` | `project_root` | canonical | Absolute project root in list response. |
| `proto:272` | `ServerStatusNotification` | `project_root` | canonical | Optional; absolute project root sent in server status events. |
| `proto:580` | `TextSearchMatch` | `file_path` | canonical | Absolute file path from `file_metadata.file_path`. Hot-path: emitted for every search result. |
| `proto:638` | `TraversalNodeProto` | `file_path` | canonical | Absolute path to file containing the traversal node. |
| `proto:662` | `ImpactNodeProto` | `file_path` | canonical | Absolute path to file containing impacted node. |
| `proto:702` | `PageRankNodeProto` | `file_path` | canonical | Absolute path to file containing the PageRank node. |
| `proto:732` | `CommunityMemberProto` | `file_path` | canonical | Absolute path to file containing the community member. |
| `proto:755` | `BetweennessNodeProto` | `file_path` | canonical | Absolute path to file containing the betweenness node. |
| `proto:968` | `CancelItemsResponse` | `project_path` | canonical | Resolved project path for display; sourced from `watch_folders.path`. |

### Canonical — nested-message paths (producer-validated per spec §7.4 item 3)

| Proto line | Message | Field | Class | Notes |
|---|---|---|---|---|
| `proto:1338` | `ProjectPayload` | `file_absolute_path` | canonical | Optional; full host-absolute path to the indexed file. Nested inside `ProjectPayload`; producer (Qdrant write path) must validate before serialization. |
| `proto:1370` | `SymbolReference` | `file_path` | canonical | Absolute path inside `LspMetadata.references`; nested two levels deep (`ProjectPayload → LspMetadata → SymbolReference`). Producer-validated rule applies. |
| `proto:1381` | `LibraryPayload` (proto) | `source_file` | canonical | Absolute path to the original library file. Nested in proto-defined `LibraryPayload` (distinct from serde `LibraryPayload`). |

### Relative (no canonical wrapping)

| Proto line | Message | Field | Anchor | Class | Notes |
|---|---|---|---|---|---|
| `proto:1337` | `ProjectPayload` | `file_path` | Relative to project root | relative | Aligns with `tracked_files.relative_path`. Must NOT be wrapped in `CanonicalPath` (would reject as non-absolute). |

### Non-path proto fields (false positives)

| Proto line | Message | Field | Class | Notes |
|---|---|---|---|---|
| `proto:641` | `TraversalNodeProto` | `path` | non-path | Traversal path description string (e.g. `"A → B → C"`); not a filesystem path. |
| `proto:558` | `TextSearchRequest` | `path_glob` | non-path | Glob pattern for filtering (e.g. `"**/*.rs"`); not a concrete path. |
| `proto:559` | `TextSearchRequest` | `path_prefix` | non-path | Prefix pattern for filtering (e.g. `"src/"`); may be a path prefix but is used as a LIKE filter, not a standalone path. |

---

## TypeScript MCP Server (`src/typescript/mcp-server/src/`)

These fields mirror the proto messages over gRPC. They are listed here for T2 coverage confirmation — the TS types do not add new path storage.

| File | Interface | Field | Class | Notes |
|---|---|---|---|---|
| `clients/grpc-types-messages-document-project.ts:42` | `RegisterProjectRequest` | `path` | canonical | Mirrors proto `RegisterProjectRequest.path`. Sent from TS → gRPC. |
| `clients/grpc-types-messages-document-project.ts:57` | `RegisterProjectResponse` | `watch_path` | canonical | Mirrors proto response. |
| `clients/grpc-types-messages-document-project.ts:62` | `DeprioritizeProjectRequest` | `watch_path` | canonical | Mirrors proto request. |
| `clients/grpc-types-messages-document-project.ts:79` | `GetProjectStatusResponse` | `project_root` | canonical | Mirrors proto response. |
| `clients/grpc-types-messages-document-project.ts:95` | `ProjectInfo` | `project_root` | canonical | Mirrors proto. |
| `clients/grpc-types-messages-system-collection.ts:72` | `ServerStatusNotification` | `project_root` | canonical | Mirrors proto. |
| `clients/grpc-types-search-graph.ts:30` | `TextSearchMatch` | `file_path` | canonical | Mirrors proto. Returned to LLM client. |
| `clients/grpc-types-search-graph.ts:58` | `TraversalNodeProto` | `file_path` | canonical | Mirrors proto. |
| `clients/grpc-types-search-graph.ts:67` | `ImpactAnalysisRequest` | `file_path` | canonical | Optional; mirrors proto. |
| `clients/grpc-types-search-graph.ts:79` | `ImpactNodeProto` | `file_path` | canonical | Mirrors proto. |
| `clients/grpc-types-search-graph.ts:61` | `TraversalNodeProto` | `path` | non-path | Traversal description string; mirrors proto field. |
| `clients/project-queries.ts:26` | `ProjectRow` (SQLite result) | `disambiguation_path` | disambiguation-suffix | Read from `watch_folders.disambiguation_path`. |

---

## Process-local Paths

These paths are used only within the current process for I/O — never serialized to SQLite as a stored canonical path, not transmitted over gRPC, not written to Qdrant. Per spec §4.1, must remain `PathBuf`/`String` and must NOT become `CanonicalPath`.

| Site | Struct / field | Rust type | Class | Notes |
|---|---|---|---|---|
| `daemon/core/src/queue_config.rs:19` | `QueueConnectionConfig::database_path` | `String` | process-local | SQLite state.db file location. Bootstrap input. |
| `daemon/core/src/config/mod.rs:85` | `DaemonConfig::log_file` | `Option<PathBuf>` | process-local | Log file path; never stored in DB or transmitted. |
| `daemon/core/src/config/mod.rs:99` | `DaemonConfig::project_path` | `Option<PathBuf>` | process-local | Working directory override for daemon; not persisted. |
| `daemon/core/src/config/mod.rs:478` | `Config::database_path` | `Option<PathBuf>` | process-local | SQLite DB path for the processing engine config. |
| `daemon/core/src/graph/schema.rs:53` | `GraphDbManager::path` | `PathBuf` | process-local | `graph.db` file location on disk. |
| `daemon/core/src/graph/ladybug_store/config.rs:8` | `LadybugConfig::db_path` | `PathBuf` | process-local | LadybugDB directory path. |
| `daemon/core/src/monitoring/logging_config.rs:35` | `LoggingConfig::log_file_path` | `Option<PathBuf>` | process-local | Log file path for file logging. |
| `daemon/core/src/project_disambiguation.rs:46` | `ProjectRecord::project_path` | `PathBuf` | process-local | In-memory path for disambiguation logic; never directly stored. Note: this struct is the daemon-side record; the stored value is in `watch_folders.path`. |
| `daemon/core/src/project_disambiguation.rs:75` | `RegisteredProject::project_path` | `PathBuf` | process-local | Deprecated struct (see module comment); `project_path` is never persisted in this form. |
| `daemon/core/src/log_pruner.rs:33` | `LogPrunerConfig::path` | `PathBuf` | process-local | Log directory path. |
| `daemon/core/src/watching/events.rs:12` | `FileEvent::path` | `PathBuf` | process-local | In-memory watcher event path; converted to canonical before DB write. Hot-path: every filesystem event. |
| `daemon/core/src/watching/move_detector/types.rs:52` | `PendingMove::old_path` | `PathBuf` | process-local | Transient; holds the MOVED_FROM path until MOVED_TO arrives. Hot-path: rename events. |
| `daemon/core/src/watching/path_validator.rs:64` | `OrphanedProject::path` | `PathBuf` | process-local | In-memory tracking for orphan detection; not persisted. |
| `daemon/core/src/watching/path_validator.rs:80` | `RegisteredProject::path` (path-validator) | `PathBuf` | process-local | Local copy of `watch_folders.path` read for validation; not a new write surface. |
| `daemon/core/src/watching/path_validator.rs:293` | `OrphanCleanupActions::path` | `PathBuf` | process-local | Used to build SQL DELETE statement; the path is the filter value, not stored. |
| `daemon/grpc/src/auth.rs:11–13` | `TlsConfig::cert_path`, `key_path`, `ca_cert_path` | `String` | process-local | TLS certificate file paths; read at startup for gRPC server; never stored or transmitted as payload. |
| `common/src/cli_profiles.rs:108` | `Profile::database_path` | `String` | process-local | CLI profile override for DB path; used at connection time. |
| `common/src/yaml_defaults/processing.rs:145` | `YamlLspConfig::user_path` | `Option<String>` | process-local | User-supplied override for LSP binary search path; used at process startup. |

---

## Non-path Fields (False Positives)

Fields whose names match `*_path`, `*_file`, or `path` but carry no filesystem path semantics:

| Site | Struct/field | Class | Notes |
|---|---|---|---|
| `proto:641` | `TraversalNodeProto::path` | non-path | Traversal description string (e.g. `"A → B → C"`). |
| `proto:558` | `TextSearchRequest::path_glob` | non-path | Glob pattern filter string; not a path to a file. |
| `proto:559` | `TextSearchRequest::path_prefix` | non-path | Path prefix used as LIKE filter; not a standalone canonical path. |
| `common/src/payloads/library.rs:35` | `LibraryPayload::source_url` (contains "source") | non-path | URL, not filesystem path. Not pattern-matched but included for completeness. |
| `daemon/core/src/watching/telemetry.rs:80` | `WatcherTelemetry::watched_paths` | non-path | Integer count of watched paths; not a path value. |
| `daemon/core/src/watching/file_watcher/mod.rs:119` | `WatchStats::watched_paths` | non-path | Integer count; not a path value. |
| `daemon/core/src/watching/path_validator.rs:88` | `MoveDetectorStats::pending_by_path` | non-path | Integer count. |
| `daemon/grpc/src/metrics_layer.rs:77` | `parse_grpc_path(path: &str)` param | non-path | gRPC method path (e.g. `/workspace.ProjectService/Register`); not filesystem. |
| `daemon/grpc/src/services/project_service/worktree.rs:28` | `WatchMetadata::watch_path` | canonical | (Reclassified: this is a canonical path — see producer note in §7 below.) |
| `common/src/constants.rs:86` | `qdrant::LIBRARY_PATH` | non-path | Qdrant payload field name string constant `"library_path"`; not a path value itself. |

---

## CLI Display/UI Path Fields

These fields hold paths for display or CLI argument parsing — they are derived from `watch_folders.path` or user input and may be truncated/formatted. They feed into gRPC requests (where they become canonical) or are display-only.

| Site | Struct / field | Class | Notes |
|---|---|---|---|
| `cli/src/commands/watch/types.rs:14` | `WatchListItem::path` | canonical | Display copy of `watch_folders.path`; sourced from DB. No separate storage. |
| `cli/src/commands/watch/types.rs:42` | `WatchListItemVerbose::path` | canonical | Same. |
| `cli/src/commands/watch/types.rs:74` | `WatchDetailItem::path` | canonical | Same. |
| `cli/src/commands/watch/types.rs:79` | `WatchDetailItem::disambiguation_path` | disambiguation-suffix | Display copy of `watch_folders.disambiguation_path`. |
| `cli/src/commands/watch/types.rs:91` | `WatchDetailItem::submodule_path` | relative | Display copy of `watch_folders.submodule_path`. |
| `cli/src/data/queries/projects.rs:10` | `ProjectInfo::path` | canonical | CLI query result; sourced from `watch_folders.path`. |
| `cli/src/tui/views/libraries_data.rs:16` | `LibraryTreeNode::display_path` | canonical | Display-formatted canonical path. |
| `cli/src/tui/views/projects_data.rs:22` | `ProjectRow::display_path` | canonical | Display-formatted canonical path. |
| `cli/src/tui/views/dashboard_popups.rs:21` | `QueueItem::rel_path` | relative | Relative path shown in queue popup. |
| `cli/src/tui/views/dashboard_popups.rs:42` | `SearchResult::file_path` | canonical | Shown in search popup; sourced from Qdrant payload. |

---

## Hot-path Sites (for T6 performance consideration)

Sites that participate in the watcher event loop or queue hot-path — flag for performance review during T6.

| Site | Reason |
|---|---|
| `unified_queue.file_path` | Every file enqueue/dedup check touches this column with a composite UNIQUE index. |
| `tracked_files.file_path` | Written on every watcher event (create, modify, rename, delete). In UNIQUE constraint. |
| `watch_folders.path` | Read by queue priority JOIN on every dequeue. Path-validator reads it in polling loop. |
| `FileEvent::path` (watching/events.rs:12) | Lives on the hot-path from OS notify callbacks through debounce to queue write. Must be converted to canonical before DB write. |
| `PendingMove::old_path` (move_detector/types.rs:52) | Renamed-file correlation is time-critical (MOVED_FROM → MOVED_TO within debounce window). |
| `FilePayload::file_path` (payloads/filesystem.rs:17) | Serialized into Qdrant point payload JSON on every file ingest. |
| `TextSearchMatch::file_path` (proto:580) | Emitted for every FTS5 search result row — can be thousands per query. |

---

## Producer/Consumer Matrix for Proto Canonical Fields

For each canonical-class proto field: which handlers produce it, which consumers decode it. Informs T8 validation placement (handler-entry vs producer-side per spec §7.4).

### Handler-entry validated (request fields — incoming data)

| Message | Field | Producing client | Handler entry point |
|---|---|---|---|
| `RegisterProjectRequest` | `path` | wqm CLI / TS MCP session-lifecycle | `project_service/registration.rs` — `canonicalize_project_path()` currently (Category A, A-2 target) |
| `DeprioritizeProjectRequest` | `watch_path` | wqm CLI | `project_service/deactivation.rs` |
| `SetIncrementalRequest` | `file_paths` | wqm CLI | daemon library write service |
| `ImpactAnalysisRequest` | `file_path` | wqm CLI / TS graph tool | `graph_service/handlers.rs` |

### Producer-validated (response fields — outgoing data built by daemon)

| Message | Field | Build site | Notes |
|---|---|---|---|
| `RegisterProjectResponse` | `watch_path` | `project_service/registration.rs` | Sourced from `watch_folders.path` (DB-read canonical). |
| `GetProjectStatusResponse` | `project_root` | `project_service/queries.rs` | DB read. |
| `GetProjectStatusResponse` | `main_worktree_path` | `project_service/worktree.rs` | Built from worktree detection; must be validated before serialization. |
| `ProjectInfo` | `project_root` | `project_service/queries.rs` | DB read. |
| `ServerStatusNotification` | `project_root` | `system_service/helpers.rs` | Built from registered project record. |
| `TextSearchMatch` | `file_path` | `text_search_service.rs` | Decoded from `file_metadata.file_path` (DB canonical). |
| `TraversalNodeProto` | `file_path` | `graph_service/handlers.rs` | Decoded from `graph_nodes.file_path` (DB canonical). |
| `ImpactNodeProto` | `file_path` | `graph_service/handlers.rs` | Same. |
| `PageRankNodeProto` | `file_path` | `graph_service/handlers.rs` | Same. |
| `CommunityMemberProto` | `file_path` | `graph_service/handlers.rs` | Same. |
| `BetweennessNodeProto` | `file_path` | `graph_service/handlers.rs` | Same. |
| `CancelItemsResponse` | `project_path` | `queue_write_service.rs` | Sourced from `watch_folders.path`; display field. |

### Nested-message paths (producer-validated, see spec §7.4 item 3)

| Message (container) | Nested type | Field | Build site |
|---|---|---|---|
| `ProjectPayload` | direct | `file_absolute_path` | Qdrant write path in daemon; built when constructing the point payload. Producer must call `CanonicalPath::from_validated`. |
| `LspMetadata → SymbolReference` | `SymbolReference` | `file_path` | LSP enrichment pipeline in daemon; built during LSP symbol extraction. |
| `LibraryPayload` (proto) | direct | `source_file` | Library ingest pipeline in daemon. |

---

## Decisions Needed

The following items require architectural input before T6 (refactor task) closes:

1. **`DaemonConfig::project_path` (config/mod.rs:99)** — This `Option<PathBuf>` is used as a daemon working-directory override at startup. It is never stored in SQLite or sent over gRPC. However, it may flow into watch-folder registration indirectly. If it does, the registration handler must normalize it before persisting. Current classification: process-local. **Decision needed:** confirm it never flows into a persistence path unvalidated, or reclassify as canonical and add a validation gate at registration.

2. **`WatchMetadata::watch_path` in `daemon/grpc/src/services/project_service/worktree.rs:28`** — This field is classified canonical (it is returned in `RegisterProjectResponse.watch_path`). The worktree handler builds it from `std::fs::canonicalize()` (spec §3.2.2 site: `worktree.rs:97` — Category A). **Decision needed:** confirm A-2 will replace this with syntactic normalization to resolve the Category A violation.

3. **`LibraryDocumentPayload::library_path` (common/src/payloads/library.rs:93)** — Classified as relative/non-path. The field is a logical hierarchy label (e.g. `"cs/design_patterns"`), not a filesystem path. It is serialized into the Qdrant payload. **Decision needed:** confirm this is intentionally a label, not derived from a filesystem path, so it needs no `CanonicalPath` treatment.

---

## Cross-Reference with Spec §6.1

| Spec §6.1 entry | Audit finding | Status |
|---|---|---|
| `watch_folders.path` — canonical | Confirmed canonical (`schema/watch_folders_schema.sql:14`). | match |
| `tracked_files.file_path` — canonical | Confirmed canonical (`tracked_files_schema/schema.rs:12`). | match |
| `file_metadata.file_path` (search.db) — canonical | Confirmed canonical (`code_lines_schema.rs:158`). | match |
| `graph_nodes.file_path` — canonical | Confirmed canonical (`graph/schema.rs:186`). | match |
| `graph_edges.source_file` — canonical | Confirmed canonical (`graph/schema.rs:216`). | match |
| `unified_queue.file_path` — canonical | Confirmed canonical (`unified_queue_schema/sql.rs:42`). | match |
| `ignore_file_mtimes.project_root` — canonical | Confirmed canonical (`schema_version/v34.rs:24`). | match |
| `ignore_file_mtimes.file_path` — canonical | Confirmed canonical (`schema_version/v34.rs:25`). | match |
| `watch_folders.submodule_path` — relative | Confirmed relative. | match |
| `watch_folders.disambiguation_path` — disambiguation-suffix | Confirmed disambiguation-suffix. | match |
| `tracked_files.relative_path` — relative | Confirmed relative. | match |
| `file_metadata.relative_path` — relative | Confirmed relative. | match |
| `FilePayload.file_path` — canonical | Confirmed canonical (`payloads/filesystem.rs:17`). | match |
| `FilePayload.old_path` — canonical | Confirmed canonical (`payloads/filesystem.rs:29`). | match |
| `FolderPayload.folder_path` — canonical | Confirmed canonical (`payloads/filesystem.rs:36`). | match |
| `FolderPayload.old_path` — canonical | Confirmed canonical (`payloads/filesystem.rs:51`). | match |
| "Other Rust payload structs TBD by A-1" | Added: `LibraryDocumentPayload.document_path` (canonical), `LibraryDocumentPayload.library_path` (relative/non-path), `ImageSearchResult.file_path` (canonical). | **added** |
| `ProjectPayload.file_absolute_path` (proto:1338) — canonical | Confirmed canonical, nested-message class. | match |
| `LibraryPayload.source_file` (proto:1381) — canonical | Confirmed canonical (`proto:1381`). | match |
| `SymbolReference.file_path` (proto:1370) — canonical | Confirmed canonical, nested inside `LspMetadata`. | match |
| `TextSearchMatch.file_path` (proto:580) — canonical | Confirmed canonical. | match |
| `TraversalNodeProto.file_path` (proto:638) — canonical | Confirmed canonical. | match |
| `ImpactNodeProto.file_path` (proto:662) — canonical | Confirmed canonical. | match |
| `PageRankNodeProto.file_path` (proto:702) — canonical | Confirmed canonical. | match |
| `CommunityMemberProto.file_path` (proto:732) — canonical | Confirmed canonical. | match |
| `BetweennessNodeProto.file_path` (proto:755) — canonical | Confirmed canonical. | match |
| `ImpactAnalysisRequest.file_path` (proto:648) — canonical | Confirmed canonical (request field). | match |
| `ProjectPayload.file_path` (proto:1337) — relative | Confirmed relative. | match |
| `queue_config.database_path` — process-local | Confirmed process-local (`queue_config.rs:19`). | match |
| `graph::path` — process-local | Confirmed process-local (`graph/schema.rs:53` → `GraphDbManager::path`). | match |
| Config file path — process-local | Confirmed process-local (env var / `paths::find_config_file()`). | match |
| `LocalPath`-flavored fs API args — process-local | Confirmed: watching events, path-validator, move-detector types all use `PathBuf`, not `String`. | match |
| **New: `RegisterProjectRequest.path` (proto:429)** | Not in spec §6.1 proto section. Canonical request field. | **added** |
| **New: `DeprioritizeProjectRequest.watch_path` (proto:450)** | Not in spec §6.1. Canonical request field. | **added** |
| **New: `SetIncrementalRequest.file_paths` (proto:1083)** | Not in spec §6.1. Canonical `repeated string`. | **added** |
| **New: `RegisterProjectResponse.watch_path` (proto:444)** | Not in spec §6.1. Canonical response field. | **added** |
| **New: `GetProjectStatusResponse.project_root` (proto:468)** | Not in spec §6.1. Canonical response field. | **added** |
| **New: `GetProjectStatusResponse.main_worktree_path` (proto:475)** | Not in spec §6.1. Canonical response field. | **added** |
| **New: `ProjectInfo.project_root` (proto:492)** | Not in spec §6.1. Canonical response field. | **added** |
| **New: `ServerStatusNotification.project_root` (proto:272)** | Not in spec §6.1. Canonical notification field. | **added** |
| **New: `CancelItemsResponse.project_path` (proto:968)** | Not in spec §6.1. Canonical display field. | **added** |
| **New: `DaemonConfig::log_file`, `project_path`, `Config::database_path`** | Not in spec §6.1 process-local list. Process-local. | **added** |
| **New: `LoggingConfig::log_file_path`** | Not in spec §6.1. Process-local. | **added** |
| **New: `LadybugConfig::db_path`** | Not in spec §6.1. Process-local. | **added** |
| **New: `TlsConfig::cert_path`, `key_path`, `ca_cert_path`** | Not in spec §6.1. Process-local. | **added** |
| **New: `Profile::database_path` (cli_profiles.rs)** | Not in spec §6.1. Process-local. | **added** |
| **New: `YamlLspConfig::user_path`** | Not in spec §6.1. Process-local. | **added** |
| **New: watcher / move-detector `PathBuf` fields** | Not in spec §6.1. Process-local (never reach persistence boundary unvalidated). | **added** |
| **New: CLI display path fields** | Not in spec §6.1. Canonical (sourced from DB). No separate storage. | **added** |
| **New: `ImageSearchResult::file_path`** | Not in spec §6.1. Canonical (decoded from Qdrant). | **added** |
| **New: TraversalNodeProto.path, TextSearchRequest.path_glob/prefix** | Not in spec §6.1. Non-path false positives. | **added** |

