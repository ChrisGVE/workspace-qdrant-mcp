# Path Abstraction Audit (A-1)

**Status:** Complete (revised under root/relative discipline)  
**Produced by:** Audit task A-1 (task-master tag: path-abstraction, task ID: 4)  
**Date:** 2026-05-14  
**Revision date:** 2026-05-14 (reclassification: only roots are canonical; file paths are relative)  
**Spec reference:** `docs/specs/16-path-abstraction.md` §6.1, §6.1.1, §6.3

---

## Summary

| Class | Count |
|---|---|
| canonical | 8 |
| relative | 30 |
| dropped-in-v37 | 2 |
| process-local | 16 |
| disambiguation-suffix | 2 |
| non-path | 10 |
| **total** | **68** |

Classification definitions (from spec §6.1.1, §6.3):

- **canonical** — host-absolute ROOT path only (`watch_folders.path`, `ignore_file_mtimes.project_root`, library roots, gRPC root fields). Must become `CanonicalPath`.
- **relative** — content path anchored to a watch_folder or library root; portable across clones and deployment modes. Must become `RelativePath`. Previously misclassified as canonical.
- **dropped-in-v37** — denormalized absolute `file_path` columns eliminated in schema v37. Replaced by `(watch_folder_id, relative_path, branch)` UNIQUE constraint.
- **process-local** — SQLite DB file path, config path, log path, etc.; never serialized over gRPC or stored in Qdrant.
- **disambiguation-suffix** — path suffix for clone disambiguation; relative semantically; see spec §6.1.
- **non-path** — field name matches pattern but is not a filesystem path (e.g. traversal description string, URL, filter pattern).

---

## SQLite Columns

### Canonical — Root Paths Only

| Site | Table | Column | Type | Class | Notes |
|---|---|---|---|---|---|
| `schema/watch_folders_schema.sql:14` | `watch_folders` | `path` | `TEXT NOT NULL UNIQUE` | canonical | Project/library root; absolute. Hot-path: read by queue priority JOIN and path-validator loop. |
| `schema_version/v34.rs:24` | `ignore_file_mtimes` | `project_root` | `TEXT NOT NULL` (PK part) | canonical | Absolute path to project root; part of composite PK. |

### Dropped in v37 — Denormalized Absolute Columns Eliminated

| Site | Table | Column | Type | Class | Notes |
|---|---|---|---|---|---|
| `tracked_files_schema/schema.rs:12` | `tracked_files` | `file_path` | `TEXT NOT NULL` | **dropped-in-v37** | Was absolute. DROPPED in schema v37. Replaced by `(watch_folder_id, relative_path, branch)` UNIQUE constraint. Previously in UNIQUE constraint `(watch_folder_id, file_path, branch)`. |
| `code_lines_schema.rs:158` | `file_metadata` (search.db) | `file_path` | `TEXT NOT NULL` | **dropped-in-v37** | Was denormalized absolute path copied from `tracked_files.file_path` for FTS5 scoping. DROPPED in schema v37. |

### Relative — Content Paths (must become `RelativePath`)

| Site | Table | Column | Anchor | Class | Notes |
|---|---|---|---|---|---|
| `tracked_files_schema/schema.rs:29` | `tracked_files` | `relative_path` | `watch_folders.path` | relative | Surviving key. UNIQUE constraint rebuilt on `(watch_folder_id, relative_path, branch)` in v37. Hot-path: every watcher event writes here. |
| `code_lines_schema.rs:168` | `file_metadata` (search.db) | `relative_path` | `watch_folders.path` | relative | Surviving key. Added in search.db migration v5. |
| `graph/schema.rs:186` | `graph_nodes` | `file_path` | `watch_folders.path` | relative | **Reclassified** from canonical. Previously stored absolute path; stores relative path going forward in v37. |
| `graph/schema.rs:216` | `graph_edges` | `source_file` | `watch_folders.path` | relative | **Reclassified** from canonical. Previously stored absolute path; stores relative path going forward in v37. |
| `unified_queue_schema/sql.rs:42` | `unified_queue` | `file_path` | `watch_folders.path` via `watch_folder_id` | relative | **Reclassified** from canonical. Relative path for per-file dedup; NULL for non-file item types. Hot-path: on every enqueue. |
| `schema_version/v34.rs:25` | `ignore_file_mtimes` | `file_path` | `ignore_file_mtimes.project_root` (same row) | relative | **Reclassified** from canonical. Anchored to `project_root` in the same row. Path to `.gitignore`/`.wqmignore` file relative to the project root. |
| `schema/watch_folders_schema.sql:21` | `watch_folders` | `submodule_path` | Parent `watch_folders.path` | relative | NULL for top-level watches. |

### Disambiguation-suffix

| Site | Table | Column | Class | Notes |
|---|---|---|---|---|
| `schema/watch_folders_schema.sql:26` | `watch_folders` | `disambiguation_path` | disambiguation-suffix | Path suffix for clone disambiguation; not a full path; see spec §6.1. Treat as `String` unless a `DisambiguationSuffix` newtype is introduced. |

---

## Qdrant Payloads (Rust serde structs — `common/src/payloads/`)

### Relative — Content Paths (must become `RelativePath`)

All file-level payload fields are content paths inside a project or library, and
are **reclassified from canonical to relative** under the corrected discipline.

| Site | Struct | Field | Rust type | Anchor | Class | Notes |
|---|---|---|---|---|---|---|
| `common/src/payloads/filesystem.rs:17` | `FilePayload` | `file_path` | `String` | `watch_folders.path` | relative | **Reclassified.** File path relative to owning watch_folder root. Previously stored absolute. |
| `common/src/payloads/filesystem.rs:29` | `FilePayload` | `old_path` | `Option<String>` | `watch_folders.path` | relative | **Reclassified.** Previous relative path before rename. Previously stored absolute. |
| `common/src/payloads/filesystem.rs:36` | `FolderPayload` | `folder_path` | `String` | `watch_folders.path` | relative | **Reclassified.** Folder path relative to owning watch_folder root. Previously stored absolute. |
| `common/src/payloads/filesystem.rs:51` | `FolderPayload` | `old_path` | `Option<String>` | `watch_folders.path` | relative | **Reclassified.** Previous relative folder path before rename. Previously stored absolute. |
| `common/src/payloads/library.rs:78` | `LibraryDocumentPayload` | `document_path` | `String` | library root | relative | **Reclassified.** Document path relative to library root. Previously stored as absolute path to library document on disk. |
| `daemon/core/src/image_search.rs:79` | `ImageSearchResult` | `file_path` | `String` | `watch_folders.path` | relative | **Reclassified.** Relative path to source document; decoded from Qdrant `images` collection payload. Previously stored absolute. |
| `common/src/payloads/library.rs:93` | `LibraryDocumentPayload` | `library_path` | Relative within library hierarchy | logical | relative | e.g. `"cs/design_patterns"`. Empty string for root-level docs. Logical hierarchy label, not a direct filesystem path — but relative semantics apply (no `..`, no leading `/`). |

### Serde non-path fields in payloads (false positives)

| Site | Struct | Field | Class | Notes |
|---|---|---|---|---|
| `common/src/payloads/library.rs:35` | `LibraryPayload` | `source_url` | non-path | URL string; not a filesystem path. |

---

## Proto-defined Messages (`.proto` source — prost-generated in Rust)

### Canonical — Root paths only (request messages, handler-validated at entry)

| Proto line | Message | Field | Class | Notes |
|---|---|---|---|---|
| `proto:429` | `RegisterProjectRequest` | `path` | canonical | Absolute path to project root; feeds `watch_folders.path`. Primary site: `project_service/registration.rs`. |
| `proto:450` | `DeprioritizeProjectRequest` | `watch_path` | canonical | Optional; absolute path to specific watch root for multi-clone disambiguation. |

### Relative — Content paths (request messages, handler-validated at entry)

| Proto line | Message | Field | Class | Notes |
|---|---|---|---|---|
| `proto:1083` | `SetIncrementalRequest` | `file_paths` | relative | **Reclassified.** `repeated string`; each element is a file path relative to the project root. See spec §7.4 — requires `extract_relative_paths!` macro. |
| `proto:648` | `ImpactAnalysisRequest` | `file_path` | relative | **Reclassified.** Optional; file path relative to project root; narrows graph query. |

### Canonical — Root paths (response messages, producer-validated)

| Proto line | Message | Field | Class | Notes |
|---|---|---|---|---|
| `proto:444` | `RegisterProjectResponse` | `watch_path` | canonical | Confirmed registered watch root; built by registration handler. |
| `proto:468` | `GetProjectStatusResponse` | `project_root` | canonical | Absolute project root from `watch_folders.path`. |
| `proto:475` | `GetProjectStatusResponse` | `main_worktree_path` | canonical | Absolute path to main working tree when project is a worktree. |
| `proto:492` | `ProjectInfo` | `project_root` | canonical | Absolute project root in list response. |
| `proto:272` | `ServerStatusNotification` | `project_root` | canonical | Optional; absolute project root sent in server status events. |
| `proto:968` | `CancelItemsResponse` | `project_path` | canonical | Display field; sourced from `watch_folders.path`. |

### Relative — Content paths (response messages, producer-validated)

| Proto line | Message | Field | Class | Notes |
|---|---|---|---|---|
| `proto:580` | `TextSearchMatch` | `file_path` | relative | **Reclassified.** File path relative to project root; sourced from `file_metadata.relative_path`. Hot-path: emitted for every search result. |
| `proto:638` | `TraversalNodeProto` | `file_path` | relative | **Reclassified.** Relative path to file containing the traversal node; sourced from `graph_nodes.file_path`. |
| `proto:662` | `ImpactNodeProto` | `file_path` | relative | **Reclassified.** Relative path to file containing impacted node; sourced from `graph_nodes.file_path`. |
| `proto:702` | `PageRankNodeProto` | `file_path` | relative | **Reclassified.** Relative path; sourced from `graph_nodes.file_path`. |
| `proto:732` | `CommunityMemberProto` | `file_path` | relative | **Reclassified.** Relative path; sourced from `graph_nodes.file_path`. |
| `proto:755` | `BetweennessNodeProto` | `file_path` | relative | **Reclassified.** Relative path; sourced from `graph_nodes.file_path`. |

### Canonical — library root (nested-message path, producer-validated per spec §7.4)

| Proto line | Message | Field | Class | Notes |
|---|---|---|---|---|
| `proto:1381` | `LibraryPayload` (proto) | `source_file` | canonical | Absolute path to the original library root document. Nested in proto-defined `LibraryPayload` (distinct from serde `LibraryPayload`). |

### Relative — Content paths (nested-message paths, producer-validated per spec §7.4)

| Proto line | Message | Field | Class | Notes |
|---|---|---|---|---|
| `proto:1337` | `ProjectPayload` | `file_path` | relative | Relative to project root. Aligns with `tracked_files.relative_path`. Must NOT be wrapped in `CanonicalPath`. |
| `proto:1338` | `ProjectPayload` | `file_absolute_path` | canonical | Optional display/reference field. Reconstructed from root + relative. Canonical when present. Producer must validate. |
| `proto:1370` | `SymbolReference` | `file_path` | relative | **Reclassified.** File path relative to project root; nested inside `LspMetadata.references` (`ProjectPayload → LspMetadata → SymbolReference`). Producer-validated rule applies. |

### Non-path proto fields (false positives)

| Proto line | Message | Field | Class | Notes |
|---|---|---|---|---|
| `proto:641` | `TraversalNodeProto` | `path` | non-path | Traversal path description string (e.g. `"A → B → C"`); not a filesystem path. |
| `proto:558` | `TextSearchRequest` | `path_glob` | non-path | Glob pattern for filtering (e.g. `"**/*.rs"`); not a concrete path. |
| `proto:559` | `TextSearchRequest` | `path_prefix` | non-path | Prefix pattern for filtering (e.g. `"src/"`); may be a path prefix but is used as a LIKE filter, not a standalone path. |

---

## TypeScript MCP Server (`src/typescript/mcp-server/src/`)

These fields mirror the proto messages over gRPC. Reclassified under root/relative discipline.

### Canonical — Root paths

| File | Interface | Field | Class | Notes |
|---|---|---|---|---|
| `clients/grpc-types-messages-document-project.ts:42` | `RegisterProjectRequest` | `path` | canonical | Mirrors proto `RegisterProjectRequest.path`. Root path. Sent from TS → gRPC. |
| `clients/grpc-types-messages-document-project.ts:57` | `RegisterProjectResponse` | `watch_path` | canonical | Mirrors proto response. Watch root. |
| `clients/grpc-types-messages-document-project.ts:62` | `DeprioritizeProjectRequest` | `watch_path` | canonical | Mirrors proto request. Watch root for disambiguation. |
| `clients/grpc-types-messages-document-project.ts:79` | `GetProjectStatusResponse` | `project_root` | canonical | Mirrors proto response. Root. |
| `clients/grpc-types-messages-document-project.ts:95` | `ProjectInfo` | `project_root` | canonical | Mirrors proto. Root. |
| `clients/grpc-types-messages-system-collection.ts:72` | `ServerStatusNotification` | `project_root` | canonical | Mirrors proto. Root. |

### Relative — Content paths

| File | Interface | Field | Class | Notes |
|---|---|---|---|---|
| `clients/grpc-types-search-graph.ts:30` | `TextSearchMatch` | `file_path` | relative | **Reclassified.** Mirrors proto. File path relative to project root. Returned to LLM client. |
| `clients/grpc-types-search-graph.ts:58` | `TraversalNodeProto` | `file_path` | relative | **Reclassified.** Mirrors proto. Relative to project root. |
| `clients/grpc-types-search-graph.ts:67` | `ImpactAnalysisRequest` | `file_path` | relative | **Reclassified.** Optional; mirrors proto. Relative to project root. |
| `clients/grpc-types-search-graph.ts:79` | `ImpactNodeProto` | `file_path` | relative | **Reclassified.** Mirrors proto. Relative to project root. |

### Non-path / disambiguation

| File | Interface | Field | Class | Notes |
|---|---|---|---|---|
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
| `unified_queue.file_path` | Every file enqueue/dedup check touches this column with a composite UNIQUE index. Now stores relative path. |
| `tracked_files.relative_path` | Written on every watcher event (create, modify, rename, delete). Surviving key after `file_path` drop; in UNIQUE constraint `(watch_folder_id, relative_path, branch)`. |
| `watch_folders.path` | Read by queue priority JOIN on every dequeue. Path-validator reads it in polling loop. Canonical root. |
| `FileEvent::path` (watching/events.rs:12) | Lives on the hot-path from OS notify callbacks through debounce to queue write. Must be converted to `RelativePath` (relative to watch root) before DB write. |
| `PendingMove::old_path` (move_detector/types.rs:52) | Renamed-file correlation is time-critical (MOVED_FROM → MOVED_TO within debounce window). |
| `FilePayload::file_path` (payloads/filesystem.rs:17) | Serialized into Qdrant point payload JSON on every file ingest. Now stores relative path. |
| `TextSearchMatch::file_path` (proto:580) | Emitted for every FTS5 search result row — can be thousands per query. Now relative path sourced from `file_metadata.relative_path`. |

---

## Producer/Consumer Matrix for Proto Path Fields

For each canonical/relative proto field: which handlers produce it, which consumers decode it. Informs T8 validation placement (handler-entry vs producer-side per spec §7.4).

### Handler-entry validated (request fields — incoming data)

| Message | Field | Class | Producing client | Handler entry point |
|---|---|---|---|---|
| `RegisterProjectRequest` | `path` | canonical | wqm CLI / TS MCP session-lifecycle | `project_service/registration.rs` — `canonicalize_project_path()` currently (Category A, A-2 target). Validate as `CanonicalPath`. |
| `DeprioritizeProjectRequest` | `watch_path` | canonical | wqm CLI | `project_service/deactivation.rs`. Validate as `CanonicalPath`. |
| `SetIncrementalRequest` | `file_paths` | relative | wqm CLI | daemon library write service. **Reclassified.** Validate as `RelativePath` via `extract_relative_paths!`. |
| `ImpactAnalysisRequest` | `file_path` | relative | wqm CLI / TS graph tool | `graph_service/handlers.rs`. **Reclassified.** Validate as `RelativePath`. |

### Producer-validated (response fields — outgoing data built by daemon)

| Message | Field | Class | Build site | Notes |
|---|---|---|---|---|
| `RegisterProjectResponse` | `watch_path` | canonical | `project_service/registration.rs` | Sourced from `watch_folders.path` (DB-read canonical). |
| `GetProjectStatusResponse` | `project_root` | canonical | `project_service/queries.rs` | DB read. |
| `GetProjectStatusResponse` | `main_worktree_path` | canonical | `project_service/worktree.rs` | Built from worktree detection; must be validated before serialization. |
| `ProjectInfo` | `project_root` | canonical | `project_service/queries.rs` | DB read. |
| `ServerStatusNotification` | `project_root` | canonical | `system_service/helpers.rs` | Built from registered project record. |
| `CancelItemsResponse` | `project_path` | canonical | `queue_write_service.rs` | Sourced from `watch_folders.path`; display field. |
| `TextSearchMatch` | `file_path` | relative | `text_search_service.rs` | **Reclassified.** Decoded from `file_metadata.relative_path` (DB relative). |
| `TraversalNodeProto` | `file_path` | relative | `graph_service/handlers.rs` | **Reclassified.** Decoded from `graph_nodes.file_path` (DB relative). |
| `ImpactNodeProto` | `file_path` | relative | `graph_service/handlers.rs` | **Reclassified.** Same source. |
| `PageRankNodeProto` | `file_path` | relative | `graph_service/handlers.rs` | **Reclassified.** Same source. |
| `CommunityMemberProto` | `file_path` | relative | `graph_service/handlers.rs` | **Reclassified.** Same source. |
| `BetweennessNodeProto` | `file_path` | relative | `graph_service/handlers.rs` | **Reclassified.** Same source. |

### Nested-message paths (producer-validated, see spec §7.4)

| Message (container) | Nested type | Field | Class | Build site | Notes |
|---|---|---|---|---|---|
| `ProjectPayload` | direct | `file_absolute_path` | canonical | Qdrant write path in daemon; optional display field reconstructed from root + relative. Producer must call `CanonicalPath::from_validated`. |
| `ProjectPayload` | direct | `file_path` | relative | Qdrant write path in daemon; relative to project root. Producer must call `RelativePath::from_validated`. |
| `LspMetadata → SymbolReference` | `SymbolReference` | `file_path` | relative | **Reclassified.** LSP enrichment pipeline in daemon; built during LSP symbol extraction. Relative to project root. |
| `LibraryPayload` (proto) | direct | `source_file` | canonical | Library ingest pipeline in daemon. Library root document; absolute. |

---

## Decisions Needed

The following items require architectural input before T6 (refactor task) closes:

1. **`DaemonConfig::project_path` (config/mod.rs:99)** — This `Option<PathBuf>` is used as a daemon working-directory override at startup. It is never stored in SQLite or sent over gRPC. However, it may flow into watch-folder registration indirectly. If it does, the registration handler must normalize it before persisting. Current classification: process-local. **Decision needed:** confirm it never flows into a persistence path unvalidated, or reclassify as canonical and add a validation gate at registration.

2. **`WatchMetadata::watch_path` in `daemon/grpc/src/services/project_service/worktree.rs:28`** — This field is classified canonical (it is returned in `RegisterProjectResponse.watch_path`). The worktree handler builds it from `std::fs::canonicalize()` (spec §3.2.2 site: `worktree.rs:97` — Category A). **Decision needed:** confirm A-2 will replace this with syntactic normalization to resolve the Category A violation.

3. **`LibraryDocumentPayload::library_path` (common/src/payloads/library.rs:93)** — Classified as relative (logical hierarchy label). The field is a hierarchy label (e.g. `"cs/design_patterns"`), not a direct filesystem path. It is serialized into the Qdrant payload. Confirmed: this is intentionally a label, not derived from a filesystem path; no `CanonicalPath` treatment needed. The `RelativePath` rules (no `..`, no leading `/`) still apply as a best-practice guard. **Resolved:** leave as `String` with input validation; no `RelativePath` newtype required for label fields.

---

## Cross-Reference with Spec §6.1

| Spec §6.1 entry | Audit finding | Status |
|---|---|---|
| `watch_folders.path` — canonical | Confirmed canonical (root only). `schema/watch_folders_schema.sql:14`. | match |
| `ignore_file_mtimes.project_root` — canonical | Confirmed canonical (root). `schema_version/v34.rs:24`. | match |
| `tracked_files.file_path` — **DROPPED in v37** | Reclassified: was absolute, now eliminated. `tracked_files_schema/schema.rs:12`. | **reclassified** |
| `file_metadata.file_path` (search.db) — **DROPPED in v37** | Reclassified: was absolute/denormalized, now eliminated. `code_lines_schema.rs:158`. | **reclassified** |
| `graph_nodes.file_path` — **relative** | **Reclassified** from canonical. Stores relative path going forward. `graph/schema.rs:186`. | **reclassified** |
| `graph_edges.source_file` — **relative** | **Reclassified** from canonical. Stores relative path going forward. `graph/schema.rs:216`. | **reclassified** |
| `unified_queue.file_path` — **relative** | **Reclassified** from canonical. Relative path for per-file dedup. `unified_queue_schema/sql.rs:42`. | **reclassified** |
| `ignore_file_mtimes.file_path` — **relative** | **Reclassified** from canonical. Anchored to `project_root` in same row. `schema_version/v34.rs:25`. | **reclassified** |
| `watch_folders.submodule_path` — relative | Confirmed relative. | match |
| `watch_folders.disambiguation_path` — disambiguation-suffix | Confirmed disambiguation-suffix. | match |
| `tracked_files.relative_path` — relative | Confirmed relative. Surviving key after `file_path` drop. | match |
| `file_metadata.relative_path` — relative | Confirmed relative. Surviving key after `file_path` drop. | match |
| `FilePayload.file_path` — **relative** | **Reclassified** from canonical. Relative to watch_folder root. `payloads/filesystem.rs:17`. | **reclassified** |
| `FilePayload.old_path` — **relative** | **Reclassified** from canonical. Relative path before rename. `payloads/filesystem.rs:29`. | **reclassified** |
| `FolderPayload.folder_path` — **relative** | **Reclassified** from canonical. Relative folder path. `payloads/filesystem.rs:36`. | **reclassified** |
| `FolderPayload.old_path` — **relative** | **Reclassified** from canonical. Relative folder path before rename. `payloads/filesystem.rs:51`. | **reclassified** |
| `LibraryDocumentPayload.document_path` — **relative** | **Reclassified** from canonical. Relative to library root. `payloads/library.rs:78`. | **reclassified** |
| `LibraryDocumentPayload.library_path` — relative | Confirmed relative (logical hierarchy label). `payloads/library.rs:93`. | match |
| `ImageSearchResult.file_path` — **relative** | **Reclassified** from canonical. Relative to watch_folder root. `daemon/core/src/image_search.rs:79`. | **reclassified** |
| `ProjectPayload.file_absolute_path` (proto:1338) — canonical | Optional display/reference field; canonical when present. Confirmed canonical. | match |
| `LibraryPayload.source_file` (proto:1381) — canonical | Library root document. Confirmed canonical. | match |
| `SymbolReference.file_path` (proto:1370) — **relative** | **Reclassified** from canonical. Relative to project root; nested inside `LspMetadata`. | **reclassified** |
| `TextSearchMatch.file_path` (proto:580) — **relative** | **Reclassified** from canonical. Sourced from `file_metadata.relative_path`. | **reclassified** |
| `TraversalNodeProto.file_path` (proto:638) — **relative** | **Reclassified** from canonical. Sourced from `graph_nodes.file_path`. | **reclassified** |
| `ImpactNodeProto.file_path` (proto:662) — **relative** | **Reclassified** from canonical. | **reclassified** |
| `PageRankNodeProto.file_path` (proto:702) — **relative** | **Reclassified** from canonical. | **reclassified** |
| `CommunityMemberProto.file_path` (proto:732) — **relative** | **Reclassified** from canonical. | **reclassified** |
| `BetweennessNodeProto.file_path` (proto:755) — **relative** | **Reclassified** from canonical. | **reclassified** |
| `ImpactAnalysisRequest.file_path` (proto:648) — **relative** | **Reclassified** from canonical. File path relative to project root. | **reclassified** |
| `SetIncrementalRequest.file_paths` (proto:1083) — **relative** | **Reclassified** from canonical. Each element relative to project root. | **reclassified** |
| `ProjectPayload.file_path` (proto:1337) — relative | Confirmed relative. | match |
| `RegisterProjectRequest.path` (proto:429) — canonical | Root path. Confirmed canonical. | match |
| `DeprioritizeProjectRequest.watch_path` (proto:450) — canonical | Root path. Confirmed canonical. | match |
| `RegisterProjectResponse.watch_path` (proto:444) — canonical | Watch root. Confirmed canonical. | match |
| `GetProjectStatusResponse.project_root` (proto:468) — canonical | Root. Confirmed canonical. | match |
| `GetProjectStatusResponse.main_worktree_path` (proto:475) — canonical | Root. Confirmed canonical. | match |
| `ProjectInfo.project_root` (proto:492) — canonical | Root. Confirmed canonical. | match |
| `ServerStatusNotification.project_root` (proto:272) — canonical | Root. Confirmed canonical. | match |
| `CancelItemsResponse.project_path` (proto:968) — canonical | Display root. Confirmed canonical. | match |
| `queue_config.database_path` — process-local | Confirmed process-local (`queue_config.rs:19`). | match |
| `graph::path` — process-local | Confirmed process-local (`graph/schema.rs:53` → `GraphDbManager::path`). | match |
| Config file path — process-local | Confirmed process-local (env var / `paths::find_config_file()`). | match |
| `LocalPath`-flavored fs API args — process-local | Confirmed: watching events, path-validator, move-detector types all use `PathBuf`, not `String`. | match |
| `DaemonConfig::log_file`, `project_path`, `Config::database_path` — process-local | Confirmed process-local. | match |
| `LoggingConfig::log_file_path` — process-local | Confirmed process-local. | match |
| `LadybugConfig::db_path` — process-local | Confirmed process-local. | match |
| `TlsConfig::cert_path`, `key_path`, `ca_cert_path` — process-local | Confirmed process-local. | match |
| `Profile::database_path` (cli_profiles.rs) — process-local | Confirmed process-local. | match |
| `YamlLspConfig::user_path` — process-local | Confirmed process-local. | match |
| watcher / move-detector `PathBuf` fields — process-local | Confirmed process-local (never reach persistence boundary unvalidated). | match |
| CLI display path fields — canonical (sourced from DB roots) | Confirmed canonical (sourced from `watch_folders.path`). No separate storage. | match |
| `TraversalNodeProto.path`, `TextSearchRequest.path_glob/prefix` — non-path | Confirmed non-path false positives. | match |

