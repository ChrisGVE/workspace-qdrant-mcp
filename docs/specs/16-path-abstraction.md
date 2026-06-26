# Spec 16: Path Abstraction

**Status:** Draft
**Owner:** core
**Last updated:** 2026-05-14 (revised: root/relative discipline)
**Depends on:** 06-file-watching, 12-configuration
**Blocks:** observability stack, docker-memexd image, host/docker parity

## 1. Goals

Establish a single, type-system-enforced discipline for filesystem paths
that allows memexd, wqm CLI, and the MCP server to be deployed in any
combination of host-native or containerized form without data loss,
duplicated configuration, or silent state divergence.

Specifically:

1. The user perceives no behavioral difference between host-only,
   docker-only, or hybrid deployments.
2. Switching deployment mode does not lose data already ingested.
3. Paths stored in SQLite, transmitted over gRPC, and returned in MCP
   responses are always expressed in a single canonical form.
4. Each module translates between canonical and process-local paths at
   exactly one boundary -- the filesystem I/O layer.
5. Mount mappings (host <-> container directory pairings) are declared in
   one place: the shared `config.yaml`.

## 2. Non-Goals

- Cross-machine portability of state. State is per-user, per-machine.
- Multi-user concurrent access to the same SQLite or Qdrant data dir.
- Path translation for content **inside** indexed files (e.g., rewriting
  source-code import paths). Only filesystem paths of indexed files
  themselves.
- Backward compatibility with pre-spec DB layouts (no users -- pre-release).

## 3. Canonical Form

The **canonical path** is a host-absolute, syntactically normalized path
as the user would type it on their own machine, regardless of whether
any individual module is currently running inside a container.

### 3.1 Normalization Rules

A canonical path:

1. Is absolute. Relative inputs are rejected.
2. Has `~` expanded to the user's home directory (`$HOME`).
3. Has `.` segments removed.
4. Contains NO `..` segments after normalization. Inputs containing
   `..` are rejected at `from_user_input`. Rationale: §3.2.1.
5. Has duplicate `/` collapsed to single `/`.
6. Preserves case exactly as input. No case folding.
7. Does NOT resolve symbolic links. Symlinks remain as written.
8. Does NOT touch the filesystem during normalization. Pure string op.
9. Has UTF-8 validity. Non-UTF-8 paths are rejected.

The rules apply identically in Rust and TypeScript implementations.

### 3.3 Relative Paths

A **relative path** names content **inside** a project or library, anchored to its owning
root. Relative paths are the correct type for all file-level content fields -- every
`file_path`, `source_file`, `folder_path`, `document_path`, and similar field that refers
to a file inside a project or library.

A `RelativePath` value:

1. Is NOT absolute. An absolute input is rejected.
2. Contains NO `..` segments. Inputs with `..` are rejected (traversal risk; the anchor
   is trusted, the content within it must not escape).
3. Contains NO `.` segments after normalization. Inputs with `.` are cleaned.
4. Has no leading `/`.
5. Has no duplicate `/` segments.
6. Is UTF-8.
7. Is syntactically normalized (rules 3-5 applied at construction time, pure string op).

`RelativePath` has NO mount-map translation -- it is deployment-independent by construction.
A relative path stored in one clone of a project is valid in every other clone sharing the
same `tenant_id`, and is equally valid inside or outside a container.

Reconstruction of the absolute path at read time:

```
absolute = watch_folders.path + "/" + relative_path
```

This JOIN replaces all denormalized absolute `file_path` columns that were previously
stored redundantly alongside `relative_path`.

### 3.2 Why Not `canonicalize()`

`Path::canonicalize` (Rust) and `realpath` (POSIX) resolve symlinks and
require fs access. This produces machine-specific output that differs
between host and container even when both view the same logical file
through different mounts. For canonical storage we need a pure
syntactic form.

#### 3.2.1 Why Reject `..` Instead of Resolving Syntactically

Syntactic `..` resolution is unsound across symlink boundaries. If
`$HOME/dev` is a symlink to `/Volumes/fast/dev`, then
`$HOME/dev/project/../other` syntactically collapses to
`$HOME/other`, but the real filesystem target is
`/Volumes/fast/other`. Since rule 7 forbids symlink resolution and
rule 4 originally required syntactic `..` resolution, the two rules
combined produce paths that do not correspond to any real location.

Resolution: reject `..` entirely in canonical-path constructors. Shells
expand `..` before arguments reach the process, so genuine user input
containing `..` is rare. The few internal call sites that synthesize
paths with `..` (e.g., test fixtures) must normalize before passing to
the constructor.

#### 3.2.2 Existing `canonicalize()` Call Sites -- Two Categories

`canonicalize()` is used in the codebase for two distinct purposes,
only one of which this spec forbids:

**Category A -- Canonical-storage derivation (FORBIDDEN).** The result
flows into a path that is persisted to SQLite, transmitted over gRPC,
written to a Qdrant payload, or returned to the MCP client. These
sites must be removed and replaced with syntactic normalization. Task
A-2 enumerates and rewrites every Category A site.

**Category B -- Process-local path arithmetic (PERMITTED with
restrictions).** The result never leaves the current process as a
stored or transmitted value. Examples: resolving `../..` written by
git into `.git/worktrees/<n>/commondir`; security-sensitive
containment checks ("is this extracted HTML image path inside
`base_dir`?"); intermediate `PathBuf` math during filesystem walks.
These sites must:

1. Use `PathBuf`, not `CanonicalPath`, for the canonicalized result.
2. Document why canonicalization is needed at the call site.
3. Never construct a `CanonicalPath` from the result without
   re-deriving via the syntactic rules in §3.1.
4. **Never convert the result to `String` via `to_string_lossy()` and
   pass the resulting `String` outside the immediate function.** A
   `to_string_lossy()` call assigned to a local binding that is then
   passed as a `String` parameter to any other function is forbidden,
   because there is no compile-time type-system signal that the value
   was canonicalize-derived and might end up in a SQL bind or gRPC
   payload. Permitted uses of `to_string_lossy()`: inline inside a
   `debug!`/`info!`/`error!` log macro argument, or inside a
   `Display`/`Debug` impl. Forbidden: `let s = p.to_string_lossy().to_string();`
   followed by passing `s` further. The CI grep job
   `path-discipline.sh` flags any binding pattern that captures
   `to_string_lossy()` output near a `canonicalize()` call.

Audit task A-2 classifies every call site below into Category A or B
and applies the appropriate treatment. The list is exhaustive as of
spec drafting; the audit verifies no further sites exist.

**CLI (`src/rust/cli/src/commands/`):**

- `project/resolver.rs:16, 147` -- Category A (project root -> stored)
- `ingest/file_folder.rs:26, 89` -- Category A (ingest target paths)
- `library/watch_cmd.rs:83` -- Category A
- `library/add.rs:24` -- Category A
- `library/ingest.rs:38` -- Category A
- `library/set_incremental.rs:16` -- Category A
- `project/register.rs:68` -- Category A

**Common (`src/rust/common/src/`):**

- `project_id/calculator.rs:52` -- Category A (project ID derivation;
  §3.2.3 covers semantic change)

**Daemon gRPC (`src/rust/daemon/grpc/src/`):**

- `services/project_service/registration.rs:72, 104, 371` -- Category A.
  `canonicalize_project_path()` derives the value stored as
  `watch_folders.path` (project root) and returned in gRPC responses.
  Migration: replace with syntactic normalization (§3.1). Line 105
  also uses the `p.to_string_lossy().to_string()` binding pattern
  forbidden under Category B rule 4 -- must be rewritten regardless of
  category.
- `services/project_service/worktree.rs:97` -- Category A and Category
  B rule 4 violation: combines `std::fs::canonicalize(...).unwrap_or_else(...)`
  with `to_string_lossy().to_string()` binding, then persists to a
  worktree record. Highest-priority site in A-2 because the path
  reaches both SQLite storage and gRPC.

**Daemon core (`src/rust/daemon/core/src/`):**

- `watching/platform/macos.rs:59` (`resolve_symlink`) -- Category A;
  test §11.1 case 8e gates the redesign
- `watching/platform/windows.rs:72` -- Category A (analogous to macOS
  resolve_symlink; affects Windows target -- currently out-of-scope
  per §13 last bullet, but the call must be annotated/removed
  consistently with macOS even if Windows tests are deferred)
- `watching/file_watcher/handle.rs:51, 86` -- likely Category A
  (registration path stored in watcher state) -- A-2 verifies
- `image_extraction/html.rs:48, 52` -- Category B (path-traversal
  containment check). Replacement: syntactic containment check using
  normalized paths plus an explicit verification that no component is
  a symlink that escapes `base_dir`; or retain `canonicalize()` here
  with a comment justifying Category B status
- `write_actor/exec_watch.rs:156` -- A-2 classifies
- `write_actor/exec_queue.rs:438` -- A-2 classifies
- `git/branch_detector.rs:67, 182` -- Category B (git-internal path
  resolution; never stored)
- `git/worktree.rs:48` -- Category B (resolves `../..` in
  `commondir`; never stored)
- `patterns/eligibility_trie.rs:105` -- A-2 classifies
- `patterns/gitignore.rs:140, 141` -- A-2 classifies

**Test files in scope (assertions rely on `canonicalize()` semantics
that change under this spec; A-2 updates the assertions):**

- `git/reflog.rs:356, 357` -- currently asserts `result.canonicalize()
  == main_git_dir.canonicalize()`; post-migration, must assert
  equality of syntactically normalized paths instead. The test files
  in `git/worktree.rs:94, 113, 123, 180` are similar and must also be
  updated.
- `daemon/grpc/src/services/project_service/tests/registration_tests.rs:71` --
  test asserts on canonicalize() output; update to assert on
  syntactic-canonical output after the production sites are migrated.

Other test files calling `canonicalize()` purely for fixture setup
(constructing test paths, no assertion on the resulting value's
canonical form) are out of scope for A-2.

CI grep guard `scripts/ci/forbid_canonicalize.sh` accepts only sites
explicitly allowlisted as Category B, with a `// CATEGORY-B:` comment
on the line. Any new `canonicalize()` call without the marker fails CI.

#### 3.2.3 Project ID Derivation

`resolve_project_id()` currently derives identity from a
canonicalize()-resolved path. Switching to the spec's syntactic
canonical form changes project IDs for projects whose root is a
symlink. Combined with the pre-release "NO MIGRATION EFFORT" policy
(CLAUDE.md), this is acceptable: existing local state is wiped on
schema-version bump, and projects re-register with new IDs.

## 4. Type System Discipline

### 4.1 Rust (`wqm-common::paths`)

```rust
/// Host-absolute, syntactically normalized, UTF-8 path. Stable across
/// deployment modes. The form persisted to SQLite, transmitted over
/// gRPC, and returned in MCP responses.
///
/// Stored as `String` internally because gRPC `string` fields and
/// SQLite TEXT columns both require UTF-8. `PathBuf` on Linux can hold
/// arbitrary bytes that would fail at serialization boundaries; storing
/// `String` shifts the validation cost to construction time where it
/// belongs.
#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct CanonicalPath(String);

/// Path as seen by the current process's filesystem. Differs between
/// host and container deployments. Used only for fs I/O calls.
/// NEVER serialized.
#[derive(Debug, Clone)]
pub struct LocalPath(PathBuf);

/// Content-relative path anchored to a watch_folder root.
/// Not absolute, no `..`, no `.`, UTF-8, normalized.
/// Stored in all file-level path columns (tracked_files.relative_path,
/// Qdrant FilePayload.file_path, etc.).  Serializes as plain string.
/// Mount-map translation is NOT needed -- relative paths are
/// deployment-independent.
#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct RelativePath(String);
```

Constructors:

```rust
impl CanonicalPath {
    /// Build from raw user input (CLI argument, config field, gRPC payload).
    /// Applies normalization rules §3.1. Errors on relative input.
    pub fn from_user_input(s: &str) -> Result<Self, PathError>;

    /// Build from a value already known to be canonical (e.g., DB row).
    /// Validates absolute + normalized; debug-asserts in release.
    pub fn from_validated(s: String) -> Result<Self, PathError>;
}

impl LocalPath {
    /// Build from a CanonicalPath by applying the active MountMap.
    /// Errors if no mount covers the canonical path.
    pub fn from_canonical(c: &CanonicalPath, mounts: &MountMap)
        -> Result<Self, PathError>;

    /// Reverse: build a CanonicalPath from a LocalPath.
    /// Errors on non-UTF-8 paths or paths outside any mount.
    pub fn to_canonical(&self, mounts: &MountMap)
        -> Result<CanonicalPath, PathError>;

    /// Final boundary: hand to fs APIs.
    pub fn as_std_path(&self) -> &Path;
}

impl RelativePath {
    /// Build from raw user input or watcher event path.
    /// Applies normalization (strips leading `/` if accidentally present,
    /// removes `.` segments, rejects `..`). Errors on absolute or traversal input.
    pub fn from_user_input(s: &str) -> Result<Self, PathError>;

    /// Build from a value already stored as relative (e.g., DB row decode).
    /// Validates: not absolute, no `..`, no `.`, UTF-8.
    pub fn from_validated(s: String) -> Result<Self, PathError>;

    /// Reconstruct absolute path by joining to its watch_folder root.
    pub fn to_absolute(&self, root: &CanonicalPath) -> CanonicalPath;

    /// Raw string reference for serialization.
    pub fn as_str(&self) -> &str;
}
```

The `from_validated` constructor is `pub` because `CanonicalPath`
lives in the `wqm-common` crate while the persistence layer (`daemon/core`)
and gRPC layer (`daemon/grpc`) are separate crates that must call it
for SQLite row decode and gRPC handler-entry validation respectively;
Rust's `pub(crate)` would block cross-crate access. The intended
"persistence layer only" discipline is enforced by the multi-layer
defense in §4.3 (CI grep + code review + type system), not by Rust
visibility. The CI grep job `path-discipline.sh` reports any call site
of `from_validated` outside the deserialization/decode entrypoints
listed in the script's allowlist.

### 4.2 TypeScript MCP (`@wqm/common/paths`)

```ts
declare const _canonical: unique symbol;
declare const _local: unique symbol;
declare const _relative: unique symbol;

export type CanonicalPath = string & { readonly [_canonical]: true };
export type LocalPath = string & { readonly [_local]: true };
/** Content path relative to a watch_folder root. Not absolute, no `..`. */
export type RelativePath = string & { readonly [_relative]: true };

export function fromUserInput(s: string): CanonicalPath;
export function fromValidated(s: string): CanonicalPath;
export function toLocal(c: CanonicalPath, mounts: MountMap): LocalPath;
export function toCanonical(l: LocalPath, mounts: MountMap): CanonicalPath;

// RelativePath constructors -- no mount-map needed.
export function relativeFromUserInput(s: string): RelativePath;
export function relativeFromValidated(s: string): RelativePath;
export function relativeToAbsolute(r: RelativePath, root: CanonicalPath): CanonicalPath;
```

Brand types provide nominal typing at compile time. Runtime validation
in `fromUserInput` and `fromValidated` is **mandatory** and must apply
the same nine normalization rules as the Rust implementation (§3.1).
A no-op cast that only changes the type tag is forbidden. The TypeScript
implementation includes:

- `fromUserInput(s)`: full normalization (`~` expand, `.` strip, `..`
  reject, dup `/` collapse, UTF-8 verify, absolute check). Throws on
  failure.
- `fromValidated(s)`: same checks as `fromUserInput` but assumes input
  came from a trusted source (DB row, gRPC). Still throws on
  validation failure -- never silently accepts. In debug builds, can
  optionally be a stricter "assert this is already in canonical form"
  check (no transformation, only validation).

`as CanonicalPath` and `as RelativePath` casts that bypass these functions
are banned by an ESLint rule (`@wqm/eslint-config` ships `no-restricted-syntax`
rules matching `TSAsExpression[typeAnnotation.typeName.name='CanonicalPath']` and
`TSAsExpression[typeAnnotation.typeName.name='RelativePath']` outside the `paths.ts`
module itself). Runtime validation in `relativeFromUserInput` and `relativeFromValidated`
applies the §3.3 rules (no absolute, no `..`, UTF-8, normalized).

### 4.3 Forbidden Patterns

After this spec lands, the following are forbidden in code review:

| Where | Forbidden | Required |
|---|---|---|
| SQLite root columns (`watch_folders.path`, `ignore_file_mtimes.project_root`) | `PathBuf`, `&Path`, `&str` | `CanonicalPath` |
| `sqlx` row decode for root columns | `String` decoded directly to `PathBuf` | `String` -> `CanonicalPath::from_validated` |
| SQLite file-content columns (`tracked_files.relative_path`, etc.) | raw `String` without newtype | `RelativePath::from_validated` |
| gRPC root path fields (`project_root`, `watch_path`, `path` register inputs) | `String` used directly as fs path | Decode to `CanonicalPath` at handler entry |
| gRPC file path fields (content inside a project) | `String` used directly | Decode to `RelativePath` or keep as `RelativePath` in payload structs |
| MCP tool response payloads -- root paths | Raw strings | `CanonicalPath` serialized as string |
| MCP tool response payloads -- file paths inside project | Raw strings | `RelativePath` serialized as string |
| `tokio::fs::*`, `std::fs::*`, `File::open` | `CanonicalPath`, `RelativePath`, raw string | Reconstruct via `RelativePath::to_absolute`, then `LocalPath::as_std_path()` |
| Qdrant payload struct root fields | `String` | `CanonicalPath` |
| Qdrant payload struct file-content fields | `String` | `RelativePath` |

`clippy::disallowed_types` cannot enforce field-level rules: it blocks
a type globally, which would break legitimate non-path `String` and
`PathBuf` uses elsewhere. Enforcement is therefore a multi-layer
defense, not a single compiler check:

1. **Type system** -- `CanonicalPath` and `LocalPath` are distinct types
   with no implicit conversion. The compiler blocks accidental mixing
   inside any function that uses them.
2. **CI grep job** -- `scripts/ci/path-discipline.sh` audits specific
   known schema-mirroring structs. Two companion allowlist files live in
   `scripts/ci/`:
   - `path-discipline-allowlist.txt` -- process-local fields exempt from
     both `CanonicalPath` and `RelativePath` requirements.
   - `path-discipline-relative-allowlist.txt` -- fields expected to use
     `RelativePath` (not `CanonicalPath`). The script verifies these
     fields are typed `RelativePath`, not `String`/`PathBuf`.
   Runs on every PR. Lists updated by audit task A-1.
3. **Custom `dylint` lint** (optional, future) -- once `dylint` adoption
   matures in the project, port the CI grep into a proper AST lint that
   inspects fields of structs annotated with a marker attribute.
4. **Code-review policy** -- forbidden-pattern table above is part of
   the PR review checklist.

ESLint side covered in §4.2.

## 5. Mount Map

### 5.1 Configuration

The mount map is a list of `(host, container)` directory pairs in
`config.yaml`:

```yaml
mounts:
  - host: $HOME/dev
    container: $HOME/dev          # mirror -- default style
  - host: /Volumes/External/books
    container: /mnt/external-books
  - host: ~/reference
    container: /mnt/reference
```

**Mirror mounts** (host path == container path) are the default
suggestion in onboarding docs. They eliminate translation entirely when
chosen well. Non-mirror mounts exist only for cases where a host path
cannot exist inside a Linux container -- primarily `/Volumes/*` on macOS
and removable media.

### 5.2 Resolution Rules

- Longest-prefix wins. If two entries both cover a path, the entry with
  the longer host prefix is selected.
- Prefix matches are component-aware. `$HOME/dev` matches
  `$HOME/dev/foo` but not `$HOME/development`.
- `~` in mount entries is expanded once on config load.
- A canonical path that no mount entry covers cannot be translated to a
  LocalPath. The originating operation fails fast with a clear message.

### 5.3 Validation and Lifecycle

- **Duplicate host prefix** in the config (two entries with identical
  `host:` after `~` expansion) is a config-load error. The daemon
  refuses to start. Same for duplicate container prefix.
- **Overlapping mounts** (one entry's host path is a prefix of
  another's) are allowed. Longest-prefix-wins is well-defined in that
  case and is the documented behavior. Document common overlap
  patterns in the user guide.
- **Immutable for process lifetime.** The MountMap loaded at startup is
  not reloaded if `config.yaml` changes on disk. Config edits require
  process restart. A `SIGHUP`-style reload is explicitly out of scope
  for v0.1.0; the daemon reads the mount map once and treats it as
  frozen. Document.

### 5.4 Cross-Process Drift

When memexd, wqm, and MCP load the config independently and one is
restarted with a different config, the processes briefly disagree
about translations. Mitigation: all path storage and wire transmission
uses `CanonicalPath`; only the `LocalPath` translation differs, and
the daemon (sole writer) is the only process that uses LocalPath for
state mutation. Readers that consume `CanonicalPath` are unaffected.

### 5.5 Per-Module Active Map

The active mount map is selected at process startup:

| Process | Active map |
|---|---|
| memexd on host | Identity (host == host) |
| memexd in docker | Mount map from config |
| wqm CLI on host | Identity |
| wqm CLI inside any container (rare) | Container's mount map |
| MCP server (host or docker) | Mount map of the deployment it runs in |

Identity maps are the trivial `host == container` case. They still flow
through the same `LocalPath::from_canonical` plumbing -- no special-case
code paths for host-only deployments.

## 6. Schema Audit and Migration

### 6.1 Path Column Inventory

This inventory reflects the corrected root/relative discipline (see §3.3 and §6.3).
Only root paths are canonical. All file-level content paths are relative to their
owning watch_folder root. Denormalized absolute `file_path` columns are DROPPED in
schema version 37 -- see §6.2.

Audit task A-1 verifies exhaustiveness by grepping every schema file and serde struct
in the workspace.

**SQLite -- Canonical (root paths only -- must become `CanonicalPath`):**

| Table | Column | Source | Notes |
|---|---|---|---|
| `watch_folders` | `path` | `schema/watch_folders_schema.sql:14` | Project/library root. Single source of truth for absolute location. |
| `ignore_file_mtimes` | `project_root` | `schema_version/v34.rs:24` | Absolute root; part of composite PK. |

**SQLite -- DROPPED in v37 (denormalized absolute columns -- eliminated; use `relative_path` + JOIN):**

| Table | Column | Source | Notes |
|---|---|---|---|
| `tracked_files` | `file_path` | `tracked_files_schema/schema.rs:12` | **DROPPED.** Was absolute; replaced by `(watch_folder_id, relative_path, branch)` UNIQUE constraint. |
| `file_metadata` (search.db) | `file_path` | `search_db/migrations.rs` | **DROPPED.** Was denormalized absolute copied from `tracked_files.file_path`. |

**SQLite -- Relative (content paths -- `RelativePath`):**

| Table | Column | Anchor | Notes |
|---|---|---|---|
| `tracked_files` | `relative_path` | `watch_folders.path` | Surviving key. UNIQUE constraint rebuilds on `(watch_folder_id, relative_path, branch)`. |
| `file_metadata` (search.db) | `relative_path` | `watch_folders.path` | Surviving key. |
| `graph_nodes` | `file_path` | `watch_folders.path` | Stores relative path going forward (previously absolute; reclassified). |
| `graph_edges` | `source_file` | `watch_folders.path` | Stores relative path going forward (previously absolute; reclassified). |
| `unified_queue` | `file_path` | `watch_folders.path` via `watch_folder_id` | Relative path for per-file dedup; NULL for non-file item types. |
| `ignore_file_mtimes` | `file_path` | `ignore_file_mtimes.project_root` (same row) | Relative to the `project_root` column in the same row. |
| `watch_folders` | `submodule_path` | Parent `watch_folders.path` | NULL for top-level watches. |
| `watch_folders` | `disambiguation_path` | Path suffix for clone disambiguation | `DisambiguationSuffix` newtype if introduced, otherwise `String`. |

**Qdrant payloads -- Relative (content paths inside a project or library -- `RelativePath`):**

| Struct | Field | Source | Notes |
|---|---|---|---|
| `FilePayload` | `file_path` | `common/src/payloads/filesystem.rs:17` | Relative to owning watch_folder root. Previously absolute; reclassified. |
| `FilePayload` | `old_path` (Option) | `common/src/payloads/filesystem.rs:29` | Previous relative path before rename. |
| `FolderPayload` | `folder_path` | `common/src/payloads/filesystem.rs:36` | Relative folder path. Previously absolute; reclassified. |
| `FolderPayload` | `old_path` (Option) | `common/src/payloads/filesystem.rs:51` | Previous relative path before rename. |
| `LibraryDocumentPayload` | `document_path` | `common/src/payloads/library.rs:78` | Relative to library root. Previously absolute; reclassified. |
| `ImageSearchResult` | `file_path` | `daemon/core/src/image_search.rs:79` | Relative to owning watch_folder root. Previously absolute; reclassified. |

**Proto-defined payload messages -- Relative (content inside a project):**

| Message | Field | Source | Notes |
|---|---|---|---|
| `ProjectPayload` | `file_path` | `proto/workspace_daemon.proto:1337` | Relative to project root. Always was relative; now typed `RelativePath`. |
| `ProjectPayload` | `file_absolute_path` | `proto/workspace_daemon.proto:1338` | **Optional reference field.** Provided for display convenience only; reconstructed from root + relative. Canonical when present. |
| `SymbolReference` | `file_path` (nested in `LspMetadata`) | `proto/workspace_daemon.proto:1370` | Relative to project root. Previously canonical; reclassified. |

**Proto-defined payload messages -- Canonical (library roots, not content):**

| Message | Field | Source | Notes |
|---|---|---|---|
| `LibraryPayload` | `source_file` | `proto/workspace_daemon.proto:1381` | Absolute path to the library root document. Canonical. |

**Proto-defined response messages -- Canonical (root paths) or Relative (file content):**

| Message | Field | Source | Class | Notes |
|---|---|---|---|---|
| `RegisterProjectResponse` | `watch_path` | `proto/workspace_daemon.proto:444` | canonical | Registered watch root. |
| `GetProjectStatusResponse` | `project_root` | `proto/workspace_daemon.proto:468` | canonical | Absolute project root. |
| `GetProjectStatusResponse` | `main_worktree_path` | `proto/workspace_daemon.proto:475` | canonical | Absolute path to main worktree. |
| `ProjectInfo` | `project_root` | `proto/workspace_daemon.proto:492` | canonical | Absolute project root. |
| `ServerStatusNotification` | `project_root` (optional) | `proto/workspace_daemon.proto:272` | canonical | Absolute project root. |
| `CancelItemsResponse` | `project_path` | `proto/workspace_daemon.proto:968` | canonical | Display field; sourced from `watch_folders.path`. |
| `TextSearchMatch` | `file_path` | `proto/workspace_daemon.proto:580` | relative | File path relative to project root; sourced from `file_metadata.relative_path`. |
| `TraversalNodeProto` | `file_path` | `proto/workspace_daemon.proto:638` | relative | Relative to project root; sourced from `graph_nodes.file_path`. |
| `ImpactNodeProto` | `file_path` | `proto/workspace_daemon.proto:662` | relative | Same. |
| `PageRankNodeProto` | `file_path` | `proto/workspace_daemon.proto:702` | relative | Same. |
| `CommunityMemberProto` | `file_path` | `proto/workspace_daemon.proto:732` | relative | Same. |
| `BetweennessNodeProto` | `file_path` | `proto/workspace_daemon.proto:755` | relative | Same. |

**Proto-defined request messages -- Canonical (root inputs) or Relative (file filters):**

| Message | Field | Source | Class | Notes |
|---|---|---|---|---|
| `RegisterProjectRequest` | `path` | `proto/workspace_daemon.proto:429` | canonical | Project root to register. |
| `DeprioritizeProjectRequest` | `watch_path` (optional) | `proto/workspace_daemon.proto:450` | canonical | Watch root for multi-clone disambiguation. |
| `SetIncrementalRequest` | `file_paths` (repeated) | `proto/workspace_daemon.proto:1083` | relative | File paths relative to project root. `extract_relative_paths!` macro. |
| `ImpactAnalysisRequest` | `file_path` (optional) | `proto/workspace_daemon.proto:648` | relative | Relative file path to narrow graph query. |

Qdrant payloads are JSON-serialized into point metadata. Relative paths in payloads
are portable across machines and deployment modes. When a consumer needs an absolute
path for I/O (e.g., reading file content), it reconstructs via:
`RelativePath::to_absolute(watch_folders.path)` and then converts to `LocalPath`.
A-1's classification scope explicitly includes both Rust struct-defined payloads AND
prost-generated payload messages, and must annotate each path field as canonical (root)
or relative (content).

**Process-local (must NOT become `CanonicalPath`):**

| Site | Notes |
|---|---|
| `queue_config.database_path` (Rust struct) | SQLite state.db file location |
| `GraphDbManager::path` (Rust struct) | graph.db file location |
| `LadybugConfig::db_path` (Rust struct) | LadybugDB directory path |
| `Config::database_path` (Rust struct) | Processing engine SQLite path |
| `DaemonConfig::log_file` (Rust struct) | Log file path; never stored or transmitted |
| `DaemonConfig::project_path` (Rust struct) | Working directory override; not persisted (see decisions-needed in audit doc) |
| `LoggingConfig::log_file_path` (Rust struct) | Log rotation file path |
| `TlsConfig::cert_path`, `key_path`, `ca_cert_path` (Rust struct) | TLS cert file paths; read at startup only |
| `Profile::database_path` (CLI cli_profiles.rs) | CLI profile DB override |
| `YamlLspConfig::user_path` (YAML config) | LSP binary search path override |
| Config file path (env var / well-known) | Bootstrap input; resolved before mount map is available |
| `LocalPath`-flavored arguments to fs APIs | Per-process view |
| Watcher `FileEvent::path`, `PendingMove::old_path` (PathBuf) | Transient; converted to canonical before DB write |
| Path-validator `OrphanedProject::path`, `RegisteredProject::path` (PathBuf) | In-memory; sourced from DB read; not a write surface |

### 6.1.1 Audit Task A-1 Output Format

Task A-1 produces `docs/specs/16-path-abstraction-audit.md` listing
every `*_path` / `*_file` / `path` / `file` field across:

- SQL schema files (`**/*.sql`, schema string constants in `.rs`)
- Serde-derived structs in `common/src/payloads/**`
- Prost-generated message types (audit from `.proto` source)
- Rust struct fields in `daemon/core/src/**` matching naming pattern

Each entry classified into one of: **canonical**, **relative**,
**process-local**, **disambiguation-suffix**, **non-path** (false
positive). Mismatches between this provisional table and the audit
report constitute spec defects to be corrected before A-1 closes.

### 6.2 Schema Version Bump

Spec landing **mandates the relative-path migration** at schema version 37, not in-place rewrite.
Version 37 also DROPS the denormalized absolute `file_path` columns
(`tracked_files.file_path` and `file_metadata.file_path`) and rebuilds UNIQUE
constraints on `(watch_folder_id, relative_path, branch)` in `tracked_files`.
The `graph_nodes.file_path`, `graph_edges.source_file`, `unified_queue.file_path`,
and `ignore_file_mtimes.file_path` columns remain but store relative paths going
forward.

On daemon startup, schema-version mismatch triggers: truncation of
ingest-derived tables (`tracked_files`, `qdrant_chunks`,
`file_metadata`, `graph_nodes`, `graph_edges`, `unified_queue`,
`ignore_file_mtimes`); retention of user-configured tables
(`watch_folders`); fresh filesystem walk to re-ingest. Qdrant
collections corresponding to ingest-derived data are also truncated.

#### 6.2.1 Crash Safety

The relative-path migration must be crash-safe so that a daemon crash
mid-migration does not leave the database in a state where the new
schema version is already recorded but ingest tables are partially
emptied. The current `SchemaManager` records each version immediately
after its migration runs, which is unsafe for the migration semantics:
a partial migration followed by a crash would mark the version bumped,
and the next startup would see "version matches, do nothing" and
operate on a half-empty DB.

Two-phase marker approach (the **relative-path migration protocol**):

1. **Phase 1 (begin migration).** In a single SQLite transaction: insert a
   row into a `relative_path_migration_in_progress` table with the target
   schema version and a `started_at` timestamp. Commit.
2. **Phase 2 (truncate).** Truncate ingest-derived tables. Also
   truncate Qdrant collections. (Cross-store atomicity is impossible;
   ordering: SQLite first, then Qdrant. A crash between leaves SQLite
   empty and Qdrant non-empty -- recovered on next startup.)
3. **Phase 3 (re-ingest).** Walk `watch_folders.path` entries; enqueue
   for ingestion. Crash here just means the queue resumes on next
   startup; the `relative_path_migration_in_progress` marker remains.
4. **Phase 4 (finalize).** When all initial-walk enqueues complete and
   the queue drains for the first time, in a single transaction:
   bump the schema-version constant; delete the
   `relative_path_migration_in_progress` row. Commit.

Recovery: on every daemon startup, if `relative_path_migration_in_progress`
has any row, resume from phase 2 (re-truncate, re-walk). Truncation is
idempotent; re-enqueue is idempotent via the existing dedup key. The
schema version is only bumped after phase 4, so a crash before phase 4
re-triggers the entire migration on next startup.

#### 6.2.2 User Experience

For large repos (100k+ files), the relative-path migration can take
30+ minutes on first run after upgrade. The daemon emits progress logs
at `INFO` level (`files_processed / files_estimated` every 1000 files),
and the `wqm status health` CLI shows a "relative-path migration in
progress (N% complete)" banner when the `relative_path_migration_in_progress`
marker is present. The user upgrade notes accompanying the spec landing
include a rough estimate per 10k files based on benchmarking.

Rationale: project policy (CLAUDE.md "NO MIGRATION EFFORT" +
pre-release status) explicitly permits data wipe; the canonicalize()
removal in A-2 changes project ID derivation (§3.2.3) and would
require a complex semantic migration if attempted in place; and the
spec-driven normalization touches every absolute path column, making
the in-place-rewrite-vs-rebuild boundary fuzzy. A clean rebuild via
the relative-path migration protocol is simpler, safer, and aligned
with project policy.

The schema-version constant lives in the `schema_version` module; the
PRD task that lands this spec selects the bump value. Re-ingest under
the relative-path migration is non-trivial (large repos); document
expected one-time delay in the user upgrade notes.

In-place rewrite is explicitly deferred to post-v1.0 work, when users
exist and migration cost becomes a real constraint. Tracked in §13.

### 6.3 Why Files Are Relative

The corrected discipline (roots canonical, content relative) is grounded in
the following architectural facts:

**Submodule frontier.** A Git submodule is the boundary between two independent
projects. Each submodule has its own `tenant_id` and its own `watch_folders.path`
(root). File paths inside the submodule are relative to the submodule's root, NOT
the parent project's root. If submodule paths were stored absolute (relative to
the parent), they would have to change every time the parent moved or was cloned
to a different location -- defeating portability.

**`tenant_id` derivation.** `tenant_id` is derived from the remote URL hash (if a
remote exists) or the absolute root path (if local-only). If a local project later
gains a remote, `tenant_id` is updated but relative content paths do NOT change --
the content is the same; only the identity anchor changes. Storing absolute paths
would require mass re-tagging of every content record on `tenant_id` update.

**Clone disambiguation.** A project indexed at multiple paths (e.g., two clones of
the same repo) shares a `tenant_id` and thus identical content in the projects
collection. Relative paths are identical across clones by construction. Only the
root stored in `watch_folders.path` differs. If file paths were stored absolute,
two clones would produce duplicate, diverging records in Qdrant, bloating the
collection and requiring clone-aware deduplication logic.

**Library docs.** Library documents (PDF, markdown, reference prose) associated
with a project are stored under a library collection. The library root is absolute
(`CanonicalPath`); the document path within the library is relative. This is the
same root/content split as for project source files.

**Docker parity.** Only roots need mount-map translation. A `CanonicalPath` root is
translated to a `LocalPath` via the `MountMap` once, and all content relative paths
are then resolved relative to that local root. Content paths themselves are
deployment-independent -- no mount-map logic is needed for file-level data, and no
rewriting occurs when switching between host and container deployments.

**Single source of truth.** The `(watch_folder_id, relative_path, branch)` tuple
uniquely identifies a content record. Reconstructing the absolute path at read time
via `SELECT watch_folders.path || '/' || tracked_files.relative_path` is a
JOIN -- a standard, explicit operation. Storing a denormalized absolute `file_path`
alongside `relative_path` (the previous approach) duplicated information, required
both columns to stay in sync, and silently diverged under mount changes or clone moves.

## 7. Module Boundaries

### 7.1 Daemon (memexd)

- Receives gRPC requests. Root path fields (`project_root`, `watch_path`, register
  `path`) are deserialized to `CanonicalPath` at handler entry. File-content path
  fields are deserialized to `RelativePath` at handler entry.
- For file I/O (read source, write content hash), converts root `CanonicalPath` to
  `LocalPath` via `MountMap`, then reconstructs absolute via
  `relative.to_absolute(root)`.
- Writes `CanonicalPath` to root columns (`watch_folders.path`). Writes
  `RelativePath` to all content columns. Never writes `LocalPath` to either.
- Emits gRPC responses with `CanonicalPath` for root fields and `RelativePath` for
  content fields. Emitters produce `RelativePath` for content, `CanonicalPath` for
  roots; consumers receive both and JOIN to reconstruct absolute paths when needed.

### 7.2 wqm CLI

- Accepts CLI arguments and config values as raw strings.
- Converts via `CanonicalPath::from_user_input` at argument parse time.
- For display to the user, prints the canonical form. The user sees
  host paths whether wqm runs on host or in a (rare) container.
- For gRPC to daemon, sends canonical.

### 7.3 MCP Server (TypeScript)

- Receives tool invocations from the LLM client. Root path arguments (project roots,
  library roots) enter as strings and are converted to `CanonicalPath` via
  `fromUserInput`. File/content path arguments are converted to `RelativePath` via
  `relativeFromUserInput`.
- Forwards canonical roots and relative content paths to daemon via gRPC.
- Tool responses: root paths serialize `CanonicalPath` as plain string; file paths
  serialize `RelativePath` as plain string. The LLM client sees host-absolute root
  paths and project-relative file paths -- unambiguous and portable.

### 7.4 gRPC Schema

Every path field in `workspace_daemon.proto` carries a comment marking
its class -- canonical (root) or relative (content):

```proto
// Canonical host-absolute root path. See docs/specs/16-path-abstraction.md §3.
string project_root = 1;

// Relative content path (anchored to watch_folder root). See §3.3.
string file_path = 4;
```

Prost does NOT support field-level deserialization hooks: it generates
plain Rust structs with `String` fields and no validation extension
point. Enforcement therefore happens at **handler entry**, not during
deserialization. Pattern:

```rust
async fn ingest_file(
    &self,
    request: Request<IngestFileRequest>,
) -> Result<Response<IngestFileResponse>, Status> {
    let req = request.into_inner();
    // Root path -> CanonicalPath at handler boundary.
    let project_root = CanonicalPath::from_user_input(&req.project_root)
        .map_err(|e| Status::invalid_argument(format!("project_root: {e}")))?;
    // File path -> RelativePath at handler boundary.
    let file_path = RelativePath::from_user_input(&req.file_path)
        .map_err(|e| Status::invalid_argument(format!("file_path: {e}")))?;
    // From this line forward, types enforce correct usage -- compiler blocks
    // passing a RelativePath where CanonicalPath is required and vice versa.
    // ...
}
```

Helper layers reduce boilerplate:

1. **`extract_canonical_path!` macro** -- single `String` root field extraction with
   `Status` wrapping, producing `CanonicalPath`.
2. **`extract_relative_path!` macro** -- single `String` content field extraction,
   producing `RelativePath`.
3. **`extract_relative_paths!` macro** -- iterates a `Vec<String>` (proto `repeated
   string`) and validates each element as `RelativePath`. Used for
   `SetIncrementalRequest.file_paths` (daemon proto line 1083) and
   `ImpactAnalysisRequest.file_path`. Failure mode: first failing element aborts
   with the element index in the error message.
4. **Nested-message paths.** Path fields inside nested messages (e.g.,
   `SymbolReference.file_path` inside `LspMetadata` inside `ProjectPayload`)
   cannot be validated at the outermost handler boundary alone. The producing site
   is responsible for validating before serializing. Consumers re-validate at first
   use via `RelativePath::from_validated` (content) or `CanonicalPath::from_validated`
   (root). A-1's classification flags every nested path field so A-2 can audit
   producers.
5. **Optional `#[validated_grpc]` procedural macro** (deferred) that decorates
   handler signatures with field annotations and emits validation code
   automatically. Tracked in §13.

Test discipline: every handler has at least one test case that feeds
an unnormalized path (relative, contains `..`, non-UTF-8, empty) and
asserts an `InvalidArgument` status. Handlers with `repeated` path
fields additionally test mixed valid/invalid arrays.

## 8. Configuration File Path

The mount map itself sits in the config file. The config file's
location is a process-local concern that cannot itself depend on the
mount map (chicken-and-egg). Resolved via convention plus env override:

| Process | Default location |
|---|---|
| Host (macOS) | `~/.config/workspace-qdrant/config.yaml` |
| Host (Linux) | `~/.config/workspace-qdrant/config.yaml` |
| Container | `/etc/wqm/config.yaml` |

Overridable via `WQM_CONFIG_PATH` env var. The compose-generated
container service bind-mounts the host config file into `/etc/wqm/`
read-only.

## 9. Compose Generation

The CLI subcommand `wqm docker generate-compose` produces a
`docker-compose.override.yaml` from `config.yaml`. Behavior:

1. Reads the active `config.yaml`.
2. For each mount entry, emits a corresponding `volumes:` line under
   each docker service that needs filesystem access (memexd, optionally
   the MCP server if dockerized).
3. Emits a config-file bind mount: host config path -> `/etc/wqm/config.yaml`.
4. Emits a state-data bind mount: host SQLite + Qdrant data dirs ->
   container-side fixed locations.
5. Embeds a content-hash of the source `config.yaml` mount-section in a
   YAML comment header of the override file: `# wqm-config-hash: <sha256>`.
6. Writes the file alongside `docker-compose.yaml`. Compose's standard
   merge behavior layers the override on top.

The command also supports `--check` mode (compare existing override
hash to current config, exit non-zero on drift) and `--clean` (delete
the override; useful when removing mounts).

### 9.1 Stale-Override Defense (Container Entrypoint Check)

A regenerated override forgotten by the user would silently start a
container with wrong mounts, causing the daemon to write canonical
paths that resolve to nothing inside the container. The Docker image
entrypoint defends against this with three layers:

1. **Hash check.** On startup, the entrypoint reads `# wqm-config-hash:`
   from `/etc/docker-compose-wqm.override.yaml` (mounted read-only via
   compose) and compares to the hash computed from the live
   `/etc/wqm/config.yaml` mount section. Mismatch aborts startup with
   a clear error pointing to `wqm docker generate-compose`. If no
   override is mounted at all -- the deployment hand-wires its volumes
   directly, as the reference/minimal/standalone composes do -- there is
   no generated override to go stale, so layer 1 is skipped and layers
   2-3 still run. (Layer 1 only guards against a *stale* mounted
   override; it is not a requirement to use the generate-compose flow.)
2. **Mount-present validation.** For each mount entry in `config.yaml`,
   the entrypoint stats the container path. If any entry's container
   path is not a directory (i.e., the corresponding `volumes:` line is
   missing from compose), startup aborts with the missing mount name.
3. **Spurious-mount detection (best-effort).** The entrypoint reads the
   container's own mountinfo (`/proc/self/mountinfo`) and reports any
   bind mount under expected mount-map prefixes that is NOT in the
   config. Reported as a warning, not a hard failure, because legitimate
   reasons exist (e.g., user-added scratch mounts for debugging).

#### 9.1.1 Known Limitations

The hash check catches the dominant failure mode (user changes
`config.yaml`, forgets to re-run `generate-compose`). It does NOT
catch:

- User manually edits `docker-compose.override.yaml` without changing
  `config.yaml` (hash matches, override drifts silently). Layer 2 catches
  *missing* mounts; layer 3 catches *extra* mounts as warnings.
- User edits the override to point a mount at a different host
  directory than the config says. The host path inside the override
  is opaque to the container; only the container-side directory
  is statable. Documented limitation; future work tracked in §13.

The hash file is informational only -- compose itself doesn't read it.
It is solely the entrypoint's drift-detection signal.

### 9.2 State Bind Mounts

Host-side data directories are fixed and conventional:

| Data | Host location | Container location |
|---|---|---|
| SQLite state.db | `~/.local/share/workspace-qdrant/` | `/var/lib/wqm/` |
| Qdrant data | `~/.local/share/qdrant/` | `/qdrant/storage` |

These paths are not part of the mount map (which is only for content
directories). They are emitted by `generate-compose` unconditionally.

## 10. Deployment Matrix

Four combinations of memexd x {MCP, wqm}:

| memexd | MCP/wqm | Translation needed |
|---|---|---|
| Host | Host | None -- identity map both sides |
| Host | Docker | MCP: canonical <-> container LocalPath. memexd: identity. |
| Docker | Host | MCP/wqm: identity. memexd: canonical <-> container LocalPath. |
| Docker | Docker | Both sides translate via their own mount map. Maps may differ. |

Same SQLite file (host-mounted) is openable by whichever memexd is
running. Switching between host and docker memexd does not corrupt
state because canonical paths in the DB are deployment-independent.

### 10.1 Cross-Process Single-Instance Enforcement

Running both host memexd and docker memexd against the same SQLite
file simultaneously must be prevented. SQLite WAL handles single
in-process writers, but POSIX advisory locks do not work reliably
across the host/container boundary under common Docker storage drivers
(overlayfs, vfs) and across bind-mounted volumes. Two daemons both
believing they hold the lock will corrupt the database.

Mitigation is a network-socket-based exclusion mechanism, NOT a
filesystem PID lockfile (which cannot detect crashes reliably across
the host/container boundary):

1. **Primary lock = TCP listen socket** on `127.0.0.1:7799` (memexd
   control port). Only one process can bind the host port at a time.
   The host kernel arbitrates uniqueness when the binder is either a
   host process or a container whose port mapping publishes `7799` to
   the host's `127.0.0.1:7799`.
2. **Mandatory port publish for docker.** `wqm docker generate-compose`
   MUST emit `ports: ["127.0.0.1:7799:7799"]` for the memexd service.
   Without the publish directive, Docker's default bridge networking
   isolates the container's port `7799` in its own network namespace,
   so a host memexd and a bridged-network docker memexd could both
   believe they hold the lock -- leading to silent SQLite corruption.
   The generated override is verified by an integration test that
   inspects its YAML for the publish line.
3. **Network-mode constraints.** Two acceptable docker network modes:
   - `network_mode: host` (Linux only) -- container shares host network
     namespace; `7799` bind is direct.
   - Default bridge with `ports: ["127.0.0.1:7799:7799"]`. The
     loopback binding prevents external network exposure.
   Any other mode (custom networks without port publish, `network_mode: none`)
   makes the lock ineffective. `generate-compose` refuses configs that
   produce such modes and emits an error pointing to docs.
4. **Liveness via socket** -- process death releases the bound socket
   immediately. No stale-lock cleanup logic required.
5. **Identity stamp** -- on bind, memexd writes its mode (`host` or
   `docker`) and PID to `~/.local/share/workspace-qdrant/memexd.lock`
   for diagnostics only. The file is informational; the socket is
   authoritative.

If port `7799` is unavailable for non-memexd reasons, memexd refuses
to start with a clear error and a `--control-port` override flag for
recovery. Override changes the bound port for BOTH host and docker
modes consistently (compose-generated override consumes the same env
var).

This elevates §12's "two memexd processes on same SQLite" risk from
Medium to addressed-Critical.

## 11. Test Matrix

Integration tests must cover ingestion in one mode and query in
another:

1. Ingest host -> query host
2. Ingest host -> query docker
3. Ingest docker -> query host
4. Ingest docker -> query docker
5. Switch midway: ingest host -> stop -> start as docker -> query
6. External volume: mount `/Volumes/External/books` (macOS), query inside docker
7. Non-mirror mount: host `~/reference` -> container `/mnt/reference`

### 11.1 Symlink Test Sub-Cases

8a. **File symlink with absolute target inside watch root.** Create a
file `foo.txt`, then `bar.txt` symlink to `foo.txt`. Watch the root.
Verify: both `foo.txt` and `bar.txt` appear in SQLite
`tracked_files.relative_path` and in Qdrant `FilePayload.file_path` as
relative paths (symlink names, NOT resolved to `foo.txt`). MCP `list`
returns both names.

8b. **Directory symlink as watch root.** Watch root itself is a
symlink, e.g., `$HOME/projects/foo` -> `/Volumes/work/foo`.
Verify: registered root in `watch_folders.path` is the symlink path
exactly as the user typed (`$HOME/projects/foo`), not the
target. Files under the root appear with paths rooted at the symlink.

8c. **Broken symlink** (target deleted after ingestion). Verify: read
operations against the broken path produce a clear "unavailable"
error; the row remains in `tracked_files` but is flagged
`needs_reconcile=1`; daemon does not crash.

8d. **Symlink target outside watch root.** Symlink under watched root
points to a file outside any watched root. Verify: the symlink path
itself is ingested (it lives under a watched root); content is read
through the symlink at I/O time; canonical path stored is the symlink,
not the target outside the root.

8e. **macOS watcher symlink behavior.** Specifically test that after
removing `resolve_symlink` from `watching/platform/macos.rs` (per
§3.2.2), FSEvents events for files inside a symlinked watch root are
correctly attributed to canonical paths. This test gates A-2 closure.

All sub-cases assert against BOTH `tracked_files.relative_path` (SQLite) AND
`FilePayload.file_path` (Qdrant). Both must be relative paths. Inconsistency between
the two stores is a hard failure. Absolute path leakage into either store is a hard
failure.

### 11.2 CI Assignment

| Test | Runner | Notes |
|---|---|---|
| 1-4 (host/docker matrix) | Linux (docker subset), macOS+Linux (host subset) | Docker on macOS too slow |
| 5 (mid-run switch) | Linux | Requires docker |
| 6 (external volume) | macOS | `/Volumes/*` is macOS-specific |
| 7 (non-mirror mount) | Linux | |
| 8a (file symlink) | macOS, Linux | |
| 8b (directory symlink as root) | macOS, Linux | |
| 8c (broken symlink) | macOS, Linux | |
| 8d (symlink target outside root) | macOS, Linux | |
| 8e (macOS FSEvents) | **macOS only** | FSEvents is macOS-specific. Gates A-2 closure. |

If macOS CI is unavailable, 8e is downgraded to "manual on dev machine
before merge"; the PR description must include the manual-test
evidence. A-2 closure requires 8e passing somewhere.

## 12. Risks and Mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| A path call site is missed and writes a `PathBuf` or container-local string to SQLite, silently corrupting state | Critical | Type system + audit task A-1 (exhaustive grep) + CI grep job (§4.3) + code review policy. Clippy `disallowed_types` is NOT used (cannot enforce field-level). |
| Two memexd processes (host + docker) open same SQLite simultaneously and corrupt it | Critical | TCP listen socket on `127.0.0.1:7799` as cross-process lock (§10.1). POSIX advisory locks deliberately not used because they fail across host/container boundary. |
| Symlinks resolve differently host vs container, producing inconsistent canonical paths | High | §3.1 rule 7: never call `canonicalize()`. §3.2.2 mandates removal of every existing `canonicalize()` call site. Tested in §11.1 sub-cases including macOS watcher. |
| Existing `canonicalize()` call sites continue to silently produce machine-specific paths | High | Audit task A-2 enumerates and removes all 8 known sites; CI grep job `forbid_canonicalize.sh` keeps regressions out of `main`. |
| MCP returns container-local path to LLM by accident | High | TS brand types block at compile; ESLint rule (§4.2) bans `as CanonicalPath` casts. Test: assert response payload paths are valid canonical form. |
| gRPC field semantics drift (handler forgets validation) | High | Proto comments mandatory. Handler-entry validation (§7.4) with `extract_canonical_path!` macro. Invalid-input test per handler. |
| Stale compose override silently drives docker memexd with wrong mounts | High | §9.1 entrypoint check: hash mismatch refuses startup; per-mount stat validation; clear error pointing at `wqm docker generate-compose`. |
| Schema audit (§6.1) misses a path column or Qdrant payload field | High | Task A-1 is exhaustive grep across `**/*.sql`, schema string constants, `payloads/**`, `.proto`. Task closure requires CI grep job passes. |
| `..` in user input bypassed (e.g., concatenated path) | Medium | §3.1 rule 4: reject `..` at construction. Tests cover synthetic inputs. |
| Non-UTF-8 paths from filesystem cannot become canonical | Medium | §3.1 rule 9 + §4.1: `CanonicalPath` is `String`; non-UTF-8 fs entries produce an error at `LocalPath::to_canonical`, surfaced to user. |
| Case folding on macOS HFS+ vs Linux ext4 in container | Medium | §3.1 rule 6: preserve case exactly. Document mixed-case caveat. Tested. |
| Mount map duplicates / conflicting overlaps | Medium | §5.3: dup host = config-load error; overlap allowed with longest-prefix-wins semantics. |
| External volume unmounted (USB removed) between ingestion and query | Low | `LocalPath::from_canonical` succeeds; fs open fails; row flagged `needs_reconcile=1`. Existing failure mode. |
| Bootstrap: container needs config path before it can read mount map | Low | Fixed `/etc/wqm/config.yaml` convention. `WQM_CONFIG_PATH` env override. |
| A content path field is reclassified as canonical (or left as `String`) instead of `RelativePath` | High | CI grep job `path-discipline.sh` checks `path-discipline-relative-allowlist.txt`; fields listed as relative must type-check as `RelativePath`. Code review policy in §4.3. A-1 audit closure requires zero class-mismatch between spec and audit doc. |

## 13. Open Items (Defer to Implementation PRD)

- Symlink stored under watched root: when the daemon walks the tree,
  does it descend into symlinks? Current `follow_symlinks` column on
  `watch_folders` exists but interacts with canonicalization. Decide
  default and document.
- In-place schema-rewrite (as alternative to §6.2's mandated
  relative-path migration): defer to post-v1.0 once users exist and
  migration cost matters.
- macOS FSEvents handling for symlinked watch roots after
  `resolve_symlink` removal: A-2 must determine final approach
  (restrict to non-symlink roots, register both paths, or translate at
  event ingestion). Test §11.1 (8e) gates the choice.
- `#[validated_grpc]` procedural macro to auto-emit handler-entry
  validation: deferred to a future iteration; initial implementation
  uses the `extract_canonical_path!` macro pattern.
- Custom `dylint` lint replacing the CI grep job: deferred until
  dylint adoption matures in the project toolchain.
- Performance of normalization on hot paths: profile if the daemon's
  watcher event loop processes thousands of paths per second and the
  pure-syntactic normalize becomes measurable. Default: assume
  negligible until proven otherwise.
- `windows` support of canonical form (drive letters, UNC paths,
  backslashes). Out of current scope -- not a supported platform for
  v0.1.0. Document as deferred.

## 14. Acceptance Criteria

The path-abstraction work is complete when:

1. `CanonicalPath` (`String`-backed, UTF-8), `LocalPath`, and `RelativePath`
   types ship in `wqm-common` and the TypeScript MCP common module with full test
   coverage including all nine normalization rules (§3.1), the §3.3 relative-path
   rules, and the §4.2 runtime validation requirement.
2. Audit task A-1 has produced `docs/specs/16-path-abstraction-audit.md`
   listing every path site classified as canonical / relative /
   process-local / disambiguation-suffix / non-path. Inventory in §6.1
   is reconciled with the audit; no class-mismatch remains.
3. Audit task A-2 has removed every `std::fs::canonicalize()` call site
   (§3.2.2 list) and replaced them with syntactic normalization or
   explicit symlink handling. The macOS FSEvents test (§11.1 case 8e)
   passes.
4. CI grep job `scripts/ci/path-discipline.sh` exists, enumerates the
   known schema-mirroring structs, and passes. CI grep job
   `forbid_canonicalize.sh` exists and passes. ESLint rule banning
   `as CanonicalPath` casts is wired into MCP server CI.
5. The proto file annotates every path field with canonical semantics
   and each gRPC handler validates path inputs at entry. Per-handler
   negative test cases exist for unnormalized inputs.
6. `wqm docker generate-compose` produces a valid override from a
   sample multi-mount config and embeds the config hash. Docker image
   entrypoint check refuses startup on hash mismatch (§9.1).
7. Cross-process single-instance lock (§10.1) is implemented and tested:
   second memexd attempting to start (host or docker) refuses with a
   clear error.
8. The integration test matrix (§11 cases 1-7 plus §11.1 sub-cases
   8a-8e) passes in CI.
9. Existing test suite passes with zero ignored / skipped tests.
10. CHANGELOG `[Unreleased]` documents the spec, the schema-version
   bump, the project-ID derivation change (§3.2.3), and the new
   control-port requirement.
11. The relative-path migration is crash-safe per §6.2.1: a test
   simulates a crash after each of phases 1, 2, 3 and verifies recovery
   on next startup leaves the DB in a consistent state.
12. The daemon emits per-1000-files progress logs during the
   relative-path migration's re-ingest phase, and `wqm status health`
   shows a "relative-path migration in progress" banner when the
   `relative_path_migration_in_progress` marker row is present.
   Verified by integration test on a synthetic 10k-file fixture.
