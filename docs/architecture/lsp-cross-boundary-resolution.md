# LSP Cross-Boundary Symbol Resolution Design

## Research Context

When a parent project has submodules registered as independent watch_folders (via `parent_watch_id`), the parent's LSP instance (e.g., rust-analyzer) naturally resolves symbols across project boundaries. This document proposes a symbol ownership model for these cross-project references.

## Language Server Behavior by Language

### Rust (rust-analyzer)

- **Single instance per workspace**: rust-analyzer indexes the entire Cargo workspace including all path dependencies and workspace members
- **Full source indexing**: Subcrate/submodule source is fully indexed, not just API surfaces
- **Symbol resolution**: Go-to-definition, find-references work seamlessly across crate boundaries within the workspace
- **Implication**: A parent project's rust-analyzer will see and index all symbols from submodule crates

### TypeScript (tsserver)

- **Project references**: Uses `.d.ts` declaration files, not full source
- **Lazy loading**: Subprojects loaded on-demand via `declarationMap`
- **Implication**: Parent tsserver sees API types but not implementation details of submodules

### Python (pyright/pylsp)

- **Workspace-based**: Indexes all Python files within the workspace root
- **Import resolution**: Walks PYTHONPATH and parent directories
- **Implication**: Similar to Rust — parent LSP sees full subpackage source

### Go (gopls)

- **Module-based**: Indexes transitive closure of imported packages
- **Workspace mode**: Can handle multi-module workspaces
- **Implication**: Indexes submodule Go packages as dependencies

## Current Architecture

### What exists

1. **Per-project LSP instances**: `LanguageServerManager` uses `(project_id, language)` as key
2. **Only root projects get LSP**: Children (`parent_watch_id IS NOT NULL`) are excluded
3. **Enrichment stored as Qdrant metadata**: Not in separate tables, attached to point payloads
4. **No cross-project resolution**: The `enrich_chunk()` pipeline loses project_id context

### The problem

When a submodule is registered as a separate watch_folder with its own `tenant_id`:
- The parent project's LSP instance indexes both parent and submodule source
- When LSP returns a definition located in a submodule file path, the system must decide: **which project_id owns this symbol?**
- Currently, enrichment queries find "any" server matching the language, losing project attribution

## Proposed Symbol Ownership Model

### Principle: File Path Determines Ownership

A symbol's owner is determined by the watch_folder that contains its file path, regardless of which LSP instance discovered it.

```
watch_folders:
  proj-root (path: /projects/main, tenant_id: main-project)
  └── sub-1 (path: /projects/main/libs/auth, tenant_id: auth-module)
  └── sub-2 (path: /projects/main/libs/db, tenant_id: db-module)

Symbol: fn authenticate() at /projects/main/libs/auth/src/lib.rs
  → Owner: auth-module (sub-1's path is the most specific match)

Reference: use auth::authenticate at /projects/main/src/main.rs
  → Owner: main-project (proj-root's path is the match)
```

### Resolution algorithm

```
fn determine_symbol_owner(file_path: &Path, watch_folders: &[WatchFolder]) -> &str {
    // Sort watch_folders by path length descending (most specific first)
    // Find the first watch_folder whose path is a prefix of file_path
    // Return that watch_folder's tenant_id
}
```

### Rules

1. **Symbol definitions** are attributed to the watch_folder that owns the file containing the definition
2. **Symbol usages/references** are attributed to the watch_folder that owns the file containing the reference
3. **Cross-references** (definition in project A, usage in project B) are stored in both projects:
   - Project B's Qdrant point gets the reference with a `source_project_id` field pointing to project A
   - Project A's Qdrant point has its own enrichment from its own processing
4. **LSP instance routing**: Only the parent project's LSP instance handles queries for the entire hierarchy

## LSP Instance Strategy

### Recommendation: Single LSP per project group

**One LSP instance per root project** (current behavior), serving all submodules:

| Approach | Pros | Cons |
|----------|------|------|
| **Single instance (recommended)** | Correct cross-boundary resolution, lower resource usage | Can't customize per-submodule |
| Separate per submodule | Independent lifecycle | Missing cross-boundary context, higher resource cost |

**Justification:**
- rust-analyzer, gopls, and pyright all work best with a single workspace root
- The parent project's workspace configuration naturally includes submodules
- Cross-boundary resolution is a first-class feature when using one instance
- Running separate instances would lose the primary benefit of LSP enrichment

### Implementation changes needed

1. **Pass project_id through enrichment pipeline**: `enrich_chunk()` should receive and use project_id context
2. **File-path-based ownership lookup**: Add helper to map file paths to their owning watch_folder
3. **Cross-reference metadata**: When LSP returns a definition in a different project's file space, store the cross-reference with source attribution

## Impact on Search Relevance

### Positive impacts

- **Contextual search**: Searching within project A returns symbols defined in project A plus cross-references to submodule symbols used by A
- **Traceability**: `source_project_id` metadata enables filtering by origin
- **Completeness**: All symbols are indexed, not just those in the immediate project

### Considerations

- **Duplicate indexing**: A function in the auth submodule might appear in both the parent project's enrichment (as a reference) and the auth-module's own content chunks. This is expected and useful — it serves different search intents.
- **Stale references**: When a submodule is updated independently, the parent project's cross-references may be stale until re-processing. The unified queue's update path handles this via hash-based change detection.

## Implementation Roadmap

### Phase 1: Foundation (minimal changes)
- Add `determine_owning_watch_folder(file_path)` helper to `DaemonStateManager`
- Pass `project_id` through `enrich_chunk()` calls in `unified_queue_processor.rs`
- No behavioral change — just plumbing for future use

### Phase 2: Ownership-aware enrichment
- Use the ownership helper when storing LSP enrichment metadata
- Add `definition_project_id` field to enrichment data when definition is in a different project
- Filter enrichment results by ownership context

### Phase 3: Cross-project search enhancement
- Enable search queries to optionally include cross-project references
- Add faceted search by `source_project_id` in the MCP search tool
- Surface cross-project symbols with appropriate ranking

## Non-Goals

- **External dependency resolution**: Libraries installed from registries (crates.io, npm, PyPI) are not project-scoped. Their LSP handling is the language server's responsibility, not ours.
- **Multi-language cross-references**: A Rust file calling a Python script via FFI is out of scope. Each language's LSP only understands its own ecosystem.
- **Real-time cross-project consistency**: Eventual consistency is acceptable. When a submodule changes, the parent project's cross-references update on the next processing cycle.

## References

- `src/rust/daemon/core/src/lsp/project_manager.rs` — LanguageServerManager
- `src/rust/daemon/core/src/lsp/lifecycle.rs` — Server instance lifecycle
- `src/rust/daemon/core/src/unified_queue_processor.rs` — Enrichment pipeline
- `src/rust/daemon/core/src/daemon_state.rs` — Watch folder and activity inheritance
- [rust-analyzer Configuration](https://rust-analyzer.github.io/book/configuration.html)
- [TypeScript Project References](https://www.typescriptlang.org/docs/handbook/project-references.html)
- [Pyright Import Resolution](https://github.com/microsoft/pyright/blob/main/docs/import-resolution.md)
- [LSP Specification v3.17](https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/)
