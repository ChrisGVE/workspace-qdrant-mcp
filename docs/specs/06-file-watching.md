## File Watching and Ingestion

### Watch Sources

| Source  | Target Collection | Trigger                                                          |
| ------- | ----------------- | ---------------------------------------------------------------- |
| **MCP** | `projects`        | `RegisterProject` → re-activates existing watch entry only       |
| **CLI** | `projects`        | `wqm project add` → explicit new project registration            |
| **CLI** | `libraries`       | `wqm library add` → explicit library registration                |

**Note:** The MCP server does NOT auto-register new projects. It only re-activates projects that were previously registered (via CLI or a prior MCP session with `register_if_new=true`). If the project is not found in `watch_folders`, MCP logs a warning and continues without registration. CLI is the primary path for registering new projects and libraries.

### Watch Table Schema

See [Watch Folders Table (Unified)](02-collection-architecture.md#watch-folders-table-unified) in the Collection Architecture section for the complete schema.

**Library modes (libraries only):**

- `sync`: Full synchronization - additions, updates, AND deletions
- `incremental`: Additions and updates only, no deletions

**Always recursive:** No depth limit configuration needed.

### Two-Layer Watching Architecture

The system uses a two-layer watching approach based on whether the project is git-tracked.

#### Layer 1: File Watcher (all projects)

- Uses `notify-debouncer-full` with platform-native watchers (inotify/FSEvents/ReadDirectoryChangesW)
- Watches project directory for file changes: create, modify, delete, rename
- For non-tracked projects: this is the sole watcher, branch = `"default"`
- For git-tracked projects: handles real-time dirty-state changes between git operations (modified files that haven't been committed)
- Exclusion patterns apply: `.git/`, `target/`, `node_modules/`, etc. (`.git/` directory is excluded from Layer 1)
- See [Folder Move Detection Strategy](#folder-move-detection-strategy) for rename handling

#### Layer 2: Git Watcher (git-tracked projects only)

- Specifically watches: `.git/HEAD` and `.git/refs/heads/` (these are excluded from Layer 1 but explicitly monitored by Layer 2)
- On change: parses last line of `.git/logs/HEAD` (reflog) to identify operation type
- Reflog format: `<old-SHA> <new-SHA> <author> <timestamp> <operation description>`
- Uses `git diff-tree <old-SHA> <new-SHA>` for exact file change detection (no recursive directory scan needed)

**Event Detection Matrix:**

| Reflog Operation | Detection Signal | Action |
|------------------|-----------------|--------|
| `checkout: moving from X to Y` | `.git/HEAD` changes | Branch switch -> diff-tree |
| `commit: <msg>` | `.git/refs/heads/X` changes | New commit -> diff-tree vs parent |
| `merge <branch>: <msg>` | `.git/refs/heads/X` changes | Merge -> diff-tree |
| `pull: Fast-forward` | `.git/refs/heads/X` changes | Pull -> diff-tree |
| `rebase (finish)` | `.git/refs/heads/X` changes | Rebase -> diff-tree |
| `reset: moving to <ref>` | `.git/HEAD` or `.git/refs/` change | Reset -> diff-tree |

**Supplementary `.git/` paths** (for enriched event detection):

| Path | Changes On | Signal |
|------|-----------|--------|
| `.git/MERGE_HEAD` | Merge (non-FF) start/finish | Merge in progress |
| `.git/rebase-merge/` | Rebase start/finish | Rebase in progress |
| `.git/rebase-apply/` | Rebase start/finish | Rebase in progress |
| `.git/refs/stash` | Stash push/pop | Stash operation |
| `.git/FETCH_HEAD` | Fetch, pull | Remote data arrived |

**Branch switch protocol** (triggered by Layer 2 detecting `.git/HEAD` change):

```
1. Parse reflog -> old_sha, new_sha
2. git diff-tree --name-status old_sha new_sha -> file changes
3. For each changed file:
   a. M (modified): reference-count old base_point -> create new
   b. A (added on new branch): create new
   c. D (removed on new branch): reference-count old base_point
4. For all unchanged files: update branch in tracked_files (no re-ingest)
5. Update watch_folder.last_commit_hash = new_sha
```

**First branch switch optimization:** On the FIRST time a project switches to a branch that is truly new (no existing `tracked_files` for that branch), batch-copy `tracked_files` entries from the old branch to the new branch (updating branch and file_hash where files differ). This avoids full re-ingestion.

**Reduced recursive watching optimization:** For git-tracked projects, the git watcher handles structural events (branch switches, commits, merges). The file watcher only needs to detect dirty-state changes (modified files between git operations). This may allow reducing the scope of recursive watching since git events cover branch switches, commits, and merges.

**Layer 2 activation:** Set automatically on project registration by checking for `.git/` directory. Stored as `is_git_tracked = 1` in `watch_folders`. Layer 2 is never activated for non-git projects or libraries.

### Ingestion Filtering

Ingestion filtering operates at two levels:

- **System-wide**: defined in the YAML configuration file — applies to every watched project/library.
- **Per-project**: defined in `.gitignore` and/or `.wqmignore` files inside the project tree — allows project-specific exclusions without touching the global config.

The system uses a multi-layered approach with the **file type allowlist** as the primary gate.

See [File Type Allowlist](#file-type-allowlist) and [Per-Project Ignore Files](#per-project-ignore-files) below for the complete specification including:
- Ingestion gate layering (ignore files → allowlist → exclusions → size limits)
- Allowed extensions by category (400+ extensions across 21 categories)
- Allowed extension-less filenames (30+ exact names)
- Size-restricted extensions (with configurable stricter limits)
- Mandatory excluded directories (50+ build/cache directories)

### File Type Allowlist

The allowlist is the **primary ingestion gate** — files not on the allowlist are silently skipped (never queued, never tracked). This prevents the system from ingesting binary files, media, build artifacts, and other non-textual content.

#### Design

- **Two-tier allowlist**: Project extensions (source code, configs, text) and library extensions (superset: project + reference formats)
- **Library allowlist** = project_extensions UNION library_only_extensions (`.pdf`, `.epub`, `.djvu`, `.docx`, `.mobi`, `.odt`, `.rtf`, `.doc`, `.ppt`, `.pptx`, `.xls`, `.xlsx`, `.csv`, `.tsv`, `.parquet`)
- **Format-based routing**: Files with library-only extensions found in project folders are routed to the `libraries` collection with `source_project_id` metadata (see [Project vs. Library Boundary](14-future-development.md#project-vs-library-boundary))
- Defined in YAML config under `watching.allowed_extensions` and `watching.allowed_filenames`
- Compile-time embedded defaults in all three components (daemon, CLI, MCP server)
- User config can override (extend or restrict) the defaults

#### Ingestion Gate Layering

Every file event passes through five gates in order:

```
0. IGNORE FILES:     Must not be matched by .gitignore or .wqmignore   → YES = skip
1. ALLOWLIST:        Extension or exact filename must be on the list    → NO  = skip
2. EXCLUSION:        Must not match directory or pattern exclusions     → YES = skip
3. SIZE LIMIT:       Must be under max_file_size_mb                    → OVER = skip
   3a. SIZE-RESTRICTED: If extension is size-restricted, stricter limit → OVER = skip
4. Queue for processing
```

The allowlist supersedes the old `media_files` exclusion for `.pdf` — PDF is explicitly on the allowlist. The exclusion rules remain as defense-in-depth for directories and patterns.

#### Per-Project Ignore Files

The folder scanner reads two optional ignore files from the project root (and recursively from subdirectories, following standard gitignore semantics):

| File | Purpose |
|------|---------|
| `.gitignore` | Standard git ignore rules — already present in most projects. The daemon respects these to avoid indexing files that git itself does not track. |
| `.wqmignore` | workspace-qdrant-specific ignore rules — same syntax as `.gitignore`. Use this to exclude paths from indexing without affecting git tracking (e.g. large datasets, scratch folders, generated caches). |

**Syntax:** Both files use standard gitignore pattern syntax (globs, negation with `!`, directory anchoring with `/`).

**Precedence:** Gate 0 is evaluated before the allowlist. A path matched by either file is skipped regardless of its extension.

**Lookup order within a folder scan:** The matcher is built once per watch-folder root before the scan begins, incorporating all `.gitignore` and `.wqmignore` files found in the tree (standard gitignore hierarchy: root → subdirectories, inner rules take precedence).

**Why two files?**

`.gitignore` already excludes files that should not be in version control. Respecting it prevents the daemon from indexing build artifacts, secrets, and large binary datasets that the project author has already excluded.

`.wqmignore` addresses cases where a file _should_ remain in git but _should not_ be indexed — for example, a large reference dataset committed for reproducibility but irrelevant to semantic search. Keeping these concerns separate means neither file needs to serve double duty.

**Implementation:** Uses the `ignore` crate (same library as ripgrep) which correctly handles nested files, negation, and directory anchoring.

#### Allowed Extensions by Category

**1. Systems Languages**

| Extension | Language |
|-----------|----------|
| `.c`, `.h` | C |
| `.cpp`, `.cxx`, `.cc`, `.c++`, `.hpp`, `.hxx`, `.hh`, `.h++`, `.ipp`, `.tpp` | C++ |
| `.rs` | Rust |
| `.go` | Go |
| `.zig` | Zig |
| `.nim`, `.nims`, `.nimble` | Nim |
| `.d`, `.di` | D |
| `.v` | V |
| `.odin` | Odin |
| `.s`, `.S`, `.asm` | Assembly |

**2. JVM Languages**

| Extension | Language |
|-----------|----------|
| `.java` | Java |
| `.kt`, `.kts` | Kotlin |
| `.scala`, `.sc`, `.sbt` | Scala |
| `.clj`, `.cljs`, `.cljc`, `.edn` | Clojure |
| `.groovy`, `.gvy`, `.gy`, `.gsh` | Groovy |

**3. .NET Languages**

| Extension | Language |
|-----------|----------|
| `.cs`, `.csx` | C# |
| `.fs`, `.fsi`, `.fsx`, `.fsscript` | F# |
| `.vb` | VB.NET |
| `.csproj`, `.fsproj`, `.vbproj`, `.sln`, `.props`, `.targets` | Project files |
| `.xaml` | XAML |
| `.razor`, `.cshtml` | Razor |
| `.nuspec` | NuGet spec |

**4. Scripting Languages**

| Extension | Language |
|-----------|----------|
| `.py`, `.pyi`, `.pyw`, `.pyx`, `.pxd` | Python |
| `.rb`, `.rbw`, `.rake`, `.gemspec` | Ruby |
| `.pl`, `.pm`, `.pod`, `.t`, `.psgi` | Perl |
| `.lua` | Lua |
| `.php`, `.phtml`, `.php3`, `.php4`, `.php5`, `.php7`, `.phps` | PHP |
| `.tcl`, `.tk` | Tcl/Tk |
| `.r`, `.R`, `.Rmd`, `.Rnw` | R |
| `.dart` | Dart |
| `.raku`, `.rakumod`, `.rakutest`, `.p6`, `.pm6` | Raku |

**5. Functional Languages**

| Extension | Language |
|-----------|----------|
| `.hs`, `.lhs` | Haskell |
| `.ml`, `.mli`, `.mll`, `.mly` | OCaml |
| `.erl`, `.hrl` | Erlang |
| `.ex`, `.exs` | Elixir |
| `.lsp`, `.lisp`, `.cl`, `.fasl` | Common Lisp |
| `.scm`, `.ss` | Scheme |
| `.rkt` | Racket |
| `.elm` | Elm |
| `.purs` | PureScript |
| `.nix` | Nix |
| `.lean`, `.olean` | Lean |
| `.agda` | Agda |
| `.idr`, `.ipkg` | Idris |
| `.sml`, `.sig`, `.fun` | Standard ML |

**6. Web Technologies**

| Extension | Language |
|-----------|----------|
| `.js`, `.mjs`, `.cjs`, `.jsx` | JavaScript |
| `.ts`, `.mts`, `.cts`, `.tsx` | TypeScript |
| `.html`, `.htm`, `.xhtml` | HTML |
| `.css` | CSS |
| `.scss`, `.sass` | SCSS/Sass |
| `.less` | Less |
| `.styl`, `.stylus` | Stylus |
| `.vue` | Vue |
| `.svelte` | Svelte |
| `.astro` | Astro |
| `.mdx` | MDX |
| `.coffee`, `.litcoffee` | CoffeeScript |
| `.wasm`, `.wat` | WebAssembly |

**7. Shell and Scripting**

| Extension | Language |
|-----------|----------|
| `.sh`, `.bash`, `.zsh`, `.fish` | Shell |
| `.ps1`, `.psm1`, `.psd1` | PowerShell |
| `.bat`, `.cmd` | Batch |
| `.mk` | Make |
| `.awk` | AWK |
| `.sed` | sed |

**8. Legacy Languages**

| Extension | Language |
|-----------|----------|
| `.cob`, `.cbl`, `.cpy` | COBOL |
| `.f`, `.f90`, `.f95`, `.f03`, `.f08`, `.for`, `.fpp` | Fortran |
| `.pas`, `.pp`, `.dpr`, `.dpk`, `.dfm`, `.lfm` | Pascal/Delphi |
| `.adb`, `.ads` | Ada |
| `.bas`, `.vbs`, `.vba`, `.cls`, `.frm` | BASIC/VBA |
| `.rpg`, `.rpgle`, `.sqlrpgle` | RPG |
| `.abap` | ABAP |

**9. Apple Ecosystem**

| Extension | Language |
|-----------|----------|
| `.swift` | Swift |
| `.m`, `.mm` | Objective-C |
| `.applescript`, `.scpt` | AppleScript |
| `.plist` | Property list |
| `.pbxproj` | Xcode project |
| `.storyboard`, `.xib` | Interface Builder |
| `.entitlements` | Entitlements |
| `.xcconfig` | Xcode config |
| `.xcscheme` | Xcode scheme |
| `.metal` | Metal shader |
| `.strings`, `.stringsdict` | Localization |
| `.xcdatamodeld` | Core Data model |

**10. Data Science and Scientific Computing**

| Extension | Language |
|-----------|----------|
| `.jl` | Julia |
| `.m` | MATLAB |
| `.wl`, `.wls`, `.nb` | Mathematica/Wolfram |
| `.mpl`, `.mw` | Maple |
| `.oct` | Octave |
| `.ipynb` | Jupyter |
| `.qmd` | Quarto |
| `.sas` | SAS |
| `.do`, `.ado` | Stata |
| `.stan` | Stan |
| `.sage`, `.spyx` | SageMath |

**11. DevOps and Configuration**

| Extension | Language |
|-----------|----------|
| `.yaml`, `.yml` | YAML |
| `.toml` | TOML |
| `.json`, `.jsonc`, `.json5` | JSON |
| `.xml` | XML |
| `.ini`, `.cfg` | INI |
| `.tf`, `.tfvars` | Terraform |
| `.hcl` | HCL |
| `.env.example`, `.env.template` | Env templates |
| `.properties` | Java properties |
| `.conf` | Config |
| `.desktop` | Desktop entry |
| `.service`, `.timer`, `.socket` | systemd |

**12. Build Systems**

| Extension | Language |
|-----------|----------|
| `.cmake` | CMake |
| `.bzl`, `.bazel` | Bazel |
| `.gradle` | Gradle |
| `.ninja` | Ninja |
| `.meson` | Meson |
| `.gn`, `.gni` | GN |
| `.spec` | RPM spec |

**13. Documents and Text**

| Extension | Language |
|-----------|----------|
| `.md`, `.markdown` | Markdown |
| `.rst` | reStructuredText |
| `.adoc`, `.asciidoc` | AsciiDoc |
| `.org` | Org-mode |
| `.tex`, `.latex`, `.sty`, `.cls`, `.bib`, `.bst` | LaTeX |
| `.pdf` | PDF |
| `.epub` | EPUB |
| `.docx` | Word |
| `.odt` | OpenDocument |
| `.rtf` | Rich Text |
| `.csv`, `.tsv`, `.tab` | Delimited data |
| `.svg` | SVG |
| `.txt`, `.text` | Plain text |
| `.man`, `.1`, `.2`, `.3`, `.4`, `.5`, `.6`, `.7`, `.8`, `.9` | Man pages |
| `.diff`, `.patch` | Patches |

**14. Templates**

| Extension | Language |
|-----------|----------|
| `.j2`, `.jinja`, `.jinja2` | Jinja2 |
| `.hbs`, `.handlebars` | Handlebars |
| `.mustache` | Mustache |
| `.ejs` | EJS |
| `.pug`, `.jade` | Pug |
| `.slim` | Slim |
| `.haml` | Haml |
| `.liquid` | Liquid |
| `.twig` | Twig |
| `.blade.php` | Blade |
| `.erb` | ERB |
| `.njk`, `.nunjucks` | Nunjucks |

**15. Protocol and Schema Definitions**

| Extension | Language |
|-----------|----------|
| `.proto` | Protocol Buffers |
| `.graphql`, `.gql` | GraphQL |
| `.thrift` | Thrift |
| `.fbs` | FlatBuffers |
| `.capnp` | Cap'n Proto |
| `.xsd`, `.xsl`, `.xslt` | XML Schema/XSLT |
| `.cue` | CUE |
| `.avsc`, `.avdl` | Avro |

**16. Database**

| Extension | Language |
|-----------|----------|
| `.sql` | SQL |
| `.plsql`, `.pls`, `.plb` | PL/SQL |
| `.tsql` | T-SQL |
| `.pgsql` | PostgreSQL |
| `.mysql` | MySQL |
| `.prisma` | Prisma |
| `.edgeql`, `.esdl` | EdgeDB |

**17. Graph Languages**

| Extension | Language |
|-----------|----------|
| `.cypher`, `.cql` | Cypher (Neo4j) |
| `.rq`, `.sparql` | SPARQL |
| `.dot`, `.gv` | Graphviz |

**18. Shaders**

| Extension | Language |
|-----------|----------|
| `.glsl`, `.vert`, `.frag`, `.geom`, `.tesc`, `.tese`, `.comp` | GLSL |
| `.hlsl`, `.fx`, `.fxh` | HLSL |
| `.metal` | Metal |
| `.wgsl` | WGSL |
| `.cg`, `.cgfx` | Cg |
| `.shader`, `.compute` | Unity |

**19. Hardware and Embedded**

| Extension | Language |
|-----------|----------|
| `.v`, `.sv`, `.svh`, `.vh` | Verilog/SystemVerilog |
| `.vhd`, `.vhdl` | VHDL |
| `.ino` | Arduino |
| `.dts`, `.dtsi` | Device Tree |
| `.ld`, `.lds` | Linker scripts |

**20. Blockchain**

| Extension | Language |
|-----------|----------|
| `.sol` | Solidity |
| `.vy` | Vyper |
| `.cairo` | Cairo |
| `.move` | Move |
| `.clar` | Clarity |

**21. Diagram and Visual**

| Extension | Language |
|-----------|----------|
| `.dot`, `.gv` | Graphviz |
| `.mmd` | Mermaid |
| `.puml`, `.plantuml` | PlantUML |
| `.d2` | D2 |

#### Allowed Extension-less Filenames

These files are recognized by exact name (case-sensitive):

| Filename | Category |
|----------|----------|
| `Makefile`, `GNUmakefile` | Build |
| `Dockerfile`, `Containerfile` | Container |
| `Jenkinsfile` | CI/CD |
| `Vagrantfile` | Virtualization |
| `Rakefile` | Ruby build |
| `Gemfile` | Ruby deps |
| `Podfile` | CocoaPods |
| `Fastfile`, `Appfile`, `Matchfile`, `Snapfile` | Fastlane |
| `Brewfile` | Homebrew |
| `Procfile` | Process |
| `Justfile`, `justfile` | Just |
| `BUILD`, `WORKSPACE` | Bazel |
| `CODEOWNERS` | GitHub |
| `LICENSE`, `LICENCE`, `COPYING` | Legal |
| `README`, `CHANGELOG`, `AUTHORS`, `CONTRIBUTORS` | Project docs |
| `CMakeLists.txt` | CMake |
| `.gitignore`, `.gitattributes`, `.gitmodules` | Git config |
| `.dockerignore` | Docker |
| `.editorconfig` | Editor config |
| `.clang-format`, `.clang-tidy` | C/C++ tools |
| `.eslintrc`, `.prettierrc`, `.stylelintrc` | JS/TS tools |
| `.rubocop.yml` | Ruby tools |
| `.flake8`, `.pylintrc`, `.mypy.ini` | Python tools |
| `.rustfmt.toml`, `.clippy.toml` | Rust tools |
| `.swiftlint.yml` | Swift tools |

#### Size-Restricted Extensions

Some extensions can contain either small config/schema files or massive datasets. These extensions are on the allowlist but subject to a **stricter size limit** (configurable, default 1 MB instead of the general `max_file_size_mb`):

| Extension | Risk | Rationale |
|-----------|------|-----------|
| `.csv`, `.tsv`, `.tab` | Dataset dumps | Can be multi-GB data exports |
| `.json`, `.jsonc`, `.json5` | Data dumps | Can be large API responses, datasets |
| `.xml`, `.xsd`, `.xsl` | Data dumps | Can be large data exports, SOAP payloads |
| `.jsonl`, `.ndjson` | Streaming data | Can be unbounded log/event streams |
| `.log` | Log files | Can grow unbounded |
| `.sql` | Database dumps | Can contain full database exports |

Config keys: `watching.size_restricted_extensions` with `watching.size_restricted_max_mb` (default: 1 MB).
Files with these extensions exceeding the restricted size limit are skipped with an INFO log.

#### Mandatory Excluded Directories

These directories are always excluded regardless of configuration. They are checked by path component (any directory segment matching is excluded):

| Directory | Reason |
|-----------|--------|
| `node_modules` | NPM packages |
| `target` | Rust/Maven build output |
| `build` | General build output |
| `dist` | Distribution output |
| `out` | Compiler output |
| `.git` | Git internals |
| `__pycache__` | Python bytecode |
| `.venv`, `venv`, `.env` | Python virtual environments |
| `.tox` | Python tox |
| `.mypy_cache` | Mypy cache |
| `.pytest_cache` | Pytest cache |
| `.ruff_cache` | Ruff cache |
| `.gradle` | Gradle cache |
| `.next` | Next.js build |
| `.nuxt` | Nuxt.js build |
| `.svelte-kit` | SvelteKit build |
| `.astro` | Astro build |
| `Pods` | CocoaPods |
| `DerivedData` | Xcode build |
| `.build` | Swift PM build |
| `.swiftpm` | Swift PM cache |
| `.fastembed_cache` | FastEmbed model cache |
| `.terraform` | Terraform state |
| `.terragrunt-cache` | Terragrunt cache |
| `coverage` | Test coverage |
| `.nyc_output` | NYC coverage |
| `.cargo` | Cargo cache |
| `.rustup` | Rustup toolchains |
| `vendor` | Vendored deps |
| `.bundle` | Ruby bundle |
| `.cache` | General cache |
| `.tmp`, `tmp` | Temporary files |
| `.DS_Store` | macOS metadata |
| `.idea`, `.vscode` | IDE settings |
| `.settings`, `.project`, `.classpath` | Eclipse |
| `bin`, `obj` | .NET build output |
| `.zig-cache` | Zig cache |
| `zig-out` | Zig output |
| `elm-stuff` | Elm packages |
| `.stack-work` | Haskell Stack |
| `_build` | Elixir/Phoenix build |
| `deps` | Elixir deps |
| `.dart_tool` | Dart tool cache |
| `.pub-cache` | Pub cache |

### Git Submodules

When a subfolder contains a `.git` directory (submodule):

1. **Detect:** Daemon detects `.git` in subfolder during scanning
2. **Separate entry:** Submodule is registered in `watch_folders` with its own `tenant_id` (project_id)
3. **Link via junction table:** Submodule relationships are stored in the `watch_folder_submodules` junction table (many-to-many). A submodule can have multiple parents, and a project can have multiple submodules.
4. **Activity inheritance:** Submodules share `is_active` and `last_activity_at` with parent via the junction table (see [Activity Inheritance](02-collection-architecture.md#watch-folders-table-unified))

#### Submodule Junction Table

The `watch_folder_submodules` junction table replaces the previous single-FK `parent_watch_id` column on `watch_folders`:

```sql
CREATE TABLE IF NOT EXISTS watch_folder_submodules (
    parent_watch_id TEXT NOT NULL,
    child_watch_id TEXT NOT NULL,
    submodule_path TEXT NOT NULL,   -- relative path within parent
    created_at TEXT NOT NULL,
    PRIMARY KEY (parent_watch_id, child_watch_id),
    FOREIGN KEY (parent_watch_id) REFERENCES watch_folders(watch_id)
        ON DELETE CASCADE,
    FOREIGN KEY (child_watch_id) REFERENCES watch_folders(watch_id)
        ON DELETE CASCADE
);
CREATE INDEX idx_submodule_parent ON watch_folder_submodules(parent_watch_id);
CREATE INDEX idx_submodule_child ON watch_folder_submodules(child_watch_id);
```

**Rationale for many-to-many:** A submodule can be used by multiple parent projects (e.g., a shared utility library referenced by several repos). The previous single-FK model (`parent_watch_id`) only captured one parent, which was insufficient for multi-project environments.

#### Submodule Commit Pin Detection

For git-tracked projects, the parent's tree entry stores the pinned commit for each submodule:

```bash
# Read submodule pinned commit from parent tree
git ls-tree HEAD path/to/submodule
# Output: 160000 commit <SHA> path/to/submodule

# Detect submodule pin changes between commits
git diff-tree old-sha new-sha -- path/to/submodule
```

The mode `160000` indicates a submodule (gitlink) entry. Changes to this entry detected via `git diff-tree` signal that the parent project has updated the submodule pin.

#### Activation Cascade

Activating a project cascades through the junction table to activate ALL submodule children:

```sql
-- Activate parent and all its submodules
UPDATE watch_folders SET is_active = 1, last_activity_at = datetime('now')
WHERE watch_id = :watch_id
   OR watch_id IN (SELECT child_watch_id FROM watch_folder_submodules WHERE parent_watch_id = :watch_id)
   OR watch_id IN (
       SELECT parent_watch_id FROM watch_folder_submodules WHERE child_watch_id = :watch_id
   )
   OR watch_id IN (
       SELECT ws2.child_watch_id FROM watch_folder_submodules ws1
       JOIN watch_folder_submodules ws2 ON ws1.parent_watch_id = ws2.parent_watch_id
       WHERE ws1.child_watch_id = :watch_id
   );
```

**Submodule archive safety:**

When archiving a project that has submodules, the system must check cross-references via the junction table before archiving submodule data:

1. Set `is_archived = 1` on the parent project's `watch_folders` entry
2. For each submodule (linked via `watch_folder_submodules`):
   a. Check if the submodule `child_watch_id` appears in any other **active** (non-archived) parent's junction table entry
   b. If yes: the submodule data stays fully active — another project still references it
   c. If no: set `is_archived = 1` on the submodule entry (data remains searchable, watching stops)
3. Junction table entries are **preserved as-is** — they are historical fact. No detaching on archive.
4. Qdrant data is **never deleted** on archive. Archived content remains fully searchable.

**Un-archiving:** Set `is_archived = 0` on the project and its submodule entries (via junction table lookup). Daemon resumes watching/ingesting.

### Daemon Polling

The daemon:

1. Polls `watch_folders` table every 5 seconds
2. Detects changes via `updated_at` timestamp
3. Updates file watchers dynamically
4. Processes file events through ingestion queue

### Ingestion Pipeline

Different operations have different pipelines:

| Event                        | Pipeline                                                  |
| ---------------------------- | --------------------------------------------------------- |
| **New file (project)**       | Debounce → Read → Parse/Chunk → Embed → Upsert            |
| **File changed (project)**   | Debounce → Read → Parse/Chunk → Embed → Upsert (replace)  |
| **File deleted (project)**   | Delete from Qdrant (filter by `file_path` + `project_id`) |
| **File renamed (project)**   | Delete old + Upsert new (simple approach)                 |
| **Library document (new)**   | Extract → Title Extraction → Token Chunk → Embed → Upsert (parent + children) |
| **Library document (changed)** | Extract → Check fingerprint → Skip if unchanged OR Upsert (replace parent + children) |

**Project File Processing Steps:**

```
Read Content → Parse/Chunk → Generate Embeddings → Upsert to Qdrant
                   │
                   ├── Tree-sitter parsing (always, for code files)
                   ├── LSP enrichment (active projects only)
                   ├── Metadata extraction (file_type, language)
                   └── Content hashing (deduplication)
```

**Library Document Processing Steps:**

```
Read File → Classify Format → Extract Text → Title Extraction → Token-Based Chunking → Generate Embeddings → Upsert
    │           │                 │              │                    │
    │           │                 │              │                    ├── Parent record (no vectors)
    │           │                 │              │                    └── Child chunks (with vectors)
    │           │                 │              │
    │           │                 │              └── Priority cascade: metadata → content → filename
    │           │                 │
    │           │                 └── Page-based: PDF, DOCX, PPTX, ODT, RTF
    │           │                     Stream-based: EPUB, HTML, Markdown, Text
    │           │
    │           └── Determines extractor and chunking strategy
    │
    └── Computes SHA256 fingerprint for change detection
```

**Library Document Idempotency:**

- `doc_fingerprint` (SHA256 of file bytes) stored in parent record
- On re-ingestion: If fingerprint matches, skip processing
- If fingerprint differs: Delete existing parent + children, create new records

---

