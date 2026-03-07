## Grammar and Runtime Management

The daemon manages tree-sitter grammars and ONNX runtime as external dependencies with automatic updates.

### Cache Locations

| Component            | Default Location                | Config Key            |
| -------------------- | ------------------------------- | --------------------- |
| Tree-sitter grammars | `~/.workspace-qdrant/grammars/` | `grammars.cache_dir`  |
| Embedding models     | `~/.workspace-qdrant/models/`   | `embedding.cache_dir` |

### Tree-sitter Grammar Management

**Language-agnostic grammar support via Language Registry:**

Tree-sitter grammars are available for hundreds of languages. The system does not pre-load any grammars — they are downloaded on demand when a file of that language is first encountered. Grammar repository mappings come from the Language Registry, which merges data from multiple upstream providers:

- **Bundled YAML definitions** (priority 255) — 44 languages with curated grammar repos
- **nvim-treesitter lockfile** (priority 20) — grammar-to-repo mappings for 200+ languages
- **tree-sitter-grammars org** (priority 15) — curated repos from the GitHub organization
- **GitHub Linguist** (priority 10) — language identity (extensions, aliases, type)

Each grammar source has a quality tier (Curated > Official > Community). When multiple sources exist for a language, the highest-quality source is preferred for download.

**Configuration:**

```yaml
grammars:
  cache_dir: "~/.workspace-qdrant/grammars/"
  auto_download: true # Download missing grammars automatically on first use
```

**Grammar discovery and installation:**

```bash
wqm language ts-search <lang>           # Search grammar sources with quality tiers
wqm language ts-install <lang> [--force] # Download and install a grammar
wqm language ts-remove <lang|all>        # Remove cached grammar(s)
wqm language refresh                     # Refresh registry from upstream providers
```

`ts-search` shows all known grammar sources for a language, their quality tier, and origin provider. `refresh` fetches the latest metadata from all upstream providers.

**Daemon behavior:**

1. When a file is encountered during ingestion, detect language from file extension
2. If grammar not in cache and `auto_download: true`, download from grammar repository
3. If grammar version mismatches tree-sitter runtime version, replace with compatible version
4. If `auto_download: false` and grammar missing, log warning and fall back to text chunking

### Manual Updates via CLI

```bash
wqm update                    # Check for updates and install if available
wqm update --check            # Check only, don't install
wqm update --force            # Force reinstall current version
wqm update --version 1.2.3    # Install specific version
```

**Update behavior:**

1. Query GitHub releases API for latest version
2. Compare with currently installed version
3. If newer version available (or `--force`):
   - Download appropriate binary for current platform
   - Verify checksum
   - Stop running daemon gracefully
   - Replace binary
   - Restart daemon
4. Report success/failure

**Configuration:**

```yaml
updates:
  check_on_startup: false # Auto-check for updates when daemon starts
  notify_only: true # If true, only notify; don't auto-install
  channel: "stable" # stable|beta|nightly
```

### Continuous Integration

**Automated releases triggered by upstream updates:**

| Trigger                  | Action                                                                                        |
| ------------------------ | --------------------------------------------------------------------------------------------- |
| New tree-sitter release  | Rebuild daemon, bump patch version, release with message "tree-sitter version bump to X.Y.Z"  |
| New ONNX Runtime release | Rebuild daemon, bump patch version, release with message "ONNX Runtime version bump to X.Y.Z" |

**Target platforms (6 binaries per release):**

| Platform            | Target Triple               |
| ------------------- | --------------------------- |
| Linux ARM64         | `aarch64-unknown-linux-gnu` |
| Linux x86_64        | `x86_64-unknown-linux-gnu`  |
| macOS Apple Silicon | `aarch64-apple-darwin`      |
| macOS Intel         | `x86_64-apple-darwin`       |
| Windows ARM64       | `aarch64-pc-windows-msvc`   |
| Windows x86_64      | `x86_64-pc-windows-msvc`    |

**CI workflow:**

1. Monitor upstream releases (tree-sitter, ONNX Runtime) via GitHub Actions or webhook
2. On new release detected, trigger build pipeline
3. Build for all 6 targets
4. Run integration tests on each platform
5. Create GitHub release with all binaries
6. Update homebrew formula / other package managers

---

