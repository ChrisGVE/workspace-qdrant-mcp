## Grammar and Runtime Management

The daemon manages tree-sitter grammars and ONNX runtime as external dependencies with automatic updates.

### Cache Locations

| Component            | Default Location                | Config Key            |
| -------------------- | ------------------------------- | --------------------- |
| Tree-sitter grammars | `~/.workspace-qdrant/grammars/` | `grammars.cache_dir`  |
| Embedding models     | `~/.workspace-qdrant/models/`   | `embedding.cache_dir` |

### Tree-sitter Grammar Management

**Language-agnostic grammar support:**

Tree-sitter grammars are available for hundreds of languages. The system does not pre-load any grammars — they are downloaded on demand when a file of that language is first encountered. The default configuration maps known languages to file extensions; the set of supported languages is limited only by the availability of tree-sitter grammars in the ecosystem.

**Configuration:**

```yaml
grammars:
  cache_dir: "~/.workspace-qdrant/grammars/"
  auto_download: true # Download missing grammars automatically on first use
```

**Grammar discovery and installation:**

```bash
wqm language ts-search <query>          # Search for available tree-sitter grammars
wqm language ts-install <language>      # Download and install a grammar
wqm language ts-list                    # List installed grammars
wqm language ts-remove <language>       # Remove a cached grammar
```

`ts-search` queries the tree-sitter grammar ecosystem to find available grammars matching the query. This allows users to discover and install grammars for any language without needing to know the exact grammar repository name.

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

