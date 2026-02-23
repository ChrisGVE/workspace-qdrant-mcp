## Grammar and Runtime Management

The daemon manages tree-sitter grammars and ONNX runtime as external dependencies with automatic updates.

### Cache Locations

| Component            | Default Location                | Config Key            |
| -------------------- | ------------------------------- | --------------------- |
| Tree-sitter grammars | `~/.workspace-qdrant/grammars/` | `grammars.cache_dir`  |
| Embedding models     | `~/.workspace-qdrant/models/`   | `embedding.cache_dir` |

### Tree-sitter Grammar Management

**Configuration:**

```yaml
grammars:
  cache_dir: "~/.workspace-qdrant/grammars/"
  required:
    - rust
    - python
    - typescript
    - javascript
  auto_download: true # Download missing grammars automatically
```

**Daemon behavior:**

1. On startup, check each required grammar exists in cache
2. If grammar missing and `auto_download: true`, download from grammar repository
3. If grammar version mismatches tree-sitter runtime version, replace with compatible version
4. If `auto_download: false` and grammar missing, log warning and skip that language

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

