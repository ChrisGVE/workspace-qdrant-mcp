# workspace-qdrant-mcp

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)

A Model Context Protocol (MCP) server providing project-scoped Qdrant vector
database operations with hybrid (dense + sparse) search, backed by a Rust daemon
for high-performance file watching, indexing, and code intelligence.

> **Status: v0.2 rebuild in progress.** The codebase is being rebuilt from the
> ground up against a locked architecture. This README is an F-00 stub and grows
> with each phase; it reaches its final form at the release phase. The previous
> v0.1 implementation is preserved at the git tag **`wqm-0.1-reference`** and is
> not part of the active tree.

## Workspace layout

The Rust workspace lives at [`src/rust/`](src/rust/) and is a single Cargo
workspace that grows phase by phase. Its final shape is **19 members** -- 16 library
crates plus 3 binaries -- arranged in a strict dependency DAG across four layers
(surfaces -> host -> kernel -> infra floor). The read/write boundary is enforced at
**crate** granularity: no client binary may link a write-path crate.

| Binary | Surface | Role |
|--------|---------|------|
| `memexd` | daemon | Owns all persistent state; serves the read/write API over gRPC. |
| `workspace-qdrant-mcp` | MCP server | Exposes the 7 JSON-RPC tools over stdio. |
| `wqm` | CLI (+ TUI) | Human command surface; reads directly, proxies writes. |

The 16 library crates (`wqm-common`, `wqm-proto`, `wqm-store-read`,
`wqm-store-write`, `wqm-search`, `wqm-embed-native`, `wqm-ingest`, `wqm-intel`,
`wqm-graph`, `wqm-facade-read`, `wqm-facade-write`, `wqm-secrets`, `wqm-host`,
`wqm-service-install`, `wqm-client`, `wqm-serve`) are declared as their phase
lands, so `cargo build --workspace` is green at every merge.

## Building

```bash
cd src/rust
cargo build --workspace     # build the currently-declared members
cargo test --workspace      # run the suite
bash ci/run_guards.sh        # run the architectural + test-discipline guards
```

Later phases add platform toolchain requirements (protobuf for the gRPC contract,
a static ONNX Runtime for native embeddings); those are documented here as they
land.

## Documentation

- **Architecture & specs:** [`docs/specs/`](docs/specs/) -- the modular design.
- **Contributor guidance:** [`CLAUDE.md`](CLAUDE.md) -- build commands, invariants,
  and development rules.
- **Test charter:** [`TESTING.md`](TESTING.md) -- the two-tier TDD discipline.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
