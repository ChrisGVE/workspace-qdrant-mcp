## Future Development (Wishlist, Not Yet Scoped)

This section documents research findings and architectural ideas that may be pursued in future development cycles. These items are exploratory and have not been scoped into the project plan.

### Graph RAG (Knowledge Graph-Enhanced Retrieval)

**What it is:** Graph RAG augments traditional vector search with knowledge graph traversal, enabling relationship-aware retrieval that understands structural connections between code entities (function calls, imports, type hierarchies, module dependencies).

**Measured benefits (external benchmarks):**

- Lettria (2024): 20-25% accuracy improvement over vector-only RAG on relational queries
- Neo4j benchmark (2024): 3.4x accuracy improvement for schema-heavy and relationship-dependent queries
- Microsoft GraphRAG (2024): Significant improvement on "global" questions requiring synthesis across documents
- Matt Ambrogi retrieval study: Smaller, focused chunks (128 tokens) achieved 3x better MRR than larger chunks (256 tokens)

**Where it adds value in this project:**

- Cross-file navigation: "What functions call this method?" or "What imports this module?"
- Impact analysis: "What would break if I change this interface?"
- Architectural understanding: "Show me the dependency chain from this entry point"
- Cross-language boundaries: Connecting TypeScript MCP server to Rust daemon via gRPC definitions
- Multi-product relationships: CLI ↔ daemon SQLite schema sharing, MCP ↔ daemon gRPC communication

**Existing building blocks already in the codebase:**

- Tree-sitter semantic chunking extracts functions, classes, methods, structs, traits, enums, and their hierarchical relationships (`parent_symbol`)
- LSP integration provides resolved references, type information, and cross-file relationships
- SQLite infrastructure is already in place (`state.db` with WAL mode, ACID guarantees)
- Hybrid search (dense + sparse + RRF) provides the vector retrieval foundation
- `tracked_files` and `qdrant_chunks` tables already track file-to-chunk relationships

**Graph database evaluation (2026-02-09, updated 2026-02-10):**

| DB | License | Embeddable | Query Language | Multi-process R/W | Verdict |
|----|---------|------------|----------------|-------------------|---------|
| Kuzu | MIT | Yes (Rust + Node.js) | Full Cypher | No concurrent R+W | ~~Best graph engine~~ **ARCHIVED Oct 2025** — see note below |
| LadybugDB | MIT | Yes (Kuzu fork) | Cypher (inherited) | TBD | Fork of Kuzu, carries legacy but under new maintenance |
| HelixDB | AGPL-3.0 | Rust-native | HelixQL (proprietary) | TBD | Graph+vector in one, YC-backed, active dev — AGPL license concern |
| SQLite adjacency | Public domain | Yes | SQL + recursive CTE | WAL: 1 writer + N readers | Viable fallback, but 3+ hop queries get expensive |
| Neo4j Community | GPLv3 + Commons Clause | No (separate JVM) | Cypher | Yes | Too heavy, licensing friction (GPLv3 + Commons Clause) |
| SurrealDB | Apache 2.0 | Yes (Rust) | SurrealQL | Yes | Multi-model, vector HNSW persistence incomplete |
| Oxigraph | Apache 2.0/MIT | Yes (Rust) | SPARQL (RDF only) | N/A | Wrong data model (RDF triples, not property graphs) |

**Kuzu archived (2026-02-10 update):** The [Kuzu GitHub repository was archived on Oct 10, 2025](https://github.com/kuzudb/kuzu). The maintainers stated they are "working on something new" with no further details. No new releases or bug fixes are expected. Existing releases remain usable but the project is effectively abandoned. All references to Kuzu in this spec are retained for historical context but should be considered superseded by the alternatives below when Graph RAG work begins.

**Alternatives to evaluate when Graph RAG is scoped:**
- **[LadybugDB](https://github.com/ladybugdb/ladybugdb)**: Fork of Kuzu under new maintenance. Inherits Kuzu's MIT license, Cypher support, and embeddable architecture. Carries Kuzu's codebase legacy (both strengths and technical debt). Maturity and long-term commitment to be assessed.
- **[HelixDB](https://github.com/helixdb/helix-db)**: Rust-native graph+vector database with built-in embeddings. Active development (YC-backed, 165+ releases). Uses proprietary HelixQL query language (not Cypher). **AGPL-3.0 license** is a concern for our MIT-licensed project — would require careful evaluation of linking/embedding implications vs. using as a separate service.
- **SQLite recursive CTEs**: Already available, no new dependency. Sufficient for shallow graph queries (1-2 hops) but expensive for deep traversals and graph algorithms.

**Qdrant confirmed** as the right choice for vector search. No multi-model database matches Qdrant's vector performance.

#### Architecture Decision: Single Daemon with Embedded Graph (2026-02-23)

After deep analysis of single-daemon (graph embedded in memexd) vs. dual-daemon (separate graphd) architectures, the **single-daemon** approach was chosen. The full analysis follows.

##### Option A: Single Daemon (graph embedded in memexd) — CHOSEN

```
memexd (Rust daemon)
    ├── File watching       → filesystem (notify)
    ├── Embedding           → ONNX Runtime (all-MiniLM-L6-v2)
    ├── Vector writes       → Qdrant (direct)
    ├── FTS5 writes         → search.db (SQLite)
    ├── Graph writes        → graph.db (embedded, in-process)
    ├── Graph queries       → graph.db (in-process, zero-IPC)
    ├── State writes        → state.db (SQLite WAL)
    └── gRPC server         → serves MCP server, CLI
```

**Pros:**

1. **Zero-IPC graph access.** Graph writes during file ingestion are in-process function calls, not gRPC round-trips. At ~10 edges/file and thousands of files, this saves significant latency. The ingestion pipeline already runs `tree_sitter::extract_symbols()` and `cooccurrence_graph::update_graph()` in-process — adding graph edge insertion as another in-process step is natural.

2. **Atomic ingestion.** File ingestion currently writes to Qdrant + search.db + state.db. Adding graph.db as a fourth destination keeps all state mutations in the same transaction scope. With a separate daemon, a crash between "edge sent to graphd" and "queue item marked done" creates consistency gaps.

3. **Single process management.** One launchd plist, one PID, one log stream, one health check. Users already manage memexd — adding graphd doubles the operational surface area (install, start, stop, status, logs, troubleshooting for two processes).

4. **Shared context.** memexd already holds the `ProcessingContext` with SQLite pool, Qdrant client, lexicon, embedding generator, LSP manager. Graph operations need all of this context. In a separate daemon, you'd replicate it or proxy it via gRPC.

5. **LadybugDB embeds well.** The `lbug` crate is designed for in-process embedding (no server). Its MVCC model (concurrent readers, single writer) maps perfectly to memexd's pattern: the queue processor writes edges sequentially, while MCP/CLI queries read concurrently via the gRPC server.

6. **Build pipeline is simpler.** One binary to build, test, deploy. The existing `ORT_LIB_LOCATION` + static linking story already handles ONNX Runtime (C++). Adding LadybugDB (also C++ underneath) is one more dependency in the same build.

7. **No inter-daemon synchronization.** With two daemons, you need to handle: graphd startup before memexd tries to send edges, graphd crashes while memexd is running, version skew between the two binaries, shared file locking on state.db.

**Cons:**

1. **Binary size growth.** LadybugDB's C++ core adds significant weight to the memexd binary (likely 10-20MB). Already large due to ONNX Runtime.

2. **Memory footprint.** Graph database keeps its own buffer pool, page cache, and index structures in-process. Combined with ONNX Runtime, Qdrant client, SQLite pools, and the embedding model, memory usage increases. On constrained systems this matters.

3. **Crash blast radius.** A bug in graph traversal (e.g., infinite loop in Cypher evaluation, memory corruption in LadybugDB C++) takes down the entire daemon — file watching, embedding, everything. With a separate daemon, a graphd crash doesn't affect file ingestion.

4. **Build complexity.** LadybugDB requires Clang/LLVM for C++ compilation. This adds to the already complex Intel Mac build story (ORT static linking). Cross-compilation for CI becomes harder.

5. **Single-writer contention.** Both graph writes (during ingestion) and graph queries (from gRPC) go through the same instance. Heavy graph queries could block graph writes during ingestion, or vice versa. Managed via read-write coordination (see concurrency model below).

6. **Testing surface.** Integration tests for graph features must spin up the full memexd environment. With a separate daemon, graph tests can run in isolation.

##### Option B: Dual Daemon (separate graphd) — REJECTED

```
memexd (Rust daemon)                    graphd (Rust daemon)
    ├── File watching → filesystem          ├── Graph storage → LadybugDB
    ├── Embedding → ONNX Runtime            ├── Write API → gRPC
    ├── Vector writes → Qdrant              ├── Query API → gRPC
    ├── Graph edges → graphd (gRPC)         └── Analytics → Cypher
    └── State writes → state.db
```

**Pros:**

1. **Fault isolation.** graphd crash doesn't affect file watching or embedding. memexd crash doesn't corrupt the graph. Each daemon can be restarted independently.

2. **Independent scaling.** Graph queries (impact analysis, community detection) are CPU-intensive. In a separate process, they run on their own thread pool without competing with embedding generation or queue processing.

3. **Clean API boundary.** The gRPC interface forces a well-defined contract between ingestion and graph operations. This is good for testing, documentation, and future extensibility (e.g., swapping graph engines behind the same gRPC API).

4. **Simpler builds per binary.** memexd doesn't need Clang for LadybugDB. graphd doesn't need ORT. Each binary has a smaller dependency closure.

5. **Graph engine swappability.** The gRPC API is engine-agnostic. If LadybugDB stalls (post-Kuzu risk), you can swap to CozoDB or SQLite CTEs without touching memexd.

6. **Memory isolation.** Each process has its own address space. LadybugDB's buffer pool doesn't compete with ONNX Runtime's memory allocator. OS can page out graphd memory when it's idle.

**Cons:**

1. **IPC overhead.** Every graph edge write during file ingestion is a gRPC call. At 10 edges/file x 1000 files/scan = 10K gRPC calls. Even with connection pooling and batching, this is ~10x slower than in-process writes. Batching mitigates but adds buffering complexity.

2. **Consistency gaps.** If memexd sends edges to graphd and then crashes before marking the queue item done, the graph has partial data for a file version. Need retry logic, idempotency keys, and graph-side deduplication.

3. **Operational complexity.** Two daemons = two launchd plists, two PIDs, two log streams, two health checks, two versions to keep in sync. Users must install/update/troubleshoot both. The CLI needs `wqm service install --graph`, `wqm service status --graph`, etc.

4. **Startup ordering.** memexd sends edges during ingestion. If graphd isn't running, those writes fail. Need retry/queue-and-replay logic in memexd, or startup dependency management (memexd waits for graphd).

5. **Version coupling.** The gRPC proto between memexd and graphd creates a tight coupling. Any schema change requires coordinated releases. Proto backward compatibility must be maintained.

6. **Network port consumption.** graphd needs its own port (e.g., 50052). Firewall rules, port conflicts, and service discovery add friction.

7. **Development velocity impact.** Every graph feature requires changes to at least three packages: the proto definition, graphd's implementation, and memexd's client code. With a single daemon, it's one crate.

##### Debugging Complexity Comparison

**Single daemon:** One process, one log stream, one debugger attachment point. All state is in-process — you can inspect graph, SQLite, and Qdrant state from the same debugging session. Stack traces show the full call chain from file watcher to queue to graph write.

**Dual daemon:** Distributed tracing needed (OpenTelemetry spans across gRPC boundaries). Debugging "why is this graph edge missing?" requires correlating logs from two processes. Race conditions between the two daemons are harder to reproduce and diagnose.

##### Roadblocks and Risks

**LadybugDB (both architectures):**
- C++ build dependency (Clang/LLVM required)
- Only 61% of API documented
- Fork from archived project — long-term sustainability uncertain
- Some bindings (Go, WASM) were broken in early releases; Rust bindings may have edge cases

**Dual daemon specific:**
- No existing graphd binary, gRPC proto, or service management code — entirely new infrastructure
- launchd dependency ordering between memexd and graphd isn't natively supported (would need health-check polling)
- Cross-daemon integration testing requires both processes running

**Single daemon specific:**
- LadybugDB C++ linking may conflict with ONNX Runtime's C++ requirements (both use C++ standard library, potentially different versions)
- Memory pressure on machines with limited RAM (8GB systems running Docker + Qdrant + memexd)

##### Decisive Factors for Single Daemon

1. **Development velocity.** Pre-release project evolving rapidly. Every graph feature change would require proto changes + graphd updates + memexd client updates — tripling the development surface. In a single daemon, graph features are regular Rust modules alongside text_search, storage, and embedding.

2. **LadybugDB embeds naturally.** The `lbug` crate is designed for exactly this use case. Its MVCC model matches memexd's access pattern. There's no technical reason to put it in a separate process — it's like how we embed SQLite (via sqlx) and ONNX Runtime (via fastembed) in the same process.

3. **Atomicity matters.** The ingestion pipeline writes to Qdrant, search.db, and state.db in a coordinated flow. Adding graph.db as a fourth in-process destination is straightforward. With a separate daemon, every write requires IPC and explicit consistency handling.

4. **Operational simplicity.** Users are developers using this for personal knowledge management. One daemon that "just works" is the right abstraction.

5. **Trait abstraction replaces process boundary.** Define a `GraphStore` trait, implement it for SQLite CTEs (Phase 1) and LadybugDB (Phase 2). The trait boundary provides the same swappability as a gRPC API, without the IPC overhead.

6. **Crash isolation is manageable.** LadybugDB runs in `tokio::task::spawn_blocking` context. Panics can be caught with `std::panic::catch_unwind`. For truly critical paths, the `catch_unwind` wrapper prevents C++ layer issues from bringing down the tokio runtime.

##### Performance Comparison

| Operation | Single Daemon | Dual Daemon |
|-----------|--------------|-------------|
| Edge write (per file) | ~50μs (in-process) | ~500μs-1ms (gRPC) |
| Batch edge write (1000 files) | ~50ms | ~500ms-1s (with batching) |
| Graph query (1-hop) | ~100μs (in-process) | ~1-2ms (gRPC) |
| Graph query (3-hop) | ~1-5ms (in-process) | ~2-6ms (gRPC) |
| Impact analysis (deep) | ~10-50ms (in-process) | ~12-52ms (gRPC) |
| Community detection | ~100ms-1s (CPU-bound) | ~100ms-1s (own thread pool) |

##### Graph Read-Write Concurrency Model

Graph access uses a self-managed read-write coordination scheme that avoids blocking the ingestion pipeline:

- **Multiple concurrent readers:** Graph read queries (from gRPC) run in parallel on a dedicated thread pool via `spawn_blocking`. Readers acquire a shared read token.
- **Single writer:** The queue processor writes graph edges sequentially during file ingestion.
- **Write-yields-to-reads:** When a write operation completes, if pending read queries are waiting, reads run first. The next write waits for all pending reads to complete before proceeding.
- **Reads-yield-to-write:** When no reads are pending, writes proceed immediately without coordination overhead.

This ensures that graph queries from MCP/CLI are never starved by sustained write bursts during project scans, while writes are only briefly delayed when queries are actively in flight. The pattern is similar to a fair read-write lock with writer starvation prevention.

##### Implementation Roadmap

**Phase 1 — SQLite CTEs (zero new dependencies):**
- Graph schema as SQLite adjacency tables in `state.db` or dedicated `graph.db`
- `GraphStore` trait with `SqliteGraphStore` implementation
- Recursive CTE queries for 1-2 hop traversals (covers ~80% of use cases: call chains, imports, type usage)
- Integration with existing ingestion pipeline (tree-sitter symbol extraction → graph edges)
- gRPC service for graph queries

**Phase 2 — LadybugDB upgrade (when deep queries needed):**
- Add `lbug` crate dependency with `LadybugGraphStore` implementation of same `GraphStore` trait
- Full Cypher support: multi-hop traversal, path finding, graph algorithms
- Community detection, centrality analysis, impact analysis
- Feature-flag or config toggle between SQLite and LadybugDB backends
- Dedicated `~/.workspace-qdrant/graph/` storage directory for LadybugDB

##### Graph Database Evaluation (2026-02-23 update)

| DB | License | Embeddable | Query Language | Concurrent R/W | Status | Verdict |
|----|---------|------------|----------------|----------------|--------|---------|
| LadybugDB (lbug) | MIT | Yes (in-process) | Cypher | MVCC / single writer | Active (Jan 2026, v0.14.2) | **Phase 2 target** — best embeddable property graph |
| HelixDB | AGPL-3.0 | No (server only) | Custom HelixQL | Yes (server) | Active (Feb 2026) | **Rejected** — AGPL incompatible with MIT, server-only |
| CozoDB | MPL-2.0 | Yes (in-process) | Datalog | RocksDB: full | Stalled (Dec 2023) | **Rejected** — stalled development, Datalog not Cypher |
| IndraDB | MPL-2.0 | Yes (in-process) | None (Rust API) | Backend-dependent | Low (Aug 2025) | **Rejected** — no query language, low maintenance |
| Oxigraph | MIT/Apache-2.0 | Yes (in-process) | SPARQL (RDF) | Single writer + readers | Active (Feb 2026) | **Wrong data model** — RDF, not property graph |
| SQLite + CTEs | Public domain | Yes | SQL + recursive CTE | WAL: 1W + NR | N/A | **Phase 1 backend** — zero deps, sufficient for shallow queries |

**LadybugDB details (v0.14.2, January 2026):**
- Fork of archived KuzuDB, under active maintenance by Ladybug Memory
- Crate: `lbug` on crates.io (MIT license)
- True in-process embedding via `Database::new()`, no server required
- Builds C++ core from source (requires Clang/LLVM)
- 61% API documented — Rust bindings are usable but edge cases may exist
- Long-term sustainability depends on Ladybug Memory's continued investment

##### Graph Schema (property graph, engine-agnostic)

Node types: `File`, `Function`, `Class`, `Method`, `Struct`, `Trait`, `Module`, `Document`

Edge types: `CALLS`, `IMPORTS`, `EXTENDS`, `IMPLEMENTS`, `USES_TYPE`, `CONTAINS`, `MENTIONS`

Each node and edge carries a `tenant_id` property for project isolation. Cross-tenant queries are explicitly forbidden.

### Cross-Project Search

**Current model:** Projects are isolated by `tenant_id` in the `projects` collection. Search is project-scoped.

**Revised model: Tiered search with automated grouping**

Search should support three scopes, selectable per query:
1. **Project scope** (default): Search within the current project's `tenant_id`
2. **Group scope**: Search within an automatically detected project group
3. **All projects scope**: Search across the entire `projects` collection (no tenant filter)

**Automated project grouping (no manual configuration required):**
- Shared dependencies: Projects using the same libraries (parsed from Cargo.toml, package.json, requirements.txt)
- Git organization: Projects under the same GitHub org/user
- Explicit cross-references: Projects linked by gRPC proto imports, shared crate dependencies, or workspace membership
- Embedding similarity: Projects whose README/description embeddings cluster together

**Relevance ranking across projects:** When searching beyond the current project, results from the current project receive full weight. Results from group projects receive a decay factor (e.g., 0.7). Results from unrelated projects receive a further decay (e.g., 0.4). This preserves signal-to-noise while enabling cross-project discovery.

**Graph RAG across projects:** Cross-project graph edges (e.g., shared dependency usage patterns, similar function signatures) enable structural code reuse discovery that vector search alone misses. GitHub research shows ~5% of code across repositories is cross-project clones, concentrated within similar domains.

### Project vs. Library Boundary

**Observation:** Not all files in a project folder are "project code." Research papers, development notes, experiment results, and reference PDFs are background knowledge that supports the project but clutters code search results.

**Revised approach — format-based routing (deterministic, no content classification):**
- **Developer-created text files** (`.md`, `.txt`, `.rst`, source code, configs, specs, tests): Stay in `projects` collection with project `tenant_id`. This includes experiment notes, PoC write-ups, and development logs — these are project context, not parasites. Vector search naturally ranks code higher for code queries and notes higher for design rationale queries.
- **Binary/downloaded reference files** (`.pdf`, `.epub`, `.djvu`, `.docx`): Route to `libraries` collection with a `source_project_id` metadata field linking them back to their project. These are external reference material, not developer-created project artifacts.

**Rationale for keeping .md experiments in projects:** Experiment notes, PoC results, and development logs answer questions like "why did we choose this algorithm?" and "what did we try that didn't work?" — genuinely valuable project context. The occasional noise in code search is far less costly than the complexity of content-based classification to sort .md files. The hybrid search system handles relevance naturally: code-related queries rank code chunks higher; design rationale queries rank notes higher.

**Configurable edge cases:** `.docx` files could exist in either context (a spec written in Word vs. a downloaded paper). The default routes `.docx` to libraries, but users can override via configuration. The routing rule is extension-based and lives in the watching configuration.

**Cross-collection search:** Qdrant does not support native cross-collection queries. Implementation requires issuing parallel queries to both collections and merging results via RRF. The MCP `search` tool would accept an optional `include_libraries: true` parameter (default false for code queries, true for general knowledge queries).

### Library Collection Tenancy (Revised)

**Current model:** Libraries isolated by `library_name`. Each library is a separate tenant.

**Revised model: Hierarchical naming + cross-library search by default**

**Hierarchical tenant naming** derived from the filesystem structure relative to the watched folder root:
```
library_name.relative_path_segments.document_name
```
Example: Library root "main" containing `computer_science/design_patterns/Gang_of_Four.pdf`:
- `library_name`: "main"
- `library_path`: "computer_science/design_patterns"
- `document_name`: "Gang_of_Four.pdf"
- Full tenant: "main.computer_science.design_patterns.Gang_of_Four"

**Search scoping via prefix matching:**
- `library_name = "main"` → search entire main library
- `library_name = "main" AND library_path LIKE "computer_science%"` → search CS subdomain
- No filter → search all libraries (default for knowledge queries)
- `source_project_id = "<project>"` → search project-associated references

**Cross-library search is the default.** The `library_name` and `library_path` fields are metadata for management (deletion, updates) and optional scoping, not mandatory isolation. Knowledge synthesis across complementary sources (e.g., multiple physics textbooks) happens naturally when search is unfiltered.

### Automated Tagging and Grouping

**Problem:** Manual tagging creates mental overhead and inconsistent coverage. Tags should be generated automatically from available signals.

**Core principle: per-chunk aggregation.** Tags are not derived from a document summary or its first N tokens. Instead, each chunk (without overlap) is independently tagged, then tags are aggregated across all chunks by frequency. Tags appearing in the highest fraction of chunks represent the document's dominant topics. This naturally weights tags by coverage — a QM textbook mentioning "Schrödinger equation" in 40% of chunks gets a strong tag; "classical limit" in 3% gets a weak tag. A configurable `min_frequency` threshold (e.g., 10% of chunks) filters noise.

**Concept normalization:** Raw implementation-level names (library names, API names) must be mapped to concept-level tags. `regex` (Rust crate), `re` (Python stdlib), `pcre` (C library), and `oniguruma` (Ruby) all map to the concept `regular-expressions`. Without normalization, two projects using the same concept in different languages appear unrelated.

Normalization sources (in order of preference):
1. **Package registry categories**: crates.io categories (`text-processing`, `parsing`), npm keywords, PyPI classifiers. These are already concept-level and maintained by the package ecosystem.
2. **Curated concept dictionary**: A mapping from common library names to concepts, generated once by an LLM and stored as a configuration file (`assets/concept_normalization.yaml`). One-time cost, periodically refreshed.
3. **Embedding-based semantic grouping**: Embed dependency descriptions (from registry metadata). Libraries with similar descriptions cluster into shared concept tags.

**Automated tagging pipeline:**

**Tier 1 — Zero-cost heuristics (always active, no ML):**
1. **Path-derived tags**: Parse directory names and filenames into normalized topic tags. `computer_science/design_patterns/` → tags: `computer-science`, `design-patterns`. Most reliable signal because directory names tend to be conceptual.
2. **PDF metadata extraction**: Extract title, author, subject, keywords from PDF document metadata fields. Available in most academic papers and textbooks without content analysis.
3. **Dependency-derived concepts** (projects only): Parse dependency files (Cargo.toml, package.json, requirements.txt), normalize via concept dictionary/registry categories. Produces project-level topic tags.

**Tier 2 — Embedding-based (uses existing model, no LLM):**
4. **TF-IDF keyword extraction per chunk**: Extract top distinctive terms from each chunk, aggregate by frequency across all chunks. Filtered against stop-word list. Produces content-level topic tags.
5. **Embedding-based clustering**: Group documents by embedding similarity (embeddings already generated during ingestion). Cluster labels derived from distinctive terms within each cluster.
6. **Zero-shot classification**: Compare document embedding against embeddings of a predefined topic taxonomy (~100 terms). If cosine similarity exceeds threshold, apply the tag. Uses existing embedding model.

**Tier 3 — LLM-assisted (optional, configurable, independent from MCP session):**
7. **LLM-based chunk tagging**: For each chunk, ask a configured LLM for 3-5 topic tags. Aggregate across chunks by frequency. Top-N by frequency = document tags.

**Tier 3 LLM configuration:** The tagging LLM is independent from the MCP-connected LLM session. Tagging happens during daemon ingestion, which may run when no LLM session is active. The model is configurable:

```yaml
# In default_configuration.yaml
tagging:
  auto_tag: true
  min_frequency: 0.1              # Tag must appear in ≥10% of chunks
  top_n_tags: 10                  # Maximum tags per document
  tier3_enabled: false            # LLM tagging off by default
  tier3_provider: "anthropic"     # anthropic, openai, ollama, none
  tier3_model: "claude-haiku"     # Cheapest appropriate model
  tier3_api_key_env: "ANTHROPIC_API_KEY"
  tier3_ollama_url: "http://localhost:11434"  # For local models
```

Supported providers (prefer subscription/local models over per-call API charges):
- **Ollama** (local models — zero marginal cost, no network dependency, recommended for users with hardware)
- **Subscription-based platforms** (e.g., Abacus.ai, or subscription plans that include API access)
- **Anthropic** (Haiku for cost efficiency, only when API calls are acceptable)
- **OpenAI** (GPT-4o-mini or similar low-cost model, only when API calls are acceptable)
- **None** (Tiers 1-2 only — default)

**Tag inheritance:** Tags propagate to contained documents. A folder tagged `physics` automatically tags all documents within it. Document-level tags extend folder-level tags.

**Applicability to projects:** The same pipeline applies to project grouping. Dependency analysis with concept normalization provides strong domain signals. README content embedding provides semantic classification. Two projects with different dependency names but the same concepts (e.g., `regex` and `re`) are correctly grouped.

**Tag evolution — lifecycle tied to ingestion pipeline:**

Tags are dynamic and must evolve as content changes. Tag updates piggyback on the existing file change processing pipeline — no separate watcher or queue needed.

| Event | Tag action |
|-------|-----------|
| File created | Chunks tagged (Tiers 1-2) → aggregate by frequency → store document-level tags |
| File modified | Chunks re-generated → re-tagged → document tags recomputed from new frequencies → old zero-frequency tags removed, new tags added |
| File deleted | Document tags removed → cluster memberships recalculated if clustering active |
| Dependency file changed | Concept tags re-derived from updated dependencies → project-level tags updated |
| Folder renamed | Path-derived tags re-derived from new path |

**Tag storage (dual):**
- **Qdrant payload**: Each point carries chunk-level tags (enables tag-filtered vector search)
- **SQLite `tracked_files`**: Document-level tag summary with frequencies (enables browsing, management, cross-document tag analysis)

**Tag drift:** A project that starts as `data-processing` and evolves toward `machine-learning` naturally reflects this because tags are re-derived from content on every re-ingestion, never manually pinned. Frequency-based aggregation means dominant topics surface automatically as the codebase evolves.

#### Automated Affinity Grouping

**Goal:** Automatically group related projects without user intervention, producing both cluster membership and human-readable group labels.

**Embedding-based affinity pipeline (LLM-free):**

1. **Per-project aggregate embedding**: Average all chunk embeddings for a project into a single vector. Chunk embeddings already exist from ingestion — no additional embedding cost.
2. **Pairwise cosine similarity**: Compare aggregate embeddings between projects. Projects above a configurable threshold (e.g., 0.7) form an affinity group.
3. **Group labeling via taxonomy matching**: Compare the group's centroid embedding against a predefined taxonomy (see below). Top-N matching taxonomy terms become the group's label.

This pipeline is LLM-free, uses only the existing FastEmbed model, and runs as a background daemon task after ingestion.

**Taxonomy source — package registry categories:**

The taxonomy is sourced from community-curated package registry categories, not manually defined:
- **crates.io**: ~70 categories (`algorithms`, `authentication`, `command-line-interface`, `concurrency`, `cryptography`, `database`, `network-programming`, `text-processing`, `web-programming`, etc.)
- **npm**: similar keyword ecosystem
- **PyPI**: detailed classifiers (`Topic :: Scientific/Engineering :: Artificial Intelligence`, etc.)

Combined and deduplicated, these provide ~150-200 concept-level terms. They are embedded once at daemon startup using FastEmbed and cached.

**Zero-shot taxonomy matching:**

For each document or project, compare its aggregate embedding against all taxonomy embeddings via cosine similarity. Top-N matches above threshold become tags. This replaces manual tagging entirely for code projects.

**Open questions and concerns (to be validated empirically):**

1. **Embedding dimensionality mismatch**: 384-dim MiniLM embeddings of short taxonomy phrases (e.g., "cryptography") may not produce reliable cosine similarity against averaged code chunk embeddings. The semantic spaces may be too different. Empirical testing required.
2. **Tier 1 heuristic quality**: Path-derived tags are unreliable (directory names are often structural, not conceptual — `src/`, `lib/`, `utils/`). Dependency-derived concepts require a concept dictionary to map library names to concepts (e.g., `tokio` → `async-runtime`, `serde` → `serialization`). Without this mapping, raw dependency names are not useful as tags. PDF/EPUB metadata is valuable when present but not universally available.
3. **TF-IDF produces terms, not concepts**: TF-IDF (Term Frequency – Inverse Document Frequency) extracts distinctive keywords by scoring words that are frequent in a specific chunk but rare across all documents. For example, in an async runtime library, `async`, `executor`, `spawn` score high while `the`, `function`, `return` are filtered out. However, TF-IDF produces raw terms (`tokio`, `serde`, `reqwest`), not concept-level tags (`async-runtime`, `serialization`, `http-client`). A concept normalization step is still needed.
4. **Concept labeling gap**: The fundamental challenge is turning raw signals (keywords, library names, embedding clusters) into meaningful human-readable concept tags. Without either an LLM or a curated mapping, embedding clustering gives groups but cannot name them. The registry-based taxonomy approach closes this gap for code projects but may be insufficient for non-code content (documents, research papers).
5. **Fallback strategy**: If zero-shot taxonomy matching proves too noisy, the fallback is TF-IDF keywords matched against the `concept_normalization.yaml` dictionary (one-time LLM cost to generate the mapping, then static and periodically refreshed).

**Implementation plan:**

Phase 1 (current): Implement zero-shot taxonomy matching using registry categories as taxonomy source. Embed taxonomy terms at startup, compare against document/project aggregate embeddings during ingestion. Evaluate quality empirically.

Phase 2 (if Phase 1 insufficient): Add TF-IDF keyword extraction + concept dictionary mapping. Generate `concept_normalization.yaml` once using LLM, store as static config.

Phase 3 (optional): Enable Tier 3 LLM-assisted tagging for users who want higher quality and have API access configured.

### Tag/Keyword Hierarchy Management — Paths Not Taken

*Decision date: 2026-02-16. Context: tagging PRD design decisions (`.taskmaster/docs/20260216-tagging-decisions.md`, Group 4).*

**Chosen approach (v1):** Nightly batch rebuild — full agglomerative clustering on canonical tag vectors per collection. Hierarchies stored in SQLite. Simple, high-quality, and cheap (clustering ~1000 tag vectors takes seconds).

**Deferred approaches:**

1. **Incremental hierarchy updates on each ingestion**: Update the hierarchy tree as each document is ingested. Deferred because: (a) agglomerative clustering degrades when applied incrementally — adding a single document can shift centroids in ways a full rebuild wouldn't; (b) inevitably requires a periodic full rebuild as a correction pass, resulting in two code paths; (c) 24-hour staleness is acceptable since hierarchies serve navigation/exploration, not real-time search.

2. **Hybrid approach (incremental adds + nightly full re-clustering)**: Incrementally add new tags as "pending" nodes during ingestion, then nightly rebuild integrates them into the full hierarchy. Deferred because: (a) two code paths to maintain (incremental insertion + full rebuild); (b) marginal improvement in freshness doesn't justify the complexity; (c) "pending" nodes in the tree create UX ambiguity (are they positioned correctly?).

**When to reconsider:** If users report that 24-hour hierarchy lag is painful (e.g., rapid library ingestion where navigation is needed immediately), revisit incremental updates. The hybrid approach becomes attractive if the nightly rebuild cost grows beyond minutes (unlikely below ~100K canonical tags).

### Symbol Co-occurrence Graph for Concept Extraction — Deferred to v2

*Decision date: 2026-02-16. Context: tagging PRD design decisions (`.taskmaster/docs/20260216-tagging-decisions.md`, Group 5).*

**Concept:** Build a symbol co-occurrence graph (nodes = extracted phrases, edges = appear in same file/module) and rank by graph centrality + semantic similarity to surface architectural concepts better than TF-IDF alone.

**Why deferred:** The two-stage pipeline (TF-IDF candidate extraction + embedding rerank with MMR diversity) is already strong for v1. Co-occurrence graph adds significant scope (graph construction, centrality computation, LSP dependency for cross-file references) for incremental quality improvement. LSP is only available for active projects, creating a quality split between active and inactive projects.

**v2 synergy with graph infrastructure:** Once LadybugDB (or equivalent graph engine) is integrated for Graph RAG, the co-occurrence graph becomes a natural extension — symbol nodes and co-occurrence edges map directly to the property graph model. Graph centrality queries (PageRank, betweenness) that would be expensive to implement ad-hoc are trivial with a graph database. Additionally, some deferred items from hierarchy management (incremental updates, hybrid approach) also benefit from native graph storage for the tag hierarchy itself.

**When to reconsider:** After LadybugDB integration is stable and the v1 tagging pipeline has been validated empirically.

### Knowledge Overlap and Complementary Sources

**Challenge:** Multiple sources covering the same topic (e.g., physics textbooks with math prerequisite chapters) overlap and complement each other. The system should synthesize across sources rather than treating each in isolation.

**Approach:**
- **Cross-library search by default** ensures overlapping content from different sources is surfaced together
- **Provenance metadata** (library_name, library_path, document_name) returned with every search result enables the consumer to assess source diversity and reliability
- **Source diversity in results**: When multiple chunks from different sources match a query, prefer diverse results (1 chunk from each of 3 books) over concentrated results (3 chunks from 1 book). This is a post-retrieval re-ranking step.
- **Contradiction handling**: Not automatically resolved. Provenance metadata enables the LLM consumer to reason about conflicting sources. Academic research confirms this remains an unsolved problem for automated systems.

### Graph Storage — Embedded in memexd (Single Daemon)

**Note (2026-02-23):** The original graphd (separate daemon) architecture was replaced by an embedded single-daemon approach after thorough analysis. See [Architecture Decision: Single Daemon with Embedded Graph](#architecture-decision-single-daemon-with-embedded-graph-2026-02-23) for the full rationale, pros/cons comparison, and performance analysis.

**Graph evolution — delete/re-ingest pattern (same as Qdrant):**

Graph data follows the same lifecycle as vector data. When a file changes, old graph edges are deleted and new ones are extracted and inserted.

| Event | Graph action |
|-------|-------------|
| File created | Tree-sitter extracts symbols (nodes) → LSP resolves references (edges) → inserted in-process |
| File modified | Delete edges WHERE `source_file = modified_file` → re-extract → re-insert. Nodes updated in place (symbol identity persists even if signature changes) |
| File deleted | Delete edges WHERE `source_file = deleted_file` → orphan node cleanup (nodes with no remaining edges pruned) |
| Project deleted | Delete all nodes/edges WHERE `tenant_id = project_id` |

**Node vs. edge ownership:**
- **Edges** are owned by the source file. When `bar.rs` changes, edges *from* symbols in `bar.rs` are deleted and re-created. Edges *to* symbols in `bar.rs` from other files remain untouched — they update when their source files are re-processed.
- **Nodes** represent symbols and are updated in place. A function may change its signature but keeps its graph identity. Orphaned nodes (no remaining edges) are pruned during cleanup.

**Pipeline integration — additions to existing ingestion flow:**
```
File change detected (file watcher)
    → Queue item created (unified_queue)
        → memexd processes:
            1. Delete old Qdrant chunks for file       (existing)
            2. Delete old graph edges for file          (NEW — in-process)
            3. Re-chunk (tree-sitter or fixed-size)     (existing)
            4. Re-embed chunks                          (existing)
            5. Extract relationships from AST/LSP       (NEW)
            6. Upsert chunks to Qdrant                  (existing)
            7. Insert graph edges                       (NEW — in-process)
            8. Update tags (per-chunk, aggregate)       (NEW)
```

Steps 2, 5, 7, 8 are additions. The queue, file watching, debouncing, and processing loop remain unchanged. All graph operations are in-process function calls (no IPC).

**Eventual consistency:** During batch operations (e.g., `git checkout` changing many files), the graph may have stale edges briefly while files are processed through the queue. This is the same trade-off accepted with Qdrant — the queue guarantees all files are eventually processed.

**Future-proofing:** The `GraphStore` trait is the stable contract. The graph engine can be swapped (SQLite CTEs → LadybugDB, or any future candidate) by implementing the trait. The trait boundary provides the same swappability as a gRPC API, without IPC overhead.

**CLI commands:**
```bash
wqm graph query "MATCH (f:Function)-[:CALLS]->(g:Function) WHERE f.name = 'main' RETURN g"
wqm graph impact --symbol parse --file document_processor.rs
wqm graph communities           # List detected code communities
wqm graph stats                 # Node/edge counts and storage size
```

### Knowledge Manager Product Potential

The capabilities being built form the foundation for a general-purpose knowledge management system that extends well beyond code intelligence:

**Current capabilities (developer tool):**
- Automatic file watching with intelligent ingestion
- Hybrid semantic + keyword search with RRF fusion
- Multi-tenant knowledge organization (projects, libraries, memory)
- Semantic code intelligence (tree-sitter, LSP)
- MCP interface (any LLM client can use it)
- CLI for direct access and diagnostics

**Planned capabilities (knowledge base):**
- Graph relationship tracking (embedded in memexd — SQLite CTEs → LadybugDB, see architecture decision)
- Automated tagging and classification (Tiers 1-3)
- Cross-collection and cross-project search
- Content-type routing (code vs. reference material)
- Hierarchical library organization with cross-library synthesis
- Source diversity in results and provenance metadata

**Product evolution path:**
1. **Current**: Developer tool — code intelligence + reference libraries for individual developers
2. **Near-term**: Personal knowledge base — automated tagging, cross-library synthesis, background knowledge routing. The user's entire document collection (textbooks, research papers, notes, code) becomes a searchable, interconnected knowledge graph.
3. **Long-term**: Team knowledge manager — multi-user access control, shared libraries, collaborative tagging, organizational knowledge graph

**The MCP interface is the critical enabler.** It means the knowledge manager is accessible from any LLM-powered tool (Claude Desktop, Claude Code, Cursor, or any future MCP client) without building a custom UI. The LLM conversation itself becomes the interface. The underlying engine (memexd + Qdrant + SQLite + embedded graph) is domain-agnostic — it handles code, documents, research papers, and any text content through the same pipeline.

### Chunk Size Optimization Research

**Finding:** 384 characters with 15% overlap (58 characters) is optimal for the all-MiniLM-L6-v2 embedding model.

**Evidence:**

- Internal benchmark (2026-02-09, 500KB text): 384 chars achieved 86.16 ms/KB, 19% better throughput than the previous 512-char default
- Matt Ambrogi study: 128 tokens had 3x better MRR than 256 tokens for retrieval
- Chroma Research: 200-token chunks had 2x better precision than 400-token chunks
- all-MiniLM-L6-v2 is trained on 128-token sequences (256-token max); 384 chars ≈ 82 tokens (prose) or ≈ 110 tokens (code)
- NVIDIA research: 15% overlap is optimal for context preservation across chunk boundaries

**Status:** Defaults updated to 384 chars / 58 overlap in configuration and `ChunkingConfig`.

**Note:** These defaults apply to fixed-size text chunking only. Tree-sitter semantic chunking uses natural code boundaries (functions, classes, methods) and has its own size limits (`DEFAULT_MAX_CHUNK_SIZE = 8000` estimated tokens).

### Distance Matrix Visualization

Qdrant's Distance Matrix API can compute pairwise distances between points using the `dense` vector. This could power interactive code intelligence visualizations showing clusters of semantically related files, functions, or documentation. Could serve as a stepping stone toward full Graph RAG by revealing natural code clusters. See the [Qdrant Dashboard Visualization](12-configuration.md#qdrant-dashboard-visualization) section for current capabilities.

---

## Related Documents

| Document                                                           | Purpose                          |
| ------------------------------------------------------------------ | -------------------------------- |
| [FIRST-PRINCIPLES.md](../../FIRST-PRINCIPLES.md)                    | Architectural philosophy         |
| [ADR-001](../adr/ADR-001-canonical-collection-architecture.md)      | Collection architecture decision |
| [ADR-002](../adr/ADR-002-daemon-only-write-policy.md)               | Write policy decision            |
| [ADR-003](../adr/ADR-003-daemon-owns-sqlite.md)                     | SQLite ownership decision        |
| [docs/ARCHITECTURE.md](../ARCHITECTURE.md)                          | Visual architecture diagrams     |
| [docs/LSP_INTEGRATION.md](../LSP_INTEGRATION.md)                    | LSP integration guide            |
| [README.md](../../README.md)                                        | User documentation               |

---

**Version:** 1.9.0
**Last Updated:** 2026-02-18
**Changes:**

- v1.9.0: State integrity redesign — base point identity model (`hash(tenant_id, branch, relative_path, file_hash)`) replacing simple `SHA256(tenant_id | branch | file_path | chunk_index)` formula; two-layer watching architecture (Layer 1: file watcher for all projects, Layer 2: git watcher for `.git/HEAD` + `.git/refs/heads/` with reflog parsing and `git diff-tree` for precise change detection); branch switch protocol with selective re-ingestion; `watch_folder_submodules` junction table replacing single-FK `parent_watch_id` for many-to-many submodule relationships with commit pin detection via `git ls-tree`; per-destination queue state machine (`qdrant_status`, `search_status`, `decision_json`) enabling parallel Qdrant + search DB execution with retry on failed destination only; search.db (FTS5) documentation (`file_metadata` + `code_lines` tables); cross-instance deduplication via reference-counting across watch folders sharing same tenant_id; disaster recovery (`wqm admin recover-state` from Qdrant payloads, `memory_mirror` table for reverse recovery, `wqm admin qdrant-snapshot` commands); tenant ID precedence (remote-based takes precedence over path-based, `path_` prefix for local projects, cascade rename on gaining remote); `is_git_tracked` and `last_commit_hash` columns on `watch_folders`; `base_point`, `relative_path`, `incremental` columns on `tracked_files`; updated Qdrant payload schema with `base_point`, `relative_path`, `absolute_path`, `file_hash`, `commit_hash` fields for recovery
- v1.8.1: Queue taxonomy expansion — added `tenant`, `collection`, `website`, `url`, `doc` item types and `add`, `rename`, `uplift`, `reset` operations; enqueue-only gRPC pattern for RegisterProject and DeleteProject (no direct SQLite mutations, all routed through unified queue); progressive single-level directory scanning replacing recursive WalkDir; BM25 IDF weighting with per-collection vocabulary persistence (schema v15: `sparse_vocabulary` + `corpus_statistics` tables); adaptive resource management with Active Processing mode (+50% resources when queue has work during user activity); collection-level uplift and reset cascade handlers; website progressive crawl with link extraction; heartbeat logging for adaptive resources
- v1.8.0: Branch-scoped point IDs with hybrid vector copy approach — new formula `SHA256(tenant_id | branch | file_path | chunk_index)`; added `is_archived` column to `watch_folders` with archive semantics (no watching/ingesting, fully searchable, no search exclusion); documented submodule archive safety with cross-reference checks; added cascade rename mechanism (queue-mediated `tenant_id` changes via SQLite-first + Qdrant eventual consistency); added remote rename detection (periodic git remote check vs stored URL); memory tool default scope changed from `global` to `project`; converted all Python code examples to Rust/TypeScript; added automated affinity grouping section (embedding-based pipeline, registry-sourced taxonomy, zero-shot classification, phased implementation plan)
- v1.7.0: Added comprehensive file type allowlist as primary ingestion gate (400+ extensions across 21 categories, 30+ extension-less filenames, size-restricted extensions, mandatory excluded directories); updated MCP registration policy — MCP server no longer auto-registers new projects, only re-activates existing entries (`register_if_new` field added to `RegisterProject` gRPC); expanded daemon startup automation from simple recovery to 6-step sequence (schema check, config reconciliation with fingerprinting, Qdrant collection verification, path validation, filesystem recovery, crash recovery); updated configuration reference — dropped legacy `.wq_config.yaml` name, documented embedded defaults via `include_str!()`, added complete `watching` section with allowlist/exclusion/size-restriction keys; added Qdrant dashboard visualization guide for named vectors; documented Distance Matrix API for graph visualization
- v1.6.7: Added comprehensive Deployment and Installation section documenting deployment architecture, platform support matrix (6 platforms), installation methods (binary, source, npm), Docker deployment modes, service management (macOS/Linux), CI/CD process, upgrade/migration procedures, and troubleshooting; renamed memory tool ruleId to label with LLM generation guidelines (max 15 chars, word-word-word format); added memory_limits config section; updated Docker compose files to reflect TypeScript/Rust architecture
- v1.6.6: Corrected session lifecycle documentation - clarified that Claude Code's SessionStart/SessionEnd are external hooks (shell commands configured in settings.json), not SDK callbacks; removed incorrect @anthropic-ai/claude-agent-sdk dependency (not needed); documented actual MCP SDK callbacks (server.onclose, onsessioninitialized for HTTP transport); clarified memory injection is via memory tool, not automatic session hooks
- v1.6.5: Updated all references from legacy `registered_projects` table to consolidated `watch_folders` table (priority calculation, activity tracking, batch processing); Python codebase removed in preparation for TypeScript MCP server
- v1.6.4: Updated unified_queue schema to match robust implementation - added status column for lifecycle tracking, lease_until/worker_id for crash recovery, idempotency_key for deduplication (supports content items without file paths), priority column for item type prioritization; documented idempotency key calculation and crash recovery procedure
- v1.6.3: Corrected SDK references from non-existent @anthropic/claude-code-sdk to actual packages (@modelcontextprotocol/sdk and @anthropic-ai/claude-agent-sdk); added SDK Architecture section explaining dual-SDK pattern
- v1.6.2: Consolidated database tables - merged `registered_projects`, `project_submodules`, and separate `watch_folders` tables into single unified `watch_folders` table; added activity inheritance for subprojects (parent and all submodules share `is_active` and `last_activity_at`); removed `project_aliases` table (dead code)
- v1.6.1: Clarified PATH configuration (expansion, merge, deduplication steps); documented session lifecycle with memory injection via Claude SDK; added Grammar and Runtime Management section (dynamic grammar loading, CLI update command, CI automation for 6 platforms)
- v1.6: MCP server rewrite decision - TypeScript instead of Python for native SessionStart/SessionEnd hook support; added TypeScript dependencies and rationale; updated architecture diagram
- v1.5: Major queue and daemon architecture update - simplified queue schema (removed status states, added failed flag and errors array); defined batch processing flow with sort alternation for anti-starvation; documented three daemon phases (initial scan, watching, removal); added daemon watch management lifecycle; defined semantic code chunking strategy; added Tree-sitter baseline + LSP enhancement architecture; defined LSP lifecycle and language server management; added PATH configuration for daemon
- v1.4: Clarified 4 MCP tools only (search, retrieve, memory, store); removed health/session as tools (health is server-internal affecting search responses with uncertainty status, session is automated); clarified memory collection is multi-tenant via nullable project_id; added detailed "table not found" graceful handling documentation; clarified "actively used" means RegisterProject received from MCP server; documented memory list action as search in disguise
- v1.3: Major API redesign - replaced `manage` tool with dedicated tools (`memory`, `health`, `session`); clarified MCP does NOT store to `projects` collection (daemon handles via file watching); added single configuration file requirement with cascade search; updated queue schema to include `memory` item_type; documented libraries as reference documentation (not programming libraries)
- v1.2: Updated API Reference to match actual implementation; clarified manage actions (removed create/delete collection as MCP actions); added pattern configuration documentation; updated gRPC services table format
- v1.1: Added comprehensive Project ID specification with duplicate handling, branch lifecycle, and registered_projects schema
