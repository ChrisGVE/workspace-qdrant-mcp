## Keyword and Tag Extraction

The daemon automatically extracts keywords and concept tags from ingested documents using a two-stage pipeline: lexical candidate extraction followed by semantic reranking with diversity enforcement.

### Extraction Pipeline

**Stage 1 — Lexical Candidate Extraction:**
1. TF-IDF scoring across document chunks to identify distinctive terms
2. N-gram extraction (1-3 tokens) for multi-word phrases
3. Stop-word filtering and minimum frequency thresholds
4. Produces ~50 keyword candidates per document

**Stage 2 — Semantic Rerank with MMR Diversity:**
1. Embed all keyword candidates using FastEmbed
2. Rank by combined score: `final = α × lexical_score + (1-α) × semantic_score`
3. Apply Maximal Marginal Relevance (MMR) to enforce diversity — penalizes candidates too similar to already-selected keywords
4. Select top-N keywords (configurable per collection)

### Keywords vs Tags

| Concept | Description | Storage |
|---------|-------------|---------|
| **Keywords** | Raw extracted terms/phrases from the document | `keywords` table + Qdrant `keywords` payload field |
| **Concept tags** | Semantic topic labels (e.g., "async-runtime", "error-handling") | `tags` table + Qdrant `concept_tags` payload field |
| **Structural tags** | Metadata-derived labels (e.g., "language:rust", "framework:tokio") | `tags` table + Qdrant `structural_tags` payload field |
| **Keyword baskets** | Mapping from each tag to its associated keywords | `keyword_baskets` table + Qdrant `keyword_baskets` payload field |

### Canonical Tag Hierarchy

Tags are deduplicated across documents into canonical tags per collection. A nightly batch job performs agglomerative clustering on canonical tag embeddings to build a topic hierarchy:

- **Level 1**: Broad domains (e.g., "systems-programming", "web-development")
- **Level 2**: Sub-domains (e.g., "async-runtime", "http-server")
- **Level 3**: Specific topics (e.g., "tokio-executor", "hyper-routing")

Hierarchy is stored in `canonical_tags` (nodes) and `tag_hierarchy_edges` (parent-child relationships). Rebuild is triggered nightly or via `wqm tags rebuild-hierarchy`.

### Search Query Expansion

When `enableTagExpansion` is true (default), keyword/hybrid search queries are automatically expanded:

1. Query tokens are matched against the `tags` table in SQLite
2. Matching tags' keyword baskets are retrieved
3. Basket keywords are used to generate an expansion sparse vector via the daemon
4. Expansion vector is merged into the original sparse vector at reduced weight (default 0.5×)
5. Overlapping indices preserve the original (higher) weight

This improves recall for queries that use different terminology than the indexed documents.

### Tag-Based Search Filtering

The `search` MCP tool supports filtering by concept tags:

- `tag: "async-runtime"` — exact match on a single concept tag
- `tags: ["async-runtime", "error-handling"]` — OR logic across multiple tags

Tag filters apply to the `concept_tags` Qdrant payload field. Combined with keyword/semantic search for precise topic-scoped retrieval.

### Per-Collection Configuration

Extraction parameters are tunable per collection via `collection_config.rs`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_keywords` | 20 | Maximum keywords extracted per document |
| `max_tags` | 10 | Maximum concept tags per document |
| `min_keyword_score` | 0.1 | Minimum combined score threshold |
| `mmr_lambda` | 0.7 | MMR diversity parameter (0=max diversity, 1=max relevance) |
| `alpha` | 0.6 | Lexical vs semantic weight balance |

### CLI Commands

```bash
wqm tags list [--collection <name>]           # List tags with document counts
wqm tags show <tag>                            # Show tag details and keyword basket
wqm tags rebuild-hierarchy [--collection <name>]  # Trigger hierarchy rebuild
wqm tags stats [--collection <name>]           # Tag extraction statistics
```

### SQLite Schema (v16)

**keywords** — Per-document keyword records:
```sql
CREATE TABLE keywords (
    keyword_id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT NOT NULL,
    keyword TEXT NOT NULL,
    score REAL NOT NULL DEFAULT 0.0,
    semantic_score REAL NOT NULL DEFAULT 0.0,
    lexical_score REAL NOT NULL DEFAULT 0.0,
    stability_count INTEGER NOT NULL DEFAULT 0,
    collection TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    created_at TEXT NOT NULL
);
```

**tags** — Per-document tag records:
```sql
CREATE TABLE tags (
    tag_id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT NOT NULL,
    tag TEXT NOT NULL,
    tag_type TEXT NOT NULL DEFAULT 'concept' CHECK (tag_type IN ('concept', 'structural')),
    score REAL NOT NULL DEFAULT 0.0,
    diversity_score REAL NOT NULL DEFAULT 0.0,
    basket_id INTEGER,
    collection TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    created_at TEXT NOT NULL
);
```

**keyword_baskets** — Tag-to-keywords mapping:
```sql
CREATE TABLE keyword_baskets (
    basket_id INTEGER PRIMARY KEY AUTOINCREMENT,
    tag_id INTEGER NOT NULL REFERENCES tags(tag_id) ON DELETE CASCADE,
    keywords_json TEXT NOT NULL DEFAULT '[]',
    tenant_id TEXT NOT NULL,
    created_at TEXT NOT NULL
);
```

**canonical_tags** — Deduplicated cross-document tag nodes:
```sql
CREATE TABLE canonical_tags (
    canonical_id INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_name TEXT NOT NULL,
    centroid_vector_id TEXT,
    level INTEGER NOT NULL DEFAULT 3,
    parent_id INTEGER REFERENCES canonical_tags(canonical_id) ON DELETE SET NULL,
    tenant_id TEXT NOT NULL,
    collection TEXT NOT NULL,
    created_at TEXT NOT NULL
);
```

**tag_hierarchy_edges** — Parent-child relationships:
```sql
CREATE TABLE tag_hierarchy_edges (
    edge_id INTEGER PRIMARY KEY AUTOINCREMENT,
    parent_tag_id INTEGER NOT NULL REFERENCES canonical_tags(canonical_id) ON DELETE CASCADE,
    child_tag_id INTEGER NOT NULL REFERENCES canonical_tags(canonical_id) ON DELETE CASCADE,
    similarity_score REAL NOT NULL DEFAULT 0.0,
    tenant_id TEXT NOT NULL,
    UNIQUE(parent_tag_id, child_tag_id)
);
```

---

