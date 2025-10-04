# Rule Fetching Mechanism Design

**Version:** 0.2.1dev1
**Date:** 2025-10-04
**Task:** 295.2 - Design rule fetching mechanism from memory collection

## Table of Contents

1. [Overview](#overview)
2. [Memory Collection Schema](#memory-collection-schema)
3. [Daemon API Interface](#daemon-api-interface)
4. [Rule Categorization and Tagging](#rule-categorization-and-tagging)
5. [Query Mechanisms](#query-mechanisms)
6. [Caching Strategies](#caching-strategies)
7. [Rule Versioning and Updates](#rule-versioning-and-updates)
8. [Implementation Plan](#implementation-plan)

## Overview

The rule fetching mechanism provides the interface between the memory collection (Qdrant) and the context injection system. It retrieves context rules via the daemon, applies filtering based on scope and project context, and delivers formatted rules for LLM injection.

### Design Goals

1. **Efficient Retrieval**: Minimize latency for rule fetching (<100ms for typical queries)
2. **Smart Filtering**: Retrieve only relevant rules based on scope/project context
3. **Caching**: Reduce Qdrant queries through intelligent caching
4. **Version Control**: Handle rule updates and maintain consistency
5. **Daemon Integration**: Leverage existing daemon infrastructure

### Architecture Position

```
Context Injector → Rule Selector → Daemon Client → gRPC → Rust Daemon → Qdrant
                       ↑
                   Rule Cache
```

## Memory Collection Schema

### Collection Structure

**Collection Name:** `_memory` (reserved system collection)

**Vector Configuration:**
```python
{
    "dense": VectorParams(size=384, distance=Distance.COSINE),  # Semantic search
    "sparse": VectorParams(size=<dynamic>, distance=Distance.DOT)  # Keyword search
}
```

### Payload Schema

Each memory rule point in Qdrant contains the following payload:

```python
{
    # Core fields (from MemoryRule dataclass)
    "id": str,              # Unique rule identifier
    "category": str,        # "preference" | "behavior" | "agent"
    "name": str,           # Short descriptive name
    "rule": str,           # The actual rule text/instruction
    "authority": str,      # "absolute" | "default"
    "scope": List[str],    # Contexts where rule applies (empty = global)
    "source": str,         # How rule was created

    # Optional fields
    "conditions": Dict[str, Any],  # Conditional logic for rule application
    "replaces": List[str],         # IDs of rules this supersedes
    "created_at": str,             # ISO 8601 timestamp
    "updated_at": str,             # ISO 8601 timestamp
    "metadata": Dict[str, Any],    # Additional metadata

    # Enhanced fields for context injection
    "project_id": Optional[str],   # Project this rule applies to
    "tool_targets": List[str],     # Target LLM tools ["claude", "codex", "gemini"]
    "priority": int,               # Priority within authority level (0-100)
    "token_cost": int,             # Estimated token cost for this rule
    "usage_count": int,            # How many times this rule has been used
    "last_used": Optional[str],    # ISO 8601 timestamp of last usage
    "tags": List[str],             # Additional tags for categorization
}
```

### Indexing Strategy

**Payload Indexes:**
```python
# Create indexes for efficient filtering
collection.create_payload_index("category", field_schema="keyword")
collection.create_payload_index("authority", field_schema="keyword")
collection.create_payload_index("scope", field_schema="keyword")
collection.create_payload_index("project_id", field_schema="keyword")
collection.create_payload_index("tool_targets", field_schema="keyword")
collection.create_payload_index("tags", field_schema="keyword")
collection.create_payload_index("priority", field_schema="integer")
```

## Daemon API Interface

### gRPC Service Definition

**New RPC Methods in IngestService:**

```protobuf
service IngestService {
    // Existing methods...

    // Memory rule retrieval
    rpc GetMemoryRules(GetMemoryRulesRequest) returns (MemoryRulesResponse);
    rpc SearchMemoryRules(SearchMemoryRulesRequest) returns (MemoryRulesResponse);
    rpc GetMemoryRuleById(GetMemoryRuleByIdRequest) returns (MemoryRule);
    rpc UpdateRuleUsage(UpdateRuleUsageRequest) returns (UpdateRuleUsageResponse);
}

// Request/Response messages
message GetMemoryRulesRequest {
    repeated string scope = 1;           // Filter by scope (empty = all scopes)
    optional string project_id = 2;      // Filter by project
    optional string category = 3;        // Filter by category
    optional string authority = 4;       // Filter by authority level
    repeated string tool_targets = 5;    // Filter by target tools
    repeated string tags = 6;            // Filter by tags
    int32 limit = 7;                    // Maximum results (default: 100)
}

message SearchMemoryRulesRequest {
    string query = 1;                   // Semantic search query
    repeated string scope = 2;          // Scope filter
    optional string project_id = 3;     // Project filter
    optional string category = 4;       // Category filter
    string search_mode = 5;             // "hybrid" | "dense" | "sparse"
    int32 limit = 6;                    // Maximum results (default: 10)
    float score_threshold = 7;          // Minimum relevance score (default: 0.5)
}

message MemoryRulesResponse {
    repeated MemoryRule rules = 1;
    int32 total_count = 2;
    optional string next_cursor = 3;    // For pagination
}

message MemoryRule {
    string id = 1;
    string category = 2;
    string name = 3;
    string rule = 4;
    string authority = 5;
    repeated string scope = 6;
    string source = 7;
    optional string conditions = 8;     // JSON string
    repeated string replaces = 9;
    string created_at = 10;
    string updated_at = 11;
    optional string metadata = 12;      // JSON string
    optional string project_id = 13;
    repeated string tool_targets = 14;
    int32 priority = 15;
    int32 token_cost = 16;
    int32 usage_count = 17;
    optional string last_used = 18;
    repeated string tags = 19;
    float relevance_score = 20;         // For search results
}

message GetMemoryRuleByIdRequest {
    string rule_id = 1;
}

message UpdateRuleUsageRequest {
    repeated string rule_ids = 1;       // Rules that were used
    string session_id = 2;              // Session identifier
    string tool = 3;                    // Which tool used the rules
}

message UpdateRuleUsageResponse {
    int32 updated_count = 1;
    bool success = 2;
}
```

### Python Daemon Client Extension

**Extend `AsyncIngestClient` in `src/python/common/grpc/client.py`:**

```python
class AsyncIngestClient:
    # ... existing methods ...

    async def get_memory_rules(
        self,
        scope: Optional[List[str]] = None,
        project_id: Optional[str] = None,
        category: Optional[str] = None,
        authority: Optional[str] = None,
        tool_targets: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100,
        timeout: float = 10.0,
    ) -> List[MemoryRule]:
        """
        Retrieve memory rules from the daemon with filtering.

        Args:
            scope: Filter by scope (empty list = all scopes)
            project_id: Filter by project
            category: Filter by category ("preference", "behavior", "agent")
            authority: Filter by authority ("absolute", "default")
            tool_targets: Filter by target tools
            tags: Filter by tags
            limit: Maximum number of rules to return
            timeout: Request timeout in seconds

        Returns:
            List of MemoryRule objects matching the filters
        """
        if not self._started:
            await self.start()

        request = GetMemoryRulesRequest(
            scope=scope or [],
            project_id=project_id,
            category=category,
            authority=authority,
            tool_targets=tool_targets or [],
            tags=tags or [],
            limit=limit,
        )

        async def _get_rules(stub: IngestServiceStub):
            pb_response = await asyncio.wait_for(
                stub.GetMemoryRules(request), timeout=timeout
            )
            return [MemoryRule.from_pb(rule) for rule in pb_response.rules]

        return await self.connection_manager.with_retry(_get_rules)

    async def search_memory_rules(
        self,
        query: str,
        scope: Optional[List[str]] = None,
        project_id: Optional[str] = None,
        category: Optional[str] = None,
        search_mode: str = "hybrid",
        limit: int = 10,
        score_threshold: float = 0.5,
        timeout: float = 10.0,
    ) -> List[Tuple[MemoryRule, float]]:
        """
        Search memory rules by semantic similarity.

        Args:
            query: Search query for semantic search
            scope: Filter by scope
            project_id: Filter by project
            category: Filter by category
            search_mode: "hybrid", "dense", or "sparse"
            limit: Maximum results
            score_threshold: Minimum relevance score
            timeout: Request timeout

        Returns:
            List of (MemoryRule, relevance_score) tuples
        """
        if not self._started:
            await self.start()

        request = SearchMemoryRulesRequest(
            query=query,
            scope=scope or [],
            project_id=project_id,
            category=category,
            search_mode=search_mode,
            limit=limit,
            score_threshold=score_threshold,
        )

        async def _search_rules(stub: IngestServiceStub):
            pb_response = await asyncio.wait_for(
                stub.SearchMemoryRules(request), timeout=timeout
            )
            return [
                (MemoryRule.from_pb(rule), rule.relevance_score)
                for rule in pb_response.rules
            ]

        return await self.connection_manager.with_retry(_search_rules)

    async def get_memory_rule_by_id(
        self,
        rule_id: str,
        timeout: float = 5.0,
    ) -> Optional[MemoryRule]:
        """
        Retrieve a specific memory rule by ID.

        Args:
            rule_id: The rule ID to retrieve
            timeout: Request timeout

        Returns:
            MemoryRule if found, None otherwise
        """
        if not self._started:
            await self.start()

        request = GetMemoryRuleByIdRequest(rule_id=rule_id)

        async def _get_rule(stub: IngestServiceStub):
            try:
                pb_response = await asyncio.wait_for(
                    stub.GetMemoryRuleById(request), timeout=timeout
                )
                return MemoryRule.from_pb(pb_response)
            except grpc.RpcError as e:
                if e.code() == grpc.StatusCode.NOT_FOUND:
                    return None
                raise

        return await self.connection_manager.with_retry(_get_rule)

    async def update_rule_usage(
        self,
        rule_ids: List[str],
        session_id: str,
        tool: str,
        timeout: float = 5.0,
    ) -> bool:
        """
        Update usage statistics for rules that were used in a session.

        Args:
            rule_ids: List of rule IDs that were used
            session_id: Session identifier
            tool: Tool that used the rules ("claude", "codex", "gemini")
            timeout: Request timeout

        Returns:
            True if update successful
        """
        if not self._started:
            await self.start()

        request = UpdateRuleUsageRequest(
            rule_ids=rule_ids,
            session_id=session_id,
            tool=tool,
        )

        async def _update_usage(stub: IngestServiceStub):
            pb_response = await asyncio.wait_for(
                stub.UpdateRuleUsage(request), timeout=timeout
            )
            return pb_response.success

        return await self.connection_manager.with_retry(_update_usage)
```

## Rule Categorization and Tagging

### Rule Categories

**MemoryCategory Enum** (existing):
- `PREFERENCE`: User preferences ("Use uv for Python")
- `BEHAVIOR`: Behavioral instructions ("Always make atomic commits")
- `AGENT`: Agent library definitions

### Additional Categorization Dimensions

**Tags System:**
```python
# Predefined tag categories
TAG_CATEGORIES = {
    "domain": ["backend", "frontend", "devops", "testing", "documentation"],
    "language": ["python", "rust", "javascript", "typescript", "sql"],
    "tool": ["git", "docker", "pytest", "uv", "npm"],
    "workflow": ["commit", "review", "deploy", "test", "refactor"],
    "priority": ["critical", "important", "optional"],
}

# Example rule with tags
rule = MemoryRule(
    id="rule_12345",
    category=MemoryCategory.BEHAVIOR,
    name="atomic_commits",
    rule="Always make atomic commits with clear messages",
    authority=AuthorityLevel.ABSOLUTE,
    scope=["git"],
    tags=["workflow:commit", "tool:git", "priority:critical"],
)
```

### Scope Matching Semantics

**Scope Interpretation:**
- **Empty scope (`[]`)**: Global rule, applies to all contexts
- **Non-empty scope**: Rule applies only when one of the scope values matches current context

**Matching Logic:**
```python
def scope_matches(rule_scope: List[str], current_context: List[str]) -> bool:
    """
    Check if rule scope matches current context.

    Args:
        rule_scope: Scope list from rule (empty = global)
        current_context: Current context scope values

    Returns:
        True if rule applies to current context
    """
    # Global rules always match
    if not rule_scope:
        return True

    # Check for scope intersection
    return bool(set(rule_scope) & set(current_context))
```

## Query Mechanisms

### Query Types

#### 1. Filter-Based Retrieval

**Use Case:** Retrieve all rules for a specific project/scope/tool

```python
# Example: Get all absolute rules for Claude Code in current project
rules = await daemon_client.get_memory_rules(
    project_id=current_project,
    tool_targets=["claude"],
    authority="absolute",
    limit=50,
)
```

**Daemon Implementation:**
```python
# Qdrant filter construction
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

filter = Filter(
    must=[
        FieldCondition(key="project_id", match=MatchValue(value=project_id)),
        FieldCondition(key="tool_targets", match=MatchAny(any=["claude"])),
        FieldCondition(key="authority", match=MatchValue(value="absolute")),
    ]
)

# Scroll through results
points, _ = qdrant_client.scroll(
    collection_name="_memory",
    scroll_filter=filter,
    limit=limit,
    with_payload=True,
    with_vectors=False,  # Don't need vectors for filter-based retrieval
)
```

#### 2. Semantic Search

**Use Case:** Find relevant rules for a specific query/task

```python
# Example: Find rules relevant to "testing Python code"
rules = await daemon_client.search_memory_rules(
    query="testing Python code with pytest",
    project_id=current_project,
    search_mode="hybrid",
    limit=10,
    score_threshold=0.6,
)
```

**Daemon Implementation:**
```python
# Generate query embeddings (dense + sparse)
dense_vector = embedding_service.embed(query)
sparse_vector = bm25_encoder.encode(query)

# Hybrid search
search_result = qdrant_client.search(
    collection_name="_memory",
    query_vector=("dense", dense_vector),
    query_filter=filter,  # Apply scope/project filters
    limit=limit,
    score_threshold=score_threshold,
    using="hybrid",  # Combine dense + sparse
)
```

#### 3. Composite Queries

**Use Case:** Complex multi-criteria retrieval

```python
# Get all absolute rules + top 5 relevant default rules
absolute_rules = await daemon_client.get_memory_rules(
    project_id=project_id,
    authority="absolute",
    tool_targets=["claude"],
)

default_rules = await daemon_client.search_memory_rules(
    query=task_context,
    project_id=project_id,
    authority="default",
    limit=5,
    score_threshold=0.7,
)

all_rules = absolute_rules + [rule for rule, score in default_rules]
```

### Query Optimization Strategies

1. **Index-First Filtering**: Always apply filters before semantic search
2. **Limit Vectors**: Don't return vectors unless needed
3. **Batch Requests**: Combine multiple queries when possible
4. **Pagination**: Use cursor-based pagination for large result sets

## Caching Strategies

### Multi-Level Cache Architecture

```
┌─────────────────────────────────────────┐
│  L1: In-Memory Process Cache            │
│  (TTL: 5 minutes, Size: 1000 rules)     │
└─────────────┬───────────────────────────┘
              │
┌─────────────▼───────────────────────────┐
│  L2: Redis Cache (Optional)             │
│  (TTL: 30 minutes, Size: 10000 rules)   │
└─────────────┬───────────────────────────┘
              │
┌─────────────▼───────────────────────────┐
│  L3: Daemon/Qdrant                      │
│  (Persistent storage)                   │
└─────────────────────────────────────────┘
```

### L1: In-Memory Cache

**Implementation:**

```python
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import threading

class RuleCache:
    """
    Thread-safe in-memory cache for memory rules.
    """

    def __init__(
        self,
        ttl_seconds: int = 300,  # 5 minutes
        max_size: int = 1000,
    ):
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self._cache: Dict[str, Tuple[List[MemoryRule], datetime]] = {}
        self._lock = threading.RLock()

    def get(self, cache_key: str) -> Optional[List[MemoryRule]]:
        """Get cached rules if not expired."""
        with self._lock:
            if cache_key not in self._cache:
                return None

            rules, cached_at = self._cache[cache_key]

            # Check TTL
            if datetime.now() - cached_at > timedelta(seconds=self.ttl_seconds):
                del self._cache[cache_key]
                return None

            return rules

    def set(self, cache_key: str, rules: List[MemoryRule]):
        """Cache rules with current timestamp."""
        with self._lock:
            # Evict oldest if at capacity
            if len(self._cache) >= self.max_size:
                oldest_key = min(self._cache.items(), key=lambda x: x[1][1])[0]
                del self._cache[oldest_key]

            self._cache[cache_key] = (rules, datetime.now())

    def invalidate(self, cache_key: Optional[str] = None):
        """Invalidate specific key or entire cache."""
        with self._lock:
            if cache_key:
                self._cache.pop(cache_key, None)
            else:
                self._cache.clear()

    def generate_cache_key(
        self,
        project_id: Optional[str] = None,
        scope: Optional[List[str]] = None,
        category: Optional[str] = None,
        authority: Optional[str] = None,
        tool_targets: Optional[List[str]] = None,
    ) -> str:
        """Generate cache key from query parameters."""
        parts = [
            f"project:{project_id or 'none'}",
            f"scope:{','.join(sorted(scope or []))}",
            f"category:{category or 'none'}",
            f"authority:{authority or 'none'}",
            f"tools:{','.join(sorted(tool_targets or []))}",
        ]
        return "|".join(parts)
```

### Cache Invalidation Strategy

**Invalidation Triggers:**

1. **Rule Updates**: Invalidate cache entries affected by rule changes
2. **Time-Based**: TTL expiration (5 minutes for in-memory)
3. **Project Changes**: Invalidate when project context changes
4. **Manual**: Explicit invalidation via API

**Implementation:**

```python
class RuleFetcherWithCache:
    """
    Rule fetcher with intelligent caching.
    """

    def __init__(
        self,
        daemon_client: AsyncIngestClient,
        cache: RuleCache,
    ):
        self.daemon = daemon_client
        self.cache = cache

    async def get_rules(
        self,
        project_id: Optional[str] = None,
        scope: Optional[List[str]] = None,
        category: Optional[str] = None,
        authority: Optional[str] = None,
        tool_targets: Optional[List[str]] = None,
        bypass_cache: bool = False,
    ) -> List[MemoryRule]:
        """
        Get rules with caching.

        Args:
            project_id: Filter by project
            scope: Filter by scope
            category: Filter by category
            authority: Filter by authority
            tool_targets: Filter by target tools
            bypass_cache: Skip cache lookup

        Returns:
            List of memory rules
        """
        # Generate cache key
        cache_key = self.cache.generate_cache_key(
            project_id=project_id,
            scope=scope,
            category=category,
            authority=authority,
            tool_targets=tool_targets,
        )

        # Check cache first
        if not bypass_cache:
            cached_rules = self.cache.get(cache_key)
            if cached_rules is not None:
                logger.debug(f"Cache hit for key: {cache_key}")
                return cached_rules

        # Cache miss - fetch from daemon
        logger.debug(f"Cache miss for key: {cache_key}")
        rules = await self.daemon.get_memory_rules(
            project_id=project_id,
            scope=scope,
            category=category,
            authority=authority,
            tool_targets=tool_targets,
        )

        # Update cache
        self.cache.set(cache_key, rules)

        return rules
```

### Cache Warmup Strategy

**Pre-populate cache on session start:**

```python
async def warmup_cache(
    rule_fetcher: RuleFetcherWithCache,
    project_id: str,
    tool: str,
):
    """
    Pre-populate cache with commonly used rules.
    """
    # Fetch absolute rules for this project
    await rule_fetcher.get_rules(
        project_id=project_id,
        authority="absolute",
        tool_targets=[tool],
    )

    # Fetch common categories
    for category in ["preference", "behavior"]:
        await rule_fetcher.get_rules(
            project_id=project_id,
            category=category,
            tool_targets=[tool],
        )
```

## Rule Versioning and Updates

### Version Tracking

**Rule Metadata for Versioning:**

```python
{
    "id": "rule_12345",
    "version": 2,  # Version number
    "replaces": ["rule_11111"],  # Previous version ID
    "metadata": {
        "version_history": [
            {
                "version": 1,
                "rule_id": "rule_11111",
                "created_at": "2025-01-01T00:00:00Z",
                "reason": "Initial version",
            },
            {
                "version": 2,
                "rule_id": "rule_12345",
                "created_at": "2025-10-04T20:00:00Z",
                "reason": "Updated to reflect new workflow",
            },
        ]
    }
}
```

### Update Handling

**Scenario 1: Rule Content Update**

1. Create new rule with incremented version
2. Set `replaces` field to point to old rule
3. Old rule remains in collection for historical queries
4. Cache invalidation for affected queries

**Scenario 2: Authority Level Change**

1. Update rule in-place (same ID)
2. Increment `updated_at` timestamp
3. Invalidate all caches containing this rule

**Scenario 3: Scope Expansion**

1. Update scope field in-place
2. Invalidate caches for old and new scopes

### Consistency Guarantees

**Read Consistency:**
- Cache TTL ensures rules don't get too stale
- Bypass cache option for critical operations

**Write Consistency:**
- Updates go through daemon (ACID guarantees)
- Cache invalidation happens synchronously after write

## Implementation Plan

### Phase 1: gRPC Protocol Extension

**Files to Modify:**
1. `src/rust/daemon/grpc/proto/ingestion.proto` - Add new RPC methods
2. `src/python/common/grpc/ingestion_pb2.py` - Regenerate from proto
3. `src/python/common/grpc/ingestion_pb2_grpc.py` - Regenerate from proto

**Tasks:**
- [x] Define gRPC service methods (GetMemoryRules, SearchMemoryRules, etc.)
- [ ] Add request/response message types
- [ ] Regenerate Python stubs from proto

### Phase 2: Daemon Implementation

**Files to Create/Modify:**
1. `src/rust/daemon/core/src/memory/` - New module for memory operations
2. `src/rust/daemon/core/src/memory/retrieval.rs` - Rule retrieval logic
3. `src/rust/daemon/core/src/memory/cache.rs` - Rust-side caching
4. `src/rust/daemon/grpc/src/service_impl.rs` - Implement gRPC methods

**Tasks:**
- [ ] Implement GetMemoryRules RPC handler
- [ ] Implement SearchMemoryRules with hybrid search
- [ ] Implement GetMemoryRuleById RPC handler
- [ ] Implement UpdateRuleUsage RPC handler
- [ ] Add Qdrant filter construction utilities
- [ ] Add error handling and logging

### Phase 3: Python Client Extension

**Files to Modify:**
1. `src/python/common/grpc/client.py` - Extend AsyncIngestClient
2. `src/python/common/grpc/types.py` - Add MemoryRule type wrappers

**Tasks:**
- [ ] Add `get_memory_rules()` method to AsyncIngestClient
- [ ] Add `search_memory_rules()` method
- [ ] Add `get_memory_rule_by_id()` method
- [ ] Add `update_rule_usage()` method
- [ ] Add MemoryRule type wrapper with from_pb/to_pb methods

### Phase 4: Caching Implementation

**Files to Create:**
1. `src/python/common/core/context_injection/cache.py` - RuleCache class
2. `src/python/common/core/context_injection/fetcher.py` - RuleFetcherWithCache

**Tasks:**
- [ ] Implement RuleCache with TTL and eviction
- [ ] Implement RuleFetcherWithCache wrapper
- [ ] Add cache warmup utilities
- [ ] Add cache invalidation hooks

### Phase 5: Integration and Testing

**Files to Create:**
1. `tests/unit/test_rule_fetching.py` - Unit tests
2. `tests/integration/test_daemon_memory_api.py` - Integration tests

**Tasks:**
- [ ] Unit tests for cache logic
- [ ] Integration tests for daemon API
- [ ] End-to-end tests for rule retrieval
- [ ] Performance benchmarks (<100ms retrieval time)
- [ ] Cache hit rate validation (target: >80%)

## References

- **llm-context-injection.md**: Overall architecture
- **src/python/common/core/memory.py**: Existing MemoryRule schema
- **src/python/common/grpc/client.py**: Existing daemon client
- **PRDv3.txt**: System specification
- **Task 294**: Memory rule data structures (dependency)
