# LLM Context Injection System Architecture

**Version:** 0.2.1dev1
**Date:** 2025-10-04
**Status:** Design Phase

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Principles](#architecture-principles)
3. [Component Architecture](#component-architecture)
4. [Data Flow](#data-flow)
5. [Integration Points](#integration-points)
6. [Design Patterns](#design-patterns)
7. [System Boundaries](#system-boundaries)

## System Overview

The LLM Context Injection System provides automatic, intelligent context injection into LLM tool sessions (Claude Code, GitHub Codex, Google Gemini) using memory rules stored in Qdrant. The system enables consistent, project-aware LLM behavior through pre-session rule injection.

### Key Capabilities

- **Automatic Rule Injection**: Pre-session context injection from memory collection
- **Multi-Tool Support**: Adapters for Claude Code, GitHub Codex, Google Gemini
- **Intelligent Formatting**: LLM-specific prompt formatting and token management
- **Rule Management**: Hierarchical rule system with authority levels and scope matching
- **Configuration-Driven**: YAML-based configuration for tool selection and behavior
- **Token Budget Management**: Intelligent rule prioritization within token constraints

### System Goals

1. Provide consistent LLM behavior across sessions and tools
2. Enable project-specific context injection without manual prompting
3. Support multiple LLM tools with tool-specific formatting
4. Manage token budgets intelligently with rule prioritization
5. Allow extensible architecture for future LLM tools

## Architecture Principles

### 1. Separation of Concerns

- **Memory Layer**: Rule storage and retrieval (Qdrant + Daemon)
- **Context Layer**: Rule selection and formatting (Context Injector)
- **Integration Layer**: LLM tool hooks and triggers (Tool Adapters)

### 2. Configuration-Driven Behavior

- All tool selection, formatting, and injection behavior driven by YAML configuration
- Runtime configuration updates without code changes
- Environment variable overrides for deployment flexibility

### 3. Extensible Multi-Tool Support

- Plugin architecture for new LLM tools
- Tool-specific adapters implementing common interface
- Capability detection and negotiation per tool

### 4. Token Budget Optimization

- Rule prioritization based on authority level and scope
- Dynamic token allocation with compression strategies
- Absolute rules always preserved regardless of budget

### 5. Fail-Safe Operation

- Graceful degradation when daemon unavailable
- Rollback mechanisms for failed injections
- Comprehensive error handling and logging

## Component Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     LLM Context Injection System                 │
└─────────────────────────────────────────────────────────────────┘
                                 │
                ┌────────────────┼────────────────┐
                │                │                │
         ┌──────▼──────┐  ┌─────▼─────┐  ┌──────▼──────┐
         │   Memory    │  │  Context  │  │ Integration │
         │   Layer     │  │   Layer   │  │    Layer    │
         └─────────────┘  └───────────┘  └─────────────┘
```

### Memory Layer

**Components:**
- **Memory Collection** (`_memory`): Qdrant collection storing context rules
- **Agent Memory Collection** (`_agent_memory`): Agent-specific rule storage
- **Daemon Client**: gRPC interface to Rust daemon for rule retrieval
- **Rule Schema**: MemoryRule dataclass with metadata

**Responsibilities:**
- Store context rules with metadata (category, authority, scope)
- Provide query interface for rule retrieval by scope/project
- Handle rule versioning and updates
- Maintain rule consistency and validation

**Key Interfaces:**
```python
class MemoryRuleRetrieval:
    async def get_rules_by_scope(
        self,
        scope: List[str],
        project_id: Optional[str] = None
    ) -> List[MemoryRule]

    async def get_rules_by_category(
        self,
        category: MemoryCategory
    ) -> List[MemoryRule]

    async def get_absolute_rules(
        self,
        scope: List[str]
    ) -> List[MemoryRule]
```

### Context Layer

**Components:**
- **Context Injector**: Core component orchestrating rule fetching and formatting
- **Rule Selector**: Filters rules based on scope, project, and relevance
- **Format Manager**: Routes rules to appropriate LLM formatter
- **Token Budget Manager**: Manages token allocation and rule prioritization
- **Rule Cache**: In-memory cache for frequently used rules

**Responsibilities:**
- Fetch rules from memory layer via daemon
- Select relevant rules based on context (project, scope, tool)
- Format rules for target LLM tool
- Manage token budgets and rule truncation
- Cache rules for performance

**Key Interfaces:**
```python
class ContextInjector:
    async def inject_context(
        self,
        tool: LLMTool,
        project_id: str,
        scope: List[str],
        token_budget: int
    ) -> InjectionResult

    async def format_rules(
        self,
        rules: List[MemoryRule],
        tool: LLMTool,
        token_budget: int
    ) -> FormattedContext
```

### Integration Layer

**Components:**
- **Tool Adapters**: LLM-specific integration (Claude, Codex, Gemini)
- **Hook Manager**: Pre-session trigger system
- **Configuration Manager**: YAML configuration loading and validation
- **Plugin Registry**: Dynamic tool adapter registration

**Responsibilities:**
- Integrate with LLM tool workflows
- Trigger pre-session context injection
- Load and validate configuration
- Register and discover tool adapters
- Handle injection failures and rollbacks

**Key Interfaces:**
```python
class LLMToolAdapter(ABC):
    @abstractmethod
    async def inject_context(
        self,
        context: FormattedContext
    ) -> bool

    @abstractmethod
    def get_capabilities(self) -> ToolCapabilities

    @abstractmethod
    def format_for_tool(
        self,
        rules: List[MemoryRule]
    ) -> str
```

## Data Flow

### Rule Injection Flow

```
┌─────────────┐
│  LLM Tool   │ (Claude Code, Codex, Gemini)
│  Session    │
│   Start     │
└──────┬──────┘
       │
       │ 1. Pre-session trigger
       ▼
┌─────────────────┐
│  Hook Manager   │
└────────┬────────┘
         │
         │ 2. Request context injection
         ▼
┌─────────────────────────┐
│   Context Injector      │
│   - Load configuration  │
│   - Determine scope     │
│   - Get token budget    │
└────────┬────────────────┘
         │
         │ 3. Fetch rules by scope/project
         ▼
┌─────────────────────────┐
│   Daemon Client         │
│   - gRPC call to daemon │
│   - Query memory coll.  │
└────────┬────────────────┘
         │
         │ 4. Return MemoryRules
         ▼
┌─────────────────────────┐
│   Rule Selector         │
│   - Filter by relevance │
│   - Apply authority     │
│   - Sort by priority    │
└────────┬────────────────┘
         │
         │ 5. Selected rules
         ▼
┌─────────────────────────┐
│  Token Budget Manager   │
│   - Count tokens        │
│   - Prioritize rules    │
│   - Truncate if needed  │
└────────┬────────────────┘
         │
         │ 6. Budget-compliant rules
         ▼
┌─────────────────────────┐
│   Format Manager        │
│   - Select tool adapter │
│   - Format for LLM      │
└────────┬────────────────┘
         │
         │ 7. Formatted context
         ▼
┌─────────────────────────┐
│   Tool Adapter          │
│   - Tool-specific inject│
│   - Verify injection    │
└────────┬────────────────┘
         │
         │ 8. Injection complete
         ▼
┌─────────────┐
│  LLM Tool   │
│  Session    │
│  (Injected) │
└─────────────┘
```

### Rule Storage Flow

```
┌─────────────┐
│  CLI/MCP    │ (User creates rule)
│   Command   │
└──────┬──────┘
       │
       │ 1. Create MemoryRule
       ▼
┌─────────────────┐
│  Daemon Client  │
│   - Validate    │
│   - Enrich meta │
└────────┬────────┘
         │
         │ 2. Store via daemon
         ▼
┌─────────────────────────┐
│   Rust Daemon           │
│   - Compute embeddings  │
│   - Add project_id      │
│   - Store in Qdrant     │
└────────┬────────────────┘
         │
         │ 3. Stored in collection
         ▼
┌─────────────────────────┐
│  Qdrant Collection      │
│   _memory               │
│   - Dense vectors       │
│   - Sparse vectors      │
│   - Metadata/payload    │
└─────────────────────────┘
```

## Integration Points

### 1. Memory Infrastructure Integration

**Integration with existing memory system:**
- Uses `_memory` collection for rule storage
- Leverages MemoryRule dataclass from `src/python/common/core/memory.py`
- Integrates with daemon client for rule retrieval
- Shares memory category and authority level enums

**Files affected:**
- `src/python/common/core/memory.py` - MemoryRule schema
- `src/python/common/core/client.py` - Daemon client extensions
- New: `src/python/common/core/context_injection/` - Context injection components

### 2. Daemon Integration

**gRPC interface for rule retrieval:**
- New gRPC method: `GetMemoryRules(scope, project_id) -> MemoryRulesResponse`
- Daemon handles query construction and result formatting
- Python client wrapper in DaemonClient class

**Protocol changes:**
```protobuf
// New message types
message GetMemoryRulesRequest {
    repeated string scope = 1;
    optional string project_id = 2;
    optional MemoryCategory category = 3;
}

message MemoryRulesResponse {
    repeated MemoryRule rules = 1;
    int32 total_count = 2;
}
```

### 3. LLM Tool Integration

**Claude Code:**
- Hook: Claude Code plugin system or filesystem watcher
- Injection method: `.claude/context.md` file creation
- Trigger: Pre-session or on-demand via slash command

**GitHub Codex:**
- Hook: VSCode extension activation
- Injection method: `.github/codex/context.txt`
- Trigger: Workspace open or project switch

**Google Gemini:**
- Hook: Gemini API initialization
- Injection method: System instructions parameter
- Trigger: Session initialization

### 4. Configuration Integration

**Configuration file structure:**
```yaml
# config/context_injection.yaml

context_injection:
  enabled: true

  tools:
    claude_code:
      enabled: true
      adapter: "claude_code_adapter"
      format: "markdown"
      injection_method: "file"
      injection_path: ".claude/context.md"
      token_budget: 8000

    github_codex:
      enabled: true
      adapter: "codex_adapter"
      format: "plain"
      injection_method: "file"
      injection_path: ".github/codex/context.txt"
      token_budget: 4000

    google_gemini:
      enabled: true
      adapter: "gemini_adapter"
      format: "system_instruction"
      injection_method: "api"
      token_budget: 6000

  rule_selection:
    include_global: true
    include_project: true
    max_rules: 50
    authority_precedence: true

  token_management:
    strategy: "priority_based"  # priority_based, round_robin, equal
    compression: "intelligent"  # none, simple, intelligent
    absolute_rules_protected: true
```

## Design Patterns

### 1. Strategy Pattern

**Token budget management strategies:**
- Priority-based: Sort by authority + priority, truncate from bottom
- Round-robin: Distribute tokens equally across rule categories
- Equal allocation: Fixed token allocation per rule with compression

### 2. Adapter Pattern

**LLM tool adapters:**
- Common interface: `LLMToolAdapter`
- Tool-specific implementations: `ClaudeCodeAdapter`, `CodexAdapter`, `GeminiAdapter`
- Capability negotiation via `get_capabilities()`

### 3. Factory Pattern

**Tool adapter factory:**
- Dynamic adapter creation based on configuration
- Plugin registration system for third-party adapters
- Lazy initialization of adapters

### 4. Observer Pattern

**Hook and trigger system:**
- Pre-session events trigger context injection
- Multiple observers can register for injection events
- Error observers for failure handling

### 5. Template Method Pattern

**Rule formatting pipeline:**
- Base template for rule-to-text conversion
- Tool-specific overrides for formatting details
- Consistent structure across tools with customization points

## System Boundaries

### In Scope

1. **Rule Retrieval**: Fetching rules from memory collection via daemon
2. **Rule Selection**: Filtering and prioritizing rules by scope/project
3. **LLM Formatting**: Converting rules to LLM-specific formats
4. **Token Management**: Managing token budgets with intelligent truncation
5. **Tool Integration**: Hooks for Claude Code, Codex, Gemini
6. **Configuration**: YAML-based configuration system
7. **Error Handling**: Graceful degradation and rollback mechanisms

### Out of Scope

1. **Rule Creation UI**: GUI for creating/managing rules (future enhancement)
2. **Rule Analytics**: Usage tracking and effectiveness metrics (future)
3. **Multi-LLM Conversations**: Coordinating context across multiple LLMs
4. **Rule Conflict Resolution**: Automatic resolution of conflicting rules
5. **Dynamic Rule Learning**: AI-powered rule generation from usage
6. **Cross-Project Rule Sharing**: Community rule repositories

### Assumptions

1. Qdrant server is available and accessible
2. Rust daemon is running and responsive
3. Memory collection schema is stable
4. LLM tools support pre-session context injection
5. Token counting is reasonably accurate for target models
6. Configuration files are valid YAML and accessible

### Constraints

1. **Token Limits**: Each LLM has different token budgets for context
2. **Injection Timing**: Must inject before LLM session initialization
3. **Performance**: Rule retrieval and formatting must complete < 2 seconds
4. **Memory Footprint**: Rule caching limited to 100MB in-memory
5. **Concurrency**: Support up to 10 concurrent injection requests
6. **Backward Compatibility**: Must work with existing memory infrastructure

## Next Steps

1. **Subtask 295.2**: Design rule fetching mechanism from memory collection
2. **Subtask 295.3**: Design LLM-specific formatting strategies
3. **Subtask 295.4**: Design token budget management architecture
4. **Subtask 295.5**: Design hook and trigger integration points
5. **Subtask 295.6**: Design configuration schema and management
6. **Subtask 295.7**: Design multi-tool support framework
7. **Subtask 295.8**: Plan extensibility for future LLM tools

## References

- **FIRST-PRINCIPLES.md**: Core architectural principles
- **PRDv3.txt**: Complete system specification
- **src/python/common/core/memory.py**: MemoryRule schema
- **Task 294**: Memory rule data structures (dependency)
