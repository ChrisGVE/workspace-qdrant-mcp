# LLM Context Injection System - Architecture Diagrams

**Version:** 0.2.1dev1
**Date:** 2025-10-04

## Component Diagram

```mermaid
graph TB
    subgraph "LLM Tools"
        Claude[Claude Code]
        Codex[GitHub Codex]
        Gemini[Google Gemini]
    end

    subgraph "Integration Layer"
        HookMgr[Hook Manager]
        ClaudeAdapter[Claude Adapter]
        CodexAdapter[Codex Adapter]
        GeminiAdapter[Gemini Adapter]
        PluginReg[Plugin Registry]
        ConfigMgr[Config Manager]
    end

    subgraph "Context Layer"
        Injector[Context Injector]
        RuleSelector[Rule Selector]
        FormatMgr[Format Manager]
        TokenMgr[Token Budget Manager]
        RuleCache[Rule Cache]
    end

    subgraph "Memory Layer"
        DaemonClient[Daemon Client]
        MemoryAPI[Memory API]
        RustDaemon[Rust Daemon]
        QdrantMemory[(Memory Collection)]
    end

    Claude --> ClaudeAdapter
    Codex --> CodexAdapter
    Gemini --> GeminiAdapter

    ClaudeAdapter --> HookMgr
    CodexAdapter --> HookMgr
    GeminiAdapter --> HookMgr

    HookMgr --> Injector
    ConfigMgr --> Injector

    Injector --> RuleSelector
    RuleSelector --> RuleCache
    RuleCache --> DaemonClient
    RuleSelector --> TokenMgr
    TokenMgr --> FormatMgr

    FormatMgr --> ClaudeAdapter
    FormatMgr --> CodexAdapter
    FormatMgr --> GeminiAdapter

    DaemonClient --> MemoryAPI
    MemoryAPI --> RustDaemon
    RustDaemon --> QdrantMemory

    PluginReg -.-> ClaudeAdapter
    PluginReg -.-> CodexAdapter
    PluginReg -.-> GeminiAdapter
```

## Data Flow Diagram - Rule Injection

```mermaid
sequenceDiagram
    participant LLM as LLM Tool
    participant Hook as Hook Manager
    participant Injector as Context Injector
    participant Selector as Rule Selector
    participant Token as Token Manager
    participant Format as Format Manager
    participant Daemon as Daemon Client
    participant Qdrant as Memory Collection

    LLM->>Hook: Session Start Event
    Hook->>Injector: trigger_injection(tool, project, scope)
    Injector->>Selector: select_rules(project, scope)
    Selector->>Daemon: get_rules_by_scope(scope, project)
    Daemon->>Qdrant: query(scope, project_id)
    Qdrant-->>Daemon: MemoryRules[]
    Daemon-->>Selector: MemoryRules[]
    Selector->>Selector: filter_by_relevance()
    Selector->>Selector: sort_by_priority()
    Selector-->>Token: MemoryRules[]
    Token->>Token: count_tokens()
    Token->>Token: prioritize_rules()
    Token->>Token: truncate_if_needed()
    Token-->>Format: Budget-compliant Rules
    Format->>Format: select_adapter(tool)
    Format->>Format: format_for_tool(rules)
    Format-->>Hook: FormattedContext
    Hook->>LLM: inject_context(formatted)
    LLM-->>Hook: Injection Success
    Hook-->>Injector: InjectionResult
```

## Component Interaction Diagram

```mermaid
graph LR
    subgraph "Context Injection Pipeline"
        A[Rule Retrieval] --> B[Rule Selection]
        B --> C[Token Management]
        C --> D[Format Conversion]
        D --> E[Tool Injection]
    end

    subgraph "Supporting Components"
        Config[Configuration]
        Cache[Rule Cache]
        Metrics[Metrics/Logging]
    end

    Config --> A
    Config --> B
    Config --> C
    Config --> D
    Config --> E

    Cache --> A
    A --> Cache

    A --> Metrics
    B --> Metrics
    C --> Metrics
    D --> Metrics
    E --> Metrics
```

## System Context Diagram

```mermaid
graph TB
    subgraph "External Systems"
        LLMTools[LLM Tools<br/>Claude/Codex/Gemini]
        QdrantServer[Qdrant Server]
        FileSystem[File System<br/>Config Files]
    end

    subgraph "Context Injection System"
        Core[Core System]
    end

    subgraph "Workspace Qdrant MCP"
        Daemon[Rust Daemon]
        MCPServer[MCP Server]
        CLI[CLI Tools]
    end

    LLMTools <--> Core
    Core <--> Daemon
    Daemon <--> QdrantServer
    Core <--> FileSystem
    MCPServer --> Core
    CLI --> Core
```

## Rule Selection Flow

```mermaid
flowchart TD
    Start[Start: Injection Request] --> GetRules[Fetch Rules from Memory]
    GetRules --> FilterScope{Scope Match?}
    FilterScope -->|No| Skip[Skip Rule]
    FilterScope -->|Yes| FilterProject{Project Match?}
    FilterProject -->|No Global| CheckGlobal{Global Rules<br/>Enabled?}
    FilterProject -->|Yes| Include[Include Rule]
    CheckGlobal -->|Yes| Include
    CheckGlobal -->|No| Skip
    Include --> CheckAuthority{Authority Level}
    CheckAuthority -->|Absolute| Absolute[Mark as Absolute]
    CheckAuthority -->|Default| Default[Mark as Default]
    Absolute --> SortRules[Sort by Priority]
    Default --> SortRules
    Skip --> NextRule{More Rules?}
    SortRules --> TokenBudget[Apply Token Budget]
    TokenBudget --> ProtectAbsolute[Protect Absolute Rules]
    ProtectAbsolute --> TruncateDefault[Truncate Default Rules]
    TruncateDefault --> Format[Format for Tool]
    Format --> End[End: Formatted Context]
    NextRule -->|Yes| GetRules
    NextRule -->|No| SortRules
```

## Token Budget Management Flow

```mermaid
flowchart TD
    Start[Start: Selected Rules] --> CountTokens[Count Total Tokens]
    CountTokens --> CheckBudget{Within Budget?}
    CheckBudget -->|Yes| Format[Format All Rules]
    CheckBudget -->|No| SeparateAbsolute[Separate Absolute Rules]
    SeparateAbsolute --> CountAbsolute[Count Absolute Tokens]
    CountAbsolute --> CheckAbsoluteBudget{Absolute > Budget?}
    CheckAbsoluteBudget -->|Yes| Warning[Log Warning:<br/>Absolute Exceeds Budget]
    CheckAbsoluteBudget -->|No| CalcRemaining[Calculate Remaining Budget]
    Warning --> IncludeAbsolute[Include All Absolute Rules]
    CalcRemaining --> IncludeAbsolute
    IncludeAbsolute --> SortDefault[Sort Default by Priority]
    SortDefault --> AllocateTokens[Allocate Remaining Tokens]
    AllocateTokens --> TruncateLoop{More Default Rules?}
    TruncateLoop -->|Yes, Budget Left| AddRule[Add Rule to Context]
    TruncateLoop -->|No Budget| SkipRule[Skip Remaining Rules]
    AddRule --> UpdateBudget[Update Remaining Budget]
    UpdateBudget --> TruncateLoop
    SkipRule --> LogSkipped[Log Skipped Rules]
    LogSkipped --> Format
    Format --> End[End: Formatted Context]
```

## Multi-Tool Support Architecture

```mermaid
graph TB
    subgraph "Tool Adapter Interface"
        IAdapter[LLMToolAdapter<br/>Abstract Interface]
    end

    subgraph "Concrete Adapters"
        ClaudeAdapter[Claude Code Adapter]
        CodexAdapter[Codex Adapter]
        GeminiAdapter[Gemini Adapter]
        CustomAdapter[Custom Adapter<br/>Plugin]
    end

    subgraph "Adapter Capabilities"
        Cap1[Format Conversion]
        Cap2[Injection Method]
        Cap3[Token Counting]
        Cap4[Validation]
    end

    subgraph "Registration"
        Registry[Plugin Registry]
        Factory[Adapter Factory]
    end

    IAdapter -.-> ClaudeAdapter
    IAdapter -.-> CodexAdapter
    IAdapter -.-> GeminiAdapter
    IAdapter -.-> CustomAdapter

    ClaudeAdapter --> Cap1
    ClaudeAdapter --> Cap2
    ClaudeAdapter --> Cap3
    ClaudeAdapter --> Cap4

    CodexAdapter --> Cap1
    CodexAdapter --> Cap2
    CodexAdapter --> Cap3
    CodexAdapter --> Cap4

    GeminiAdapter --> Cap1
    GeminiAdapter --> Cap2
    GeminiAdapter --> Cap3
    GeminiAdapter --> Cap4

    CustomAdapter --> Cap1
    CustomAdapter --> Cap2
    CustomAdapter --> Cap3
    CustomAdapter --> Cap4

    Registry --> Factory
    Factory --> ClaudeAdapter
    Factory --> CodexAdapter
    Factory --> GeminiAdapter
    Factory --> CustomAdapter
```

## Configuration Flow

```mermaid
flowchart LR
    subgraph "Configuration Sources"
        Default[Default Config<br/>assets/default_config.yaml]
        User[User Config<br/>config/context_injection.yaml]
        Env[Environment Variables]
    end

    subgraph "Configuration Manager"
        Loader[Config Loader]
        Validator[Schema Validator]
        Merger[Config Merger]
    end

    subgraph "Runtime"
        Cache[Config Cache]
        Runtime[Runtime Config]
    end

    Default --> Loader
    User --> Loader
    Env --> Loader
    Loader --> Validator
    Validator --> Merger
    Merger --> Cache
    Cache --> Runtime
```

## Error Handling Flow

```mermaid
flowchart TD
    Start[Context Injection Request] --> TryFetch{Fetch Rules}
    TryFetch -->|Success| TrySelect[Select Rules]
    TryFetch -->|Daemon Unavailable| FallbackCache{Cache Available?}
    FallbackCache -->|Yes| UseCache[Use Cached Rules]
    FallbackCache -->|No| LogError1[Log Error: No Daemon]
    LogError1 --> GracefulDegrade1[Return Empty Context]
    UseCache --> TrySelect
    TrySelect -->|Success| TryFormat[Format Rules]
    TrySelect -->|Error| LogError2[Log Error: Selection Failed]
    LogError2 --> GracefulDegrade2[Use Partial Rules]
    GracefulDegrade2 --> TryFormat
    TryFormat -->|Success| TryInject[Inject to Tool]
    TryFormat -->|Error| LogError3[Log Error: Format Failed]
    LogError3 --> GracefulDegrade3[Use Plain Text]
    GracefulDegrade3 --> TryInject
    TryInject -->|Success| End[Return Success]
    TryInject -->|Error| LogError4[Log Error: Injection Failed]
    LogError4 --> Rollback[Rollback Injection]
    Rollback --> GracefulDegrade1
    GracefulDegrade1 --> End
```

## Deployment Architecture

```mermaid
graph TB
    subgraph "Developer Machine"
        subgraph "LLM Tools"
            Claude[Claude Code]
            VSCode[VSCode + Codex]
        end

        subgraph "Workspace Qdrant MCP"
            MCP[MCP Server<br/>Context Injection]
            Daemon[Rust Daemon]
            Config[Config Files]
        end

        subgraph "Local Services"
            QdrantLocal[Qdrant<br/>localhost:6333]
        end
    end

    subgraph "Cloud Services Optional"
        QdrantCloud[Qdrant Cloud]
    end

    Claude --> MCP
    VSCode --> MCP
    MCP --> Daemon
    Daemon --> QdrantLocal
    Daemon -.-> QdrantCloud
    Config --> MCP
```

## Extension Points

```mermaid
graph LR
    subgraph "Core System"
        Core[Context Injector]
    end

    subgraph "Extension Points"
        EP1[Tool Adapter Plugin]
        EP2[Format Strategy Plugin]
        EP3[Token Counter Plugin]
        EP4[Rule Filter Plugin]
        EP5[Hook Provider Plugin]
    end

    Core --> EP1
    Core --> EP2
    Core --> EP3
    Core --> EP4
    Core --> EP5

    EP1 -.-> Custom1[Custom LLM Tool]
    EP2 -.-> Custom2[Custom Format]
    EP3 -.-> Custom3[Custom Token Logic]
    EP4 -.-> Custom4[Custom Filters]
    EP5 -.-> Custom5[Custom Triggers]
```

## References

- **llm-context-injection.md**: Detailed architecture documentation
- **FIRST-PRINCIPLES.md**: Core architectural principles
- **PRDv3.txt**: System specification
