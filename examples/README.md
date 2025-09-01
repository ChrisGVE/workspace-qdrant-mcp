# workspace-qdrant-mcp Examples Suite

This comprehensive examples suite demonstrates practical applications of workspace-qdrant-mcp across different domains and use cases. Each example includes working code, configuration files, sample data, and automation scripts.

## 🎯 Quick Navigation

### 📚 Domain Examples
- **[Software Development](software_development/README.md)** - Code documentation, architecture decisions, project onboarding
- **[Research](research/README.md)** - Academic paper management, citation tracking, literature reviews
- **[Business](business/README.md)** - Meeting notes, knowledge bases, document management
- **[Personal](personal/README.md)** - Personal wikis, learning notes, idea management

### 🔧 Integration Examples
- **[VS Code](integrations/vscode/README.md)** - Workspace setup, task integration, snippets
- **[Cursor IDE](integrations/cursor/README.md)** - AI-powered development workflows
- **[Automation](integrations/automation/README.md)** - CLI scripts, batch processing, workflows

### ⚡ Performance Optimization
- **[Performance Optimization](performance_optimization/README.md)** - Large datasets, memory optimization, search tuning

## 🚀 Getting Started

### 1. Basic Setup

Each example includes its own setup instructions, but all require workspace-qdrant-mcp to be installed and configured:

```bash
# Install the package
uv tool install workspace-qdrant-mcp

# Run setup wizard for configuration
workspace-qdrant-setup
```

### 2. Choose Your Use Case

Browse the examples above to find workflows that match your needs. Each example is self-contained and includes:

- **Sample data** - Real-world examples you can use immediately
- **Configuration files** - Ready-to-use configurations
- **Working scripts** - Automation and utility scripts
- **Documentation** - Step-by-step guides and best practices

### 3. Claude Integration

All examples work seamlessly with Claude Desktop and Claude Code. Basic configuration examples:

#### Claude Desktop (`claude_desktop_config.json`)
```json
{
  "mcpServers": {
    "workspace-qdrant-mcp": {
      "command": "workspace-qdrant-mcp",
      "env": {
        "QDRANT_URL": "http://localhost:6333",
        "COLLECTIONS": "project,docs",
        "GLOBAL_COLLECTIONS": "references,templates"
      }
    }
  }
}
```

#### Claude Code
```bash
# Add MCP server
claude mcp add workspace-qdrant-mcp

# Configure environment variables in your shell
export QDRANT_URL="http://localhost:6333"
export COLLECTIONS="project,docs"
export GLOBAL_COLLECTIONS="references,templates"
```

## 📋 Example Categories

### Domain-Specific Workflows

| Domain | Primary Use Cases | Key Features |
|--------|------------------|-------------|
| **Software Development** | Code docs, architecture, onboarding | Git integration, code snippets, decision tracking |
| **Research** | Papers, citations, reviews | Academic workflows, reference management |
| **Business** | Meetings, knowledge bases | Team collaboration, document management |
| **Personal** | Notes, learning, ideas | Personal knowledge management, journaling |

### Integration & Automation

| Integration | Purpose | Key Benefits |
|------------|---------|-------------|
| **VS Code** | Workspace integration | Seamless editor experience |
| **Cursor** | AI development | Enhanced AI workflows |
| **Automation** | CLI tools, scripts | Batch processing, workflows |
| **Performance** | Optimization | Large-scale operations |

## 🏗️ Architecture Overview

```
examples/
├── domain_examples/           # Real-world use cases
│   ├── software_development/  # Code documentation workflows
│   ├── research/             # Academic workflows
│   ├── business/             # Team and business processes
│   └── personal/             # Personal knowledge management
├── integrations/             # Tool integrations
│   ├── vscode/               # VS Code workspace setup
│   ├── cursor/               # Cursor IDE integration
│   └── automation/           # CLI automation scripts
└── performance_optimization/ # Scale and optimization
```

Each domain includes:
- **Real sample data** for immediate testing
- **Complete configurations** ready to use
- **Automation scripts** for common tasks
- **Best practices** and optimization tips

## 🧪 Testing Examples

Before diving into specific examples, verify your setup:

```bash
# Test system health
workspace-qdrant-test

# Verify workspace detection
wqutil workspace-status

# Test with sample data
cd examples/software_development
python sample_ingestion.py
```

## 📖 Configuration Reference

### Environment Variables

| Variable | Example | Description |
|----------|---------|-------------|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server URL |
| `COLLECTIONS` | `project,docs,tests` | Project-scoped collections |
| `GLOBAL_COLLECTIONS` | `references,templates` | Shared collections |
| `GITHUB_USER` | `yourusername` | Filter subprojects |
| `FASTEMBED_MODEL` | `BAAI/bge-base-en-v1.5` | Embedding model |

### Collection Strategy

**Automatic Collections:**
- `{project-name}-scratchbook` - Always created for notes and ideas

**Configurable Collections:**
- Project collections: `{project-name}-{suffix}` (from `COLLECTIONS`)
- Global collections: `{name}` (from `GLOBAL_COLLECTIONS`)
- Subproject support with GitHub user filtering

## 🎓 Learning Path

**New Users:**
1. Start with [Personal Examples](personal/README.md) for basic concepts
2. Try [Software Development](software_development/README.md) for practical workflows
3. Explore [Integrations](integrations/README.md) for editor setup

**Advanced Users:**
1. Review [Performance Optimization](performance_optimization/README.md)
2. Customize [Automation Scripts](integrations/automation/README.md)
3. Adapt examples for your specific domain

**Teams:**
1. Begin with [Business Examples](business/README.md)
2. Set up shared [Global Collections](software_development/README.md#global-collections)
3. Implement [Team Workflows](business/team_workflows/)

## 🔧 Troubleshooting

Common issues and solutions:

```bash
# Connection problems
workspace-qdrant-test --component qdrant

# Configuration validation
workspace-qdrant-validate

# Performance issues
workspace-qdrant-health --analyze

# Collection management
wqutil list-collections
wqutil workspace-status
```

For detailed troubleshooting, see the [API Reference](../API.md#troubleshooting).

---

**Legacy Configuration Examples:**
- [Claude Desktop Config](claude_desktop_config.json) - Production setup
- [Claude Desktop Dev Config](claude_desktop_dev_config.json) - Development setup
- [Memory System Demo](memory_system_demo.py) - Basic usage patterns
- [YAML Metadata Example](yaml_metadata_example.yaml) - Document metadata
