# Claude Code Hooks Integration (CLI-based)

## Overview

This guide documents how to integrate workspace-qdrant-mcp with Claude Code's hook system using `wqm` CLI commands for automatic session lifecycle management and memory context injection.

**Key architecture insight:** Claude Code hooks must be pre-configured in `.claude/settings.json` or shipped with the project. They cannot be programmatically created by MCP servers at runtime.

**Separation of concerns:**
- **MCP server** provides tools (`search`, `store`, `memory`, `retrieve`)
- **Hooks** call `wqm` CLI commands for session lifecycle events
- **Daemon** manages persistent state and file watching

This replaces the HTTP-based approach documented in `docs/claude_code_integration.md`.

## Architecture

```
                    ┌─────────────────────────┐
                    │       Claude Code        │
                    │                          │
                    │  SessionStart / End      │
                    └───────────┬──────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                  │
              ▼                 ▼                  ▼
     ┌────────────────┐  ┌──────────┐  ┌────────────────┐
     │ SessionStart   │  │   MCP    │  │  SessionEnd    │
     │ Hook           │  │  Server  │  │  Hook          │
     │ wqm session    │  │  tools   │  │  wqm session   │
     │   start        │  │          │  │   end          │
     └───────┬────────┘  └──────────┘  └───────┬────────┘
             │                                  │
             │ gRPC                      gRPC   │
             ▼                                  ▼
     ┌──────────────────────────────────────────────────┐
     │                  Rust Daemon                      │
     │  - RegisterProject / DeprioritizeProject          │
     │  - File Watch (always-on for enabled folders)     │
     │  - Ingestion / Processing                         │
     │  - Memory collection management                   │
     └──────────────────────────────────────────────────┘
```

## Prerequisites

1. **memexd daemon** running (`wqm service status`)
2. **wqm CLI** installed and in PATH
3. **Claude Code** with hooks support
4. **Qdrant** running (for memory collection access)

## Hook Configuration

Add to `.claude/settings.json` (project-level) or `~/.claude/settings.json` (global):

```json
{
  "hooks": {
    "SessionStart": [
      {
        "matcher": "startup|resume",
        "hooks": [
          {
            "type": "command",
            "command": "wqm session start --json",
            "timeout": 10,
            "statusMessage": "Activating project..."
          }
        ]
      },
      {
        "matcher": "clear|compact",
        "hooks": [
          {
            "type": "command",
            "command": "wqm session start --json --lightweight",
            "timeout": 5,
            "statusMessage": "Refreshing context..."
          }
        ]
      }
    ],
    "SessionEnd": [
      {
        "matcher": "logout|prompt_input_exit|other",
        "hooks": [
          {
            "type": "command",
            "command": "wqm session end",
            "timeout": 5,
            "async": true
          }
        ]
      }
    ]
  }
}
```

### Matcher Patterns Explained

**SessionStart matchers:**
| Matcher | When | Action |
|---------|------|--------|
| `startup` | New session begins | Full initialization: activate project, inject memories |
| `resume` | Session resumed from pause | Full initialization (same as startup) |
| `clear` | User runs `/clear` | Lightweight re-check (triggers both SessionEnd + SessionStart) |
| `compact` | Context auto-compaction | Lightweight re-check |

**SessionEnd matchers:**
| Matcher | When | Action |
|---------|------|--------|
| `logout` | User logs out | Send DeprioritizeProject |
| `prompt_input_exit` | User exits prompt | Send DeprioritizeProject |
| `other` | Other termination | Send DeprioritizeProject |
| `clear` | User runs `/clear` | Do NOT deactivate (session continues) |

Note: `clear` is intentionally excluded from SessionEnd hooks because `/clear` triggers both SessionEnd and SessionStart in sequence. Deactivating on clear would immediately re-activate, wasting a round-trip.

## CLI Commands (Planned)

### `wqm session start`

Detects the current project, activates it in the daemon, fetches relevant memories, and outputs context for Claude Code injection.

```bash
# Full initialization (startup/resume)
wqm session start --json

# Lightweight refresh (clear/compact)
wqm session start --json --lightweight
```

**Full initialization flow:**
1. Detect project from current working directory (git analysis)
2. Send `RegisterProject` gRPC to daemon (sets `is_active = 1` for project group)
3. Fetch global memories from the `memory` collection
4. Fetch project-scoped memories (if any)
5. Output JSON with `additionalContext` containing formatted memories

**Lightweight refresh flow:**
1. Send heartbeat to daemon (refresh `last_activity_at`)
2. Output JSON with minimal context

**Output format (SessionStart hooks can return additionalContext):**

```json
{
  "hookSpecificOutput": {
    "hookEventName": "SessionStart",
    "additionalContext": "## Project Memories\n\n- Always use snake_case for Rust functions\n- Run tests with: cargo test --package workspace-qdrant-core\n\n## Global Memories\n\n- Prefer explicit error handling over unwrap()\n"
  }
}
```

### `wqm session end`

Notifies the daemon that the session has ended, allowing the project to be deprioritized.

```bash
wqm session end
```

**Flow:**
1. Detect project from current working directory
2. Send `DeprioritizeProject` gRPC to daemon (sets `is_active = 0` for project group)
3. Exit silently (no output needed for SessionEnd hooks)

### Performance Requirements

Both commands must complete within their timeout:
- `wqm session start --json`: < 10 seconds (full), < 5 seconds (lightweight)
- `wqm session end`: < 5 seconds (async, non-blocking)

Error handling must be graceful — hooks should never crash Claude Code. If the daemon is unavailable, commands should exit cleanly with exit code 0.

## Memory Formatting

Memories injected via `additionalContext` follow this format:

```markdown
## Project Memories ({project_name})

- {memory_rule_1}
- {memory_rule_2}

## Global Memories

- {global_rule_1}
- {global_rule_2}
```

Memory categories may include:
- **Coding standards** (naming conventions, patterns)
- **Project-specific rules** (architecture decisions, constraints)
- **Build/test commands** (frequently used commands)
- **Known issues** (workarounds, gotchas)

## Activity Inheritance

When a project is activated or deactivated via hooks, the operation applies to the entire project group using recursive SQL:

```sql
WITH RECURSIVE project_group AS (
    SELECT watch_id FROM watch_folders WHERE watch_id = ?
    UNION
    SELECT wf.watch_id FROM watch_folders wf
    JOIN project_group pg ON wf.parent_watch_id = pg.watch_id
)
UPDATE watch_folders
SET is_active = ?, updated_at = datetime('now')
WHERE watch_id IN (SELECT watch_id FROM project_group)
```

This ensures submodules inherit their parent project's active state. See the daemon's `DaemonStateManager::activate_project_group()` and `deactivate_project_group()` methods.

## Inactivity Timeout

Projects that remain active but have no session activity are automatically deactivated by the daemon after a configurable timeout (default: 12 hours). This handles edge cases where SessionEnd hooks fail to fire.

Configure via environment variable:
```bash
WQM_INACTIVITY_TIMEOUT_SECS=43200  # 12 hours (default)
```

The daemon checks every 5 minutes for stale projects.

## Testing and Debugging

### Verify daemon is running

```bash
wqm service status
```

### Test session commands manually

```bash
# Simulate session start
wqm session start --json
# Should output JSON with additionalContext

# Simulate session end
wqm session end
# Should exit silently with code 0

# Check project activation state
wqm watch list
# Active projects show is_active=true
```

### Debug hook execution

Claude Code shows hook output in verbose mode. Check for:
- Exit codes (0 = success, non-zero = error)
- JSON formatting issues in stdout
- Timeout errors (hook took too long)

### Verify hook configuration

```bash
# Check settings.json is valid JSON
python3 -c "import json; json.load(open('.claude/settings.json'))"

# Verify wqm is in PATH
which wqm
```

## Comparison with HTTP-based Approach

| Aspect | HTTP Approach | CLI Approach |
|--------|--------------|-------------|
| **Dependency** | Python HTTP server on port 8765 | `wqm` binary in PATH |
| **Setup** | Install scripts, start server | Configure hooks in settings.json |
| **Reliability** | Requires server to be running | Works as long as daemon runs |
| **Performance** | HTTP round-trip | Direct gRPC to daemon |
| **Maintenance** | Separate process to manage | Part of existing daemon infrastructure |

The CLI approach is recommended as the primary integration method. It eliminates the need for a separate HTTP server process and leverages the existing daemon infrastructure.

## File Reference

| File | Purpose |
|------|---------|
| `.claude/settings.json` | Hook configuration (project-level) |
| `~/.claude/settings.json` | Hook configuration (global) |
| `src/rust/cli/src/commands/session.rs` | CLI session command implementation |
| `src/rust/daemon/core/src/daemon_state.rs` | Activity inheritance SQL |
| `src/rust/daemon/grpc/src/services/project_service.rs` | RegisterProject/DeprioritizeProject handlers |
| `docs/claude_code_integration.md` | Legacy HTTP-based approach |
