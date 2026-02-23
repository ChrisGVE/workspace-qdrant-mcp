## Memory System

### Purpose

The memory collection stores LLM behavioral rules that persist across sessions. Rules are injected into Claude's context at session start.

### Rule Schema

```json
{
  "label": "prefer-uv", // Human-readable identifier
  "content": "Use uv instead of pip for Python packages",
  "scope": "global", // global | project
  "project_id": null, // null for global, project_id for project-specific
  "created_at": "2026-01-30T12:00:00Z"
}
```

**Uniqueness constraint:** `label` + `scope` must be unique. A global rule and a project rule can have the same label.

### Rule Scope

| Scope     | Application           | project_id    |
| --------- | --------------------- | ------------- |
| `global`  | All projects          | `null`        |
| `project` | Specific project only | `"abc123..."` |

### Context Injection

At session start:

1. MCP server queries `memory` collection
2. Filters: all global rules + current project's rules
3. Orders: global rules first (by creation date), then project rules (by creation date)
4. Formatted and injected into system context

### Rule Management

**Via CLI:**

```bash
wqm memory list                      # List all rules (global + all projects)
wqm memory list --global             # List global rules only
wqm memory list --project <path>     # List rules for specific project
wqm memory add --label "prefer-uv" --content "Use uv instead of pip" --global
wqm memory add --label "use-pytest" --content "Use pytest for testing" --project .
wqm memory remove --label "prefer-uv" --global
```

**Via MCP:**

```typescript
memory({ action: "list" });                // List global + current project rules
memory({ action: "add", label: "...", content: "...", scope: "project" });
memory({ action: "remove", label: "...", scope: "global" });
```

### Conversational Updates

Rules can be added conversationally:

```
User: "For future reference, always use uv instead of pip"
→ Creates memory rule: {label: "prefer-uv", content: "Use uv for Python packages", scope: "global"}
```

---

