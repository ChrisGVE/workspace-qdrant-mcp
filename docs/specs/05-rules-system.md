## Rules System

### Purpose

The rules collection stores LLM behavioral rules that persist across sessions. Rules are injected into Claude's context at session start.

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

1. MCP server queries `rules` collection
2. Filters: all global rules + current project's rules
3. Orders: global rules first (by creation date), then project rules (by creation date)
4. Formatted and injected into system context

### Rule Management

**Via CLI:**

```bash
wqm rules list                      # List all rules (global + all projects)
wqm rules list --global             # List global rules only
wqm rules list --project <path>     # List rules for specific project
wqm rules add --label "prefer-uv" --content "Use uv instead of pip" --global
wqm rules add --label "use-pytest" --content "Use pytest for testing" --project .
wqm rules remove --label "prefer-uv" --global
```

**Via MCP:**

```typescript
rules({ action: "list" });                // List global + current project rules
rules({ action: "add", label: "...", content: "...", scope: "project" });
rules({ action: "remove", label: "...", scope: "global" });
```

### Conversational Updates

Rules can be added conversationally:

```
User: "For future reference, always use uv instead of pip"
→ Creates rule: {label: "prefer-uv", content: "Use uv for Python packages", scope: "global"}
```

---

## Known Issues & Planned Improvements (rules tool)

Found 2026-05-29 while reviewing/repairing behavioral rules. Tracked for fixing
(see the spawned "Fix rules update/remove" task).

1. **`update` creates a duplicate instead of editing in place.** Calling
   `rules action=update` on an existing rule (same label+scope) inserts a NEW
   point with a new id and leaves the original — violating the "label + scope
   must be unique" invariant above. Fix: make `update` an upsert keyed on
   (label, scope, project_id), reusing the existing point id.

2. **`remove` cannot disambiguate two rules sharing (label, scope, project_id).**
   When duplicates exist with identical keys, `remove` deletes neither
   (idempotency-keyed; three calls left both copies in place). Fix: `remove`
   should delete ALL rows matching the filter, and/or accept an explicit rule
   `id` to target a single point.

3. **Default `list` is project-scoped and hides global rules.** `rules action=list`
   without `scope` returns only the current project's rules — EMPTY for a
   project with no project-rules, even though that project DOES receive all
   global rules at injection time (see Context Injection above). This misleads:
   an agent concludes "no rules" when the global baseline applies. Fix: default
   `list` should reflect the injection set (global + current project), or clearly
   signal "use scope=global to see global rules".

4. **Rules writes are NOT gated.** `add` / `update` / `remove` mutate persistent,
   cross-session, cross-project shared state with no confirmation — unlike
   mutating `workspace_index` actions, which require double opt-in (the
   `confirm-mut` global rule). An agent can silently rewrite the rules that every
   future session (and, for global rules, every project) inherits. Improvement:
   gate rules writes behind explicit user confirmation (same philosophy as
   `confirm-mut`) — agent proposes, user approves, agent applies.

**Related drift observed:** the `rules_mirror` SQLite table and the Qdrant
`rules` collection can fall out of sync (8 mirror rows vs 17 Qdrant points
observed 2026-05-29). Any write-path fix must keep both consistent.

**Recovery:** the canonical rule set is versioned at `assets/rules.example.json`
(global + project arrays). On DB/volume loss, re-apply with `rules action=add`
on a clean collection (NOT `update` — see issue 1). See
`docs/runbooks/restore-rules-and-global-ignore.md` for the procedure.

---
