# workspace-qdrant usage for agents

When a task depends on this repository's structure, implementation details, history, or prior notes, use the `workspace-qdrant` MCP server before manually walking files.

Recommended order:

1. `list` with `format="summary"` for broad orientation.
2. `grep` for exact symbols, filenames, constants and error messages.
3. `search` for semantic/conceptual queries.
4. `retrieve` only when a result ID is already known.
5. `store` a short scratchpad note when you discover durable context that should survive the session.

For indexed-project and agent-branch coordination, use `workspace_index`.
Call read-only actions first (`list_projects`, `project_status`, `list_branches`,
`agent_branch_status`, `observe_project`, `incremental_check`) before any mutation.

Codex-friendly calls may use this context envelope:

```json
{
  "action": "agent_branch_status",
  "projectId": "project-tenant-id",
  "branch": "agent/example-20260525",
  "worktree": "C:/dev/example-agent",
  "payload": {
    "purpose": "describe the task"
  }
}
```

`projectId` maps to the workspace-qdrant tenant ID. Use `projectName` instead
when only the local `.wqm-fork/indexed-projects.json` name is known. `branch`
is an alias for `branchName`, and `worktree` is an alias for `worktreePath`.
Top-level values win over fields inside `payload`.

Mutating actions (`add_project`, `start_agent_branch`, `finish_agent_branch`,
`abandon_agent_branch`, `cleanup_orphans`, etc.) require both
`WQM_INDEX_MANAGER_ALLOW_MUTATION=1` in the MCP server environment and
`allowMutation: true` in the tool call. Never merge an agent branch back to the
original branch automatically.

Do not store secrets, tokens, private keys, `.env` contents, database dumps or user-private data.

If workspace-qdrant is unavailable, continue with normal file tools and report the MCP failure briefly.
