## workspace-qdrant

Use `workspace-qdrant` first when context is uncertain or the task involves codebase exploration.

Preferred flow:

1. `list` for project overview.
2. `grep` for exact strings/symbols.
3. `search` for semantic search across indexed code and notes.
4. `store` relevant findings in scratchpad only when they are useful for future sessions.

Never store secrets or sensitive local data.

If the MCP server is unavailable, mention it briefly and fall back to normal file inspection.
