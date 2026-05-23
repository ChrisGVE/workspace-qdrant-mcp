# workspace-qdrant usage for agents

When a task depends on this repository's structure, implementation details, history, or prior notes, use the `workspace-qdrant` MCP server before manually walking files.

Recommended order:

1. `list` with `format="summary"` for broad orientation.
2. `grep` for exact symbols, filenames, constants and error messages.
3. `search` for semantic/conceptual queries.
4. `retrieve` only when a result ID is already known.
5. `store` a short scratchpad note when you discover durable context that should survive the session.

Do not store secrets, tokens, private keys, `.env` contents, database dumps or user-private data.

If workspace-qdrant is unavailable, continue with normal file tools and report the MCP failure briefly.
