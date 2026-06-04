/**
 * MCP tool schema definition for the 'embedding' tool.
 *
 * Surfaces the active embedding provider's configuration and live probe
 * status. Handy for `/health` style introspection from the MCP client and
 * for verifying that a provider migration (`wqm admin reembed`) succeeded.
 */
export const embeddingToolDefinition = {
    name: 'embedding',
    description: 'Report the active embedding provider used by the daemon: provider id, model, configured output dimensionality, base URL (for remote providers), and the live probe status.',
    inputSchema: {
        type: 'object',
        properties: {},
    },
};
//# sourceMappingURL=embedding.js.map