/**
 * MCP tool schema definition for the 'embedding' tool.
 *
 * Surfaces the active embedding provider's configuration and live probe
 * status. Handy for `/health` style introspection from the MCP client and
 * for verifying that a provider migration (`wqm admin reembed`) succeeded.
 */
export declare const embeddingToolDefinition: {
    name: string;
    description: string;
    inputSchema: {
        type: "object";
        properties: {};
    };
};
//# sourceMappingURL=embedding.d.ts.map