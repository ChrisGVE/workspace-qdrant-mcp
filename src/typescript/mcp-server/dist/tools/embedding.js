/**
 * `embedding` MCP tool handler.
 *
 * Calls `SystemService.GetEmbeddingProviderStatus` over gRPC and returns
 * the active provider's configuration plus its current probe state.
 */
/**
 * Handle the `embedding` tool. The schema currently has no input parameters.
 */
export async function handleEmbedding(_args, daemonClient) {
    try {
        const response = await daemonClient.getEmbeddingProviderStatus();
        return {
            success: true,
            provider: response.provider,
            model: response.model,
            output_dim: response.output_dim,
            base_url: response.base_url,
            probe_status: response.probe_status,
            probe_message: response.probe_message,
        };
    }
    catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        return {
            success: false,
            error: `Failed to fetch embedding provider status: ${message}`,
        };
    }
}
//# sourceMappingURL=embedding.js.map