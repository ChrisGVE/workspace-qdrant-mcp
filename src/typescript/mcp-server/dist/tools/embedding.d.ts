/**
 * `embedding` MCP tool handler.
 *
 * Calls `SystemService.GetEmbeddingProviderStatus` over gRPC and returns
 * the active provider's configuration plus its current probe state.
 */
import type { DaemonClient } from '../clients/daemon-client.js';
export interface EmbeddingToolResult {
    success: boolean;
    provider?: string;
    model?: string;
    output_dim?: number;
    base_url?: string;
    probe_status?: string;
    probe_message?: string;
    error?: string;
}
/**
 * Handle the `embedding` tool. The schema currently has no input parameters.
 */
export declare function handleEmbedding(_args: Record<string, unknown> | undefined, daemonClient: DaemonClient): Promise<EmbeddingToolResult>;
//# sourceMappingURL=embedding.d.ts.map