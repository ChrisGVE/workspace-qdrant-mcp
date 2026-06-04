/**
 * MCP over Streamable HTTP transport.
 *
 * Wraps `@modelcontextprotocol/sdk`'s `StreamableHTTPServerTransport` in a
 * Node.js HTTP server so the MCP server can be reached over the network
 * (Docker deployments, remote agents) instead of a per-process stdio pipe.
 *
 * The transport is created in stateful mode: session IDs are generated per
 * connection and returned via the `Mcp-Session-Id` response header. Initial
 * `initialize` requests are anonymous; subsequent requests must echo the
 * session ID back as a request header. This is enforced by the SDK.
 *
 * Graceful shutdown closes the HTTP listener first (stops accepting new
 * connections), then closes the transport (drains in-flight SSE streams).
 */
import { type Server as NodeHttpServer } from 'node:http';
import { StreamableHTTPServerTransport } from '@modelcontextprotocol/sdk/server/streamableHttp.js';
import type { Server as McpServer } from '@modelcontextprotocol/sdk/server/index.js';
import type { HttpTransportOptions } from './server-types.js';
import type { AuthConfig } from './auth-middleware.js';
/**
 * Running HTTP-mode transport plus the Node listener that fronts it. Held by
 * `WorkspaceQdrantMcpServer` so `stop()` can tear both down in order.
 */
export interface McpHttpServerHandle {
    transport: StreamableHTTPServerTransport;
    httpServer: NodeHttpServer;
    host: string;
    port: number;
    path: string;
    /** `true` when native TLS termination is active (`https` server). */
    tlsEnabled: boolean;
}
export declare function startMcpHttpServer(mcpServer: McpServer, options: HttpTransportOptions, authConfig: AuthConfig): Promise<McpHttpServerHandle>;
/**
 * Close the HTTP MCP transport. Stops accepting new connections, then drains
 * the transport's in-flight SSE streams.
 */
export declare function stopMcpHttpServer(handle: McpHttpServerHandle): Promise<void>;
//# sourceMappingURL=mcp-http-server.d.ts.map