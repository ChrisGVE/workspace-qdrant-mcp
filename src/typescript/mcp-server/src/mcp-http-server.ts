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

import {
  createServer as createHttpServer,
  type IncomingMessage,
  type Server as NodeHttpServer,
  type ServerResponse,
} from 'node:http';
import { randomUUID } from 'node:crypto';

import { StreamableHTTPServerTransport } from '@modelcontextprotocol/sdk/server/streamableHttp.js';
import type { Server as McpServer } from '@modelcontextprotocol/sdk/server/index.js';
import type { Transport } from '@modelcontextprotocol/sdk/shared/transport.js';

import type { HttpTransportOptions } from './server-types.js';
import { logInfo, logError } from './utils/logger.js';

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
}

/**
 * Bring up the HTTP MCP transport and connect it to `mcpServer`.
 *
 * Returns after `listen()` has fired its callback — at that point the port is
 * bound and the server will accept requests. The caller keeps the returned
 * handle so it can be closed on shutdown.
 */
export async function startMcpHttpServer(
  mcpServer: McpServer,
  options: HttpTransportOptions
): Promise<McpHttpServerHandle> {
  const transport = new StreamableHTTPServerTransport({
    sessionIdGenerator: (): string => randomUUID(),
  });

  // Attach the transport to the MCP server (registers message handlers).
  // Cast through Transport: StreamableHTTPServerTransport's optional-property
  // signatures do not line up with `exactOptionalPropertyTypes` inference
  // against the base interface, but the runtime contract is satisfied.
  await mcpServer.connect(transport as unknown as Transport);

  const httpServer = createHttpServer((req: IncomingMessage, res: ServerResponse): void => {
    void handleRequest(req, res, transport, options.path);
  });

  await new Promise<void>((resolve, reject) => {
    httpServer.once('error', reject);
    httpServer.listen(options.port, options.host, () => {
      httpServer.off('error', reject);
      resolve();
    });
  });

  logInfo('MCP HTTP transport listening', {
    host: options.host,
    port: options.port,
    path: options.path,
  });

  return { transport, httpServer, host: options.host, port: options.port, path: options.path };
}

/**
 * Close the HTTP MCP transport. Stops accepting new connections, then drains
 * the transport's in-flight SSE streams.
 */
export async function stopMcpHttpServer(handle: McpHttpServerHandle): Promise<void> {
  await new Promise<void>((resolve) => {
    handle.httpServer.close(() => resolve());
    // `close` only waits for idle connections — force-close active SSE streams.
    handle.httpServer.closeAllConnections?.();
  });
  await handle.transport.close();
  logInfo('MCP HTTP transport stopped');
}

async function handleRequest(
  req: IncomingMessage,
  res: ServerResponse,
  transport: StreamableHTTPServerTransport,
  mcpPath: string
): Promise<void> {
  if (req.url === '/healthz' && req.method === 'GET') {
    res.writeHead(200, { 'Content-Type': 'text/plain' });
    res.end('ok');
    return;
  }

  // StreamableHTTP uses a single endpoint (default `/mcp`) for POST/GET/DELETE.
  const urlPath = (req.url ?? '').split('?')[0];
  if (urlPath !== mcpPath) {
    res.writeHead(404, { 'Content-Type': 'text/plain' });
    res.end('Not Found');
    return;
  }

  try {
    await transport.handleRequest(req, res);
  } catch (error) {
    logError('MCP HTTP transport error', error);
    if (!res.headersSent) {
      res.writeHead(500, { 'Content-Type': 'text/plain' });
      res.end('Internal Server Error');
    }
  }
}
