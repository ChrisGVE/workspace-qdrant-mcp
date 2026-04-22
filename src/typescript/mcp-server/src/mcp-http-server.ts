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
import { createServer as createHttpsServer } from 'node:https';
import { readFileSync } from 'node:fs';
import { randomUUID } from 'node:crypto';

import { StreamableHTTPServerTransport } from '@modelcontextprotocol/sdk/server/streamableHttp.js';
import type { Server as McpServer } from '@modelcontextprotocol/sdk/server/index.js';
import type { Transport } from '@modelcontextprotocol/sdk/shared/transport.js';

import type { HttpTlsOptions, HttpTransportOptions } from './server-types.js';
import { recordHttpRequest } from './telemetry/metrics.js';
import { logInfo, logError } from './utils/logger.js';
import type { AuthConfig } from './auth-middleware.js';
import { createAuthMiddleware } from './auth-middleware.js';

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

/**
 * Bring up the HTTP MCP transport and connect it to `mcpServer`.
 *
 * Returns after `listen()` has fired its callback — at that point the port is
 * bound and the server will accept requests. The caller keeps the returned
 * handle so it can be closed on shutdown.
 */
export async function startMcpHttpServer(
  mcpServer: McpServer,
  options: HttpTransportOptions,
  authConfig: AuthConfig
): Promise<McpHttpServerHandle> {
  const transport = new StreamableHTTPServerTransport({
    sessionIdGenerator: (): string => randomUUID(),
  });

  // Attach the transport to the MCP server (registers message handlers).
  // Cast through Transport: StreamableHTTPServerTransport's optional-property
  // signatures do not line up with `exactOptionalPropertyTypes` inference
  // against the base interface, but the runtime contract is satisfied.
  await mcpServer.connect(transport as unknown as Transport);

  const authMiddleware = createAuthMiddleware(authConfig);

  const requestHandler = (req: IncomingMessage, res: ServerResponse): void => {
    void handleRequest(req, res, transport, options.path, authMiddleware);
  };

  const tlsEnabled = options.tls !== undefined;
  const httpServer: NodeHttpServer = tlsEnabled
    ? // Node's https.Server extends http.Server — safe to type through it.
      (createHttpsServer(
        loadTlsCredentials(options.tls!),
        requestHandler
      ) as unknown as NodeHttpServer)
    : createHttpServer(requestHandler);

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
    scheme: tlsEnabled ? 'https' : 'http',
  });

  return {
    transport,
    httpServer,
    host: options.host,
    port: options.port,
    path: options.path,
    tlsEnabled,
  };
}

/**
 * Read the certificate, key, and optional CA bundle from disk. Throws with a
 * specific error message if any path is unreadable so operators get actionable
 * output instead of a stack trace deep in `tls.createSecureContext`.
 */
function loadTlsCredentials(tls: HttpTlsOptions): { cert: Buffer; key: Buffer; ca?: Buffer } {
  const cert = readPem(tls.certPath, 'MCP_HTTP_TLS_CERT');
  const key = readPem(tls.keyPath, 'MCP_HTTP_TLS_KEY');
  const result: { cert: Buffer; key: Buffer; ca?: Buffer } = { cert, key };
  if (tls.caPath !== undefined && tls.caPath !== '') {
    result.ca = readPem(tls.caPath, 'MCP_HTTP_TLS_CA');
  }
  return result;
}

function readPem(path: string, envName: string): Buffer {
  try {
    return readFileSync(path);
  } catch (error) {
    const reason = error instanceof Error ? error.message : String(error);
    throw new Error(`Failed to read ${envName} from ${path}: ${reason}`);
  }
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
  mcpPath: string,
  authMiddleware: (req: IncomingMessage, res: ServerResponse) => { authorized: boolean }
): Promise<void> {
  // Record the final status once, no matter which branch terminates the
  // response. Hooking `res.writeHead` keeps middleware + handler + SDK
  // transport code paths consistent without threading a recorder through
  // each of them.
  instrumentHttpResponse(req, res);

  // Liveness probe: always 200, no auth. Intentionally narrow — only exact
  // match on `/healthz` GET. Anything else goes through auth.
  if (req.url === '/healthz' && req.method === 'GET') {
    res.writeHead(200, { 'Content-Type': 'text/plain' });
    res.end('ok');
    return;
  }

  // Auth + rate-limit + CORS. Middleware writes its own terminal response on
  // failure (401 / 429 / 204 for preflight) and tells us whether to proceed.
  const decision = authMiddleware(req, res);
  if (!decision.authorized) {
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

/**
 * Wrap `res.writeHead` so the first status-bearing call records a Prometheus
 * counter sample. Repeated calls (e.g. the SDK re-setting headers on stream
 * continuations) are ignored — the first response line is what clients see.
 */
function instrumentHttpResponse(req: IncomingMessage, res: ServerResponse): void {
  let recorded = false;
  const originalWriteHead = res.writeHead.bind(res);
  res.writeHead = ((...args: unknown[]): ServerResponse => {
    const statusCode = typeof args[0] === 'number' ? args[0] : res.statusCode;
    if (!recorded) {
      recorded = true;
      recordHttpRequest(req.url, statusCode);
    }
    // Delegate to the original implementation with the original arguments.
    // The signature is overloaded (writeHead(code), writeHead(code, headers),
    // writeHead(code, message), writeHead(code, message, headers)), so cast
    // through `any` rather than re-declaring every overload.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return (originalWriteHead as any)(...args);
  }) as typeof res.writeHead;
}
