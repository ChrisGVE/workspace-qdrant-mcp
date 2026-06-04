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
import { isInitializeRequest } from '@modelcontextprotocol/sdk/types.js';

import type { HttpTlsOptions, HttpTransportOptions } from './server-types.js';
import { recordHttpRequest } from './telemetry/metrics.js';
import { logInfo, logError } from './utils/logger.js';
import type { AuthConfig } from './auth-middleware.js';
import { createAuthMiddleware } from './auth-middleware.js';
import { handleAdminRequest, type AdminDeps } from './admin/handler.js';
import { runWithRequestContext, type RequestContext } from './utils/request-context.js';

/** HTTP header carrying the host-side working directory of the MCP client.
 *  Lower-case form because Node normalizes incoming header names. */
const HOST_CWD_HEADER = 'x-mcp-host-cwd';

/** Evict a stateful session whose last request is older than this. A reconnect
 *  that lost its SSE stream re-`initialize`s with a NEW session id and never
 *  DELETEs the old one, so without this the per-session transport + MCP server
 *  instance (and its daemon gRPC connection) would leak forever. */
const SESSION_IDLE_TTL_MS = ((): number => {
  const raw = Number(process.env['WQM_MCP_SESSION_TTL_MS']);
  return Number.isFinite(raw) && raw > 0 ? raw : 10 * 60_000; // 10 min default
})();
/** How often the idle-session reaper runs. */
const SESSION_REAP_INTERVAL_MS = 60_000;

/**
 * Close sessions whose last activity is older than `ttlMs`. Returns the evicted
 * session ids. Closing each transport fires its `onclose`, which removes it from
 * `transports`/`lastSeen` and shuts down the per-session MCP server instance.
 *
 * Exported for unit testing the eviction predicate without a live HTTP server.
 */
export function reapIdleSessions(
  transports: Map<string, StreamableHTTPServerTransport>,
  lastSeen: Map<string, number>,
  nowMs: number,
  ttlMs: number
): string[] {
  const evicted: string[] = [];
  for (const sessionId of transports.keys()) {
    if (nowMs - (lastSeen.get(sessionId) ?? 0) > ttlMs) evicted.push(sessionId);
  }
  // Close AFTER collecting so we never mutate `transports` mid-iteration.
  for (const sessionId of evicted) {
    void transports.get(sessionId)?.close();
  }
  return evicted;
}

/** Extract the per-request context (currently just the host CWD) from headers. */
function buildRequestContext(req: IncomingMessage): RequestContext {
  const raw = req.headers[HOST_CWD_HEADER];
  const hostCwd = Array.isArray(raw) ? raw[0] : raw;
  return hostCwd && hostCwd.length > 0 ? { hostCwd } : {};
}

/**
 * Running HTTP-mode transport plus the Node listener that fronts it. Held by
 * `WorkspaceQdrantMcpServer` so `stop()` can tear both down in order.
 */
export interface McpHttpServerHandle {
  transports: Map<string, StreamableHTTPServerTransport>;
  httpServer: NodeHttpServer;
  host: string;
  port: number;
  path: string;
  /** `true` when native TLS termination is active (`https` server). */
  tlsEnabled: boolean;
  /** Periodic idle-session reaper; cleared on shutdown. */
  reaper: NodeJS.Timeout;
}

/**
 * Bring up the HTTP MCP transport and connect it to `mcpServer`.
 *
 * Returns after `listen()` has fired its callback — at that point the port is
 * bound and the server will accept requests. The caller keeps the returned
 * handle so it can be closed on shutdown.
 */
/** Create and bind the Node HTTP(S) server, resolving when the port is ready. */
async function createBoundHttpServer(
  options: HttpTransportOptions,
  requestHandler: (req: IncomingMessage, res: ServerResponse) => void
): Promise<{ httpServer: NodeHttpServer; tlsEnabled: boolean }> {
  const tlsEnabled = options.tls !== undefined;
  const httpServer: NodeHttpServer = tlsEnabled
    ? (createHttpsServer(
        loadTlsCredentials(options.tls!),
        requestHandler
      ) as unknown as NodeHttpServer)
    : createHttpServer(requestHandler);

  // Streamable HTTP keeps a long-lived SSE stream open for server->client
  // messages. Node's DEFAULT per-request timeout (`requestTimeout`, 300s on
  // Node >= 18) aborts that stream after 5 minutes — the client's undici then
  // reports `SSE stream disconnected: TypeError: terminated` and reconnects,
  // spawning a fresh MCP instance each time (a reconnect storm). Disable the
  // request/header/keep-alive timeouts so streaming requests are never capped,
  // and enable TCP keepalive so an idle stream survives the localhost relay
  // (Windows -> WSL2 -> docker), which silently drops idle connections.
  httpServer.requestTimeout = 0; // no cap on a single (streaming) request
  httpServer.headersTimeout = 0; // header phase is instant for SSE; don't cap
  httpServer.keepAliveTimeout = 0; // keep idle keep-alive sockets open
  httpServer.timeout = 0; // no socket inactivity timeout
  httpServer.on('connection', (socket) => {
    socket.setKeepAlive(true, 30_000); // probe every 30s so relays keep the conn
    socket.setNoDelay(true);
  });

  await new Promise<void>((resolve, reject) => {
    httpServer.once('error', reject);
    httpServer.listen(options.port, options.host, () => {
      httpServer.off('error', reject);
      resolve();
    });
  });
  return { httpServer, tlsEnabled };
}

export async function startMcpHttpServer(
  createMcpServer: () => Promise<McpServer>,
  options: HttpTransportOptions,
  authConfig: AuthConfig,
  adminDeps?: AdminDeps
): Promise<McpHttpServerHandle> {
  const transports = new Map<string, StreamableHTTPServerTransport>();
  const lastSeen = new Map<string, number>();

  const authMiddleware = createAuthMiddleware(authConfig);
  const requestHandler = (req: IncomingMessage, res: ServerResponse): void => {
    const ctx = buildRequestContext(req);
    runWithRequestContext(ctx, () => {
      void handleRequest(
        req,
        res,
        transports,
        lastSeen,
        createMcpServer,
        options.path,
        authMiddleware,
        adminDeps
      );
    });
  };

  const { httpServer, tlsEnabled } = await createBoundHttpServer(options, requestHandler);

  // Reclaim leaked sessions (reconnect-without-DELETE). unref() so the timer
  // never keeps the process alive on its own.
  const reaper = setInterval(() => {
    const evicted = reapIdleSessions(transports, lastSeen, Date.now(), SESSION_IDLE_TTL_MS);
    if (evicted.length > 0) {
      logInfo('Reaped idle MCP sessions', { count: evicted.length });
    }
  }, SESSION_REAP_INTERVAL_MS);
  reaper.unref();

  logInfo('MCP HTTP transport listening', {
    host: options.host,
    port: options.port,
    path: options.path,
    scheme: tlsEnabled ? 'https' : 'http',
    sessionIdleTtlMs: SESSION_IDLE_TTL_MS,
  });

  return {
    transports,
    httpServer,
    host: options.host,
    port: options.port,
    path: options.path,
    tlsEnabled,
    reaper,
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
  clearInterval(handle.reaper);
  await new Promise<void>((resolve) => {
    handle.httpServer.close(() => resolve());
    // `close` only waits for idle connections — force-close active SSE streams.
    handle.httpServer.closeAllConnections?.();
  });
  await Promise.all(Array.from(handle.transports.values(), (transport) => transport.close()));
  handle.transports.clear();
  logInfo('MCP HTTP transport stopped');
}

/** Dispatch to the MCP transport, writing 500 on unexpected errors. */
async function dispatchToTransport(
  req: IncomingMessage,
  res: ServerResponse,
  transport: StreamableHTTPServerTransport,
  parsedBody?: unknown
): Promise<void> {
  try {
    await transport.handleRequest(req, res, parsedBody);
  } catch (error) {
    logError('MCP HTTP transport error', error);
    if (!res.headersSent) {
      res.writeHead(500, { 'Content-Type': 'text/plain' });
      res.end('Internal Server Error');
    }
  }
}

async function handleRequest(
  req: IncomingMessage,
  res: ServerResponse,
  transports: Map<string, StreamableHTTPServerTransport>,
  lastSeen: Map<string, number>,
  createMcpServer: () => Promise<McpServer>,
  mcpPath: string,
  authMiddleware: (req: IncomingMessage, res: ServerResponse) => { authorized: boolean },
  adminDeps?: AdminDeps
): Promise<void> {
  instrumentHttpResponse(req, res);

  if (req.url === '/healthz' && req.method === 'GET') {
    res.writeHead(200, { 'Content-Type': 'text/plain' });
    res.end('ok');
    return;
  }

  const urlPath = (req.url ?? '').split('?')[0] ?? '';

  // Admin UI static assets (HTML/JS/CSS) are served WITHOUT the Bearer
  // auth middleware. Browsers cannot inject the `Authorization` header
  // when the user just opens the page — auth runs inside the SPA itself
  // (see `admin/static/app.js`: prompt for token, store in
  // sessionStorage, attach to every `/admin/api/*` fetch).
  //
  // `/admin/api/*` keeps full auth because it's where the actual data
  // lives. The static handler refuses anything outside the static root
  // so this can't be used to bypass auth for arbitrary URLs.
  if (adminDeps && urlPath.startsWith('/admin') && !urlPath.startsWith('/admin/api/')) {
    const handled = await handleAdminRequest(req, res, urlPath, adminDeps);
    if (handled) return;
  }

  const decision = authMiddleware(req, res);
  if (!decision.authorized) return;

  // Admin REST under /admin/api/*. Authenticated; same Bearer token as
  // the MCP transport.
  if (adminDeps && urlPath.startsWith('/admin/api/')) {
    const handled = await handleAdminRequest(req, res, urlPath, adminDeps);
    if (handled) return;
  }

  if (urlPath !== mcpPath) {
    res.writeHead(404, { 'Content-Type': 'text/plain' });
    res.end('Not Found');
    return;
  }

  const sessionId = getSessionId(req);
  const existingTransport = sessionId ? transports.get(sessionId) : undefined;
  if (existingTransport && sessionId) {
    lastSeen.set(sessionId, Date.now()); // refresh idle TTL on every request
    await dispatchToTransport(req, res, existingTransport);
    return;
  }

  if (sessionId) {
    writeJsonRpcError(res, 404, -32001, 'Session not found');
    return;
  }

  if (req.method !== 'POST') {
    writeJsonRpcError(res, 400, -32000, 'Bad Request: No valid session ID provided');
    return;
  }

  const body = await readJsonBody(req);
  const messages = Array.isArray(body) ? body : [body];
  if (!messages.some(isInitializeRequest)) {
    writeJsonRpcError(res, 400, -32000, 'Bad Request: No valid session ID provided');
    return;
  }

  let transport!: StreamableHTTPServerTransport;
  transport = new StreamableHTTPServerTransport({
    sessionIdGenerator: (): string => randomUUID(),
    onsessioninitialized: (newSessionId: string): void => {
      transports.set(newSessionId, transport);
      lastSeen.set(newSessionId, Date.now());
      logInfo('MCP HTTP session initialized', { sessionId: newSessionId });
    },
  });

  const mcpServer = await createMcpServer();
  transport.onclose = (): void => {
    const closedSessionId = transport.sessionId;
    if (closedSessionId) {
      transports.delete(closedSessionId);
      lastSeen.delete(closedSessionId);
      logInfo('MCP HTTP session closed', { sessionId: closedSessionId });
    }
    void mcpServer.close();
  };

  // Cast through Transport: StreamableHTTPServerTransport's optional-property
  // signatures do not line up with `exactOptionalPropertyTypes` inference
  // against the base interface, but the runtime contract is satisfied.
  await mcpServer.connect(transport as unknown as Transport);
  await dispatchToTransport(req, res, transport, body);
}

function getSessionId(req: IncomingMessage): string | undefined {
  const raw = req.headers['mcp-session-id'];
  if (Array.isArray(raw)) return raw[0];
  return raw;
}

async function readJsonBody(req: IncomingMessage): Promise<unknown> {
  const chunks: Buffer[] = [];
  for await (const chunk of req) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
  }
  const raw = Buffer.concat(chunks).toString('utf8');
  return raw ? JSON.parse(raw) : undefined;
}

function writeJsonRpcError(
  res: ServerResponse,
  httpStatus: number,
  code: number,
  message: string
): void {
  res.writeHead(httpStatus, { 'Content-Type': 'application/json' });
  res.end(
    JSON.stringify({
      jsonrpc: '2.0',
      error: { code, message },
      id: null,
    })
  );
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
