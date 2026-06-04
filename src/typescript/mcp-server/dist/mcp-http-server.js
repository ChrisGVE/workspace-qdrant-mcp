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
import { createServer as createHttpServer, } from 'node:http';
import { createServer as createHttpsServer } from 'node:https';
import { readFileSync } from 'node:fs';
import { randomUUID } from 'node:crypto';
import { StreamableHTTPServerTransport } from '@modelcontextprotocol/sdk/server/streamableHttp.js';
import { recordHttpRequest } from './telemetry/metrics.js';
import { logInfo, logError } from './utils/logger.js';
import { createAuthMiddleware } from './auth-middleware.js';
/**
 * Bring up the HTTP MCP transport and connect it to `mcpServer`.
 *
 * Returns after `listen()` has fired its callback — at that point the port is
 * bound and the server will accept requests. The caller keeps the returned
 * handle so it can be closed on shutdown.
 */
/** Create and bind the Node HTTP(S) server, resolving when the port is ready. */
async function createBoundHttpServer(options, requestHandler) {
    const tlsEnabled = options.tls !== undefined;
    const httpServer = tlsEnabled
        ? createHttpsServer(loadTlsCredentials(options.tls), requestHandler)
        : createHttpServer(requestHandler);
    await new Promise((resolve, reject) => {
        httpServer.once('error', reject);
        httpServer.listen(options.port, options.host, () => {
            httpServer.off('error', reject);
            resolve();
        });
    });
    return { httpServer, tlsEnabled };
}
export async function startMcpHttpServer(mcpServer, options, authConfig) {
    const transport = new StreamableHTTPServerTransport({
        sessionIdGenerator: () => randomUUID(),
    });
    // Cast through Transport: StreamableHTTPServerTransport's optional-property
    // signatures do not line up with `exactOptionalPropertyTypes` inference
    // against the base interface, but the runtime contract is satisfied.
    await mcpServer.connect(transport);
    const authMiddleware = createAuthMiddleware(authConfig);
    const requestHandler = (req, res) => {
        void handleRequest(req, res, transport, options.path, authMiddleware);
    };
    const { httpServer, tlsEnabled } = await createBoundHttpServer(options, requestHandler);
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
function loadTlsCredentials(tls) {
    const cert = readPem(tls.certPath, 'MCP_HTTP_TLS_CERT');
    const key = readPem(tls.keyPath, 'MCP_HTTP_TLS_KEY');
    const result = { cert, key };
    if (tls.caPath !== undefined && tls.caPath !== '') {
        result.ca = readPem(tls.caPath, 'MCP_HTTP_TLS_CA');
    }
    return result;
}
function readPem(path, envName) {
    try {
        return readFileSync(path);
    }
    catch (error) {
        const reason = error instanceof Error ? error.message : String(error);
        throw new Error(`Failed to read ${envName} from ${path}: ${reason}`);
    }
}
/**
 * Close the HTTP MCP transport. Stops accepting new connections, then drains
 * the transport's in-flight SSE streams.
 */
export async function stopMcpHttpServer(handle) {
    await new Promise((resolve) => {
        handle.httpServer.close(() => resolve());
        // `close` only waits for idle connections — force-close active SSE streams.
        handle.httpServer.closeAllConnections?.();
    });
    await handle.transport.close();
    logInfo('MCP HTTP transport stopped');
}
/** Dispatch to the MCP transport, writing 500 on unexpected errors. */
async function dispatchToTransport(req, res, transport) {
    try {
        await transport.handleRequest(req, res);
    }
    catch (error) {
        logError('MCP HTTP transport error', error);
        if (!res.headersSent) {
            res.writeHead(500, { 'Content-Type': 'text/plain' });
            res.end('Internal Server Error');
        }
    }
}
async function handleRequest(req, res, transport, mcpPath, authMiddleware) {
    instrumentHttpResponse(req, res);
    if (req.url === '/healthz' && req.method === 'GET') {
        res.writeHead(200, { 'Content-Type': 'text/plain' });
        res.end('ok');
        return;
    }
    const decision = authMiddleware(req, res);
    if (!decision.authorized)
        return;
    const urlPath = (req.url ?? '').split('?')[0];
    if (urlPath !== mcpPath) {
        res.writeHead(404, { 'Content-Type': 'text/plain' });
        res.end('Not Found');
        return;
    }
    await dispatchToTransport(req, res, transport);
}
/**
 * Wrap `res.writeHead` so the first status-bearing call records a Prometheus
 * counter sample. Repeated calls (e.g. the SDK re-setting headers on stream
 * continuations) are ignored — the first response line is what clients see.
 */
function instrumentHttpResponse(req, res) {
    let recorded = false;
    const originalWriteHead = res.writeHead.bind(res);
    res.writeHead = ((...args) => {
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
        return originalWriteHead(...args);
    });
}
//# sourceMappingURL=mcp-http-server.js.map