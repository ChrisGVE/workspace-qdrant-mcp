/**
 * DaemonClientBase — connection lifecycle, retry logic, and gRPC plumbing.
 *
 * Handles proto loading, service-client instantiation, exponential-backoff
 * retry, and the reentrancy-safe auto-reconnect guard (issue #55).
 */
import * as grpc from '@grpc/grpc-js';
import * as protoLoader from '@grpc/proto-loader';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
export const PROTO_PATH = join(__dirname, '..', '..', 'proto', 'workspace_daemon.proto');
const DEFAULT_HOST = 'localhost';
const DEFAULT_PORT = 50051;
const DEFAULT_TIMEOUT_MS = 5000;
const MAX_RETRIES = 3;
const INITIAL_RETRY_DELAY_MS = 100;
/**
 * Generic gRPC unary call wrapper — turns callback-style into Promise.
 * The `method` must be a function(request, callback) on the service client.
 */
export function grpcUnary(client, methodName, request) {
    return new Promise((resolve, reject) => {
        if (!client) {
            reject(new Error('Client not connected'));
            return;
        }
        const fn = client[methodName];
        if (typeof fn !== 'function') {
            reject(new Error(`Unknown method: ${methodName}`));
            return;
        }
        fn.call(client, request, (error, response) => {
            if (error)
                reject(error);
            else
                resolve(response);
        });
    });
}
/**
 * gRPC unary call with a hard wall-clock timeout.
 *
 * Races the underlying `grpcUnary` promise against a timer.  The `finally`
 * block clears the timer so no leak occurs when the call resolves first.
 *
 * @param operationName - Human-readable label used in the rejection message
 *   (falls back to `methodName` when omitted).
 */
export async function grpcUnaryWithTimeout(client, methodName, request, timeoutMs, operationName) {
    const label = operationName ?? methodName;
    let timer;
    const timeoutPromise = new Promise((_, reject) => {
        timer = setTimeout(() => {
            reject(new Error(`gRPC call ${label} timed out after ${timeoutMs}ms`));
        }, timeoutMs);
    });
    try {
        return await Promise.race([grpcUnary(client, methodName, request), timeoutPromise]);
    }
    finally {
        if (timer !== undefined)
            clearTimeout(timer);
    }
}
export class DaemonClientBase {
    host;
    port;
    timeoutMs;
    maxRetries;
    /**
     * Per-method timeout overrides keyed by operation name.
     *
     * Subclasses may populate this map to give individual RPC methods a
     * different ceiling from the default `timeoutMs`.  For example, search
     * operations typically need a longer window than write-enqueue calls:
     *
     * ```ts
     * // In a subclass constructor — set search to 2× the default ceiling.
     * this.methodTimeouts['search'] = this.timeoutMs * 2;
     * ```
     *
     * The map is intentionally left empty in the base class so that
     * `getMethodTimeout` falls through to `timeoutMs` for every method
     * unless an override is explicitly registered.
     */
    methodTimeouts = {};
    /**
     * Return the effective timeout for a gRPC call.
     *
     * Resolution order:
     * 1. `override` argument — caller-supplied one-shot ceiling.
     * 2. `this.methodTimeouts[methodName]` — per-method class-level ceiling.
     * 3. `this.timeoutMs` — global default.
     *
     * The `search` operation uses `2 × timeoutMs` to accommodate larger
     * result sets and heavier embedding work; all other methods use the
     * default.
     */
    getMethodTimeout(methodName, override) {
        if (typeof override === 'number')
            return override;
        const explicit = this.methodTimeouts[methodName];
        if (typeof explicit === 'number')
            return explicit;
        if (methodName === 'search')
            return this.timeoutMs * 2;
        return this.timeoutMs;
    }
    systemClient;
    projectClient;
    documentClient;
    embeddingClient;
    textSearchClient;
    graphClient;
    queueWriteClient;
    trackingWriteClient;
    connectionState = { connected: false };
    /**
     * Reentrancy guard: prevents `ensureConnected()` from recursing back
     * through `healthCheck()` while `connect()` is still in flight.
     * Without it the auto-reconnect path in `callWithRetry` bounces
     * between connect → healthCheck → callWithRetry → ensureConnected →
     * connect and blows the call stack. See issue #55.
     */
    connecting = false;
    constructor(config = {}) {
        this.host = config.host ?? DEFAULT_HOST;
        this.port = config.port ?? DEFAULT_PORT;
        this.timeoutMs = config.timeoutMs ?? DEFAULT_TIMEOUT_MS;
        this.maxRetries = config.maxRetries ?? MAX_RETRIES;
    }
    getConnectionState() {
        return { ...this.connectionState };
    }
    isConnected() {
        return this.connectionState.connected;
    }
    /** Instantiate all gRPC service clients from the loaded proto. */
    instantiateClients(proto, address, credentials) {
        this.systemClient = new proto.workspace_daemon.SystemService(address, credentials);
        this.projectClient = new proto.workspace_daemon.ProjectService(address, credentials);
        this.documentClient = new proto.workspace_daemon.DocumentService(address, credentials);
        this.embeddingClient = new proto.workspace_daemon.EmbeddingService(address, credentials);
        this.textSearchClient = new proto.workspace_daemon.TextSearchService(address, credentials);
        this.graphClient = new proto.workspace_daemon.GraphService(address, credentials);
        this.queueWriteClient = new proto.workspace_daemon.QueueWriteService(address, credentials);
        this.trackingWriteClient = new proto.workspace_daemon.TrackingWriteService(address, credentials);
    }
    async connect() {
        const packageDefinition = protoLoader.loadSync(PROTO_PATH, {
            keepCase: true,
            longs: String,
            enums: Number,
            defaults: true,
            oneofs: true,
            includeDirs: [join(__dirname, '..', '..', 'proto')],
        });
        const proto = grpc.loadPackageDefinition(packageDefinition);
        const address = `${this.host}:${this.port}`;
        const credentials = grpc.credentials.createInsecure();
        this.instantiateClients(proto, address, credentials);
        try {
            await this.healthCheck();
            this.connectionState = { connected: true, lastHealthCheck: new Date() };
        }
        catch (error) {
            this.connectionState = {
                connected: false,
                lastError: error instanceof Error ? error.message : 'Unknown error',
            };
            throw error;
        }
    }
    close() {
        const clients = [
            this.systemClient,
            this.projectClient,
            this.documentClient,
            this.embeddingClient,
            this.textSearchClient,
            this.graphClient,
            this.queueWriteClient,
            this.trackingWriteClient,
        ];
        for (const c of clients) {
            if (c)
                grpc.closeClient(c);
        }
        this.connectionState = { connected: false };
    }
    /**
     * Ensure the gRPC clients are initialised before issuing an RPC.
     *
     * When the MCP server's initial `connect()` fails (daemon not running,
     * proto load error, etc.) we still hold a `DaemonClient` with
     * undefined service clients. Without this helper, every subsequent
     * call would reject with "Client not connected" even after the daemon
     * comes back up. Calling `connect()` here lets the client self-heal
     * on the next user action. See issue #55.
     */
    async ensureConnected() {
        if (this.connecting)
            return; // connect() is in flight — let it finish
        if (this.connectionState.connected && this.systemClient)
            return;
        this.connecting = true;
        try {
            await this.connect();
        }
        finally {
            this.connecting = false;
        }
    }
    async callWithRetry(fn) {
        let lastError;
        let delay = INITIAL_RETRY_DELAY_MS;
        for (let attempt = 0; attempt < this.maxRetries; attempt++) {
            try {
                await this.ensureConnected();
                const result = await fn();
                this.connectionState = { connected: true, lastHealthCheck: new Date() };
                return result;
            }
            catch (error) {
                lastError = error instanceof Error ? error : new Error(String(error));
                if (!this.isRetryableError(lastError))
                    throw lastError;
                // A transient failure invalidates our notion of being connected —
                // the next iteration's `ensureConnected` will reconnect.
                this.connectionState = { connected: false, lastError: lastError.message };
                if (attempt < this.maxRetries - 1) {
                    await new Promise((r) => setTimeout(r, delay));
                    delay *= 2;
                }
            }
        }
        this.connectionState = { connected: false, lastError: lastError?.message };
        throw lastError;
    }
    isRetryableError(error) {
        const retryableCodes = [
            grpc.status.UNAVAILABLE,
            grpc.status.DEADLINE_EXCEEDED,
            grpc.status.RESOURCE_EXHAUSTED,
        ];
        const grpcError = error;
        if (typeof grpcError.code === 'number')
            return retryableCodes.includes(grpcError.code);
        return (error.message.includes('ECONNREFUSED') ||
            error.message.includes('ETIMEDOUT') ||
            error.message.includes('ENOTFOUND') ||
            // Issue #55: retry when the local client handle is stale (never
            // connected, closed, or a dropped channel) so the next iteration's
            // ensureConnected() can heal the connection.
            error.message.includes('Client not connected') ||
            error.message.includes('channel has been closed') ||
            error.message.includes('Channel has been shut down'));
    }
}
//# sourceMappingURL=connection.js.map