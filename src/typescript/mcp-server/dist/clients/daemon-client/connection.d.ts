/**
 * DaemonClientBase — connection lifecycle, retry logic, and gRPC plumbing.
 *
 * Handles proto loading, service-client instantiation, exponential-backoff
 * retry, and the reentrancy-safe auto-reconnect guard (issue #55).
 */
import * as grpc from '@grpc/grpc-js';
import type { SystemServiceClient, ProjectServiceClient, DocumentServiceClient, EmbeddingServiceClient, TextSearchServiceClient, GraphServiceClient, QueueWriteServiceClient, TrackingWriteServiceClient } from '../grpc-types.js';
export declare const PROTO_PATH: string;
export interface DaemonClientConfig {
    host?: string;
    port?: number;
    timeoutMs?: number;
    maxRetries?: number;
}
export interface ConnectionState {
    connected: boolean;
    lastHealthCheck?: Date | undefined;
    lastError?: string | undefined;
}
type GrpcServiceDefinition = grpc.ServiceClientConstructor;
export interface ProtoGrpcType {
    workspace_daemon: {
        SystemService: GrpcServiceDefinition;
        ProjectService: GrpcServiceDefinition;
        DocumentService: GrpcServiceDefinition;
        EmbeddingService: GrpcServiceDefinition;
        TextSearchService: GrpcServiceDefinition;
        GraphService: GrpcServiceDefinition;
        QueueWriteService: GrpcServiceDefinition;
        TrackingWriteService: GrpcServiceDefinition;
    };
}
/**
 * Generic gRPC unary call wrapper — turns callback-style into Promise.
 * The `method` must be a function(request, callback) on the service client.
 */
export declare function grpcUnary<TReq, TRes>(client: unknown, methodName: string, request: TReq): Promise<TRes>;
/**
 * gRPC unary call with a hard wall-clock timeout.
 *
 * Races the underlying `grpcUnary` promise against a timer.  The `finally`
 * block clears the timer so no leak occurs when the call resolves first.
 *
 * @param operationName - Human-readable label used in the rejection message
 *   (falls back to `methodName` when omitted).
 */
export declare function grpcUnaryWithTimeout<TReq, TRes>(client: unknown, methodName: string, request: TReq, timeoutMs: number, operationName?: string): Promise<TRes>;
export declare class DaemonClientBase {
    protected readonly host: string;
    protected readonly port: number;
    protected readonly timeoutMs: number;
    protected readonly maxRetries: number;
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
    protected methodTimeouts: Record<string, number>;
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
    protected getMethodTimeout(methodName: string, override?: number): number;
    protected systemClient?: SystemServiceClient;
    protected projectClient?: ProjectServiceClient;
    protected documentClient?: DocumentServiceClient;
    protected embeddingClient?: EmbeddingServiceClient;
    protected textSearchClient?: TextSearchServiceClient;
    protected graphClient?: GraphServiceClient;
    protected queueWriteClient?: QueueWriteServiceClient;
    protected trackingWriteClient?: TrackingWriteServiceClient;
    protected connectionState: ConnectionState;
    /**
     * Reentrancy guard: prevents `ensureConnected()` from recursing back
     * through `healthCheck()` while `connect()` is still in flight.
     * Without it the auto-reconnect path in `callWithRetry` bounces
     * between connect → healthCheck → callWithRetry → ensureConnected →
     * connect and blows the call stack. See issue #55.
     */
    private connecting;
    constructor(config?: DaemonClientConfig);
    getConnectionState(): ConnectionState;
    isConnected(): boolean;
    /** Instantiate all gRPC service clients from the loaded proto. */
    private instantiateClients;
    connect(): Promise<void>;
    close(): void;
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
    protected ensureConnected(): Promise<void>;
    protected callWithRetry<T>(fn: () => Promise<T>): Promise<T>;
    protected isRetryableError(error: Error): boolean;
}
export {};
//# sourceMappingURL=connection.d.ts.map