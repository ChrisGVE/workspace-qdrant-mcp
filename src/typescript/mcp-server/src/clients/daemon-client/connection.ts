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

import type {
  SystemServiceClient,
  ProjectServiceClient,
  DocumentServiceClient,
  EmbeddingServiceClient,
  TextSearchServiceClient,
  GraphServiceClient,
  QueueWriteServiceClient,
  TrackingWriteServiceClient,
  WatchWriteServiceClient,
  AdminWriteServiceClient,
  HealthCheckResponse,
} from '../grpc-types.js';
import { DEFAULT_CONFIG } from '../../types/generated-defaults.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

export const PROTO_PATH = join(__dirname, '..', '..', 'proto', 'workspace_daemon.proto');
// Daemon host/port defaults come from assets/default_configuration.yaml via
// generated-defaults.ts — do not declare local copies (drift risk).
const DEFAULT_HOST = DEFAULT_CONFIG.daemon.grpcHost;
const DEFAULT_PORT = DEFAULT_CONFIG.daemon.grpcPort;
const DEFAULT_TIMEOUT_MS = 5000;
const MAX_RETRIES = 3;
const INITIAL_RETRY_DELAY_MS = 100;

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
    WatchWriteService: GrpcServiceDefinition;
    AdminWriteService: GrpcServiceDefinition;
  };
}

/**
 * Generic gRPC unary call wrapper — turns callback-style into Promise.
 * The `method` must be a function(request, callback) on the service client.
 */
export function grpcUnary<TReq, TRes>(
  client: unknown,
  methodName: string,
  request: TReq
): Promise<TRes> {
  return new Promise<TRes>((resolve, reject) => {
    if (!client) {
      reject(new Error('Client not connected'));
      return;
    }
    const fn = (client as Record<string, unknown>)[methodName];
    if (typeof fn !== 'function') {
      reject(new Error(`Unknown method: ${methodName}`));
      return;
    }
    (fn as (req: TReq, cb: (err: Error | null, res: TRes) => void) => void).call(
      client,
      request,
      (error, response) => {
        if (error) reject(error);
        else resolve(response);
      }
    );
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
export async function grpcUnaryWithTimeout<TReq, TRes>(
  client: unknown,
  methodName: string,
  request: TReq,
  timeoutMs: number,
  operationName?: string
): Promise<TRes> {
  const label = operationName ?? methodName;
  let timer: ReturnType<typeof setTimeout> | undefined;
  const timeoutPromise = new Promise<TRes>((_, reject) => {
    timer = setTimeout(() => {
      reject(new Error(`gRPC call ${label} timed out after ${timeoutMs}ms`));
    }, timeoutMs);
  });
  try {
    return await Promise.race([grpcUnary<TReq, TRes>(client, methodName, request), timeoutPromise]);
  } finally {
    if (timer !== undefined) clearTimeout(timer);
  }
}

export class DaemonClientBase {
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
  protected methodTimeouts: Record<string, number> = {};

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
  protected getMethodTimeout(methodName: string, override?: number): number {
    if (typeof override === 'number') return override;
    const explicit = this.methodTimeouts[methodName];
    if (typeof explicit === 'number') return explicit;
    if (methodName === 'search') return this.timeoutMs * 2;
    return this.timeoutMs;
  }

  protected systemClient?: SystemServiceClient;
  protected projectClient?: ProjectServiceClient;
  protected documentClient?: DocumentServiceClient;
  protected embeddingClient?: EmbeddingServiceClient;
  protected textSearchClient?: TextSearchServiceClient;
  protected graphClient?: GraphServiceClient;
  protected queueWriteClient?: QueueWriteServiceClient;
  protected trackingWriteClient?: TrackingWriteServiceClient;
  protected watchWriteClient?: WatchWriteServiceClient;
  protected adminWriteClient?: AdminWriteServiceClient;

  protected connectionState: ConnectionState = { connected: false };

  /**
   * Reentrancy guard: prevents `ensureConnected()` from recursing back
   * through `healthCheck()` while `connect()` is still in flight.
   * Without it the auto-reconnect path in `callWithRetry` bounces
   * between connect → healthCheck → callWithRetry → ensureConnected →
   * connect and blows the call stack. See issue #55.
   */
  private connecting = false;

  /**
   * Set once `close()` runs. Stops the auto-reconnect machinery from
   * resurrecting a torn-down client (e.g. a stray RPC racing session cleanup
   * after a daemon restart), which otherwise re-enters connect → healthCheck →
   * callWithRetry and can recurse/throw during teardown.
   */
  private closed = false;

  constructor(config: DaemonClientConfig = {}) {
    this.host = config.host ?? DEFAULT_HOST;
    this.port = config.port ?? DEFAULT_PORT;
    this.timeoutMs = config.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    this.maxRetries = config.maxRetries ?? MAX_RETRIES;
  }

  getConnectionState(): ConnectionState {
    return { ...this.connectionState };
  }

  isConnected(): boolean {
    return this.connectionState.connected;
  }

  /** Instantiate all gRPC service clients from the loaded proto. */
  private instantiateClients(
    proto: ProtoGrpcType,
    address: string,
    credentials: grpc.ChannelCredentials
  ): void {
    this.systemClient = new proto.workspace_daemon.SystemService(
      address,
      credentials
    ) as unknown as SystemServiceClient;
    this.projectClient = new proto.workspace_daemon.ProjectService(
      address,
      credentials
    ) as unknown as ProjectServiceClient;
    this.documentClient = new proto.workspace_daemon.DocumentService(
      address,
      credentials
    ) as unknown as DocumentServiceClient;
    this.embeddingClient = new proto.workspace_daemon.EmbeddingService(
      address,
      credentials
    ) as unknown as EmbeddingServiceClient;
    this.textSearchClient = new proto.workspace_daemon.TextSearchService(
      address,
      credentials
    ) as unknown as TextSearchServiceClient;
    this.graphClient = new proto.workspace_daemon.GraphService(
      address,
      credentials
    ) as unknown as GraphServiceClient;
    this.queueWriteClient = new proto.workspace_daemon.QueueWriteService(
      address,
      credentials
    ) as unknown as QueueWriteServiceClient;
    this.trackingWriteClient = new proto.workspace_daemon.TrackingWriteService(
      address,
      credentials
    ) as unknown as TrackingWriteServiceClient;
    this.watchWriteClient = new proto.workspace_daemon.WatchWriteService(
      address,
      credentials
    ) as unknown as WatchWriteServiceClient;
    this.adminWriteClient = new proto.workspace_daemon.AdminWriteService(
      address,
      credentials
    ) as unknown as AdminWriteServiceClient;
  }

  async connect(): Promise<void> {
    const packageDefinition = protoLoader.loadSync(PROTO_PATH, {
      keepCase: true,
      longs: String,
      enums: Number,
      defaults: true,
      oneofs: true,
      includeDirs: [join(__dirname, '..', '..', 'proto')],
    });

    const proto = grpc.loadPackageDefinition(packageDefinition) as unknown as ProtoGrpcType;
    const address = `${this.host}:${this.port}`;
    const credentials = grpc.credentials.createInsecure();

    this.instantiateClients(proto, address, credentials);

    try {
      await (this as unknown as { healthCheck(): Promise<HealthCheckResponse> }).healthCheck();
      this.connectionState = { connected: true, lastHealthCheck: new Date() };
    } catch (error) {
      this.connectionState = {
        connected: false,
        lastError: error instanceof Error ? error.message : 'Unknown error',
      };
      throw error;
    }
  }

  close(): void {
    this.closed = true;
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
      // Best-effort: closing an already-broken channel can throw; teardown must
      // never propagate (it runs from session cleanup / onclose).
      if (c) {
        try {
          grpc.closeClient(c as unknown as grpc.Client);
        } catch {
          /* ignore — channel already gone */
        }
      }
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
  protected async ensureConnected(): Promise<void> {
    if (this.closed) return; // torn down — do not resurrect during/after cleanup
    if (this.connecting) return; // connect() is in flight — let it finish
    if (this.connectionState.connected && this.systemClient) return;
    this.connecting = true;
    try {
      await this.connect();
    } finally {
      this.connecting = false;
    }
  }

  protected async callWithRetry<T>(fn: () => Promise<T>): Promise<T> {
    let lastError: Error | undefined;
    let delay = INITIAL_RETRY_DELAY_MS;

    for (let attempt = 0; attempt < this.maxRetries; attempt++) {
      try {
        await this.ensureConnected();
        const result = await fn();
        this.connectionState = { connected: true, lastHealthCheck: new Date() };
        return result;
      } catch (error) {
        lastError = error instanceof Error ? error : new Error(String(error));
        if (!this.isRetryableError(lastError)) throw lastError;
        // A transient failure invalidates our notion of being connected —
        // the next iteration's `ensureConnected` will reconnect.
        this.connectionState = { connected: false, lastError: lastError.message };
        if (attempt < this.maxRetries - 1) {
          await new Promise<void>((r) => setTimeout(r, delay));
          delay *= 2;
        }
      }
    }

    this.connectionState = { connected: false, lastError: lastError?.message };
    throw lastError ?? new Error('daemon RPC failed after retries');
  }

  protected isRetryableError(error: Error): boolean {
    const retryableCodes = [
      grpc.status.UNAVAILABLE,
      grpc.status.DEADLINE_EXCEEDED,
      grpc.status.RESOURCE_EXHAUSTED,
    ];
    const grpcError = error as { code?: number };
    if (typeof grpcError.code === 'number') return retryableCodes.includes(grpcError.code);
    return (
      error.message.includes('ECONNREFUSED') ||
      error.message.includes('ETIMEDOUT') ||
      error.message.includes('ENOTFOUND') ||
      // Issue #55: retry when the local client handle is stale (never
      // connected, closed, or a dropped channel) so the next iteration's
      // ensureConnected() can heal the connection.
      error.message.includes('Client not connected') ||
      error.message.includes('channel has been closed') ||
      error.message.includes('Channel has been shut down')
    );
  }
}
