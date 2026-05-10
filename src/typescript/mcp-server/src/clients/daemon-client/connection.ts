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
  HealthCheckResponse,
} from '../grpc-types.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

export const PROTO_PATH = join(__dirname, '..', '..', 'proto', 'workspace_daemon.proto');
const DEFAULT_HOST = 'localhost';
const DEFAULT_PORT = 50051;
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

export class DaemonClientBase {
  protected readonly host: string;
  protected readonly port: number;
  protected readonly timeoutMs: number;
  protected readonly maxRetries: number;

  protected systemClient?: SystemServiceClient;
  protected projectClient?: ProjectServiceClient;
  protected documentClient?: DocumentServiceClient;
  protected embeddingClient?: EmbeddingServiceClient;
  protected textSearchClient?: TextSearchServiceClient;
  protected graphClient?: GraphServiceClient;
  protected queueWriteClient?: QueueWriteServiceClient;
  protected trackingWriteClient?: TrackingWriteServiceClient;

  protected connectionState: ConnectionState = { connected: false };

  /**
   * Reentrancy guard: prevents `ensureConnected()` from recursing back
   * through `healthCheck()` while `connect()` is still in flight.
   * Without it the auto-reconnect path in `callWithRetry` bounces
   * between connect → healthCheck → callWithRetry → ensureConnected →
   * connect and blows the call stack. See issue #55.
   */
  private connecting = false;

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
      if (c) grpc.closeClient(c as unknown as grpc.Client);
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
    throw lastError;
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
