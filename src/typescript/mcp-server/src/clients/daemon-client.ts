/**
 * gRPC client for communicating with the Rust daemon (memexd).
 *
 * Provides type-safe wrappers around daemon RPC methods with automatic
 * retry (exponential backoff), connection health monitoring, and timeouts.
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
  SystemStatusResponse,
  MetricsResponse,
  RegisterProjectRequest,
  RegisterProjectResponse,
  DeprioritizeProjectRequest,
  DeprioritizeProjectResponse,
  HeartbeatRequest,
  HeartbeatResponse,
  IngestTextRequest,
  IngestTextResponse,
  ServerState,
  ServerStatusNotification,
  EmbedTextRequest,
  EmbedTextResponse,
  SparseVectorRequest,
  SparseVectorResponse,
  TextSearchRequest,
  TextSearchResponse,
  TextSearchCountResponse,
  QueryRelatedRequest,
  QueryRelatedResponse,
  EnqueueItemRequest,
  EnqueueItemResponse,
  LogSearchEventRequest,
  UpdateSearchEventRequest,
  UpsertRuleMirrorRequest,
  DeleteRuleMirrorRequest,
  UpsertScratchpadMirrorRequest,
  DeleteScratchpadMirrorRequest,
} from './grpc-types.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const PROTO_PATH = join(__dirname, '..', 'proto', 'workspace_daemon.proto');
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

interface ProtoGrpcType {
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
function grpcUnary<TReq, TRes>(client: unknown, methodName: string, request: TReq): Promise<TRes> {
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

export class DaemonClient {
  private readonly host: string;
  private readonly port: number;
  private readonly timeoutMs: number;
  private readonly maxRetries: number;

  private systemClient?: SystemServiceClient;
  private projectClient?: ProjectServiceClient;
  private documentClient?: DocumentServiceClient;
  private embeddingClient?: EmbeddingServiceClient;
  private textSearchClient?: TextSearchServiceClient;
  private graphClient?: GraphServiceClient;
  private queueWriteClient?: QueueWriteServiceClient;
  private trackingWriteClient?: TrackingWriteServiceClient;

  private connectionState: ConnectionState = { connected: false };
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

  async connect(): Promise<void> {
    const packageDefinition = protoLoader.loadSync(PROTO_PATH, {
      keepCase: true,
      longs: String,
      enums: Number,
      defaults: true,
      oneofs: true,
      includeDirs: [join(__dirname, '..', 'proto')],
    });

    const proto = grpc.loadPackageDefinition(packageDefinition) as unknown as ProtoGrpcType;
    const address = `${this.host}:${this.port}`;
    const credentials = grpc.credentials.createInsecure();

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

    try {
      await this.healthCheck();
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

  // ── SystemService ──

  async healthCheck(): Promise<HealthCheckResponse> {
    return this.callWithRetry(
      () =>
        new Promise<HealthCheckResponse>((resolve, reject) => {
          if (!this.systemClient) {
            reject(new Error('Client not connected'));
            return;
          }
          const deadline = new Date(Date.now() + this.timeoutMs);
          (this.systemClient as unknown as grpc.Client).waitForReady(deadline, (err) => {
            if (err) {
              reject(err);
              return;
            }
            this.systemClient!.health({}, (error, response) => {
              if (error) reject(error);
              else resolve(response);
            });
          });
        })
    );
  }

  async getStatus(): Promise<SystemStatusResponse> {
    return this.callWithRetry(() => grpcUnary(this.systemClient, 'getStatus', {}));
  }

  async getMetrics(): Promise<MetricsResponse> {
    return this.callWithRetry(() => grpcUnary(this.systemClient, 'getMetrics', {}));
  }

  async notifyServerStatus(
    state: ServerState,
    projectName?: string,
    projectRoot?: string
  ): Promise<void> {
    const notification: ServerStatusNotification = { state };
    if (projectName !== undefined) notification.project_name = projectName;
    if (projectRoot !== undefined) notification.project_root = projectRoot;
    return this.callWithRetry(() =>
      grpcUnary(this.systemClient, 'notifyServerStatus', notification)
    );
  }

  // ── ProjectService ──

  async registerProject(request: RegisterProjectRequest): Promise<RegisterProjectResponse> {
    return this.callWithRetry(() => grpcUnary(this.projectClient, 'registerProject', request));
  }

  async deprioritizeProject(
    request: DeprioritizeProjectRequest
  ): Promise<DeprioritizeProjectResponse> {
    return this.callWithRetry(() => grpcUnary(this.projectClient, 'deprioritizeProject', request));
  }

  async heartbeat(request: HeartbeatRequest): Promise<HeartbeatResponse> {
    return this.callWithRetry(() => grpcUnary(this.projectClient, 'heartbeat', request));
  }

  // ── DocumentService ──

  async ingestText(request: IngestTextRequest): Promise<IngestTextResponse> {
    return this.callWithRetry(() => grpcUnary(this.documentClient, 'ingestText', request));
  }

  // ── EmbeddingService ──

  async embedText(request: EmbedTextRequest): Promise<EmbedTextResponse> {
    return this.callWithRetry(() => grpcUnary(this.embeddingClient, 'embedText', request));
  }

  async generateSparseVector(request: SparseVectorRequest): Promise<SparseVectorResponse> {
    return this.callWithRetry(() =>
      grpcUnary(this.embeddingClient, 'generateSparseVector', request)
    );
  }

  // ── TextSearchService ──

  async textSearch(request: TextSearchRequest): Promise<TextSearchResponse> {
    return this.callWithRetry(() => grpcUnary(this.textSearchClient, 'search', request));
  }

  async textSearchCount(request: TextSearchRequest): Promise<TextSearchCountResponse> {
    return this.callWithRetry(() => grpcUnary(this.textSearchClient, 'countMatches', request));
  }

  // ── GraphService ──

  async queryRelated(request: QueryRelatedRequest): Promise<QueryRelatedResponse> {
    return this.callWithRetry(() => grpcUnary(this.graphClient, 'queryRelated', request));
  }

  // ── QueueWriteService ──

  async enqueueItem(request: EnqueueItemRequest): Promise<EnqueueItemResponse> {
    return this.callWithRetry(() => grpcUnary(this.queueWriteClient, 'enqueueItem', request));
  }

  // ── TrackingWriteService ──

  async logSearchEvent(request: LogSearchEventRequest): Promise<void> {
    return this.callWithRetry(() => grpcUnary(this.trackingWriteClient, 'logSearchEvent', request));
  }

  async updateSearchEvent(request: UpdateSearchEventRequest): Promise<void> {
    return this.callWithRetry(() =>
      grpcUnary(this.trackingWriteClient, 'updateSearchEvent', request)
    );
  }

  async upsertRuleMirror(request: UpsertRuleMirrorRequest): Promise<void> {
    return this.callWithRetry(() =>
      grpcUnary(this.trackingWriteClient, 'upsertRuleMirror', request)
    );
  }

  async deleteRuleMirror(request: DeleteRuleMirrorRequest): Promise<void> {
    return this.callWithRetry(() =>
      grpcUnary(this.trackingWriteClient, 'deleteRuleMirror', request)
    );
  }

  async upsertScratchpadMirror(request: UpsertScratchpadMirrorRequest): Promise<void> {
    return this.callWithRetry(() =>
      grpcUnary(this.trackingWriteClient, 'upsertScratchpadMirror', request)
    );
  }

  async deleteScratchpadMirror(request: DeleteScratchpadMirrorRequest): Promise<void> {
    return this.callWithRetry(() =>
      grpcUnary(this.trackingWriteClient, 'deleteScratchpadMirror', request)
    );
  }

  // ── Retry logic ──

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
  private async ensureConnected(): Promise<void> {
    if (this.connecting) return; // connect() is in flight — let it finish
    if (this.connectionState.connected && this.systemClient) return;
    this.connecting = true;
    try {
      await this.connect();
    } finally {
      this.connecting = false;
    }
  }

  private async callWithRetry<T>(fn: () => Promise<T>): Promise<T> {
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

  private isRetryableError(error: Error): boolean {
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

// Re-export types for convenience
export { ServiceStatus } from './grpc-types.js';
export type {
  HealthCheckResponse,
  SystemStatusResponse,
  MetricsResponse,
  RegisterProjectRequest,
  RegisterProjectResponse,
  DeprioritizeProjectRequest,
  DeprioritizeProjectResponse,
  HeartbeatRequest,
  HeartbeatResponse,
  IngestTextRequest,
  IngestTextResponse,
  EmbedTextRequest,
  EmbedTextResponse,
  SparseVectorRequest,
  SparseVectorResponse,
  TextSearchRequest,
  TextSearchResponse,
  TextSearchCountResponse,
  TextSearchMatch,
  QueryRelatedRequest,
  QueryRelatedResponse,
  TraversalNodeProto,
  EnqueueItemRequest,
  EnqueueItemResponse,
  LogSearchEventRequest,
  UpdateSearchEventRequest,
  UpsertRuleMirrorRequest,
  DeleteRuleMirrorRequest,
  UpsertScratchpadMirrorRequest,
  DeleteScratchpadMirrorRequest,
} from './grpc-types.js';
