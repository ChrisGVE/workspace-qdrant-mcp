/**
 * gRPC client for communicating with the Rust daemon (memexd)
 *
 * Provides type-safe wrappers around daemon RPC methods with:
 * - Automatic retry with exponential backoff
 * - Connection health monitoring
 * - Promise-based API
 * - Timeout handling
 */

import * as grpc from '@grpc/grpc-js';
import * as protoLoader from '@grpc/proto-loader';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import type {
  SystemServiceClient,
  ProjectServiceClient,
  DocumentServiceClient,
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
} from './grpc-types.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Proto file location
const PROTO_PATH = join(__dirname, '..', 'proto', 'workspace_daemon.proto');

// Default configuration
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
  };
}

/**
 * Daemon client for gRPC communication with memexd
 *
 * Usage:
 * ```typescript
 * const client = new DaemonClient({ port: 50051 });
 * await client.connect();
 *
 * const health = await client.healthCheck();
 * if (health.status === ServiceStatus.SERVICE_STATUS_HEALTHY) {
 *   await client.registerProject({ path: '/my/project', project_id: 'abc123' });
 * }
 *
 * await client.close();
 * ```
 */
export class DaemonClient {
  private readonly host: string;
  private readonly port: number;
  private readonly timeoutMs: number;
  private readonly maxRetries: number;

  private systemClient?: SystemServiceClient;
  private projectClient?: ProjectServiceClient;
  private documentClient?: DocumentServiceClient;

  private connectionState: ConnectionState = { connected: false };

  constructor(config: DaemonClientConfig = {}) {
    this.host = config.host ?? DEFAULT_HOST;
    this.port = config.port ?? DEFAULT_PORT;
    this.timeoutMs = config.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    this.maxRetries = config.maxRetries ?? MAX_RETRIES;
  }

  /**
   * Get current connection state
   */
  getConnectionState(): ConnectionState {
    return { ...this.connectionState };
  }

  /**
   * Check if connected to daemon
   */
  isConnected(): boolean {
    return this.connectionState.connected;
  }

  /**
   * Connect to the daemon
   */
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

    // Create service clients
    const SystemService = proto.workspace_daemon.SystemService;
    const ProjectService = proto.workspace_daemon.ProjectService;
    const DocumentService = proto.workspace_daemon.DocumentService;

    this.systemClient = new SystemService(
      address,
      credentials
    ) as unknown as SystemServiceClient;
    this.projectClient = new ProjectService(
      address,
      credentials
    ) as unknown as ProjectServiceClient;
    this.documentClient = new DocumentService(
      address,
      credentials
    ) as unknown as DocumentServiceClient;

    // Test connection with health check
    try {
      await this.healthCheck();
      this.connectionState = {
        connected: true,
        lastHealthCheck: new Date(),
      };
    } catch (error) {
      this.connectionState = {
        connected: false,
        lastError: error instanceof Error ? error.message : 'Unknown error',
      };
      throw error;
    }
  }

  /**
   * Close the connection
   */
  close(): void {
    if (this.systemClient) {
      grpc.closeClient(this.systemClient as unknown as grpc.Client);
    }
    if (this.projectClient) {
      grpc.closeClient(this.projectClient as unknown as grpc.Client);
    }
    if (this.documentClient) {
      grpc.closeClient(this.documentClient as unknown as grpc.Client);
    }
    this.connectionState = { connected: false };
  }

  // ============================================================================
  // SystemService Methods
  // ============================================================================

  /**
   * Check daemon health
   */
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
            this.systemClient!.healthCheck({}, (error, response) => {
              if (error) reject(error);
              else resolve(response);
            });
          });
        })
    );
  }

  /**
   * Get comprehensive system status
   */
  async getStatus(): Promise<SystemStatusResponse> {
    return this.callWithRetry(
      () =>
        new Promise<SystemStatusResponse>((resolve, reject) => {
          if (!this.systemClient) {
            reject(new Error('Client not connected'));
            return;
          }
          this.systemClient.getStatus({}, (error, response) => {
            if (error) reject(error);
            else resolve(response);
          });
        })
    );
  }

  /**
   * Get system metrics
   */
  async getMetrics(): Promise<MetricsResponse> {
    return this.callWithRetry(
      () =>
        new Promise<MetricsResponse>((resolve, reject) => {
          if (!this.systemClient) {
            reject(new Error('Client not connected'));
            return;
          }
          this.systemClient.getMetrics({}, (error, response) => {
            if (error) reject(error);
            else resolve(response);
          });
        })
    );
  }

  /**
   * Notify daemon of server status (UP/DOWN)
   */
  async notifyServerStatus(
    state: ServerState,
    projectName?: string,
    projectRoot?: string
  ): Promise<void> {
    return this.callWithRetry(
      () =>
        new Promise<void>((resolve, reject) => {
          if (!this.systemClient) {
            reject(new Error('Client not connected'));
            return;
          }
          const notification: ServerStatusNotification = { state };
          if (projectName !== undefined) {
            notification.project_name = projectName;
          }
          if (projectRoot !== undefined) {
            notification.project_root = projectRoot;
          }
          this.systemClient.notifyServerStatus(notification, (error) => {
            if (error) reject(error);
            else resolve();
          });
        })
    );
  }

  // ============================================================================
  // ProjectService Methods
  // ============================================================================

  /**
   * Register a project for high-priority processing
   * Called when MCP server starts for a project
   */
  async registerProject(request: RegisterProjectRequest): Promise<RegisterProjectResponse> {
    return this.callWithRetry(
      () =>
        new Promise<RegisterProjectResponse>((resolve, reject) => {
          if (!this.projectClient) {
            reject(new Error('Client not connected'));
            return;
          }
          this.projectClient.registerProject(request, (error, response) => {
            if (error) reject(error);
            else resolve(response);
          });
        })
    );
  }

  /**
   * Deprioritize a project
   * Called when MCP server stops
   */
  async deprioritizeProject(
    request: DeprioritizeProjectRequest
  ): Promise<DeprioritizeProjectResponse> {
    return this.callWithRetry(
      () =>
        new Promise<DeprioritizeProjectResponse>((resolve, reject) => {
          if (!this.projectClient) {
            reject(new Error('Client not connected'));
            return;
          }
          this.projectClient.deprioritizeProject(request, (error, response) => {
            if (error) reject(error);
            else resolve(response);
          });
        })
    );
  }

  /**
   * Send heartbeat to keep session alive
   * Should be called periodically (recommended: every 30s)
   */
  async heartbeat(request: HeartbeatRequest): Promise<HeartbeatResponse> {
    return this.callWithRetry(
      () =>
        new Promise<HeartbeatResponse>((resolve, reject) => {
          if (!this.projectClient) {
            reject(new Error('Client not connected'));
            return;
          }
          this.projectClient.heartbeat(request, (error, response) => {
            if (error) reject(error);
            else resolve(response);
          });
        })
    );
  }

  // ============================================================================
  // DocumentService Methods
  // ============================================================================

  /**
   * Ingest text content directly (synchronous)
   * Use for content not from files: user input, web content, notes
   *
   * Note: Per ADR-002, prefer using the unified queue for writes.
   * This method is provided for admin/diagnostic use.
   */
  async ingestText(request: IngestTextRequest): Promise<IngestTextResponse> {
    return this.callWithRetry(
      () =>
        new Promise<IngestTextResponse>((resolve, reject) => {
          if (!this.documentClient) {
            reject(new Error('Client not connected'));
            return;
          }
          this.documentClient.ingestText(request, (error, response) => {
            if (error) reject(error);
            else resolve(response);
          });
        })
    );
  }

  // ============================================================================
  // EmbeddingService Methods (Stubs - Not implemented in daemon proto)
  // ============================================================================
  // These methods are stubs that throw errors. The search tool has fallback
  // logic that catches these errors and uses alternative search methods.

  /**
   * Generate dense embedding for text
   * @throws Error - EmbeddingService not implemented in daemon
   */
  async embedText(_request: { text: string }): Promise<{
    success: boolean;
    embedding: number[];
    dimensions: number;
    model_name: string;
    error_message?: string;
  }> {
    throw new Error('EmbeddingService not implemented in daemon - use fallback search');
  }

  /**
   * Generate sparse vector using BM25
   * @throws Error - EmbeddingService not implemented in daemon
   */
  async generateSparseVector(_request: { text: string }): Promise<{
    success: boolean;
    indices_values: Record<number, number>;
    vocab_size: number;
    error_message?: string;
  }> {
    throw new Error('EmbeddingService not implemented in daemon - use fallback search');
  }

  // ============================================================================
  // Private Methods
  // ============================================================================

  /**
   * Execute a call with exponential backoff retry
   */
  private async callWithRetry<T>(fn: () => Promise<T>): Promise<T> {
    let lastError: Error | undefined;
    let delay = INITIAL_RETRY_DELAY_MS;

    for (let attempt = 0; attempt < this.maxRetries; attempt++) {
      try {
        const result = await fn();
        // Update connection state on success
        this.connectionState = {
          connected: true,
          lastHealthCheck: new Date(),
        };
        return result;
      } catch (error) {
        lastError = error instanceof Error ? error : new Error(String(error));

        // Check if error is retryable
        if (!this.isRetryableError(lastError)) {
          throw lastError;
        }

        // Wait before retry (exponential backoff)
        if (attempt < this.maxRetries - 1) {
          await this.sleep(delay);
          delay *= 2; // Exponential backoff
        }
      }
    }

    // Update connection state on failure
    this.connectionState = {
      connected: false,
      lastError: lastError?.message,
    };

    throw lastError;
  }

  /**
   * Check if an error is retryable
   */
  private isRetryableError(error: Error): boolean {
    // gRPC error codes that are retryable
    const retryableCodes = [
      grpc.status.UNAVAILABLE,
      grpc.status.DEADLINE_EXCEEDED,
      grpc.status.RESOURCE_EXHAUSTED,
    ];

    // Check if it's a gRPC error with a retryable code
    const grpcError = error as { code?: number };
    if (typeof grpcError.code === 'number') {
      return retryableCodes.includes(grpcError.code);
    }

    // Retry on connection errors
    return (
      error.message.includes('ECONNREFUSED') ||
      error.message.includes('ETIMEDOUT') ||
      error.message.includes('ENOTFOUND')
    );
  }

  /**
   * Sleep for a specified duration
   */
  private sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
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
} from './grpc-types.js';
