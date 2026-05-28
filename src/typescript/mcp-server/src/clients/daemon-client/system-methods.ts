/**
 * DaemonClientSystem — SystemService and ProjectService RPC methods.
 */

import * as grpc from '@grpc/grpc-js';

import type {
  HealthCheckResponse,
  SystemStatusResponse,
  MetricsResponse,
  GetEmbeddingProviderStatusResponse,
  RebuildIndexRequest,
  RebuildIndexResponse,
  ServerState,
  ServerStatusNotification,
  RegisterProjectRequest,
  RegisterProjectResponse,
  DeprioritizeProjectRequest,
  DeprioritizeProjectResponse,
  GetProjectStatusRequest,
  GetProjectStatusResponse,
  HeartbeatRequest,
  HeartbeatResponse,
  ListProjectsRequest,
  ListProjectsResponse,
} from '../grpc-types.js';

import { DaemonClientBase, grpcUnaryWithTimeout } from './connection.js';

export class DaemonClientSystem extends DaemonClientBase {
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
            grpcUnaryWithTimeout<{}, HealthCheckResponse>(
              this.systemClient,
              'health',
              {},
              this.timeoutMs,
              'health'
            ).then(resolve, reject);
          });
        })
    );
  }

  async getStatus(): Promise<SystemStatusResponse> {
    return this.callWithRetry(() =>
      grpcUnaryWithTimeout(this.systemClient, 'getStatus', {}, this.getMethodTimeout('getStatus'))
    );
  }

  async getMetrics(): Promise<MetricsResponse> {
    return this.callWithRetry(() =>
      grpcUnaryWithTimeout(this.systemClient, 'getMetrics', {}, this.getMethodTimeout('getMetrics'))
    );
  }

  async getEmbeddingProviderStatus(): Promise<GetEmbeddingProviderStatusResponse> {
    return this.callWithRetry(() =>
      grpcUnaryWithTimeout(
        this.systemClient,
        'getEmbeddingProviderStatus',
        {},
        this.getMethodTimeout('getEmbeddingProviderStatus')
      )
    );
  }

  /**
   * Rebuild computed indexes (FTS5, tags, sparse vectors, components,
   * keywords) for one tenant. Recomputes from already-indexed content —
   * does not re-read files or regenerate dense embeddings. Uses a longer
   * ceiling than the 5s default since a full per-project rebuild is heavier.
   */
  async rebuildIndex(request: RebuildIndexRequest): Promise<RebuildIndexResponse> {
    return this.callWithRetry(() =>
      grpcUnaryWithTimeout(this.systemClient, 'rebuildIndex', request, 60_000, 'rebuildIndex')
    );
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
      grpcUnaryWithTimeout(
        this.systemClient,
        'notifyServerStatus',
        notification,
        this.getMethodTimeout('notifyServerStatus')
      )
    );
  }

  // ── ProjectService ──

  async registerProject(request: RegisterProjectRequest): Promise<RegisterProjectResponse> {
    return this.callWithRetry(() =>
      grpcUnaryWithTimeout(
        this.projectClient,
        'registerProject',
        request,
        this.getMethodTimeout('registerProject')
      )
    );
  }

  async deprioritizeProject(
    request: DeprioritizeProjectRequest
  ): Promise<DeprioritizeProjectResponse> {
    return this.callWithRetry(() =>
      grpcUnaryWithTimeout(
        this.projectClient,
        'deprioritizeProject',
        request,
        this.getMethodTimeout('deprioritizeProject')
      )
    );
  }

  async heartbeat(request: HeartbeatRequest): Promise<HeartbeatResponse> {
    return this.callWithRetry(() =>
      grpcUnaryWithTimeout(
        this.projectClient,
        'heartbeat',
        request,
        this.getMethodTimeout('heartbeat')
      )
    );
  }

  /**
   * Fetch project status — registration metadata plus per-project indexing
   * counts (pending / in_progress / failed / done / total / percent_complete).
   *
   * Drives the `indexing` block on `SearchResponse` and the `indexing_status`
   * action on the `workspace_index` MCP tool. Cheap call (two COUNT queries
   * on indexed tables), but `search-helpers` caches the result with a short
   * TTL anyway since it can fire on every tool invocation.
   */
  async getProjectStatus(request: GetProjectStatusRequest): Promise<GetProjectStatusResponse> {
    return this.callWithRetry(() =>
      grpcUnaryWithTimeout(
        this.projectClient,
        'getProjectStatus',
        request,
        this.getMethodTimeout('getProjectStatus')
      )
    );
  }

  /**
   * List registered projects via the daemon's `ListProjects` RPC.
   *
   * Prefer this over `SqliteStateManager.listAllProjects()` when the MCP
   * server runs in a different container/host from the daemon — the
   * daemon owns the SQLite file and reading it through a bind-mount on
   * Docker Desktop fails with `SQLITE_CANTOPEN` because the 9P fs does
   * not implement the shared-memory locks SQLite needs to coordinate
   * with the writer.
   */
  async listProjects(request: ListProjectsRequest = {}): Promise<ListProjectsResponse> {
    return this.callWithRetry(() =>
      grpcUnaryWithTimeout(
        this.projectClient,
        'listProjects',
        request,
        this.getMethodTimeout('listProjects')
      )
    );
  }
}
