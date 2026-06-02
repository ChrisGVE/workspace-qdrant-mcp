/**
 * DaemonClientSystem — SystemService and ProjectService RPC methods.
 */

import * as grpc from '@grpc/grpc-js';

import type {
  HealthCheckResponse,
  SystemStatusResponse,
  MetricsResponse,
  QueueStatsResponse,
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
  ListWatchesRequest,
  ListWatchesResponse,
  ListFailedItemsRequest,
  ListFailedItemsResponse,
} from '../grpc-types.js';

import { DaemonClientBase, grpcUnaryWithTimeout } from './connection.js';

/** Coerce a proto `int64` (decoded as a string by the gRPC client) to a number. */
function int64ToNumber(value: unknown, fallback = 0): number {
  if (typeof value === 'number') return Number.isFinite(value) ? value : fallback;
  if (typeof value === 'string' && value.trim() !== '') {
    const n = Number(value);
    return Number.isFinite(n) ? n : fallback;
  }
  return fallback;
}

/** Like {@link int64ToNumber} but preserves "absent" — used for the optional ETA. */
function optionalInt64ToNumber(value: unknown): number | undefined {
  if (value === undefined || value === null || value === '') return undefined;
  const n = typeof value === 'number' ? value : Number(value);
  return Number.isFinite(n) ? n : undefined;
}

/**
 * Normalize the int64 indexing-progress counts of a `GetProjectStatusResponse`
 * from gRPC's string encoding into real numbers, leaving every other field
 * untouched. Without this the ETA never renders (its `typeof === 'number'`
 * guard rejects the string) and count arithmetic concatenates.
 */
function normalizeProjectStatusCounts(
  resp: GetProjectStatusResponse
): GetProjectStatusResponse {
  // Drop eta_seconds from the spread so we never assign it `undefined`
  // explicitly — `exactOptionalPropertyTypes` forbids that for an optional
  // field; absence must omit the key entirely.
  const { eta_seconds, ...rest } = resp;
  const eta = optionalInt64ToNumber(eta_seconds);
  const normalized: GetProjectStatusResponse = {
    ...rest,
    pending_count: int64ToNumber(resp.pending_count),
    in_progress_count: int64ToNumber(resp.in_progress_count),
    failed_count: int64ToNumber(resp.failed_count),
    done_count: int64ToNumber(resp.done_count),
    total_count: int64ToNumber(resp.total_count),
    percent_complete: int64ToNumber(resp.percent_complete, 100),
  };
  if (eta !== undefined) normalized.eta_seconds = eta;
  return normalized;
}

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

  async getQueueStats(): Promise<QueueStatsResponse> {
    return this.callWithRetry(() =>
      grpcUnaryWithTimeout(
        this.systemClient,
        'getQueueStats',
        {},
        this.getMethodTimeout('getQueueStats')
      )
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
    const resp = await this.callWithRetry(() =>
      grpcUnaryWithTimeout(
        this.projectClient,
        'getProjectStatus',
        request,
        this.getMethodTimeout('getProjectStatus')
      )
    );
    // The gRPC client decodes proto `int64` fields as STRINGS (to avoid JS
    // number-precision loss). The whole indexing-progress block is int64, so
    // coerce it to real numbers here — otherwise consumers' `typeof === 'number'`
    // guards reject the ETA (perpetual "warming up") and `pending + in_progress`
    // string-concatenates instead of adding (e.g. 1 + 0 -> "10" in flight).
    return normalizeProjectStatusCounts(resp as GetProjectStatusResponse);
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

  /**
   * List watched folders via the daemon's `ListWatches` RPC. Read-only gRPC
   * equivalent of `wqm watch list` — lets the dockerized MCP server enumerate
   * watches without a local wqm binary.
   */
  async listWatches(request: ListWatchesRequest = {}): Promise<ListWatchesResponse> {
    return this.callWithRetry(() =>
      grpcUnaryWithTimeout(
        this.projectClient,
        'listWatches',
        request,
        this.getMethodTimeout('listWatches')
      )
    );
  }

  /**
   * List queue items in the 'failed' state via the daemon's `ListFailedItems`
   * RPC. Read-only; backs the admin UI's failed-items drill-down. Retry is a
   * separate QueueWriteService mutation (`retryAll` / `retryItem`).
   */
  async listFailedItems(
    request: ListFailedItemsRequest = {}
  ): Promise<ListFailedItemsResponse> {
    return this.callWithRetry(() =>
      grpcUnaryWithTimeout(
        this.projectClient,
        'listFailedItems',
        request,
        this.getMethodTimeout('listFailedItems')
      )
    );
  }
}
