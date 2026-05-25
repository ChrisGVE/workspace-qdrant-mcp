/**
 * DaemonClientSystem — SystemService and ProjectService RPC methods.
 */

import * as grpc from '@grpc/grpc-js';

import type {
  HealthCheckResponse,
  SystemStatusResponse,
  MetricsResponse,
  GetEmbeddingProviderStatusResponse,
  ServerState,
  ServerStatusNotification,
  RegisterProjectRequest,
  RegisterProjectResponse,
  DeprioritizeProjectRequest,
  DeprioritizeProjectResponse,
  HeartbeatRequest,
  HeartbeatResponse,
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
}
