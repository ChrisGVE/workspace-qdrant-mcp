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

import { DaemonClientBase, grpcUnary } from './connection.js';

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

  async getEmbeddingProviderStatus(): Promise<GetEmbeddingProviderStatusResponse> {
    return this.callWithRetry(() => grpcUnary(this.systemClient, 'getEmbeddingProviderStatus', {}));
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
}
