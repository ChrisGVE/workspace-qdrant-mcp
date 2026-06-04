/**
 * DaemonClientSystem — SystemService and ProjectService RPC methods.
 */
import * as grpc from '@grpc/grpc-js';
import { DaemonClientBase, grpcUnaryWithTimeout } from './connection.js';
export class DaemonClientSystem extends DaemonClientBase {
    // ── SystemService ──
    async healthCheck() {
        return this.callWithRetry(() => new Promise((resolve, reject) => {
            if (!this.systemClient) {
                reject(new Error('Client not connected'));
                return;
            }
            const deadline = new Date(Date.now() + this.timeoutMs);
            this.systemClient.waitForReady(deadline, (err) => {
                if (err) {
                    reject(err);
                    return;
                }
                this.systemClient.health({}, (error, response) => {
                    if (error)
                        reject(error);
                    else
                        resolve(response);
                });
            });
        }));
    }
    async getStatus() {
        return this.callWithRetry(() => grpcUnaryWithTimeout(this.systemClient, 'getStatus', {}, this.getMethodTimeout('getStatus')));
    }
    async getMetrics() {
        return this.callWithRetry(() => grpcUnaryWithTimeout(this.systemClient, 'getMetrics', {}, this.getMethodTimeout('getMetrics')));
    }
    async getEmbeddingProviderStatus() {
        return this.callWithRetry(() => grpcUnaryWithTimeout(this.systemClient, 'getEmbeddingProviderStatus', {}, this.getMethodTimeout('getEmbeddingProviderStatus')));
    }
    async notifyServerStatus(state, projectName, projectRoot) {
        const notification = { state };
        if (projectName !== undefined)
            notification.project_name = projectName;
        if (projectRoot !== undefined)
            notification.project_root = projectRoot;
        return this.callWithRetry(() => grpcUnaryWithTimeout(this.systemClient, 'notifyServerStatus', notification, this.getMethodTimeout('notifyServerStatus')));
    }
    // ── ProjectService ──
    async registerProject(request) {
        return this.callWithRetry(() => grpcUnaryWithTimeout(this.projectClient, 'registerProject', request, this.getMethodTimeout('registerProject')));
    }
    async deprioritizeProject(request) {
        return this.callWithRetry(() => grpcUnaryWithTimeout(this.projectClient, 'deprioritizeProject', request, this.getMethodTimeout('deprioritizeProject')));
    }
    async heartbeat(request) {
        return this.callWithRetry(() => grpcUnaryWithTimeout(this.projectClient, 'heartbeat', request, this.getMethodTimeout('heartbeat')));
    }
    async resolveSearchScope(request) {
        return this.callWithRetry(() => grpcUnaryWithTimeout(this.projectClient, 'resolveSearchScope', request, this.getMethodTimeout('resolveSearchScope')));
    }
}
//# sourceMappingURL=system-methods.js.map