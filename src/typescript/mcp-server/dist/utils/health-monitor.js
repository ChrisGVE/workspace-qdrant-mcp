/**
 * Health Monitor for system health tracking
 *
 * Provides background health checks for daemon and Qdrant connectivity.
 * Augments search responses with health status metadata when system
 * is in an uncertain state.
 *
 * Health Status:
 * - healthy: Both daemon and Qdrant are available
 * - uncertain: One or both services are unavailable
 */
import { QdrantClient } from '@qdrant/js-client-rest';
import { ServiceStatus } from '../clients/grpc-types.js';
/**
 * Health Monitor for tracking system health
 *
 * Usage:
 * ```typescript
 * const monitor = new HealthMonitor(config, daemonClient);
 * monitor.start();
 *
 * // Later, augment search results
 * const results = await searchTool.search(options);
 * const augmented = monitor.augmentSearchResults(results);
 *
 * // Cleanup
 * monitor.stop();
 * ```
 */
export class HealthMonitor {
    daemonClient;
    qdrantClient;
    checkIntervalMs;
    intervalId = null;
    state;
    constructor(config, daemonClient) {
        this.daemonClient = daemonClient;
        this.checkIntervalMs = config.checkIntervalMs ?? 30000; // Default 30 seconds
        const clientConfig = {
            url: config.qdrantUrl,
            timeout: config.qdrantTimeout ?? 5000,
        };
        if (config.qdrantApiKey) {
            clientConfig.apiKey = config.qdrantApiKey;
        }
        this.qdrantClient = new QdrantClient(clientConfig);
        // Initialize as healthy (will be updated on first check)
        this.state = {
            status: 'healthy',
            daemonAvailable: true,
            qdrantAvailable: true,
        };
    }
    /**
     * Start background health monitoring
     */
    start() {
        if (this.intervalId) {
            return; // Already running
        }
        // Perform initial check immediately
        this.performHealthCheck();
        // Schedule periodic checks
        this.intervalId = setInterval(() => {
            this.performHealthCheck();
        }, this.checkIntervalMs);
    }
    /**
     * Stop background health monitoring
     */
    stop() {
        if (this.intervalId) {
            clearInterval(this.intervalId);
            this.intervalId = null;
        }
    }
    /**
     * Get current health state
     */
    getState() {
        return { ...this.state };
    }
    /**
     * Check if system is healthy
     */
    isHealthy() {
        return this.state.status === 'healthy';
    }
    /**
     * Perform health check for daemon and Qdrant
     */
    async performHealthCheck() {
        const [daemonAvailable, qdrantAvailable] = await Promise.all([
            this.checkDaemonHealth(),
            this.checkQdrantHealth(),
        ]);
        this.state = this.computeState(daemonAvailable, qdrantAvailable);
        return this.state;
    }
    /**
     * Check daemon health
     */
    async checkDaemonHealth() {
        try {
            if (!this.daemonClient.isConnected()) {
                return false;
            }
            const response = await this.daemonClient.healthCheck();
            // Check if status is healthy or degraded (still operational)
            return (response.status === ServiceStatus.SERVICE_STATUS_HEALTHY ||
                response.status === ServiceStatus.SERVICE_STATUS_DEGRADED);
        }
        catch {
            return false;
        }
    }
    /**
     * Check Qdrant health by listing collections
     */
    async checkQdrantHealth() {
        try {
            await this.qdrantClient.getCollections();
            return true;
        }
        catch {
            return false;
        }
    }
    /**
     * Compute health state from component availability
     */
    computeState(daemonAvailable, qdrantAvailable) {
        const lastCheck = new Date();
        if (daemonAvailable && qdrantAvailable) {
            return {
                status: 'healthy',
                daemonAvailable: true,
                qdrantAvailable: true,
                lastCheck,
            };
        }
        let reason;
        let message;
        if (!daemonAvailable && !qdrantAvailable) {
            reason = 'both_unavailable';
            message = 'Both daemon and Qdrant are unavailable. Search results may be incomplete or unavailable.';
        }
        else if (!daemonAvailable) {
            reason = 'daemon_unavailable';
            message = 'Daemon is unavailable. Search results may use cached data and new content cannot be indexed.';
        }
        else {
            reason = 'qdrant_unavailable';
            message = 'Qdrant is unavailable. Search functionality is limited.';
        }
        return {
            status: 'uncertain',
            daemonAvailable,
            qdrantAvailable,
            reason,
            message,
            lastCheck,
        };
    }
    /**
     * Get health metadata for search response augmentation
     */
    getHealthMetadata() {
        if (this.state.status === 'healthy') {
            return null; // No metadata needed for healthy state
        }
        // Build metadata object conditionally to satisfy exactOptionalPropertyTypes
        const metadata = {
            status: this.state.status,
        };
        if (this.state.reason) {
            metadata.reason = this.state.reason;
        }
        if (this.state.message) {
            metadata.message = this.state.message;
        }
        return metadata;
    }
    /**
     * Augment search results with health status metadata
     *
     * When system is uncertain, adds status information to the response
     * to inform users that results may be incomplete.
     */
    augmentSearchResults(results) {
        const metadata = this.getHealthMetadata();
        if (!metadata) {
            return results;
        }
        return {
            ...results,
            health: metadata,
        };
    }
    /**
     * Force immediate health check
     */
    async forceCheck() {
        return this.performHealthCheck();
    }
}
//# sourceMappingURL=health-monitor.js.map