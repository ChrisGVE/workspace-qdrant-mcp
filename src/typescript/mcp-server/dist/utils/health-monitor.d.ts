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
import type { DaemonClient } from '../clients/daemon-client.js';
export type HealthStatus = 'healthy' | 'uncertain';
export type UncertainReason = 'daemon_unavailable' | 'qdrant_unavailable' | 'both_unavailable';
export interface HealthState {
    status: HealthStatus;
    daemonAvailable: boolean;
    qdrantAvailable: boolean;
    reason?: UncertainReason;
    message?: string;
    lastCheck?: Date;
}
export interface HealthMetadata {
    status: HealthStatus;
    reason?: UncertainReason;
    message?: string;
}
export interface HealthMonitorConfig {
    checkIntervalMs?: number;
    qdrantUrl: string;
    qdrantApiKey?: string;
    qdrantTimeout?: number;
}
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
export declare class HealthMonitor {
    private readonly daemonClient;
    private readonly qdrantClient;
    private readonly checkIntervalMs;
    private intervalId;
    private state;
    constructor(config: HealthMonitorConfig, daemonClient: DaemonClient);
    /**
     * Start background health monitoring
     */
    start(): void;
    /**
     * Stop background health monitoring
     */
    stop(): void;
    /**
     * Get current health state
     */
    getState(): HealthState;
    /**
     * Check if system is healthy
     */
    isHealthy(): boolean;
    /**
     * Perform health check for daemon and Qdrant
     */
    performHealthCheck(): Promise<HealthState>;
    /**
     * Check daemon health
     */
    private checkDaemonHealth;
    /**
     * Check Qdrant health by listing collections
     */
    private checkQdrantHealth;
    /**
     * Compute health state from component availability
     */
    private computeState;
    /**
     * Get health metadata for search response augmentation
     */
    getHealthMetadata(): HealthMetadata | null;
    /**
     * Augment search results with health status metadata
     *
     * When system is uncertain, adds status information to the response
     * to inform users that results may be incomplete.
     */
    augmentSearchResults<T extends {
        success: boolean;
    }>(results: T): T & {
        health?: HealthMetadata;
    };
    /**
     * Force immediate health check
     */
    forceCheck(): Promise<HealthState>;
}
//# sourceMappingURL=health-monitor.d.ts.map