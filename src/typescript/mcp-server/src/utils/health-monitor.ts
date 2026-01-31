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
import type { DaemonClient } from '../clients/daemon-client.js';
import { ServiceStatus } from '../clients/grpc-types.js';

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
export class HealthMonitor {
  private readonly daemonClient: DaemonClient;
  private readonly qdrantClient: QdrantClient;
  private readonly checkIntervalMs: number;
  private intervalId: NodeJS.Timeout | null = null;
  private state: HealthState;

  constructor(config: HealthMonitorConfig, daemonClient: DaemonClient) {
    this.daemonClient = daemonClient;
    this.checkIntervalMs = config.checkIntervalMs ?? 30000; // Default 30 seconds

    const clientConfig: { url: string; apiKey?: string; timeout?: number } = {
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
  start(): void {
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
  stop(): void {
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = null;
    }
  }

  /**
   * Get current health state
   */
  getState(): HealthState {
    return { ...this.state };
  }

  /**
   * Check if system is healthy
   */
  isHealthy(): boolean {
    return this.state.status === 'healthy';
  }

  /**
   * Perform health check for daemon and Qdrant
   */
  async performHealthCheck(): Promise<HealthState> {
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
  private async checkDaemonHealth(): Promise<boolean> {
    try {
      if (!this.daemonClient.isConnected()) {
        return false;
      }
      const response = await this.daemonClient.healthCheck();
      // Check if status is healthy or degraded (still operational)
      return (
        response.status === ServiceStatus.SERVICE_STATUS_HEALTHY ||
        response.status === ServiceStatus.SERVICE_STATUS_DEGRADED
      );
    } catch {
      return false;
    }
  }

  /**
   * Check Qdrant health by listing collections
   */
  private async checkQdrantHealth(): Promise<boolean> {
    try {
      await this.qdrantClient.getCollections();
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Compute health state from component availability
   */
  private computeState(daemonAvailable: boolean, qdrantAvailable: boolean): HealthState {
    const lastCheck = new Date();

    if (daemonAvailable && qdrantAvailable) {
      return {
        status: 'healthy',
        daemonAvailable: true,
        qdrantAvailable: true,
        lastCheck,
      };
    }

    let reason: UncertainReason;
    let message: string;

    if (!daemonAvailable && !qdrantAvailable) {
      reason = 'both_unavailable';
      message = 'Both daemon and Qdrant are unavailable. Search results may be incomplete or unavailable.';
    } else if (!daemonAvailable) {
      reason = 'daemon_unavailable';
      message = 'Daemon is unavailable. Search results may use cached data and new content cannot be indexed.';
    } else {
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
  getHealthMetadata(): HealthMetadata | null {
    if (this.state.status === 'healthy') {
      return null; // No metadata needed for healthy state
    }

    // Build metadata object conditionally to satisfy exactOptionalPropertyTypes
    const metadata: HealthMetadata = {
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
  augmentSearchResults<T extends { success: boolean }>(results: T): T & { health?: HealthMetadata } {
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
  async forceCheck(): Promise<HealthState> {
    return this.performHealthCheck();
  }
}
