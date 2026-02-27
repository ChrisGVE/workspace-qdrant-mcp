/**
 * Tests for HealthMonitor: metadata, augmentSearchResults, forceCheck,
 * state transitions, and error handling
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { HealthMonitor } from '../../src/utils/health-monitor.js';
import type { DaemonClient } from '../../src/clients/daemon-client.js';
import { ServiceStatus } from '../../src/clients/grpc-types.js';

// Mock the Qdrant client
vi.mock('@qdrant/js-client-rest', () => ({
  QdrantClient: vi.fn().mockImplementation(() => ({
    getCollections: vi.fn().mockResolvedValue({ collections: [] }),
  })),
}));

function createMockDaemonClient(connected = true, healthy = true): DaemonClient {
  const status = healthy ? ServiceStatus.SERVICE_STATUS_HEALTHY : ServiceStatus.SERVICE_STATUS_UNHEALTHY;
  return {
    isConnected: vi.fn().mockReturnValue(connected),
    healthCheck: vi.fn().mockResolvedValue({ status, components: [] }),
    ingestText: vi.fn(),
    embedText: vi.fn(),
    generateSparseVector: vi.fn(),
    connect: vi.fn(),
    close: vi.fn(),
    getConnectionState: vi.fn(),
    getStatus: vi.fn(),
    getMetrics: vi.fn(),
    notifyServerStatus: vi.fn(),
    registerProject: vi.fn(),
    deprioritizeProject: vi.fn(),
    heartbeat: vi.fn(),
  } as unknown as DaemonClient;
}

describe('HealthMonitor', () => {
  let healthMonitor: HealthMonitor;
  let mockDaemonClient: DaemonClient;

  beforeEach(() => {
    vi.clearAllMocks();
    vi.useFakeTimers();
    mockDaemonClient = createMockDaemonClient();

    healthMonitor = new HealthMonitor(
      { qdrantUrl: 'http://localhost:6333' },
      mockDaemonClient
    );
  });

  afterEach(() => {
    healthMonitor.stop();
    vi.useRealTimers();
  });

  describe('health metadata', () => {
    it('should return null metadata when healthy', () => {
      const metadata = healthMonitor.getHealthMetadata();
      expect(metadata).toBeNull();
    });

    it('should return metadata when uncertain', async () => {
      const unhealthyDaemon = createMockDaemonClient(false, false);
      const monitor = new HealthMonitor(
        { qdrantUrl: 'http://localhost:6333' },
        unhealthyDaemon
      );

      await monitor.performHealthCheck();

      const metadata = monitor.getHealthMetadata();

      expect(metadata).not.toBeNull();
      expect(metadata?.status).toBe('uncertain');
      expect(metadata?.reason).toBe('daemon_unavailable');
      expect(metadata?.message).toBeDefined();

      monitor.stop();
    });
  });

  describe('augmentSearchResults', () => {
    it('should not add metadata to healthy results', () => {
      const searchResults = {
        success: true,
        results: [{ id: '1', content: 'test' }],
      };

      const augmented = healthMonitor.augmentSearchResults(searchResults);

      expect(augmented.health).toBeUndefined();
      expect(augmented.success).toBe(true);
      expect(augmented.results).toEqual(searchResults.results);
    });

    it('should add health metadata to uncertain results', async () => {
      const unhealthyDaemon = createMockDaemonClient(false, false);
      const monitor = new HealthMonitor(
        { qdrantUrl: 'http://localhost:6333' },
        unhealthyDaemon
      );

      await monitor.performHealthCheck();

      const searchResults = {
        success: true,
        results: [{ id: '1', content: 'test' }],
      };

      const augmented = monitor.augmentSearchResults(searchResults);

      expect(augmented.health).toBeDefined();
      expect(augmented.health?.status).toBe('uncertain');
      expect(augmented.health?.reason).toBe('daemon_unavailable');
      expect(augmented.success).toBe(true);

      monitor.stop();
    });

    it('should preserve original results while adding metadata', async () => {
      const unhealthyDaemon = createMockDaemonClient(false, false);
      const monitor = new HealthMonitor(
        { qdrantUrl: 'http://localhost:6333' },
        unhealthyDaemon
      );

      await monitor.performHealthCheck();

      const searchResults = {
        success: true,
        results: [{ id: '1', score: 0.9 }],
        total: 1,
        customField: 'preserved',
      };

      const augmented = monitor.augmentSearchResults(searchResults);

      expect(augmented.success).toBe(true);
      expect(augmented.results).toEqual(searchResults.results);
      expect(augmented.total).toBe(1);
      expect(augmented.customField).toBe('preserved');
      expect(augmented.health).toBeDefined();

      monitor.stop();
    });
  });

  describe('forceCheck', () => {
    it('should perform immediate health check', async () => {
      vi.clearAllMocks();

      const state = await healthMonitor.forceCheck();

      expect(state.status).toBe('healthy');
      expect(mockDaemonClient.healthCheck).toHaveBeenCalled();
    });
  });

  describe('state transitions', () => {
    it('should transition from healthy to uncertain', async () => {
      // Start healthy
      let state = await healthMonitor.performHealthCheck();
      expect(state.status).toBe('healthy');

      // Make daemon fail
      vi.mocked(mockDaemonClient.isConnected).mockReturnValue(false);

      // Should now be uncertain
      state = await healthMonitor.performHealthCheck();
      expect(state.status).toBe('uncertain');
    });

    it('should transition from uncertain to healthy', async () => {
      // Make daemon unavailable
      vi.mocked(mockDaemonClient.isConnected).mockReturnValue(false);
      let state = await healthMonitor.performHealthCheck();
      expect(state.status).toBe('uncertain');

      // Restore daemon
      vi.mocked(mockDaemonClient.isConnected).mockReturnValue(true);
      vi.mocked(mockDaemonClient.healthCheck).mockResolvedValue({
        status: ServiceStatus.SERVICE_STATUS_HEALTHY,
        components: [],
      });

      // Should now be healthy
      state = await healthMonitor.performHealthCheck();
      expect(state.status).toBe('healthy');
    });
  });

  describe('error handling', () => {
    it('should handle daemon healthCheck throwing error', async () => {
      vi.mocked(mockDaemonClient.healthCheck).mockRejectedValue(new Error('Connection error'));

      const state = await healthMonitor.performHealthCheck();

      expect(state.status).toBe('uncertain');
      expect(state.daemonAvailable).toBe(false);
    });
  });
});
