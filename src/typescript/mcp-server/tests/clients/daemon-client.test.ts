/**
 * Tests for DaemonClient
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { DaemonClient, type ConnectionState } from '../../src/clients/daemon-client.js';

describe('DaemonClient', () => {
  let client: DaemonClient;

  beforeEach(() => {
    client = new DaemonClient({ host: 'localhost', port: 50051 });
  });

  describe('constructor', () => {
    it('should create client with default configuration', () => {
      const defaultClient = new DaemonClient();
      expect(defaultClient).toBeDefined();
      expect(defaultClient.isConnected()).toBe(false);
    });

    it('should create client with custom configuration', () => {
      const customClient = new DaemonClient({
        host: '127.0.0.1',
        port: 50052,
        timeoutMs: 10000,
        maxRetries: 5,
      });
      expect(customClient).toBeDefined();
    });
  });

  describe('getConnectionState', () => {
    it('should return disconnected state initially', () => {
      const state: ConnectionState = client.getConnectionState();
      expect(state.connected).toBe(false);
      expect(state.lastHealthCheck).toBeUndefined();
      expect(state.lastError).toBeUndefined();
    });
  });

  describe('isConnected', () => {
    it('should return false when not connected', () => {
      expect(client.isConnected()).toBe(false);
    });
  });

  describe('connect', () => {
    it('should fail to connect when daemon is not running', async () => {
      // Use a port that's unlikely to have a daemon running
      const testClient = new DaemonClient({ port: 59999, timeoutMs: 1000, maxRetries: 1 });

      await expect(testClient.connect()).rejects.toThrow();
      expect(testClient.isConnected()).toBe(false);
    });
  });

  describe('close', () => {
    it('should handle close when not connected', () => {
      // Should not throw
      expect(() => client.close()).not.toThrow();
      expect(client.isConnected()).toBe(false);
    });
  });

  describe('auto-reconnect (issue #55)', () => {
    it('should attempt to connect on RPC call when never connected', async () => {
      // When the MCP server starts before the daemon, the initial connect()
      // fails and the client is left with undefined service handles. The
      // user's first RPC must retry via ensureConnected() rather than
      // failing outright with "Client not connected".
      const testClient = new DaemonClient({ port: 59998, timeoutMs: 500, maxRetries: 1 });

      await expect(testClient.healthCheck()).rejects.toThrow();
      // The surfaced error must be the connection failure, not the stale
      // "Client not connected" guard.
      try {
        await testClient.healthCheck();
      } catch (err) {
        expect((err as Error).message).not.toContain('Client not connected');
      }
    });

    it('should re-attempt connect after close()', async () => {
      const testClient = new DaemonClient({ port: 59997, timeoutMs: 500, maxRetries: 1 });
      // Simulate the lifecycle: close first (no-op), then call — the call
      // should attempt a fresh connect(), not silently use stale handles.
      testClient.close();
      expect(testClient.isConnected()).toBe(false);
      await expect(testClient.healthCheck()).rejects.toThrow();
    });
  });

  describe('getQueueStats', () => {
    it('should reject when daemon is not running', async () => {
      const testClient = new DaemonClient({ port: 59996, timeoutMs: 500, maxRetries: 1 });
      await expect(testClient.getQueueStats()).rejects.toThrow();
    });
  });
});

describe('DaemonClient integration', () => {
  // These tests require a running daemon
  // Skip if daemon is not available

  it.skip('should connect to running daemon', async () => {
    const client = new DaemonClient();
    await client.connect();
    expect(client.isConnected()).toBe(true);

    const health = await client.healthCheck();
    expect(health).toBeDefined();
    expect(health.status).toBeDefined();

    client.close();
  });

  it.skip('should register and deprioritize project', async () => {
    const client = new DaemonClient();
    await client.connect();

    const registerResponse = await client.registerProject({
      path: '/tmp/test-project',
      project_id: 'test123456ab',
    });
    expect(registerResponse.project_id).toBe('test123456ab');

    const deprioritizeResponse = await client.deprioritizeProject({
      project_id: 'test123456ab',
    });
    expect(deprioritizeResponse.success).toBe(true);

    client.close();
  });

  it.skip('should fetch queue stats from running daemon', async () => {
    const client = new DaemonClient();
    await client.connect();

    const stats = await client.getQueueStats();
    expect(stats).toBeDefined();
    expect(typeof stats.pending_count).toBe('number');

    client.close();
  });
});
