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
});
