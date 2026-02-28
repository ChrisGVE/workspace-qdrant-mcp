/**
 * Tests for StoreTool - queue operations and idempotency
 *
 * Per ADR-002: Store tool ONLY uses unified_queue for writes.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { StoreTool, type StoreOptions } from '../../src/tools/store.js';
import type { SqliteStateManager } from '../../src/clients/sqlite-state-manager.js';

function createMockStateManager(): SqliteStateManager {
  return {
    initialize: vi.fn().mockReturnValue({ status: 'ok' }),
    close: vi.fn(),
    enqueueUnified: vi.fn().mockResolvedValue({
      status: 'ok',
      data: {
        queueId: 'queue-456',
        isNew: true,
        idempotencyKey: 'test-key',
      },
    }),
  } as unknown as SqliteStateManager;
}

describe('StoreTool - queue operations', () => {
  let storeTool: StoreTool;
  let mockStateManager: SqliteStateManager;

  beforeEach(() => {
    vi.clearAllMocks();
    mockStateManager = createMockStateManager();

    storeTool = new StoreTool(
      {},
      mockStateManager
    );
  });

  it('should return queue_id on successful enqueue', async () => {
    const options: StoreOptions = {
      content: 'Test content',
      libraryName: 'test-lib',
    };

    const result = await storeTool.store(options);

    expect(result.queue_id).toBe('queue-456');
    expect(result.fallback_mode).toBe('unified_queue');
  });

  it('should call enqueueUnified with correct item type and operation', async () => {
    const options: StoreOptions = {
      content: 'Test content',
      libraryName: 'test-lib',
    };

    await storeTool.store(options);

    expect(mockStateManager.enqueueUnified).toHaveBeenCalledWith(
      'tenant',
      'add', // operation
      expect.any(String),
      'libraries',
      expect.any(Object),
      1, // PRIORITY_HIGH
      undefined, // No branch for libraries
      expect.objectContaining({ source: 'mcp_store_tool' })
    );
  });

  it('should handle queue errors gracefully', async () => {
    vi.mocked(mockStateManager.enqueueUnified).mockRejectedValue(
      new Error('Database connection failed')
    );

    const options: StoreOptions = {
      content: 'Test content',
      libraryName: 'test-lib',
    };

    const result = await storeTool.store(options);

    expect(result.success).toBe(false);
    expect(result.message).toContain('Failed to queue content');
    expect(result.message).toContain('Database connection failed');
  });

  it('should handle queue returning degraded status with no data', async () => {
    vi.mocked(mockStateManager.enqueueUnified).mockResolvedValue({
      status: 'degraded',
      message: 'Queue full',
      data: null,  // Simulate missing data
    } as { status: 'ok' | 'degraded'; message: string; data: null });

    const options: StoreOptions = {
      content: 'Test content',
      libraryName: 'test-lib',
    };

    const result = await storeTool.store(options);

    expect(result.success).toBe(false);
    expect(result.message).toContain('Failed to queue');
  });
});

describe('StoreTool - idempotency', () => {
  let storeTool: StoreTool;
  let mockStateManager: SqliteStateManager;

  beforeEach(() => {
    vi.clearAllMocks();
    mockStateManager = createMockStateManager();

    storeTool = new StoreTool(
      {},
      mockStateManager
    );
  });

  it('should generate consistent document ID for same content and tenant', async () => {
    const options: StoreOptions = {
      content: 'Identical content',
      libraryName: 'react',
    };

    // Store twice
    const result1 = await storeTool.store(options);
    const result2 = await storeTool.store(options);

    // Both should have the same documentId
    expect(result1.documentId).toBe(result2.documentId);
  });

  it('should generate different document ID for different content', async () => {
    const result1 = await storeTool.store({
      content: 'Content A',
      libraryName: 'react',
    });

    const result2 = await storeTool.store({
      content: 'Content B',
      libraryName: 'react',
    });

    expect(result1.documentId).not.toBe(result2.documentId);
  });

  it('should generate different document ID for different libraries', async () => {
    const result1 = await storeTool.store({
      content: 'Same content',
      libraryName: 'react',
    });

    const result2 = await storeTool.store({
      content: 'Same content',
      libraryName: 'vue',
    });

    expect(result1.documentId).not.toBe(result2.documentId);
  });

  it('should generate 32-character hex document ID', async () => {
    const result = await storeTool.store({
      content: 'Test content',
      libraryName: 'react',
    });

    expect(result.documentId).toMatch(/^[a-f0-9]{32}$/);
  });
});
