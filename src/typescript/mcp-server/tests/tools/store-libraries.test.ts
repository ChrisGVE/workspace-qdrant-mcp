/**
 * Tests for StoreTool - library storage and validation
 *
 * Per ADR-002: Store tool ONLY uses unified_queue for writes.
 * Per spec: MCP can only store to 'libraries' collection.
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

describe('StoreTool - store to libraries collection', () => {
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

  it('should queue library content successfully', async () => {
    const options: StoreOptions = {
      content: 'Library documentation content',
      libraryName: 'react',
      sourceType: 'web',
      url: 'https://react.dev/docs',
    };

    const result = await storeTool.store(options);

    expect(result.success).toBe(true);
    expect(result.collection).toBe('libraries');
    expect(result.fallback_mode).toBe('unified_queue');
    expect(result.queue_id).toBe('queue-456');
    expect(result.message).toContain('queued');
    expect(mockStateManager.enqueueUnified).toHaveBeenCalled();
  });

  it('should use libraryName as tenant_id', async () => {
    const options: StoreOptions = {
      content: 'React documentation',
      libraryName: 'react',
    };

    await storeTool.store(options);

    expect(mockStateManager.enqueueUnified).toHaveBeenCalledWith(
      'tenant',
      'add',
      'react', // tenant_id = libraryName
      'libraries',
      expect.any(Object),
      1, // PRIORITY_HIGH from native-bridge
      undefined, // No branch for libraries
      expect.objectContaining({ source: 'mcp_store_tool' })
    );
  });

  it('should include metadata in queue payload', async () => {
    const options: StoreOptions = {
      content: 'Test content',
      libraryName: 'lodash',
      title: 'Lodash Documentation',
      url: 'https://lodash.com/docs',
      filePath: '/docs/lodash.md',
      sourceType: 'web',
      metadata: { version: '4.17.21' },
    };

    await storeTool.store(options);

    expect(mockStateManager.enqueueUnified).toHaveBeenCalled();
    const call = vi.mocked(mockStateManager.enqueueUnified).mock.calls[0];

    // Check basic args
    expect(call[0]).toBe('tenant');
    expect(call[1]).toBe('add');
    expect(call[2]).toBe('lodash');
    expect(call[3]).toBe('libraries');

    // Check payload structure
    const payload = call[4] as Record<string, unknown>;
    expect(payload.content).toBe('Test content');
    expect(payload.source_type).toBe('web');

    const metadata = payload.metadata as Record<string, string>;
    expect(metadata.title).toBe('Lodash Documentation');
    expect(metadata.url).toBe('https://lodash.com/docs');
    expect(metadata.file_path).toBe('/docs/lodash.md');
    expect(metadata.source_type).toBe('web');
    expect(metadata.version).toBe('4.17.21');

    // Check priority and branch
    expect(call[5]).toBe(1); // PRIORITY_HIGH
    expect(call[6]).toBeUndefined();
  });

  it('should reject storage without libraryName', async () => {
    const options: StoreOptions = {
      content: 'Library content',
      libraryName: '', // Empty
    };

    const result = await storeTool.store(options);

    expect(result.success).toBe(false);
    expect(result.message).toContain('libraryName is required');
    expect(mockStateManager.enqueueUnified).not.toHaveBeenCalled();
  });

  it('should reject storage with whitespace-only libraryName', async () => {
    const options: StoreOptions = {
      content: 'Library content',
      libraryName: '   \t  ',
    };

    const result = await storeTool.store(options);

    expect(result.success).toBe(false);
    expect(result.message).toContain('libraryName is required');
  });

  it('should trim libraryName', async () => {
    const options: StoreOptions = {
      content: 'Test content',
      libraryName: '  react  ',
    };

    await storeTool.store(options);

    // Check first two args (item_type, op)
    expect(mockStateManager.enqueueUnified).toHaveBeenCalled();
    const call = vi.mocked(mockStateManager.enqueueUnified).mock.calls[0];
    expect(call[0]).toBe('tenant');
    expect(call[1]).toBe('add');
    expect(call[2]).toBe('react'); // Trimmed tenant_id
    expect(call[3]).toBe('libraries');
  });
});

describe('StoreTool - validation', () => {
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

  it('should reject empty content', async () => {
    const options: StoreOptions = {
      content: '',
      libraryName: 'react',
    };

    const result = await storeTool.store(options);

    expect(result.success).toBe(false);
    expect(result.message).toContain('Content is required');
  });

  it('should reject whitespace-only content', async () => {
    const options: StoreOptions = {
      content: '   \n\t  ',
      libraryName: 'react',
    };

    const result = await storeTool.store(options);

    expect(result.success).toBe(false);
    expect(result.message).toContain('Content is required');
  });
});
