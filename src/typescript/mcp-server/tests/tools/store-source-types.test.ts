/**
 * Tests for StoreTool - source types and response format
 *
 * Per ADR-002: Store tool ONLY uses unified_queue for writes.
 * Per spec: MCP can only store to 'libraries' collection.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { StoreTool } from '../../src/tools/store.js';
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

describe('StoreTool - source types', () => {
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

  it('should handle user_input source type', async () => {
    await storeTool.store({
      content: 'User input content',
      libraryName: 'test-lib',
      sourceType: 'user_input',
    });

    expect(mockStateManager.enqueueUnified).toHaveBeenCalled();
    const payload = vi.mocked(mockStateManager.enqueueUnified).mock.calls[0][4] as Record<string, unknown>;
    expect(payload.source_type).toBe('user_input');
  });

  it('should handle web source type', async () => {
    await storeTool.store({
      content: 'Web content',
      libraryName: 'test-lib',
      sourceType: 'web',
      url: 'https://example.com',
    });

    expect(mockStateManager.enqueueUnified).toHaveBeenCalled();
    const payload = vi.mocked(mockStateManager.enqueueUnified).mock.calls[0][4] as Record<string, unknown>;
    expect(payload.source_type).toBe('web');
    const metadata = payload.metadata as Record<string, string>;
    expect(metadata.url).toBe('https://example.com');
  });

  it('should handle file source type', async () => {
    await storeTool.store({
      content: 'File content',
      libraryName: 'test-lib',
      sourceType: 'file',
      filePath: '/path/to/doc.md',
    });

    expect(mockStateManager.enqueueUnified).toHaveBeenCalled();
    const payload = vi.mocked(mockStateManager.enqueueUnified).mock.calls[0][4] as Record<string, unknown>;
    expect(payload.source_type).toBe('file');
    const metadata = payload.metadata as Record<string, string>;
    expect(metadata.file_path).toBe('/path/to/doc.md');
  });

  it('should handle scratchbook source type', async () => {
    await storeTool.store({
      content: 'Scratchbook note',
      libraryName: 'test-lib',
      sourceType: 'scratchbook',
    });

    expect(mockStateManager.enqueueUnified).toHaveBeenCalled();
    const payload = vi.mocked(mockStateManager.enqueueUnified).mock.calls[0][4] as Record<string, unknown>;
    expect(payload.source_type).toBe('scratchbook');
  });

  it('should handle note source type', async () => {
    await storeTool.store({
      content: 'Quick note',
      libraryName: 'test-lib',
      sourceType: 'note',
    });

    expect(mockStateManager.enqueueUnified).toHaveBeenCalled();
    const payload = vi.mocked(mockStateManager.enqueueUnified).mock.calls[0][4] as Record<string, unknown>;
    expect(payload.source_type).toBe('note');
  });

  it('should default to user_input source type', async () => {
    await storeTool.store({
      content: 'Content without source type',
      libraryName: 'test-lib',
    });

    expect(mockStateManager.enqueueUnified).toHaveBeenCalled();
    const payload = vi.mocked(mockStateManager.enqueueUnified).mock.calls[0][4] as Record<string, unknown>;
    expect(payload.source_type).toBe('user_input');
  });
});

describe('StoreTool - response format', () => {
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

  it('should always include fallback_mode as unified_queue', async () => {
    const result = await storeTool.store({
      content: 'Test',
      libraryName: 'lib',
    });

    expect(result.fallback_mode).toBe('unified_queue');
  });

  it('should always include collection as libraries', async () => {
    const result = await storeTool.store({
      content: 'Test',
      libraryName: 'lib',
    });

    expect(result.collection).toBe('libraries');
  });

  it('should include documentId on success', async () => {
    const result = await storeTool.store({
      content: 'Test',
      libraryName: 'lib',
    });

    expect(result.documentId).toBeDefined();
    expect(result.documentId).toMatch(/^[a-f0-9]{32}$/);
  });

  it('should not include documentId on validation failure', async () => {
    const result = await storeTool.store({
      content: '',
      libraryName: 'lib',
    });

    expect(result.documentId).toBeUndefined();
  });
});
