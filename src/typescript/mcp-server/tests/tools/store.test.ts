/**
 * Tests for StoreTool
 *
 * Per ADR-002: Store tool ONLY uses unified_queue for writes.
 * Per spec: MCP can only store to 'libraries' collection.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { StoreTool, type StoreOptions, type StoreResponse } from '../../src/tools/store.js';
import type { SqliteStateManager } from '../../src/clients/sqlite-state-manager.js';
import type { ProjectDetector } from '../../src/utils/project-detector.js';

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

function createMockProjectDetector(): ProjectDetector {
  return {
    findProjectRoot: vi.fn().mockReturnValue('/test/project'),
    getProjectInfo: vi.fn().mockResolvedValue({
      projectId: 'test-project-123',
      projectPath: '/test/project',
      name: 'test-project',
    }),
  } as unknown as ProjectDetector;
}

describe('StoreTool', () => {
  let storeTool: StoreTool;
  let mockStateManager: SqliteStateManager;
  let mockProjectDetector: ProjectDetector;

  beforeEach(() => {
    vi.clearAllMocks();
    mockStateManager = createMockStateManager();
    mockProjectDetector = createMockProjectDetector();

    storeTool = new StoreTool(
      {},
      mockStateManager,
      mockProjectDetector
    );
  });

  describe('store to libraries collection', () => {
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
        'content',
        'ingest',
        'react', // tenant_id = libraryName
        'libraries',
        expect.any(Object),
        8, // MCP content priority
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
      expect(call[0]).toBe('content');
      expect(call[1]).toBe('ingest');
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
      expect(call[5]).toBe(8);
      expect(call[6]).toBeUndefined();
    });

    it('should reject storage without libraryName', async () => {
      const options: StoreOptions = {
        content: 'Library content',
        libraryName: '', // Empty
      };

      const result = await storeTool.store(options);

      expect(result.success).toBe(false);
      expect(result.message).toContain('Library name is required');
      expect(mockStateManager.enqueueUnified).not.toHaveBeenCalled();
    });

    it('should reject storage with whitespace-only libraryName', async () => {
      const options: StoreOptions = {
        content: 'Library content',
        libraryName: '   \t  ',
      };

      const result = await storeTool.store(options);

      expect(result.success).toBe(false);
      expect(result.message).toContain('Library name is required');
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
      expect(call[0]).toBe('content');
      expect(call[1]).toBe('ingest');
      expect(call[2]).toBe('react'); // Trimmed tenant_id
      expect(call[3]).toBe('libraries');
    });
  });

  describe('validation', () => {
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

  describe('queue operations', () => {
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
        'content', // item_type
        'ingest', // operation
        expect.any(String),
        'libraries',
        expect.any(Object),
        8, // Priority 8 for MCP content
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

    it('should handle queue returning error status', async () => {
      vi.mocked(mockStateManager.enqueueUnified).mockResolvedValue({
        status: 'error',
        message: 'Queue full',
        data: null,
      });

      const options: StoreOptions = {
        content: 'Test content',
        libraryName: 'test-lib',
      };

      const result = await storeTool.store(options);

      expect(result.success).toBe(false);
      expect(result.message).toContain('Failed to queue content');
    });
  });

  describe('idempotency', () => {
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

  describe('source types', () => {
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

  describe('response format', () => {
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
});
