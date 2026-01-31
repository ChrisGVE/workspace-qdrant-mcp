/**
 * Tests for StoreTool
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { StoreTool, type StoreOptions } from '../../src/tools/store.js';
import type { DaemonClient } from '../../src/clients/daemon-client.js';
import type { SqliteStateManager } from '../../src/clients/sqlite-state-manager.js';
import type { ProjectDetector } from '../../src/utils/project-detector.js';

function createMockDaemonClient(): DaemonClient {
  return {
    isConnected: vi.fn().mockReturnValue(true),
    ingestText: vi.fn().mockResolvedValue({
      success: true,
      document_id: 'doc-123',
      chunks_created: 3,
    }),
    embedText: vi.fn(),
    generateSparseVector: vi.fn(),
    connect: vi.fn(),
    close: vi.fn(),
    getConnectionState: vi.fn(),
    healthCheck: vi.fn(),
    getStatus: vi.fn(),
    getMetrics: vi.fn(),
    notifyServerStatus: vi.fn(),
    registerProject: vi.fn(),
    deprioritizeProject: vi.fn(),
    heartbeat: vi.fn(),
  } as unknown as DaemonClient;
}

function createMockStateManager(): SqliteStateManager {
  return {
    initialize: vi.fn().mockReturnValue({ status: 'ok' }),
    close: vi.fn(),
    enqueueUnified: vi.fn().mockReturnValue({
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
  let mockDaemonClient: DaemonClient;
  let mockStateManager: SqliteStateManager;
  let mockProjectDetector: ProjectDetector;

  beforeEach(() => {
    vi.clearAllMocks();
    mockDaemonClient = createMockDaemonClient();
    mockStateManager = createMockStateManager();
    mockProjectDetector = createMockProjectDetector();

    storeTool = new StoreTool(
      { defaultCollection: 'projects' },
      mockDaemonClient,
      mockStateManager,
      mockProjectDetector
    );
  });

  describe('store to projects collection', () => {
    it('should store content via daemon successfully', async () => {
      const options: StoreOptions = {
        content: 'Test content for storage',
        collection: 'projects',
        title: 'Test Document',
        sourceType: 'user_input',
      };

      const result = await storeTool.store(options);

      expect(result.success).toBe(true);
      expect(result.documentId).toBe('doc-123');
      expect(result.collection).toBe('projects');
      expect(result.message).toContain('3 chunks');
      expect(result.fallback_mode).toBeUndefined();
      expect(mockDaemonClient.ingestText).toHaveBeenCalled();
    });

    it('should use project detector when projectId not provided', async () => {
      const options: StoreOptions = {
        content: 'Test content',
        collection: 'projects',
      };

      await storeTool.store(options);

      expect(mockProjectDetector.getProjectInfo).toHaveBeenCalled();
      expect(mockDaemonClient.ingestText).toHaveBeenCalledWith(
        expect.objectContaining({
          tenant_id: 'test-project-123',
          collection_basename: 'code',
        })
      );
    });

    it('should use provided projectId when specified', async () => {
      const options: StoreOptions = {
        content: 'Test content',
        collection: 'projects',
        projectId: 'explicit-project-id',
      };

      await storeTool.store(options);

      expect(mockProjectDetector.getProjectInfo).not.toHaveBeenCalled();
      expect(mockDaemonClient.ingestText).toHaveBeenCalledWith(
        expect.objectContaining({
          tenant_id: 'explicit-project-id',
        })
      );
    });

    it('should include metadata in daemon request', async () => {
      const options: StoreOptions = {
        content: 'Test content',
        collection: 'projects',
        title: 'My Title',
        url: 'https://example.com',
        filePath: '/path/to/file.ts',
        branch: 'main',
        fileType: 'typescript',
        metadata: { custom_key: 'custom_value' },
      };

      await storeTool.store(options);

      expect(mockDaemonClient.ingestText).toHaveBeenCalledWith(
        expect.objectContaining({
          metadata: expect.objectContaining({
            title: 'My Title',
            url: 'https://example.com',
            file_path: '/path/to/file.ts',
            branch: 'main',
            file_type: 'typescript',
            custom_key: 'custom_value',
            source_type: 'user_input',
          }),
        })
      );
    });
  });

  describe('store to libraries collection', () => {
    it('should store library content via daemon', async () => {
      const options: StoreOptions = {
        content: 'Library documentation content',
        collection: 'libraries',
        libraryName: 'react',
        sourceType: 'web',
        url: 'https://react.dev/docs',
      };

      const result = await storeTool.store(options);

      expect(result.success).toBe(true);
      expect(result.collection).toBe('libraries');
      expect(mockDaemonClient.ingestText).toHaveBeenCalledWith(
        expect.objectContaining({
          tenant_id: 'react',
          collection_basename: 'lib',
        })
      );
    });

    it('should reject libraries storage without libraryName', async () => {
      const options: StoreOptions = {
        content: 'Library content',
        collection: 'libraries',
      };

      const result = await storeTool.store(options);

      expect(result.success).toBe(false);
      expect(result.message).toContain('Library name is required');
      expect(mockDaemonClient.ingestText).not.toHaveBeenCalled();
    });
  });

  describe('validation', () => {
    it('should reject empty content', async () => {
      const options: StoreOptions = {
        content: '',
        collection: 'projects',
      };

      const result = await storeTool.store(options);

      expect(result.success).toBe(false);
      expect(result.message).toContain('Content is required');
    });

    it('should reject whitespace-only content', async () => {
      const options: StoreOptions = {
        content: '   \n\t  ',
        collection: 'projects',
      };

      const result = await storeTool.store(options);

      expect(result.success).toBe(false);
      expect(result.message).toContain('Content is required');
    });
  });

  describe('daemon fallback to queue', () => {
    it('should fallback to queue when daemon fails', async () => {
      vi.mocked(mockDaemonClient.ingestText).mockRejectedValue(
        new Error('Daemon unavailable')
      );

      const options: StoreOptions = {
        content: 'Test content for queue',
        collection: 'projects',
      };

      const result = await storeTool.store(options);

      expect(result.success).toBe(true);
      expect(result.fallback_mode).toBe('unified_queue');
      expect(result.queue_id).toBe('queue-456');
      expect(result.message).toContain('queued');
      expect(mockStateManager.enqueueUnified).toHaveBeenCalled();
    });

    it('should fallback when daemon returns failure', async () => {
      vi.mocked(mockDaemonClient.ingestText).mockResolvedValue({
        success: false,
        document_id: '',
        chunks_created: 0,
      });

      const options: StoreOptions = {
        content: 'Test content',
        collection: 'projects',
      };

      const result = await storeTool.store(options);

      expect(result.success).toBe(true);
      expect(result.fallback_mode).toBe('unified_queue');
      expect(mockStateManager.enqueueUnified).toHaveBeenCalled();
    });

    it('should call enqueueUnified with correct parameters', async () => {
      vi.mocked(mockDaemonClient.ingestText).mockRejectedValue(
        new Error('Daemon unavailable')
      );

      const options: StoreOptions = {
        content: 'Test content',
        collection: 'projects',
        title: 'My Title',
        sourceType: 'file',
        branch: 'develop',
      };

      await storeTool.store(options);

      expect(mockStateManager.enqueueUnified).toHaveBeenCalledWith(
        'content',
        'ingest',
        'test-project-123', // tenant_id from project detector
        'projects',
        expect.objectContaining({
          content: 'Test content',
          source_type: 'file',
        }),
        5, // Normal priority
        'develop', // branch
        expect.objectContaining({ source: 'mcp_store_tool' })
      );
    });

    it('should use default branch when not specified', async () => {
      vi.mocked(mockDaemonClient.ingestText).mockRejectedValue(
        new Error('Daemon unavailable')
      );

      const options: StoreOptions = {
        content: 'Test content',
        collection: 'projects',
      };

      await storeTool.store(options);

      expect(mockStateManager.enqueueUnified).toHaveBeenCalledWith(
        'content',
        'ingest',
        expect.any(String),
        'projects',
        expect.any(Object),
        5,
        'main', // default branch
        expect.any(Object)
      );
    });
  });

  describe('idempotency', () => {
    it('should generate consistent document ID for same content', async () => {
      const options: StoreOptions = {
        content: 'Identical content',
        collection: 'projects',
        projectId: 'my-project',
      };

      // Store twice
      await storeTool.store(options);
      await storeTool.store(options);

      // Both calls should use the same document_id
      const calls = vi.mocked(mockDaemonClient.ingestText).mock.calls;
      expect(calls[0][0].document_id).toBe(calls[1][0].document_id);
    });

    it('should generate different document ID for different content', async () => {
      await storeTool.store({
        content: 'Content A',
        collection: 'projects',
        projectId: 'my-project',
      });

      await storeTool.store({
        content: 'Content B',
        collection: 'projects',
        projectId: 'my-project',
      });

      const calls = vi.mocked(mockDaemonClient.ingestText).mock.calls;
      expect(calls[0][0].document_id).not.toBe(calls[1][0].document_id);
    });

    it('should generate different document ID for different tenants', async () => {
      await storeTool.store({
        content: 'Same content',
        collection: 'projects',
        projectId: 'project-a',
      });

      await storeTool.store({
        content: 'Same content',
        collection: 'projects',
        projectId: 'project-b',
      });

      const calls = vi.mocked(mockDaemonClient.ingestText).mock.calls;
      expect(calls[0][0].document_id).not.toBe(calls[1][0].document_id);
    });
  });

  describe('source types', () => {
    it('should handle user_input source type', async () => {
      await storeTool.store({
        content: 'User input content',
        collection: 'projects',
        sourceType: 'user_input',
      });

      expect(mockDaemonClient.ingestText).toHaveBeenCalledWith(
        expect.objectContaining({
          metadata: expect.objectContaining({
            source_type: 'user_input',
          }),
        })
      );
    });

    it('should handle web source type', async () => {
      await storeTool.store({
        content: 'Web content',
        collection: 'projects',
        sourceType: 'web',
        url: 'https://example.com',
      });

      expect(mockDaemonClient.ingestText).toHaveBeenCalledWith(
        expect.objectContaining({
          metadata: expect.objectContaining({
            source_type: 'web',
            url: 'https://example.com',
          }),
        })
      );
    });

    it('should handle file source type', async () => {
      await storeTool.store({
        content: 'File content',
        collection: 'projects',
        sourceType: 'file',
        filePath: '/path/to/file.ts',
        fileType: 'typescript',
      });

      expect(mockDaemonClient.ingestText).toHaveBeenCalledWith(
        expect.objectContaining({
          metadata: expect.objectContaining({
            source_type: 'file',
            file_path: '/path/to/file.ts',
            file_type: 'typescript',
          }),
        })
      );
    });

    it('should handle scratchbook source type', async () => {
      await storeTool.store({
        content: 'Scratchbook note',
        collection: 'projects',
        sourceType: 'scratchbook',
      });

      expect(mockDaemonClient.ingestText).toHaveBeenCalledWith(
        expect.objectContaining({
          metadata: expect.objectContaining({
            source_type: 'scratchbook',
          }),
        })
      );
    });

    it('should handle note source type', async () => {
      await storeTool.store({
        content: 'Quick note',
        collection: 'projects',
        sourceType: 'note',
      });

      expect(mockDaemonClient.ingestText).toHaveBeenCalledWith(
        expect.objectContaining({
          metadata: expect.objectContaining({
            source_type: 'note',
          }),
        })
      );
    });

    it('should default to user_input source type', async () => {
      await storeTool.store({
        content: 'Content without source type',
        collection: 'projects',
      });

      expect(mockDaemonClient.ingestText).toHaveBeenCalledWith(
        expect.objectContaining({
          metadata: expect.objectContaining({
            source_type: 'user_input',
          }),
        })
      );
    });
  });

  describe('default collection', () => {
    it('should use default collection when not specified', async () => {
      const result = await storeTool.store({
        content: 'Test content',
      });

      expect(result.success).toBe(true);
      expect(result.collection).toBe('projects');
    });

    it('should respect custom default collection', async () => {
      const customStoreTool = new StoreTool(
        { defaultCollection: 'libraries' },
        mockDaemonClient,
        mockStateManager,
        mockProjectDetector
      );

      // This should fail because libraries requires libraryName
      const result = await customStoreTool.store({
        content: 'Test content',
      });

      expect(result.success).toBe(false);
      expect(result.message).toContain('Library name is required');
    });
  });
});

describe('StoreTool queue error handling', () => {
  it('should throw when queue enqueue fails', async () => {
    const mockDaemonClient = createMockDaemonClient();
    const mockStateManager = createMockStateManager();
    const mockProjectDetector = createMockProjectDetector();

    // Make daemon fail to trigger queue fallback
    vi.mocked(mockDaemonClient.ingestText).mockRejectedValue(
      new Error('Daemon unavailable')
    );

    // Make queue fail
    vi.mocked(mockStateManager.enqueueUnified).mockReturnValue({
      status: 'error',
      message: 'Database error',
      data: null,
    });

    const storeTool = new StoreTool(
      { defaultCollection: 'projects' },
      mockDaemonClient,
      mockStateManager,
      mockProjectDetector
    );

    await expect(
      storeTool.store({
        content: 'Test content',
        collection: 'projects',
      })
    ).rejects.toThrow('Database error');
  });
});
