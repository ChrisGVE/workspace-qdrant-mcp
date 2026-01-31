/**
 * Tests for MemoryTool
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { MemoryTool, type MemoryOptions } from '../../src/tools/memory.js';
import type { DaemonClient } from '../../src/clients/daemon-client.js';
import type { SqliteStateManager } from '../../src/clients/sqlite-state-manager.js';
import type { ProjectDetector } from '../../src/utils/project-detector.js';

// Mock the Qdrant client
vi.mock('@qdrant/js-client-rest', () => ({
  QdrantClient: vi.fn().mockImplementation(() => ({
    scroll: vi.fn().mockResolvedValue({
      points: [
        {
          id: 'rule-1',
          payload: {
            content: 'Always use TypeScript',
            scope: 'global',
            title: 'TypeScript Rule',
            priority: '10',
          },
        },
        {
          id: 'rule-2',
          payload: {
            content: 'Follow TDD',
            scope: 'project',
            project_id: 'test-project',
            tags: 'testing,quality',
          },
        },
      ],
    }),
  })),
}));

function createMockDaemonClient(): DaemonClient {
  return {
    isConnected: vi.fn().mockReturnValue(true),
    ingestText: vi.fn().mockResolvedValue({
      success: true,
      document_id: 'new-rule-id',
      chunks_created: 1,
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
        queueId: 'queued-rule-id',
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

describe('MemoryTool', () => {
  let memoryTool: MemoryTool;
  let mockDaemonClient: DaemonClient;
  let mockStateManager: SqliteStateManager;
  let mockProjectDetector: ProjectDetector;

  beforeEach(() => {
    vi.clearAllMocks();
    mockDaemonClient = createMockDaemonClient();
    mockStateManager = createMockStateManager();
    mockProjectDetector = createMockProjectDetector();

    memoryTool = new MemoryTool(
      { qdrantUrl: 'http://localhost:6333' },
      mockDaemonClient,
      mockStateManager,
      mockProjectDetector
    );
  });

  describe('add action', () => {
    it('should add a global rule via daemon', async () => {
      const options: MemoryOptions = {
        action: 'add',
        content: 'Always write tests',
        scope: 'global',
        title: 'Testing Rule',
      };

      const result = await memoryTool.execute(options);

      expect(result.success).toBe(true);
      expect(result.action).toBe('add');
      expect(result.ruleId).toBe('new-rule-id');
      expect(result.fallback_mode).toBeUndefined();
      expect(mockDaemonClient.ingestText).toHaveBeenCalled();
    });

    it('should add a project-scoped rule', async () => {
      const options: MemoryOptions = {
        action: 'add',
        content: 'Project-specific rule',
        scope: 'project',
      };

      const result = await memoryTool.execute(options);

      expect(result.success).toBe(true);
      expect(mockProjectDetector.getProjectInfo).toHaveBeenCalled();
    });

    it('should fallback to queue when daemon fails', async () => {
      vi.mocked(mockDaemonClient.ingestText).mockRejectedValue(
        new Error('Daemon unavailable')
      );

      const options: MemoryOptions = {
        action: 'add',
        content: 'Test rule',
        scope: 'global',
      };

      const result = await memoryTool.execute(options);

      expect(result.success).toBe(true);
      expect(result.fallback_mode).toBe('unified_queue');
      expect(result.queue_id).toBe('queued-rule-id');
      expect(mockStateManager.enqueueUnified).toHaveBeenCalled();
    });

    it('should reject empty content', async () => {
      const options: MemoryOptions = {
        action: 'add',
        content: '',
        scope: 'global',
      };

      const result = await memoryTool.execute(options);

      expect(result.success).toBe(false);
      expect(result.message).toContain('Content is required');
    });

    it('should include tags in metadata', async () => {
      const options: MemoryOptions = {
        action: 'add',
        content: 'Rule with tags',
        scope: 'global',
        tags: ['testing', 'quality'],
      };

      await memoryTool.execute(options);

      expect(mockDaemonClient.ingestText).toHaveBeenCalledWith(
        expect.objectContaining({
          metadata: expect.objectContaining({
            tags: 'testing,quality',
          }),
        })
      );
    });

    it('should include priority in metadata', async () => {
      const options: MemoryOptions = {
        action: 'add',
        content: 'High priority rule',
        scope: 'global',
        priority: 10,
      };

      await memoryTool.execute(options);

      expect(mockDaemonClient.ingestText).toHaveBeenCalledWith(
        expect.objectContaining({
          metadata: expect.objectContaining({
            priority: '10',
          }),
        })
      );
    });
  });

  describe('update action', () => {
    it('should update an existing rule via daemon', async () => {
      const options: MemoryOptions = {
        action: 'update',
        ruleId: 'existing-rule-id',
        content: 'Updated rule content',
      };

      const result = await memoryTool.execute(options);

      expect(result.success).toBe(true);
      expect(result.action).toBe('update');
      expect(result.ruleId).toBe('existing-rule-id');
    });

    it('should fallback to queue when daemon fails', async () => {
      vi.mocked(mockDaemonClient.ingestText).mockRejectedValue(
        new Error('Daemon unavailable')
      );

      const options: MemoryOptions = {
        action: 'update',
        ruleId: 'existing-rule-id',
        content: 'Updated content',
      };

      const result = await memoryTool.execute(options);

      expect(result.success).toBe(true);
      expect(result.fallback_mode).toBe('unified_queue');
    });

    it('should reject missing rule ID', async () => {
      const options: MemoryOptions = {
        action: 'update',
        content: 'Updated content',
      };

      const result = await memoryTool.execute(options);

      expect(result.success).toBe(false);
      expect(result.message).toContain('Rule ID is required');
    });

    it('should reject empty content', async () => {
      const options: MemoryOptions = {
        action: 'update',
        ruleId: 'existing-rule-id',
        content: '',
      };

      const result = await memoryTool.execute(options);

      expect(result.success).toBe(false);
      expect(result.message).toContain('Content is required');
    });
  });

  describe('remove action', () => {
    it('should queue removal (always uses queue)', async () => {
      const options: MemoryOptions = {
        action: 'remove',
        ruleId: 'rule-to-remove',
      };

      const result = await memoryTool.execute(options);

      expect(result.success).toBe(true);
      expect(result.action).toBe('remove');
      expect(result.fallback_mode).toBe('unified_queue');
      expect(mockStateManager.enqueueUnified).toHaveBeenCalledWith(
        'content',
        'delete',
        'global',
        'memory',
        expect.objectContaining({
          rule_id: 'rule-to-remove',
          action: 'remove',
        }),
        8,
        'main',
        expect.any(Object)
      );
    });

    it('should reject missing rule ID', async () => {
      const options: MemoryOptions = {
        action: 'remove',
      };

      const result = await memoryTool.execute(options);

      expect(result.success).toBe(false);
      expect(result.message).toContain('Rule ID is required');
    });
  });

  describe('list action', () => {
    it('should list global rules', async () => {
      const options: MemoryOptions = {
        action: 'list',
        scope: 'global',
      };

      const result = await memoryTool.execute(options);

      expect(result.success).toBe(true);
      expect(result.action).toBe('list');
      expect(result.rules).toBeDefined();
      expect(result.rules!.length).toBe(2);
    });

    it('should list project-scoped rules', async () => {
      const options: MemoryOptions = {
        action: 'list',
        scope: 'project',
      };

      const result = await memoryTool.execute(options);

      expect(result.success).toBe(true);
      expect(mockProjectDetector.getProjectInfo).toHaveBeenCalled();
    });

    it('should parse tags from comma-separated string', async () => {
      const options: MemoryOptions = {
        action: 'list',
        scope: 'global',
      };

      const result = await memoryTool.execute(options);

      const ruleWithTags = result.rules?.find((r) => r.id === 'rule-2');
      expect(ruleWithTags?.tags).toEqual(['testing', 'quality']);
    });

    it('should parse priority from string', async () => {
      const options: MemoryOptions = {
        action: 'list',
        scope: 'global',
      };

      const result = await memoryTool.execute(options);

      const ruleWithPriority = result.rules?.find((r) => r.id === 'rule-1');
      expect(ruleWithPriority?.priority).toBe(10);
    });

    it('should handle Qdrant errors gracefully', async () => {
      const QdrantClientMock = await import('@qdrant/js-client-rest');
      vi.mocked(QdrantClientMock.QdrantClient).mockImplementationOnce(
        () =>
          ({
            scroll: vi.fn().mockRejectedValue(new Error('Collection not found')),
          }) as unknown as ReturnType<typeof QdrantClientMock.QdrantClient>
      );

      const newTool = new MemoryTool(
        { qdrantUrl: 'http://localhost:6333' },
        mockDaemonClient,
        mockStateManager,
        mockProjectDetector
      );

      const result = await newTool.execute({ action: 'list', scope: 'global' });

      expect(result.success).toBe(false);
      expect(result.message).toContain('Failed to list rules');
    });
  });

  describe('unknown action', () => {
    it('should return error for unknown action', async () => {
      const options = {
        action: 'unknown' as unknown as 'add',
      };

      const result = await memoryTool.execute(options);

      expect(result.success).toBe(false);
      expect(result.message).toContain('Unknown action');
    });
  });
});

describe('MemoryTool queue integration', () => {
  it('should call enqueueUnified with correct parameters', async () => {
    const mockDaemonClient = createMockDaemonClient();
    const mockStateManager = createMockStateManager();
    const mockProjectDetector = createMockProjectDetector();

    // Make daemon fail to trigger queue fallback
    vi.mocked(mockDaemonClient.ingestText).mockRejectedValue(
      new Error('Daemon unavailable')
    );

    const memoryTool = new MemoryTool(
      { qdrantUrl: 'http://localhost:6333' },
      mockDaemonClient,
      mockStateManager,
      mockProjectDetector
    );

    await memoryTool.execute({
      action: 'add',
      content: 'Test rule',
      scope: 'project',
      projectId: 'my-project',
      title: 'Test Title',
      tags: ['tag1', 'tag2'],
      priority: 5,
    });

    expect(mockStateManager.enqueueUnified).toHaveBeenCalledWith(
      'content',
      'ingest',
      'my-project',
      'memory',
      expect.objectContaining({
        content: 'Test rule',
        scope: 'project',
        project_id: 'my-project',
        title: 'Test Title',
        tags: ['tag1', 'tag2'],
        priority: 5,
      }),
      8,
      'main',
      expect.objectContaining({ source: 'mcp_memory_tool' })
    );
  });
});
