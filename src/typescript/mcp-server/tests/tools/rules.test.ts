/**
 * Tests for RulesTool
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { RulesTool, type RuleOptions } from '../../src/tools/rules.js';
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
    search: vi.fn().mockResolvedValue([]),
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
    upsertMemoryMirror: vi.fn(),
    deleteMemoryMirror: vi.fn(),
    listMemoryMirror: vi.fn().mockReturnValue([]),
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

describe('RulesTool', () => {
  let rulesTool: RulesTool;
  let mockDaemonClient: DaemonClient;
  let mockStateManager: SqliteStateManager;
  let mockProjectDetector: ProjectDetector;

  beforeEach(() => {
    vi.clearAllMocks();
    mockDaemonClient = createMockDaemonClient();
    mockStateManager = createMockStateManager();
    mockProjectDetector = createMockProjectDetector();

    rulesTool = new RulesTool(
      { qdrantUrl: 'http://localhost:6333' },
      mockDaemonClient,
      mockStateManager,
      mockProjectDetector
    );
  });

  describe('add action', () => {
    it('should add a global rule via daemon', async () => {
      const options: RuleOptions = {
        action: 'add',
        label: 'write-tests',
        content: 'Always write tests',
        scope: 'global',
        title: 'Testing Rule',
      };

      const result = await rulesTool.execute(options);

      expect(result.success).toBe(true);
      expect(result.action).toBe('add');
      expect(result.label).toBe('new-rule-id');
      expect(result.fallback_mode).toBeUndefined();
      expect(mockDaemonClient.ingestText).toHaveBeenCalled();
    });

    it('should add a project-scoped rule', async () => {
      const options: RuleOptions = {
        action: 'add',
        label: 'proj-rule',
        content: 'Project-specific rule',
        scope: 'project',
      };

      const result = await rulesTool.execute(options);

      expect(result.success).toBe(true);
      expect(mockProjectDetector.getProjectInfo).toHaveBeenCalled();
    });

    it('should fallback to queue when daemon fails', async () => {
      vi.mocked(mockDaemonClient.ingestText).mockRejectedValue(
        new Error('Daemon unavailable')
      );

      const options: RuleOptions = {
        action: 'add',
        label: 'test-rule',
        content: 'Test rule',
        scope: 'global',
      };

      const result = await rulesTool.execute(options);

      expect(result.success).toBe(true);
      expect(result.fallback_mode).toBe('unified_queue');
      expect(result.queue_id).toBe('queued-rule-id');
      expect(mockStateManager.enqueueUnified).toHaveBeenCalled();
    });

    it('should reject empty content', async () => {
      const options: RuleOptions = {
        action: 'add',
        label: 'empty-rule',
        content: '',
        scope: 'global',
      };

      const result = await rulesTool.execute(options);

      expect(result.success).toBe(false);
      expect(result.message).toContain('Content is required');
    });

    it('should reject missing label', async () => {
      const options: RuleOptions = {
        action: 'add',
        content: 'Some rule content',
        scope: 'global',
      };

      const result = await rulesTool.execute(options);

      expect(result.success).toBe(false);
      expect(result.message).toContain('Label is required');
    });

    it('should include tags in metadata', async () => {
      const options: RuleOptions = {
        action: 'add',
        label: 'tagged-rule',
        content: 'Rule with tags',
        scope: 'global',
        tags: ['testing', 'quality'],
      };

      await rulesTool.execute(options);

      expect(mockDaemonClient.ingestText).toHaveBeenCalledWith(
        expect.objectContaining({
          metadata: expect.objectContaining({
            tags: 'testing,quality',
          }),
        })
      );
    });

    it('should include priority in metadata', async () => {
      const options: RuleOptions = {
        action: 'add',
        label: 'high-prio',
        content: 'High priority rule',
        scope: 'global',
        priority: 10,
      };

      await rulesTool.execute(options);

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
      const options: RuleOptions = {
        action: 'update',
        label: 'existing-rule-id',
        content: 'Updated rule content',
      };

      const result = await rulesTool.execute(options);

      expect(result.success).toBe(true);
      expect(result.action).toBe('update');
      expect(result.label).toBe('existing-rule-id');
    });

    it('should fallback to queue when daemon fails', async () => {
      vi.mocked(mockDaemonClient.ingestText).mockRejectedValue(
        new Error('Daemon unavailable')
      );

      const options: RuleOptions = {
        action: 'update',
        label: 'existing-rule-id',
        content: 'Updated content',
      };

      const result = await rulesTool.execute(options);

      expect(result.success).toBe(true);
      expect(result.fallback_mode).toBe('unified_queue');
    });

    it('should reject missing label', async () => {
      const options: RuleOptions = {
        action: 'update',
        content: 'Updated content',
      };

      const result = await rulesTool.execute(options);

      expect(result.success).toBe(false);
      expect(result.message).toContain('Label is required');
    });

    it('should reject empty content', async () => {
      const options: RuleOptions = {
        action: 'update',
        label: 'existing-rule-id',
        content: '',
      };

      const result = await rulesTool.execute(options);

      expect(result.success).toBe(false);
      expect(result.message).toContain('Content is required for updating');
    });
  });

  describe('remove action', () => {
    it('should queue removal (always uses queue)', async () => {
      const options: RuleOptions = {
        action: 'remove',
        label: 'rule-to-remove',
      };

      const result = await rulesTool.execute(options);

      expect(result.success).toBe(true);
      expect(result.action).toBe('remove');
      expect(result.fallback_mode).toBe('unified_queue');
      expect(mockStateManager.enqueueUnified).toHaveBeenCalledWith(
        'text',
        'delete',
        'global',
        'memory',
        expect.objectContaining({
          label: 'rule-to-remove',
          action: 'remove',
        }),
        1, // PRIORITY_HIGH
        'main',
        expect.any(Object)
      );
    });

    it('should reject missing label', async () => {
      const options: RuleOptions = {
        action: 'remove',
      };

      const result = await rulesTool.execute(options);

      expect(result.success).toBe(false);
      expect(result.message).toContain('Label is required');
    });
  });

  describe('list action', () => {
    it('should list global rules', async () => {
      const options: RuleOptions = {
        action: 'list',
        scope: 'global',
      };

      const result = await rulesTool.execute(options);

      expect(result.success).toBe(true);
      expect(result.action).toBe('list');
      expect(result.rules).toBeDefined();
      expect(result.rules!.length).toBe(2);
    });

    it('should list project-scoped rules', async () => {
      const options: RuleOptions = {
        action: 'list',
        scope: 'project',
      };

      const result = await rulesTool.execute(options);

      expect(result.success).toBe(true);
      expect(mockProjectDetector.getProjectInfo).toHaveBeenCalled();
    });

    it('should parse tags from comma-separated string', async () => {
      const options: RuleOptions = {
        action: 'list',
        scope: 'global',
      };

      const result = await rulesTool.execute(options);

      const ruleWithTags = result.rules?.find((r) => r.id === 'rule-2');
      expect(ruleWithTags?.tags).toEqual(['testing', 'quality']);
    });

    it('should parse priority from string', async () => {
      const options: RuleOptions = {
        action: 'list',
        scope: 'global',
      };

      const result = await rulesTool.execute(options);

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

      const newTool = new RulesTool(
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

      const result = await rulesTool.execute(options);

      expect(result.success).toBe(false);
      expect(result.message).toContain('Unknown action');
    });
  });
});

describe('RulesTool duplication detection', () => {
  let mockDaemonClient: DaemonClient;
  let mockStateManager: SqliteStateManager;
  let mockProjectDetector: ProjectDetector;

  beforeEach(() => {
    vi.clearAllMocks();
    mockDaemonClient = createMockDaemonClient();
    mockStateManager = createMockStateManager();
    mockProjectDetector = createMockProjectDetector();
  });

  it('should block add when similar rules exist above threshold', async () => {
    // Configure embedText to return a valid embedding
    vi.mocked(mockDaemonClient.embedText).mockResolvedValue({
      embedding: [0.1, 0.2, 0.3],
    } as never);

    // Configure Qdrant search to return a similar rule
    const QdrantMock = await import('@qdrant/js-client-rest');
    vi.mocked(QdrantMock.QdrantClient).mockImplementationOnce(
      () =>
        ({
          scroll: vi.fn().mockResolvedValue({ points: [] }),
          search: vi.fn().mockResolvedValue([
            {
              id: 'existing-rule-1',
              score: 0.85,
              payload: {
                content: 'Always write tests before code',
                scope: 'global',
                label: 'tdd-rule',
                title: 'TDD Rule',
              },
            },
          ]),
        }) as unknown as ReturnType<typeof QdrantMock.QdrantClient>
    );

    const rulesTool = new RulesTool(
      { qdrantUrl: 'http://localhost:6333', duplicationThreshold: 0.7 },
      mockDaemonClient,
      mockStateManager,
      mockProjectDetector
    );

    const result = await rulesTool.execute({
      action: 'add',
      label: 'write-tests',
      content: 'Always write tests before writing code',
      scope: 'global',
    });

    expect(result.success).toBe(false);
    expect(result.similar_rules).toBeDefined();
    expect(result.similar_rules!.length).toBe(1);
    expect(result.similar_rules![0]!.similarity).toBe(0.85);
    expect(result.similar_rules![0]!.content).toBe('Always write tests before code');
    expect(result.message).toContain('similar rule');
    // Should NOT have called ingestText since duplication was detected
    expect(mockDaemonClient.ingestText).not.toHaveBeenCalled();
  });

  it('should allow add when no similar rules exist', async () => {
    vi.mocked(mockDaemonClient.embedText).mockResolvedValue({
      embedding: [0.1, 0.2, 0.3],
    } as never);

    // search returns empty — no similar rules
    const QdrantMock = await import('@qdrant/js-client-rest');
    vi.mocked(QdrantMock.QdrantClient).mockImplementationOnce(
      () =>
        ({
          scroll: vi.fn().mockResolvedValue({ points: [] }),
          search: vi.fn().mockResolvedValue([]),
        }) as unknown as ReturnType<typeof QdrantMock.QdrantClient>
    );

    const rulesTool = new RulesTool(
      { qdrantUrl: 'http://localhost:6333', duplicationThreshold: 0.7 },
      mockDaemonClient,
      mockStateManager,
      mockProjectDetector
    );

    const result = await rulesTool.execute({
      action: 'add',
      label: 'unique-rule',
      content: 'A completely unique rule',
      scope: 'global',
    });

    expect(result.success).toBe(true);
    expect(result.similar_rules).toBeUndefined();
    expect(mockDaemonClient.ingestText).toHaveBeenCalled();
  });

  it('should proceed with add when embedding fails', async () => {
    // Embedding service is down
    vi.mocked(mockDaemonClient.embedText).mockRejectedValue(
      new Error('Embedding service unavailable')
    );

    const rulesTool = new RulesTool(
      { qdrantUrl: 'http://localhost:6333', duplicationThreshold: 0.7 },
      mockDaemonClient,
      mockStateManager,
      mockProjectDetector
    );

    const result = await rulesTool.execute({
      action: 'add',
      label: 'fallback-rule',
      content: 'Rule added despite embedding failure',
      scope: 'global',
    });

    // Should still succeed — embedding failure doesn't block the add
    expect(result.success).toBe(true);
    expect(result.similar_rules).toBeUndefined();
    expect(mockDaemonClient.ingestText).toHaveBeenCalled();
  });

  it('should proceed with add when embedText returns empty embedding', async () => {
    vi.mocked(mockDaemonClient.embedText).mockResolvedValue({
      embedding: [],
    } as never);

    const rulesTool = new RulesTool(
      { qdrantUrl: 'http://localhost:6333', duplicationThreshold: 0.7 },
      mockDaemonClient,
      mockStateManager,
      mockProjectDetector
    );

    const result = await rulesTool.execute({
      action: 'add',
      label: 'no-embed-rule',
      content: 'Rule added with empty embedding',
      scope: 'global',
    });

    expect(result.success).toBe(true);
    expect(mockDaemonClient.ingestText).toHaveBeenCalled();
  });

  it('should respect custom duplication threshold', async () => {
    vi.mocked(mockDaemonClient.embedText).mockResolvedValue({
      embedding: [0.1, 0.2, 0.3],
    } as never);

    // Return a result with score 0.75 — above default 0.7 but below custom 0.9
    const QdrantMock = await import('@qdrant/js-client-rest');
    vi.mocked(QdrantMock.QdrantClient).mockImplementationOnce(
      () =>
        ({
          scroll: vi.fn().mockResolvedValue({ points: [] }),
          search: vi.fn().mockResolvedValue([
            {
              id: 'rule-mid',
              score: 0.75,
              payload: {
                content: 'Moderately similar rule',
                scope: 'global',
              },
            },
          ]),
        }) as unknown as ReturnType<typeof QdrantMock.QdrantClient>
    );

    const rulesTool = new RulesTool(
      { qdrantUrl: 'http://localhost:6333', duplicationThreshold: 0.9 },
      mockDaemonClient,
      mockStateManager,
      mockProjectDetector
    );

    const result = await rulesTool.execute({
      action: 'add',
      label: 'threshold-rule',
      content: 'Test custom threshold',
      scope: 'global',
    });

    // 0.75 < 0.9 threshold — should allow the add
    expect(result.success).toBe(true);
    expect(mockDaemonClient.ingestText).toHaveBeenCalled();
  });
});

describe('RulesTool queue integration', () => {
  it('should call enqueueUnified with correct parameters', async () => {
    const mockDaemonClient = createMockDaemonClient();
    const mockStateManager = createMockStateManager();
    const mockProjectDetector = createMockProjectDetector();

    // Make daemon fail to trigger queue fallback
    vi.mocked(mockDaemonClient.ingestText).mockRejectedValue(
      new Error('Daemon unavailable')
    );

    const rulesTool = new RulesTool(
      { qdrantUrl: 'http://localhost:6333' },
      mockDaemonClient,
      mockStateManager,
      mockProjectDetector
    );

    await rulesTool.execute({
      action: 'add',
      label: 'test-rule',
      content: 'Test rule',
      scope: 'project',
      projectId: 'my-project',
      title: 'Test Title',
      tags: ['tag1', 'tag2'],
      priority: 5,
    });

    expect(mockStateManager.enqueueUnified).toHaveBeenCalledWith(
      'text',
      'add',
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
      1, // PRIORITY_HIGH
      'main',
      expect.objectContaining({ source: 'mcp_rules_tool' })
    );
  });
});
