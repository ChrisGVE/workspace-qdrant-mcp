/**
 * Tests for RulesTool - list, remove, and unknown actions
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
    upsertRulesMirror: vi.fn(),
    deleteRulesMirror: vi.fn(),
    listRulesMirror: vi.fn().mockReturnValue([]),
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
        'rules',
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
