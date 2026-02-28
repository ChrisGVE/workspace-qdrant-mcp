/**
 * Tests for RulesTool - add and update actions
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
});
