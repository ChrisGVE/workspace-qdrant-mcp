/**
 * Tests for RulesTool - duplication detection and queue integration
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { RulesTool } from '../../src/tools/rules.js';
import type { DaemonClient } from '../../src/clients/daemon-client.js';
import type { SqliteStateManager } from '../../src/clients/sqlite-state-manager.js';
import type { ProjectDetector } from '../../src/utils/project-detector.js';

// Mock the Qdrant client
vi.mock('@qdrant/js-client-rest', () => ({
  QdrantClient: vi.fn().mockImplementation(() => ({
    scroll: vi.fn().mockResolvedValue({ points: [] }),
    search: vi.fn().mockResolvedValue([]),
  })),
}));

function createMockDaemonClient(): DaemonClient {
  return {
    isConnected: vi.fn().mockReturnValue(true),
    ingestText: vi.fn().mockResolvedValue({ success: true, document_id: 'new-rule-id', chunks_created: 1 }),
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
      data: { queueId: 'queued-rule-id', isNew: true, idempotencyKey: 'test-key' },
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
      'rules',
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
