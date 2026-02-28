/**
 * Tests for SearchTool tag filtering and component filtering.
 */

import { describe, it, expect, vi } from 'vitest';
import { SearchTool } from '../../src/tools/search.js';
import type { DaemonClient } from '../../src/clients/daemon-client.js';
import type { SqliteStateManager } from '../../src/clients/sqlite-state-manager.js';
import type { ProjectDetector } from '../../src/utils/project-detector.js';

// Mock the Qdrant client
vi.mock('@qdrant/js-client-rest', () => ({
  QdrantClient: vi.fn().mockImplementation(() => ({
    search: vi.fn().mockResolvedValue([]),
    scroll: vi.fn().mockResolvedValue({ points: [] }),
    retrieve: vi.fn().mockResolvedValue([]),
    getCollection: vi.fn().mockResolvedValue({ status: 'green' }),
  })),
}));

function createMockDaemonClient(): DaemonClient {
  return {
    connect: vi.fn().mockResolvedValue(undefined),
    close: vi.fn(),
    isConnected: vi.fn().mockReturnValue(true),
    getConnectionState: vi.fn().mockReturnValue({ connected: true }),
    healthCheck: vi.fn().mockResolvedValue({ status: 1 }),
    getStatus: vi.fn().mockResolvedValue({}),
    getMetrics: vi.fn().mockResolvedValue({}),
    notifyServerStatus: vi.fn().mockResolvedValue(undefined),
    registerProject: vi.fn().mockResolvedValue({ created: true }),
    deprioritizeProject: vi.fn().mockResolvedValue({ success: true }),
    heartbeat: vi.fn().mockResolvedValue({ acknowledged: true }),
    ingestText: vi.fn().mockResolvedValue({ success: true }),
    embedText: vi.fn().mockResolvedValue({
      embedding: new Array(384).fill(0).map((_, i) => i / 384),
      dimensions: 384,
      model_name: 'all-MiniLM-L6-v2',
      success: true,
    }),
    generateSparseVector: vi.fn().mockResolvedValue({
      indices_values: { 1: 0.5, 2: 0.3, 3: 0.2 },
      vocab_size: 1000,
      success: true,
    }),
  } as unknown as DaemonClient;
}

function createMockStateManager(): SqliteStateManager {
  return {
    initialize: vi.fn().mockReturnValue({ status: 'ok' }),
    close: vi.fn(),
    getProjectByPath: vi.fn().mockResolvedValue(null),
    listProjects: vi.fn().mockResolvedValue([]),
    logSearchEvent: vi.fn(),
    updateSearchEvent: vi.fn(),
    getMatchingTags: vi.fn().mockReturnValue([]),
    getKeywordBasketsForTags: vi.fn().mockReturnValue([]),
    listTags: vi.fn().mockReturnValue([]),
    getTagHierarchy: vi.fn().mockReturnValue([]),
    getWatchFolderIdByTenantId: vi.fn().mockReturnValue(null),
    getActiveBasePoints: vi.fn().mockReturnValue([]),
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

// ── Tag filtering tests (Task 37) ──

describe('SearchTool — tag filtering', () => {
  it('should accept a single tag in search options', async () => {
    const tool = new SearchTool(
      { qdrantUrl: 'http://localhost:6333', enableTagExpansion: false },
      createMockDaemonClient(),
      createMockStateManager(),
      createMockProjectDetector(),
    );

    const result = await tool.search({
      query: 'test query',
      tag: 'async runtime',
      projectId: 'test-project-123',
    });

    expect(result).toBeDefined();
    expect(result.query).toBe('test query');
  });

  it('should accept a tags array in search options', async () => {
    const tool = new SearchTool(
      { qdrantUrl: 'http://localhost:6333', enableTagExpansion: false },
      createMockDaemonClient(),
      createMockStateManager(),
      createMockProjectDetector(),
    );

    const result = await tool.search({
      query: 'test query',
      tags: ['async runtime', 'error handling'],
      projectId: 'test-project-123',
    });

    expect(result).toBeDefined();
    expect(result.query).toBe('test query');
  });

  it('should pass tag as a must condition on concept_tags', async () => {
    const QdrantClientMock = await import('@qdrant/js-client-rest');
    const searchFn = vi.fn().mockResolvedValue([]);
    vi.mocked(QdrantClientMock.QdrantClient).mockImplementationOnce(
      () =>
        ({
          search: searchFn,
          getCollection: vi.fn().mockResolvedValue({}),
        }) as unknown as ReturnType<typeof QdrantClientMock.QdrantClient>
    );

    const tool = new SearchTool(
      { qdrantUrl: 'http://localhost:6333', enableTagExpansion: false },
      createMockDaemonClient(),
      createMockStateManager(),
      createMockProjectDetector(),
    );

    await tool.search({
      query: 'test',
      tag: 'async runtime',
      mode: 'semantic',
      projectId: 'test-project-123',
    });

    expect(searchFn).toHaveBeenCalled();
    const filter = searchFn.mock.calls[0][1].filter;

    expect(filter).toBeDefined();
    expect(filter.must).toBeDefined();
    const tagCondition = filter.must.find(
      (c: Record<string, unknown>) => c.key === 'concept_tags'
    );
    expect(tagCondition).toBeDefined();
    expect(tagCondition.match.value).toBe('async runtime');
  });

  it('should pass tags array as a should sub-condition', async () => {
    const QdrantClientMock = await import('@qdrant/js-client-rest');
    const searchFn = vi.fn().mockResolvedValue([]);
    vi.mocked(QdrantClientMock.QdrantClient).mockImplementationOnce(
      () =>
        ({
          search: searchFn,
          getCollection: vi.fn().mockResolvedValue({}),
        }) as unknown as ReturnType<typeof QdrantClientMock.QdrantClient>
    );

    const tool = new SearchTool(
      { qdrantUrl: 'http://localhost:6333', enableTagExpansion: false },
      createMockDaemonClient(),
      createMockStateManager(),
      createMockProjectDetector(),
    );

    await tool.search({
      query: 'test',
      tags: ['async runtime', 'error handling'],
      mode: 'semantic',
      projectId: 'test-project-123',
    });

    expect(searchFn).toHaveBeenCalled();
    const filter = searchFn.mock.calls[0][1].filter;

    expect(filter).toBeDefined();
    expect(filter.must).toBeDefined();
    const shouldCondition = filter.must.find(
      (c: Record<string, unknown>) => c.should !== undefined
    );
    expect(shouldCondition).toBeDefined();
    expect(shouldCondition.should).toHaveLength(2);
  });
});

// ── Component filtering tests ──

describe('SearchTool — component filtering', () => {
  it('should pass component as exact + prefix match conditions', async () => {
    const QdrantClientMock = await import('@qdrant/js-client-rest');
    const searchFn = vi.fn().mockResolvedValue([]);
    vi.mocked(QdrantClientMock.QdrantClient).mockImplementationOnce(
      () =>
        ({
          search: searchFn,
          getCollection: vi.fn().mockResolvedValue({}),
        }) as unknown as ReturnType<typeof QdrantClientMock.QdrantClient>
    );

    const tool = new SearchTool(
      { qdrantUrl: 'http://localhost:6333', enableTagExpansion: false },
      createMockDaemonClient(),
      createMockStateManager(),
      createMockProjectDetector(),
    );

    await tool.search({
      query: 'test',
      component: 'daemon.core',
      mode: 'semantic',
      projectId: 'test-project-123',
    });

    expect(searchFn).toHaveBeenCalled();
    const filter = searchFn.mock.calls[0][1].filter;

    expect(filter).toBeDefined();
    expect(filter.must).toBeDefined();

    const componentCondition = filter.must.find(
      (c: Record<string, unknown>) =>
        Array.isArray(c.should) &&
        c.should.some((s: Record<string, unknown>) => s.key === 'component_id')
    );
    expect(componentCondition).toBeDefined();
    expect(componentCondition.should).toHaveLength(2);
    expect(componentCondition.should[0]).toEqual({
      key: 'component_id',
      match: { value: 'daemon.core' },
    });
    expect(componentCondition.should[1]).toEqual({
      key: 'component_id',
      match: { text: 'daemon.core.' },
    });
  });

  it('should not add component_id condition when component is not specified', async () => {
    const QdrantClientMock = await import('@qdrant/js-client-rest');
    const searchFn = vi.fn().mockResolvedValue([]);
    vi.mocked(QdrantClientMock.QdrantClient).mockImplementationOnce(
      () =>
        ({
          search: searchFn,
          getCollection: vi.fn().mockResolvedValue({}),
        }) as unknown as ReturnType<typeof QdrantClientMock.QdrantClient>
    );

    const tool = new SearchTool(
      { qdrantUrl: 'http://localhost:6333', enableTagExpansion: false },
      createMockDaemonClient(),
      createMockStateManager(),
      createMockProjectDetector(),
    );

    await tool.search({
      query: 'test',
      mode: 'semantic',
      projectId: 'test-project-123',
    });

    expect(searchFn).toHaveBeenCalled();
    const filter = searchFn.mock.calls[0][1].filter;

    if (filter && filter.must) {
      const componentCondition = filter.must.find(
        (c: Record<string, unknown>) =>
          Array.isArray(c.should) &&
          c.should.some((s: Record<string, unknown>) => s.key === 'component_id')
      );
      expect(componentCondition).toBeUndefined();
    }
  });
});
