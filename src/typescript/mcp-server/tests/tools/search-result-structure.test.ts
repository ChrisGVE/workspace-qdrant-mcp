/**
 * Tests for SearchTool response structure and parent context expansion.
 */

import { describe, it, expect, vi } from 'vitest';
import { SearchTool, type ParentContext } from '../../src/tools/search.js';
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

describe('Search result structure', () => {
  it('should have correct response structure', async () => {
    const searchTool = new SearchTool(
      { qdrantUrl: 'http://localhost:6333' },
      createMockDaemonClient(),
      createMockStateManager(),
      createMockProjectDetector()
    );

    const result = await searchTool.search({ query: 'test' });

    expect(result).toHaveProperty('results');
    expect(result).toHaveProperty('total');
    expect(result).toHaveProperty('query');
    expect(result).toHaveProperty('mode');
    expect(result).toHaveProperty('scope');
    expect(result).toHaveProperty('collections_searched');
    expect(Array.isArray(result.results)).toBe(true);
    expect(typeof result.total).toBe('number');
    expect(result.query).toBe('test');
  });
});

describe('Parent context expansion', () => {
  it('should not expand context when expandContext is false', async () => {
    const searchTool = new SearchTool(
      { qdrantUrl: 'http://localhost:6333' },
      createMockDaemonClient(),
      createMockStateManager(),
      createMockProjectDetector()
    );

    const result = await searchTool.search({ query: 'test', expandContext: false });

    expect(result.results.every((r) => r.parent_context === undefined)).toBe(true);
  });

  it('should retrieve parent by ID', async () => {
    const QdrantClientMock = await import('@qdrant/js-client-rest');
    vi.mocked(QdrantClientMock.QdrantClient).mockImplementationOnce(
      () =>
        ({
          search: vi.fn(),
          scroll: vi.fn(),
          retrieve: vi.fn().mockResolvedValue([
            {
              id: 'parent-123',
              payload: {
                unit_type: 'pdf_page',
                unit_text: 'Full page content here',
                locator: { page: 1 },
              },
            },
          ]),
          getCollection: vi.fn().mockResolvedValue({ status: 'green' }),
        }) as unknown as ReturnType<typeof QdrantClientMock.QdrantClient>
    );

    const searchTool = new SearchTool(
      { qdrantUrl: 'http://localhost:6333' },
      createMockDaemonClient(),
      createMockStateManager(),
      createMockProjectDetector()
    );

    const parent: ParentContext | null = await searchTool.retrieveParent('parent-123', 'libraries');

    expect(parent).not.toBeNull();
    expect(parent!.parent_unit_id).toBe('parent-123');
    expect(parent!.unit_type).toBe('pdf_page');
    expect(parent!.unit_text).toBe('Full page content here');
    expect(parent!.locator).toEqual({ page: 1 });
  });

  it('should return null when parent does not exist', async () => {
    const QdrantClientMock = await import('@qdrant/js-client-rest');
    vi.mocked(QdrantClientMock.QdrantClient).mockImplementationOnce(
      () =>
        ({
          search: vi.fn(),
          scroll: vi.fn(),
          retrieve: vi.fn().mockResolvedValue([]),
          getCollection: vi.fn().mockResolvedValue({ status: 'green' }),
        }) as unknown as ReturnType<typeof QdrantClientMock.QdrantClient>
    );

    const searchTool = new SearchTool(
      { qdrantUrl: 'http://localhost:6333' },
      createMockDaemonClient(),
      createMockStateManager(),
      createMockProjectDetector()
    );

    const parent = await searchTool.retrieveParent('nonexistent', 'libraries');

    expect(parent).toBeNull();
  });
});
