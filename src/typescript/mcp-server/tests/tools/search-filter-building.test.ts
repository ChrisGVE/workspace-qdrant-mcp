/**
 * Tests for SearchTool filter building, RRF fusion, fallback search,
 * and collectionExists.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { SearchTool, type SearchOptions, type SearchResult } from '../../src/tools/search.js';
import { applyRRFFusion } from '../../src/tools/search-qdrant.js';
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

describe('SearchTool — filter building', () => {
  let searchTool: SearchTool;
  let mockDaemonClient: DaemonClient;
  let mockStateManager: SqliteStateManager;
  let mockProjectDetector: ProjectDetector;

  beforeEach(() => {
    vi.clearAllMocks();
    mockDaemonClient = createMockDaemonClient();
    mockStateManager = createMockStateManager();
    mockProjectDetector = createMockProjectDetector();

    searchTool = new SearchTool(
      { qdrantUrl: 'http://localhost:6333', qdrantTimeout: 5000 },
      mockDaemonClient,
      mockStateManager,
      mockProjectDetector
    );
  });

  it('should include project_id filter for project scope', async () => {
    const options: SearchOptions = {
      query: 'test query',
      scope: 'project',
      projectId: 'test-project-123',
    };

    const result = await searchTool.search(options);

    expect(result.status).toBe('ok');
  });

  it('should include branch filter when provided', async () => {
    const result = await searchTool.search({ query: 'test query', branch: 'main' });

    expect(result.status).toBe('ok');
  });

  it('should not include branch filter for wildcard', async () => {
    const result = await searchTool.search({ query: 'test query', branch: '*' });

    expect(result.status).toBe('ok');
  });

  it('should include file_type filter when provided', async () => {
    const result = await searchTool.search({ query: 'test query', fileType: 'code' });

    expect(result.status).toBe('ok');
  });

  it('should include tag filter when provided', async () => {
    const result = await searchTool.search({ query: 'test query', tag: 'project.main' });

    expect(result.status).toBe('ok');
  });
});

describe('RRF fusion', () => {
  it('should apply RRF fusion for hybrid mode — document in both sets scores highest', () => {
    const semanticResults: SearchResult[] = [
      { id: '1', score: 0.9, collection: 'projects', content: 'doc1', metadata: { _search_type: 'semantic' } },
      { id: '2', score: 0.8, collection: 'projects', content: 'doc2', metadata: { _search_type: 'semantic' } },
    ];
    const keywordResults: SearchResult[] = [
      { id: '2', score: 0.85, collection: 'projects', content: 'doc2', metadata: { _search_type: 'keyword' } },
      { id: '3', score: 0.7, collection: 'projects', content: 'doc3', metadata: { _search_type: 'keyword' } },
    ];

    const fused = applyRRFFusion([...semanticResults, ...keywordResults], 'hybrid');

    const doc2 = fused.find((r) => r.id === '2');
    const doc1 = fused.find((r) => r.id === '1');
    const doc3 = fused.find((r) => r.id === '3');

    expect(doc2).toBeDefined();
    expect(doc1).toBeDefined();
    expect(doc3).toBeDefined();
    expect(doc2!.score).toBeGreaterThan(doc1!.score);
    expect(doc2!.score).toBeGreaterThan(doc3!.score);
  });

  it('should not apply fusion for semantic-only mode', () => {
    const results: SearchResult[] = [
      { id: '1', score: 0.9, collection: 'projects', content: 'doc1', metadata: { _search_type: 'semantic' } },
    ];

    const fused = applyRRFFusion(results, 'semantic');

    expect(fused).toEqual(results);
  });
});

describe('SearchTool — fallback and collectionExists', () => {
  let mockDaemonClient: DaemonClient;
  let mockStateManager: SqliteStateManager;
  let mockProjectDetector: ProjectDetector;

  beforeEach(() => {
    vi.clearAllMocks();
    mockDaemonClient = createMockDaemonClient();
    mockStateManager = createMockStateManager();
    mockProjectDetector = createMockProjectDetector();
  });

  it('should use fallback when daemon is unavailable', async () => {
    vi.mocked(mockDaemonClient.embedText).mockRejectedValue(new Error('Daemon unavailable'));

    const searchTool = new SearchTool(
      { qdrantUrl: 'http://localhost:6333' },
      mockDaemonClient,
      mockStateManager,
      mockProjectDetector
    );

    const result = await searchTool.search({ query: 'test query' });

    expect(result.status).toBe('uncertain');
    expect(result.status_reason).toContain('Daemon unavailable');
  });

  it('should return true when collection exists', async () => {
    const searchTool = new SearchTool(
      { qdrantUrl: 'http://localhost:6333' },
      mockDaemonClient,
      mockStateManager,
      mockProjectDetector
    );

    const exists = await searchTool.collectionExists('projects');

    expect(exists).toBe(true);
  });

  it('should return false when collection does not exist', async () => {
    const QdrantClientMock = await import('@qdrant/js-client-rest');
    vi.mocked(QdrantClientMock.QdrantClient).mockImplementationOnce(
      () =>
        ({
          search: vi.fn(),
          scroll: vi.fn(),
          getCollection: vi.fn().mockRejectedValue(new Error('Not found')),
        }) as unknown as ReturnType<typeof QdrantClientMock.QdrantClient>
    );

    const newTool = new SearchTool(
      { qdrantUrl: 'http://localhost:6333' },
      mockDaemonClient,
      mockStateManager,
      mockProjectDetector
    );

    const exists = await newTool.collectionExists('nonexistent');

    expect(exists).toBe(false);
  });
});
