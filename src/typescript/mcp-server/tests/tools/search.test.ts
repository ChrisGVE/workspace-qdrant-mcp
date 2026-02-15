/**
 * Tests for SearchTool
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { SearchTool, type SearchOptions, type SearchResult, type ParentContext } from '../../src/tools/search.js';
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

// Create mock services
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

describe('SearchTool', () => {
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
      {
        qdrantUrl: 'http://localhost:6333',
        qdrantTimeout: 5000,
      },
      mockDaemonClient,
      mockStateManager,
      mockProjectDetector
    );
  });

  describe('search', () => {
    it('should call embedText for semantic mode', async () => {
      const options: SearchOptions = {
        query: 'test query',
        mode: 'semantic',
      };

      await searchTool.search(options);

      expect(mockDaemonClient.embedText).toHaveBeenCalledWith({
        text: 'test query',
      });
      expect(mockDaemonClient.generateSparseVector).not.toHaveBeenCalled();
    });

    it('should call generateSparseVector for keyword mode', async () => {
      const options: SearchOptions = {
        query: 'test query',
        mode: 'keyword',
      };

      await searchTool.search(options);

      expect(mockDaemonClient.generateSparseVector).toHaveBeenCalledWith({
        text: 'test query',
      });
      expect(mockDaemonClient.embedText).not.toHaveBeenCalled();
    });

    it('should call both for hybrid mode', async () => {
      const options: SearchOptions = {
        query: 'test query',
        mode: 'hybrid',
      };

      await searchTool.search(options);

      expect(mockDaemonClient.embedText).toHaveBeenCalledWith({
        text: 'test query',
      });
      expect(mockDaemonClient.generateSparseVector).toHaveBeenCalledWith({
        text: 'test query',
      });
    });

    it('should use default hybrid mode when not specified', async () => {
      const options: SearchOptions = {
        query: 'test query',
      };

      const result = await searchTool.search(options);

      expect(result.mode).toBe('hybrid');
    });

    it('should use project scope by default', async () => {
      const options: SearchOptions = {
        query: 'test query',
      };

      const result = await searchTool.search(options);

      expect(result.scope).toBe('project');
    });

    it('should search projects collection for project scope', async () => {
      const options: SearchOptions = {
        query: 'test query',
        scope: 'project',
      };

      const result = await searchTool.search(options);

      expect(result.collections_searched).toContain('projects');
      expect(result.collections_searched).not.toContain('libraries');
    });

    it('should search both collections for all scope', async () => {
      const options: SearchOptions = {
        query: 'test query',
        scope: 'all',
      };

      const result = await searchTool.search(options);

      expect(result.collections_searched).toContain('projects');
      expect(result.collections_searched).toContain('libraries');
    });

    it('should include libraries when includeLibraries is true', async () => {
      const options: SearchOptions = {
        query: 'test query',
        scope: 'project',
        includeLibraries: true,
      };

      const result = await searchTool.search(options);

      expect(result.collections_searched).toContain('projects');
      expect(result.collections_searched).toContain('libraries');
    });

    it('should use explicit collection when provided', async () => {
      const options: SearchOptions = {
        query: 'test query',
        collection: 'memory',
      };

      const result = await searchTool.search(options);

      expect(result.collections_searched).toEqual(['memory']);
    });

    it('should get project ID for project scope', async () => {
      const options: SearchOptions = {
        query: 'test query',
        scope: 'project',
      };

      await searchTool.search(options);

      expect(mockProjectDetector.getProjectInfo).toHaveBeenCalled();
    });

    it('should not get project ID when explicitly provided', async () => {
      const options: SearchOptions = {
        query: 'test query',
        scope: 'project',
        projectId: 'explicit-project-id',
      };

      await searchTool.search(options);

      expect(mockProjectDetector.getProjectInfo).not.toHaveBeenCalled();
    });
  });

  describe('filter building', () => {
    it('should include project_id filter for project scope', async () => {
      // We'll test filter building indirectly through the search results
      const options: SearchOptions = {
        query: 'test query',
        scope: 'project',
        projectId: 'test-project-123',
      };

      const result = await searchTool.search(options);

      // Search should complete without error
      expect(result.status).toBe('ok');
    });

    it('should include branch filter when provided', async () => {
      const options: SearchOptions = {
        query: 'test query',
        branch: 'main',
      };

      const result = await searchTool.search(options);

      expect(result.status).toBe('ok');
    });

    it('should not include branch filter for wildcard', async () => {
      const options: SearchOptions = {
        query: 'test query',
        branch: '*',
      };

      const result = await searchTool.search(options);

      expect(result.status).toBe('ok');
    });

    it('should include file_type filter when provided', async () => {
      const options: SearchOptions = {
        query: 'test query',
        fileType: 'code',
      };

      const result = await searchTool.search(options);

      expect(result.status).toBe('ok');
    });

    it('should include tag filter when provided', async () => {
      const options: SearchOptions = {
        query: 'test query',
        tag: 'project.main',
      };

      const result = await searchTool.search(options);

      expect(result.status).toBe('ok');
    });
  });

  describe('RRF fusion', () => {
    it('should apply RRF fusion for hybrid mode with both result types', () => {
      // Test the fusion algorithm directly
      const semanticResults: SearchResult[] = [
        { id: '1', score: 0.9, collection: 'projects', content: 'doc1', metadata: { _search_type: 'semantic' } },
        { id: '2', score: 0.8, collection: 'projects', content: 'doc2', metadata: { _search_type: 'semantic' } },
      ];
      const keywordResults: SearchResult[] = [
        { id: '2', score: 0.85, collection: 'projects', content: 'doc2', metadata: { _search_type: 'keyword' } },
        { id: '3', score: 0.7, collection: 'projects', content: 'doc3', metadata: { _search_type: 'keyword' } },
      ];

      // Access the private method via any cast for testing
      const tool = searchTool as unknown as {
        applyRRFFusion: (results: SearchResult[], mode: string) => SearchResult[];
      };
      const combined = [...semanticResults, ...keywordResults];
      const fused = tool.applyRRFFusion(combined, 'hybrid');

      // doc2 should have highest score (appears in both)
      const doc2 = fused.find((r) => r.id === '2');
      const doc1 = fused.find((r) => r.id === '1');
      const doc3 = fused.find((r) => r.id === '3');

      expect(doc2).toBeDefined();
      expect(doc1).toBeDefined();
      expect(doc3).toBeDefined();

      // doc2 should have higher RRF score than doc1 and doc3
      expect(doc2!.score).toBeGreaterThan(doc1!.score);
      expect(doc2!.score).toBeGreaterThan(doc3!.score);
    });

    it('should not apply fusion for semantic-only mode', () => {
      const results: SearchResult[] = [
        { id: '1', score: 0.9, collection: 'projects', content: 'doc1', metadata: { _search_type: 'semantic' } },
      ];

      const tool = searchTool as unknown as {
        applyRRFFusion: (results: SearchResult[], mode: string) => SearchResult[];
      };
      const fused = tool.applyRRFFusion(results, 'semantic');

      // Results should be unchanged
      expect(fused).toEqual(results);
    });
  });

  describe('fallback search', () => {
    it('should use fallback when daemon is unavailable', async () => {
      // Make daemon client throw
      vi.mocked(mockDaemonClient.embedText).mockRejectedValue(
        new Error('Daemon unavailable')
      );

      const options: SearchOptions = {
        query: 'test query',
      };

      const result = await searchTool.search(options);

      expect(result.status).toBe('uncertain');
      expect(result.status_reason).toContain('Daemon unavailable');
    });
  });

  describe('collectionExists', () => {
    it('should return true when collection exists', async () => {
      const exists = await searchTool.collectionExists('projects');

      expect(exists).toBe(true);
    });

    it('should return false when collection does not exist', async () => {
      // Mock getCollection to throw
      const QdrantClientMock = await import('@qdrant/js-client-rest');
      vi.mocked(QdrantClientMock.QdrantClient).mockImplementationOnce(
        () =>
          ({
            search: vi.fn(),
            scroll: vi.fn(),
            getCollection: vi.fn().mockRejectedValue(new Error('Not found')),
          }) as unknown as ReturnType<typeof QdrantClientMock.QdrantClient>
      );

      // Create new instance with mocked client
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
});

describe('Search result structure', () => {
  it('should have correct response structure', async () => {
    const mockDaemonClient = createMockDaemonClient();
    const mockStateManager = createMockStateManager();
    const mockProjectDetector = createMockProjectDetector();

    const searchTool = new SearchTool(
      { qdrantUrl: 'http://localhost:6333' },
      mockDaemonClient,
      mockStateManager,
      mockProjectDetector
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
    const mockDaemonClient = createMockDaemonClient();
    const mockStateManager = createMockStateManager();
    const mockProjectDetector = createMockProjectDetector();

    const searchTool = new SearchTool(
      { qdrantUrl: 'http://localhost:6333' },
      mockDaemonClient,
      mockStateManager,
      mockProjectDetector
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

    const mockDaemonClient = createMockDaemonClient();
    const mockStateManager = createMockStateManager();
    const mockProjectDetector = createMockProjectDetector();

    const searchTool = new SearchTool(
      { qdrantUrl: 'http://localhost:6333' },
      mockDaemonClient,
      mockStateManager,
      mockProjectDetector
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

    const mockDaemonClient = createMockDaemonClient();
    const mockStateManager = createMockStateManager();
    const mockProjectDetector = createMockProjectDetector();

    const searchTool = new SearchTool(
      { qdrantUrl: 'http://localhost:6333' },
      mockDaemonClient,
      mockStateManager,
      mockProjectDetector
    );

    const parent = await searchTool.retrieveParent('nonexistent', 'libraries');

    expect(parent).toBeNull();
  });
});
