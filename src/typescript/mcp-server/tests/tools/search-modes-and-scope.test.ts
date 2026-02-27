/**
 * Tests for SearchTool search modes and scope/collection selection.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { SearchTool, type SearchOptions } from '../../src/tools/search.js';
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

describe('SearchTool — search modes and scope', () => {
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

  describe('search modes', () => {
    it('should call embedText for semantic mode', async () => {
      const options: SearchOptions = { query: 'test query', mode: 'semantic' };

      await searchTool.search(options);

      expect(mockDaemonClient.embedText).toHaveBeenCalledWith({ text: 'test query' });
      expect(mockDaemonClient.generateSparseVector).not.toHaveBeenCalled();
    });

    it('should call generateSparseVector for keyword mode', async () => {
      const options: SearchOptions = { query: 'test query', mode: 'keyword' };

      await searchTool.search(options);

      expect(mockDaemonClient.generateSparseVector).toHaveBeenCalledWith({ text: 'test query' });
      expect(mockDaemonClient.embedText).not.toHaveBeenCalled();
    });

    it('should call both embedText and generateSparseVector for hybrid mode', async () => {
      const options: SearchOptions = { query: 'test query', mode: 'hybrid' };

      await searchTool.search(options);

      expect(mockDaemonClient.embedText).toHaveBeenCalledWith({ text: 'test query' });
      expect(mockDaemonClient.generateSparseVector).toHaveBeenCalledWith({ text: 'test query' });
    });

    it('should use default hybrid mode when not specified', async () => {
      const result = await searchTool.search({ query: 'test query' });

      expect(result.mode).toBe('hybrid');
    });
  });

  describe('scope and collection selection', () => {
    it('should use project scope by default', async () => {
      const result = await searchTool.search({ query: 'test query' });

      expect(result.scope).toBe('project');
    });

    it('should search projects collection for project scope', async () => {
      const result = await searchTool.search({ query: 'test query', scope: 'project' });

      expect(result.collections_searched).toContain('projects');
      expect(result.collections_searched).not.toContain('libraries');
    });

    it('should search both collections for all scope', async () => {
      const result = await searchTool.search({ query: 'test query', scope: 'all' });

      expect(result.collections_searched).toContain('projects');
      expect(result.collections_searched).toContain('libraries');
    });

    it('should include libraries when includeLibraries is true', async () => {
      const result = await searchTool.search({
        query: 'test query',
        scope: 'project',
        includeLibraries: true,
      });

      expect(result.collections_searched).toContain('projects');
      expect(result.collections_searched).toContain('libraries');
    });

    it('should use explicit collection when provided', async () => {
      const result = await searchTool.search({ query: 'test query', collection: 'memory' });

      expect(result.collections_searched).toEqual(['memory']);
    });

    it('should get project ID for project scope', async () => {
      await searchTool.search({ query: 'test query', scope: 'project' });

      expect(mockProjectDetector.getProjectInfo).toHaveBeenCalled();
    });

    it('should not get project ID when explicitly provided', async () => {
      await searchTool.search({
        query: 'test query',
        scope: 'project',
        projectId: 'explicit-project-id',
      });

      expect(mockProjectDetector.getProjectInfo).not.toHaveBeenCalled();
    });
  });
});
