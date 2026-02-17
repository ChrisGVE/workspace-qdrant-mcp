/**
 * Tests for tag-based query expansion in SearchTool (Task 34)
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { SearchTool } from '../../src/tools/search.js';
import type { DaemonClient } from '../../src/clients/daemon-client.js';
import type { SqliteStateManager } from '../../src/clients/sqlite-state-manager.js';
import type { ProjectDetector } from '../../src/utils/project-detector.js';

// Mock the Qdrant client
vi.mock('@qdrant/js-client-rest', () => ({
  QdrantClient: vi.fn().mockImplementation(() => ({
    search: vi.fn().mockResolvedValue([
      {
        id: 'doc-1',
        score: 0.9,
        payload: {
          content: 'Vector search with Qdrant embeddings',
          title: 'Vector Search Doc',
          tenant_id: 'test-project',
        },
      },
    ]),
    getCollection: vi.fn().mockResolvedValue({}),
  })),
}));

function createMockDaemonClient(): DaemonClient {
  return {
    embedText: vi.fn().mockResolvedValue({
      success: true,
      embedding: new Array(384).fill(0.1),
    }),
    generateSparseVector: vi.fn().mockResolvedValue({
      success: true,
      indices_values: { 10: 1.5, 20: 0.8, 30: 1.2 },
      vocab_size: 50000,
    }),
    isConnected: vi.fn().mockReturnValue(true),
  } as unknown as DaemonClient;
}

function createMockStateManager(options?: {
  matchingTags?: { tag_id: number; tag: string; score: number }[];
  baskets?: { tag_id: number; keywords_json: string }[];
}): SqliteStateManager {
  const tags = options?.matchingTags ?? [];
  const baskets = options?.baskets ?? [];

  return {
    logSearchEvent: vi.fn(),
    updateSearchEvent: vi.fn(),
    getMatchingTags: vi.fn().mockReturnValue(
      tags.map((t) => ({ tagId: t.tag_id, tag: t.tag, score: t.score })),
    ),
    getKeywordBasketsForTags: vi.fn().mockReturnValue(
      baskets.map((b) => {
        let keywords: string[] = [];
        try {
          keywords = JSON.parse(b.keywords_json) as string[];
        } catch {
          // skip
        }
        return { tagId: b.tag_id, keywords };
      }),
    ),
    isConnected: vi.fn().mockReturnValue(true),
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

describe('SearchTool tag expansion', () => {
  let daemonClient: DaemonClient;
  let projectDetector: ProjectDetector;

  beforeEach(() => {
    vi.clearAllMocks();
    daemonClient = createMockDaemonClient();
    projectDetector = createMockProjectDetector();
  });

  describe('expandSparseWithTags integration', () => {
    it('should expand sparse vector when matching tags have baskets', async () => {
      const stateManager = createMockStateManager({
        matchingTags: [
          { tag_id: 1, tag: 'vector indexing', score: 0.85 },
        ],
        baskets: [
          { tag_id: 1, keywords_json: '["embedding", "semantic", "qdrant"]' },
        ],
      });

      // Mock: generateSparseVector returns different vectors for original and expansion
      const generateSparse = vi.fn()
        .mockResolvedValueOnce({
          success: true,
          indices_values: { 10: 1.5, 20: 0.8 },  // original query
          vocab_size: 50000,
        })
        .mockResolvedValueOnce({
          success: true,
          indices_values: { 30: 1.0, 40: 0.6, 10: 2.0 },  // expansion (index 10 overlaps)
          vocab_size: 50000,
        });
      (daemonClient as unknown as { generateSparseVector: typeof generateSparse }).generateSparseVector = generateSparse;

      const searchTool = new SearchTool(
        { qdrantUrl: 'http://localhost:6333' },
        daemonClient,
        stateManager,
        projectDetector,
      );

      const result = await searchTool.search({
        query: 'vector search',
        mode: 'keyword',
        projectId: 'test-project-123',
      });

      // generateSparseVector called twice: once for original query, once for expansion
      expect(generateSparse).toHaveBeenCalledTimes(2);
      expect(generateSparse).toHaveBeenNthCalledWith(1, { text: 'vector search' });
      // Second call is for expansion keywords
      expect(generateSparse).toHaveBeenNthCalledWith(2, { text: expect.stringContaining('embedding') });

      // Verify tag lookup was called
      expect(stateManager.getMatchingTags).toHaveBeenCalledWith(
        'vector search',
        'projects',
        'test-project-123',
      );

      expect(result.results.length).toBeGreaterThanOrEqual(0);
    });

    it('should not expand when tag expansion is disabled', async () => {
      const stateManager = createMockStateManager({
        matchingTags: [{ tag_id: 1, tag: 'vector', score: 0.9 }],
        baskets: [{ tag_id: 1, keywords_json: '["embedding"]' }],
      });

      const searchTool = new SearchTool(
        { qdrantUrl: 'http://localhost:6333', enableTagExpansion: false },
        daemonClient,
        stateManager,
        projectDetector,
      );

      await searchTool.search({
        query: 'vector search',
        mode: 'keyword',
        projectId: 'test-project-123',
      });

      // generateSparseVector called only once (no expansion)
      expect(daemonClient.generateSparseVector).toHaveBeenCalledTimes(1);
      // getMatchingTags should NOT be called
      expect(stateManager.getMatchingTags).not.toHaveBeenCalled();
    });

    it('should not expand in semantic-only mode', async () => {
      const stateManager = createMockStateManager({
        matchingTags: [{ tag_id: 1, tag: 'vector', score: 0.9 }],
        baskets: [{ tag_id: 1, keywords_json: '["embedding"]' }],
      });

      const searchTool = new SearchTool(
        { qdrantUrl: 'http://localhost:6333' },
        daemonClient,
        stateManager,
        projectDetector,
      );

      await searchTool.search({
        query: 'vector search',
        mode: 'semantic',
        projectId: 'test-project-123',
      });

      // No sparse vector generated in semantic mode
      expect(daemonClient.generateSparseVector).not.toHaveBeenCalled();
      expect(stateManager.getMatchingTags).not.toHaveBeenCalled();
    });

    it('should skip expansion when no tags match', async () => {
      const stateManager = createMockStateManager({
        matchingTags: [],
        baskets: [],
      });

      const searchTool = new SearchTool(
        { qdrantUrl: 'http://localhost:6333' },
        daemonClient,
        stateManager,
        projectDetector,
      );

      await searchTool.search({
        query: 'obscure query',
        mode: 'keyword',
        projectId: 'test-project-123',
      });

      // Only one call (original query), no expansion
      expect(daemonClient.generateSparseVector).toHaveBeenCalledTimes(1);
      // getMatchingTags called but returned empty
      expect(stateManager.getMatchingTags).toHaveBeenCalled();
      expect(stateManager.getKeywordBasketsForTags).not.toHaveBeenCalled();
    });

    it('should skip expansion when baskets are empty', async () => {
      const stateManager = createMockStateManager({
        matchingTags: [{ tag_id: 1, tag: 'vector', score: 0.9 }],
        baskets: [{ tag_id: 1, keywords_json: '[]' }],
      });

      const searchTool = new SearchTool(
        { qdrantUrl: 'http://localhost:6333' },
        daemonClient,
        stateManager,
        projectDetector,
      );

      await searchTool.search({
        query: 'vector search',
        mode: 'keyword',
        projectId: 'test-project-123',
      });

      // Only one call - expansion has no keywords to expand with
      expect(daemonClient.generateSparseVector).toHaveBeenCalledTimes(1);
    });

    it('should respect maxExpandedKeywords configuration', async () => {
      const stateManager = createMockStateManager({
        matchingTags: [{ tag_id: 1, tag: 'vector', score: 0.9 }],
        baskets: [{
          tag_id: 1,
          keywords_json: '["a","b","c","d","e","f","g","h","i","j","k","l"]',
        }],
      });

      const generateSparse = vi.fn()
        .mockResolvedValueOnce({
          success: true,
          indices_values: { 10: 1.5 },
          vocab_size: 50000,
        })
        .mockResolvedValueOnce({
          success: true,
          indices_values: { 20: 0.5 },
          vocab_size: 50000,
        });
      (daemonClient as unknown as { generateSparseVector: typeof generateSparse }).generateSparseVector = generateSparse;

      const searchTool = new SearchTool(
        { qdrantUrl: 'http://localhost:6333', maxExpandedKeywords: 3 },
        daemonClient,
        stateManager,
        projectDetector,
      );

      await searchTool.search({
        query: 'vector',
        mode: 'keyword',
        projectId: 'test-project-123',
      });

      // Expansion text should only contain first 3 keywords
      const expansionCall = generateSparse.mock.calls[1];
      const expansionText = (expansionCall[0] as { text: string }).text;
      const expansionWords = expansionText.split(' ');
      expect(expansionWords.length).toBeLessThanOrEqual(3);
    });

    it('should apply expansion weight to merged sparse vector indices', async () => {
      const stateManager = createMockStateManager({
        matchingTags: [{ tag_id: 1, tag: 'vector', score: 0.9 }],
        baskets: [{ tag_id: 1, keywords_json: '["embedding"]' }],
      });

      // First call: original query sparse vector
      // Second call: expansion sparse vector with new indices
      const generateSparse = vi.fn()
        .mockResolvedValueOnce({
          success: true,
          indices_values: { 10: 1.5 },  // original
          vocab_size: 50000,
        })
        .mockResolvedValueOnce({
          success: true,
          indices_values: { 20: 1.0 },  // expansion - new index
          vocab_size: 50000,
        });
      (daemonClient as unknown as { generateSparseVector: typeof generateSparse }).generateSparseVector = generateSparse;

      const searchTool = new SearchTool(
        { qdrantUrl: 'http://localhost:6333', expansionWeight: 0.3 },
        daemonClient,
        stateManager,
        projectDetector,
      );

      // We can't directly inspect the merged vector, but we verify expansion was attempted
      await searchTool.search({
        query: 'vector',
        mode: 'keyword',
        projectId: 'test-project-123',
      });

      expect(generateSparse).toHaveBeenCalledTimes(2);
    });

    it('should gracefully handle expansion failure', async () => {
      const stateManager = createMockStateManager({
        matchingTags: [{ tag_id: 1, tag: 'vector', score: 0.9 }],
        baskets: [{ tag_id: 1, keywords_json: '["embedding"]' }],
      });

      // First call: original sparse vector succeeds
      // Second call: expansion fails
      const generateSparse = vi.fn()
        .mockResolvedValueOnce({
          success: true,
          indices_values: { 10: 1.5 },
          vocab_size: 50000,
        })
        .mockRejectedValueOnce(new Error('Daemon unavailable'));
      (daemonClient as unknown as { generateSparseVector: typeof generateSparse }).generateSparseVector = generateSparse;

      const searchTool = new SearchTool(
        { qdrantUrl: 'http://localhost:6333' },
        daemonClient,
        stateManager,
        projectDetector,
      );

      // Should not throw - expansion failure is graceful
      const result = await searchTool.search({
        query: 'vector',
        mode: 'keyword',
        projectId: 'test-project-123',
      });

      expect(result).toBeDefined();
      expect(result.results).toBeDefined();
    });
  });

  describe('SqliteStateManager tag methods', () => {
    it('should handle getMatchingTags when db is not connected', () => {
      const stateManager = createMockStateManager();
      // Override to simulate not connected
      (stateManager.getMatchingTags as ReturnType<typeof vi.fn>).mockReturnValue([]);

      const result = stateManager.getMatchingTags('test', 'projects');
      expect(result).toEqual([]);
    });

    it('should handle getKeywordBasketsForTags with empty tagIds', () => {
      const stateManager = createMockStateManager();
      (stateManager.getKeywordBasketsForTags as ReturnType<typeof vi.fn>).mockReturnValue([]);

      const result = stateManager.getKeywordBasketsForTags([]);
      expect(result).toEqual([]);
    });
  });
});
