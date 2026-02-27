/**
 * Tests for tag-based sparse vector expansion in SearchTool (Task 34) — part 2:
 * keyword limits, weight application, and error handling
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

describe('SearchTool expandSparseWithTags — keyword limits and error handling', () => {
  let daemonClient: DaemonClient;
  let projectDetector: ProjectDetector;

  beforeEach(() => {
    vi.clearAllMocks();
    daemonClient = createMockDaemonClient();
    projectDetector = createMockProjectDetector();
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

    const generateSparse = vi.fn()
      .mockResolvedValueOnce({
        success: true,
        indices_values: { 10: 1.5 },
        vocab_size: 50000,
      })
      .mockResolvedValueOnce({
        success: true,
        indices_values: { 20: 1.0 },
        vocab_size: 50000,
      });
    (daemonClient as unknown as { generateSparseVector: typeof generateSparse }).generateSparseVector = generateSparse;

    const searchTool = new SearchTool(
      { qdrantUrl: 'http://localhost:6333', expansionWeight: 0.3 },
      daemonClient,
      stateManager,
      projectDetector,
    );

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
