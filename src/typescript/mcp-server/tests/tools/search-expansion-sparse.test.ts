/**
 * Tests for tag-based sparse vector expansion in SearchTool (Task 34) — part 1:
 * expansion triggers, mode guards, and empty/missing data conditions
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

describe('SearchTool expandSparseWithTags — triggers and mode guards', () => {
  let daemonClient: DaemonClient;
  let projectDetector: ProjectDetector;

  beforeEach(() => {
    vi.clearAllMocks();
    daemonClient = createMockDaemonClient();
    projectDetector = createMockProjectDetector();
  });

  it('should expand sparse vector when matching tags have baskets', async () => {
    const stateManager = createMockStateManager({
      matchingTags: [
        { tag_id: 1, tag: 'vector indexing', score: 0.85 },
      ],
      baskets: [
        { tag_id: 1, keywords_json: '["embedding", "semantic", "qdrant"]' },
      ],
    });

    const generateSparse = vi.fn()
      .mockResolvedValueOnce({
        success: true,
        indices_values: { 10: 1.5, 20: 0.8 },
        vocab_size: 50000,
      })
      .mockResolvedValueOnce({
        success: true,
        indices_values: { 30: 1.0, 40: 0.6, 10: 2.0 },
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

    expect(generateSparse).toHaveBeenCalledTimes(2);
    expect(generateSparse).toHaveBeenNthCalledWith(1, { text: 'vector search' });
    expect(generateSparse).toHaveBeenNthCalledWith(2, { text: expect.stringContaining('embedding') });
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

    expect(daemonClient.generateSparseVector).toHaveBeenCalledTimes(1);
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

    expect(daemonClient.generateSparseVector).toHaveBeenCalledTimes(1);
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

    expect(daemonClient.generateSparseVector).toHaveBeenCalledTimes(1);
  });
});
