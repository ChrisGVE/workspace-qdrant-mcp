/**
 * Tests for the project-memory scratchpad recall lane on SearchTool.
 *
 * The lane appends a small, tenant-filtered scratchpad result set to
 * project-scoped searches so notes surface alongside code without displacing
 * it. It must:
 *   - run by default for scope="project" and append notes AFTER code,
 *   - mark `scratchpad` in collections_searched only when notes are returned,
 *   - collapse the dense+sparse legs so a note appears at most once,
 *   - stay off for includeScratchpad:false, an explicit collection, or
 *     non-project scopes.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { SearchTool, type SearchOptions } from '../../src/tools/search.js';
import type { DaemonClient } from '../../src/clients/daemon-client.js';
import type { SqliteStateManager } from '../../src/clients/sqlite-state-manager.js';
import type { ProjectDetector } from '../../src/utils/project-detector.js';

// Qdrant mock: returns one scratchpad note for the `scratchpad` collection and
// nothing for any other collection. Both the dense and sparse legs of
// searchCollection hit this, exercising the lane's per-id dedup.
vi.mock('@qdrant/js-client-rest', () => ({
  QdrantClient: vi.fn().mockImplementation(() => ({
    search: vi
      .fn()
      .mockImplementation((collection: string) =>
        Promise.resolve(
          collection === 'scratchpad'
            ? [{ id: 'note-1', score: 0.7, payload: { content: 'remember: ext4 for repos' } }]
            : []
        )
      ),
    scroll: vi.fn().mockResolvedValue({ points: [] }),
    retrieve: vi.fn().mockResolvedValue([]),
    getCollection: vi.fn().mockResolvedValue({ status: 'green' }),
  })),
}));

function createMockDaemonClient(): DaemonClient {
  return {
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
    logSearchEvent: vi.fn(),
    updateSearchEvent: vi.fn(),
    updateSearchEventEconomy: vi.fn(),
    getMatchingTags: vi.fn().mockReturnValue([]),
    getKeywordBasketsForTags: vi.fn().mockReturnValue([]),
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

describe('SearchTool — scratchpad recall lane', () => {
  let searchTool: SearchTool;

  beforeEach(() => {
    vi.clearAllMocks();
    searchTool = new SearchTool(
      { qdrantUrl: 'http://localhost:6333', qdrantTimeout: 5000 },
      createMockDaemonClient(),
      createMockStateManager(),
      createMockProjectDetector()
    );
  });

  it('appends scratchpad notes (deduped) after code for default project scope', async () => {
    const result = await searchTool.search({ query: 'ext4' });

    const scratchpadHits = result.results.filter((r) => r.collection === 'scratchpad');
    expect(scratchpadHits).toHaveLength(1); // dense+sparse legs collapsed to one
    expect(scratchpadHits[0]?.content).toContain('ext4');
    // Lane hits are appended after code; with no code results here the note is
    // the only hit, but it is still tagged as scratchpad and the collection is
    // recorded as searched.
    expect(result.collections_searched).toContain('scratchpad');
  });

  it('does not run the lane when includeScratchpad is false', async () => {
    const result = await searchTool.search({ query: 'ext4', includeScratchpad: false });

    expect(result.results.some((r) => r.collection === 'scratchpad')).toBe(false);
    expect(result.collections_searched).not.toContain('scratchpad');
  });

  it('does not run the lane when an explicit collection is targeted', async () => {
    const result = await searchTool.search({ query: 'ext4', collection: 'projects' });

    expect(result.results.some((r) => r.collection === 'scratchpad')).toBe(false);
    expect(result.collections_searched).toEqual(['projects']);
  });

  it('does not run the lane for non-project scopes', async () => {
    const result = await searchTool.search({ query: 'ext4', scope: 'global' });

    expect(result.results.some((r) => r.collection === 'scratchpad')).toBe(false);
    expect(result.collections_searched).not.toContain('scratchpad');
  });
});
