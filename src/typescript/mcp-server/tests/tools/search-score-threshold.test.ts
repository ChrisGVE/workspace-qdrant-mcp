/**
 * Tests for SearchTool scoreThreshold parameter (F-014).
 *
 * Verifies that:
 * - scoreThreshold from options is passed through to Qdrant search calls
 * - Default threshold (0.3) is used when parameter is omitted
 * - Explicit threshold overrides the default
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { SearchTool, type SearchOptions } from '../../src/tools/search.js';
import { DEFAULT_SCORE_THRESHOLD } from '../../src/tools/search-types.js';
import {
  buildSearchOptions,
  type SearchOptions as BuilderSearchOptions,
} from '../../src/tool-builders/search.js';
import type { DaemonClient } from '../../src/clients/daemon-client.js';
import type { SqliteStateManager } from '../../src/clients/sqlite-state-manager.js';
import type { ProjectDetector } from '../../src/utils/project-detector.js';

// Track calls made to qdrantClient.search so we can inspect the score_threshold arg
let capturedQdrantSearchCalls: unknown[] = [];

vi.mock('@qdrant/js-client-rest', () => ({
  QdrantClient: vi.fn().mockImplementation(() => ({
    search: vi.fn().mockImplementation((collection: unknown, params: unknown) => {
      capturedQdrantSearchCalls.push({ collection, params });
      return Promise.resolve([]);
    }),
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

describe('SearchTool — scoreThreshold (F-014)', () => {
  let searchTool: SearchTool;

  beforeEach(() => {
    vi.clearAllMocks();
    capturedQdrantSearchCalls = [];
    searchTool = new SearchTool(
      { qdrantUrl: 'http://localhost:6333', qdrantTimeout: 5000 },
      createMockDaemonClient(),
      createMockStateManager(),
      createMockProjectDetector()
    );
  });

  it('passes DEFAULT_SCORE_THRESHOLD to qdrant when scoreThreshold is omitted', async () => {
    const options: SearchOptions = { query: 'test query', mode: 'semantic' };
    await searchTool.search(options);

    expect(capturedQdrantSearchCalls.length).toBeGreaterThan(0);
    for (const call of capturedQdrantSearchCalls) {
      const params = (call as { params: Record<string, unknown> }).params;
      expect(params['score_threshold']).toBe(DEFAULT_SCORE_THRESHOLD);
    }
  });

  it('passes explicit scoreThreshold=0.7 to qdrant search', async () => {
    const options: SearchOptions = { query: 'test query', mode: 'semantic', scoreThreshold: 0.7 };
    await searchTool.search(options);

    expect(capturedQdrantSearchCalls.length).toBeGreaterThan(0);
    for (const call of capturedQdrantSearchCalls) {
      const params = (call as { params: Record<string, unknown> }).params;
      expect(params['score_threshold']).toBe(0.7);
    }
  });

  it('passes scoreThreshold=0 when explicitly set to 0', async () => {
    const options: SearchOptions = { query: 'test query', mode: 'semantic', scoreThreshold: 0 };
    await searchTool.search(options);

    expect(capturedQdrantSearchCalls.length).toBeGreaterThan(0);
    for (const call of capturedQdrantSearchCalls) {
      const params = (call as { params: Record<string, unknown> }).params;
      expect(params['score_threshold']).toBe(0);
    }
  });

  it('builder SearchOptions type accepts scoreThreshold field', () => {
    // Type-level test: ensure the builder type includes scoreThreshold
    const opts: BuilderSearchOptions = {
      query: 'test',
      scoreThreshold: 0.5,
    };
    expect(opts.scoreThreshold).toBe(0.5);
  });

  it('buildSearchOptions extracts scoreThreshold from raw args', () => {
    const result = buildSearchOptions({ query: 'hello', scoreThreshold: 0.85 });
    expect(result.scoreThreshold).toBe(0.85);
  });

  it('buildSearchOptions leaves scoreThreshold undefined when not in args', () => {
    const result = buildSearchOptions({ query: 'hello' });
    expect(result.scoreThreshold).toBeUndefined();
  });
});
