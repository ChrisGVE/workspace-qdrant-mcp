/**
 * Tests for provenance metadata on search results and the `diverse` parameter.
 */

import { describe, it, expect, vi } from 'vitest';
import { buildSearchOptions } from '../../src/tool-builders/search.js';
import { SearchTool } from '../../src/tools/search.js';
import type { DaemonClient } from '../../src/clients/daemon-client.js';
import type { SqliteStateManager } from '../../src/clients/sqlite-state-manager.js';
import type { ProjectDetector } from '../../src/utils/project-detector.js';

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
    resolveSearchScope: vi.fn().mockResolvedValue({ tenant_ids: [], filter_by_tenant: false }),
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

// ---------------------------------------------------------------------------
// buildSearchOptions — diverse parameter forwarding
// ---------------------------------------------------------------------------

describe('buildSearchOptions — diverse parameter', () => {
  it('omits diverse when not provided', () => {
    const opts = buildSearchOptions({ query: 'hello' });
    expect(opts.diverse).toBeUndefined();
  });

  it('forwards diverse=true', () => {
    const opts = buildSearchOptions({ query: 'hello', diverse: true });
    expect(opts.diverse).toBe(true);
  });

  it('forwards diverse=false', () => {
    const opts = buildSearchOptions({ query: 'hello', diverse: false });
    expect(opts.diverse).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// SearchTool.search — provenance fields on results
// ---------------------------------------------------------------------------

describe('SearchTool.search — provenance on results', () => {
  it('attaches provenance with source=projects for projects collection results', async () => {
    const { QdrantClient } = await import('@qdrant/js-client-rest');
    vi.mocked(QdrantClient).mockImplementationOnce(
      () =>
        ({
          search: vi.fn().mockResolvedValue([
            {
              id: 'pt-1',
              score: 0.9,
              payload: {
                content: 'some code',
                tenant_id: 'tenant-abc',
                _search_type: 'semantic',
              },
            },
          ]),
          scroll: vi.fn().mockResolvedValue({ points: [] }),
          retrieve: vi.fn().mockResolvedValue([]),
          getCollection: vi.fn().mockResolvedValue({ status: 'green' }),
        }) as unknown as InstanceType<typeof QdrantClient>
    );

    const tool = new SearchTool(
      { qdrantUrl: 'http://localhost:6333' },
      createMockDaemonClient(),
      createMockStateManager(),
      createMockProjectDetector()
    );

    const result = await tool.search({ query: 'test', collection: 'projects', mode: 'semantic' });
    const hit = result.results[0];
    expect(hit).toBeDefined();
    expect(hit!.provenance).toBeDefined();
    expect(hit!.provenance!.source).toBe('projects');
    expect(hit!.provenance!.source_project_id).toBe('tenant-abc');
  });

  it('attaches provenance with source=libraries for libraries collection results', async () => {
    const { QdrantClient } = await import('@qdrant/js-client-rest');
    vi.mocked(QdrantClient).mockImplementationOnce(
      () =>
        ({
          search: vi.fn().mockResolvedValue([
            {
              id: 'lib-1',
              score: 0.85,
              payload: {
                content: 'library doc',
                library_name: 'my-lib',
                library_path: 'docs/api',
                document_name: 'API Reference',
              },
            },
          ]),
          scroll: vi.fn().mockResolvedValue({ points: [] }),
          retrieve: vi.fn().mockResolvedValue([]),
          getCollection: vi.fn().mockResolvedValue({ status: 'green' }),
        }) as unknown as InstanceType<typeof QdrantClient>
    );

    const tool = new SearchTool(
      { qdrantUrl: 'http://localhost:6333' },
      createMockDaemonClient(),
      createMockStateManager(),
      createMockProjectDetector()
    );

    const result = await tool.search({ query: 'test', collection: 'libraries', mode: 'semantic' });
    const hit = result.results[0];
    expect(hit).toBeDefined();
    expect(hit!.provenance).toBeDefined();
    expect(hit!.provenance!.source).toBe('libraries');
    expect(hit!.provenance!.library_name).toBe('my-lib');
    expect(hit!.provenance!.library_path).toBe('docs/api');
    expect(hit!.provenance!.doc_title).toBe('API Reference');
  });

  it('uses title field as doc_title when document_name is absent', async () => {
    const { QdrantClient } = await import('@qdrant/js-client-rest');
    vi.mocked(QdrantClient).mockImplementationOnce(
      () =>
        ({
          search: vi.fn().mockResolvedValue([
            {
              id: 'lib-2',
              score: 0.8,
              payload: {
                content: 'some content',
                library_name: 'other-lib',
                title: 'Chapter 3',
              },
            },
          ]),
          scroll: vi.fn().mockResolvedValue({ points: [] }),
          retrieve: vi.fn().mockResolvedValue([]),
          getCollection: vi.fn().mockResolvedValue({ status: 'green' }),
        }) as unknown as InstanceType<typeof QdrantClient>
    );

    const tool = new SearchTool(
      { qdrantUrl: 'http://localhost:6333' },
      createMockDaemonClient(),
      createMockStateManager(),
      createMockProjectDetector()
    );

    const result = await tool.search({ query: 'test', collection: 'libraries', mode: 'semantic' });
    const hit = result.results[0];
    expect(hit!.provenance!.doc_title).toBe('Chapter 3');
  });

  it('attaches provenance with source=scratchpad for scratchpad collection results', async () => {
    const { QdrantClient } = await import('@qdrant/js-client-rest');
    vi.mocked(QdrantClient).mockImplementationOnce(
      () =>
        ({
          search: vi.fn().mockResolvedValue([
            {
              id: 'sc-1',
              score: 0.75,
              payload: {
                content: 'scratch note',
                tenant_id: 'tenant-xyz',
              },
            },
          ]),
          scroll: vi.fn().mockResolvedValue({ points: [] }),
          retrieve: vi.fn().mockResolvedValue([]),
          getCollection: vi.fn().mockResolvedValue({ status: 'green' }),
        }) as unknown as InstanceType<typeof QdrantClient>
    );

    const tool = new SearchTool(
      { qdrantUrl: 'http://localhost:6333' },
      createMockDaemonClient(),
      createMockStateManager(),
      createMockProjectDetector()
    );

    const result = await tool.search({
      query: 'test',
      collection: 'scratchpad',
      mode: 'semantic',
    });
    const hit = result.results[0];
    expect(hit).toBeDefined();
    expect(hit!.provenance!.source).toBe('scratchpad');
    expect(hit!.provenance!.source_project_id).toBe('tenant-xyz');
  });
});

// ---------------------------------------------------------------------------
// SearchTool.search — diverse=false skips diversity re-ranking
// ---------------------------------------------------------------------------

describe('SearchTool.search — diverse parameter controls re-ranking', () => {
  it('diverse=false is accepted and does not error', async () => {
    const tool = new SearchTool(
      { qdrantUrl: 'http://localhost:6333' },
      createMockDaemonClient(),
      createMockStateManager(),
      createMockProjectDetector()
    );

    // No results but the call itself should succeed without throwing
    const result = await tool.search({
      query: 'test',
      includeLibraries: true,
      diverse: false,
    });
    expect(result).toHaveProperty('results');
    expect(result.diversity_score).toBeUndefined();
  });

  it('diverse=true (default) does not error', async () => {
    const tool = new SearchTool(
      { qdrantUrl: 'http://localhost:6333' },
      createMockDaemonClient(),
      createMockStateManager(),
      createMockProjectDetector()
    );

    const result = await tool.search({ query: 'test', diverse: true });
    expect(result).toHaveProperty('results');
  });
});
