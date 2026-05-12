/**
 * Tests for RetrieveTool scratchpad collection support (F-015).
 *
 * Verifies that:
 * - collection='scratchpad' is accepted by the type and builder
 * - Retrieve routes to the correct scratchpad Qdrant collection
 * - Tenant isolation: only documents matching resolvedProjectId are returned
 * - Unresolvable project ID returns an error response, not a broad scan
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { RetrieveTool } from '../../src/tools/retrieve.js';
import { buildRetrieveOptions } from '../../src/tool-builders/retrieve.js';
import type { RetrieveCollectionType } from '../../src/tools/retrieve-types.js';
import type { ProjectDetector } from '../../src/utils/project-detector.js';
import { COLLECTION_SCRATCHPAD } from '../../src/common/native-bridge.js';

// Track which Qdrant collection names were used in scroll/retrieve calls
let scrollCalls: { collection: string; filter?: unknown }[] = [];
let retrieveCalls: { collection: string }[] = [];

vi.mock('@qdrant/js-client-rest', () => ({
  QdrantClient: vi.fn().mockImplementation(() => ({
    retrieve: vi.fn().mockImplementation((collection: string) => {
      retrieveCalls.push({ collection });
      return Promise.resolve([
        {
          id: 'sp-doc-1',
          payload: {
            content: 'scratchpad content',
            title: 'Scratch Note',
            tenant_id: 'proj-abc',
          },
        },
      ]);
    }),
    scroll: vi.fn().mockImplementation((collection: string, params: unknown) => {
      scrollCalls.push({ collection, filter: (params as Record<string, unknown>)?.['filter'] });
      return Promise.resolve({
        points: [
          {
            id: 'sp-doc-1',
            payload: {
              content: 'scratchpad content',
              title: 'Scratch Note',
              tenant_id: 'proj-abc',
            },
          },
        ],
        next_page_offset: null,
      });
    }),
  })),
}));

function makeProjectDetector(projectId: string | undefined): ProjectDetector {
  return {
    findProjectRoot: vi.fn().mockReturnValue('/test/project'),
    getProjectInfo: vi
      .fn()
      .mockResolvedValue(
        projectId ? { projectId, projectPath: '/test/project', name: 'test' } : null
      ),
  } as unknown as ProjectDetector;
}

function makeTool(projectId: string | undefined): RetrieveTool {
  return new RetrieveTool(
    { qdrantUrl: 'http://localhost:6333', qdrantTimeout: 5000 },
    makeProjectDetector(projectId)
  );
}

describe('RetrieveTool — scratchpad collection (F-015)', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    scrollCalls = [];
    retrieveCalls = [];
  });

  it('builder accepts scratchpad as collection value', () => {
    const result = buildRetrieveOptions({ collection: 'scratchpad' });
    expect(result.collection).toBe('scratchpad');
  });

  it('builder rejects unknown collection value', () => {
    const result = buildRetrieveOptions({ collection: 'unknown_coll' });
    expect(result.collection).toBeUndefined();
  });

  it('retrieve with collection=scratchpad scrolls the scratchpad Qdrant collection', async () => {
    const tool = makeTool('proj-abc');
    const result = await tool.retrieve({ collection: 'scratchpad', projectId: 'proj-abc' });

    expect(result.success).toBe(true);
    expect(scrollCalls.length).toBeGreaterThan(0);
    expect(scrollCalls[0].collection).toBe(COLLECTION_SCRATCHPAD);
  });

  it('scroll filter includes tenant_id for scratchpad tenant isolation', async () => {
    const tool = makeTool('proj-abc');
    await tool.retrieve({ collection: 'scratchpad', projectId: 'proj-abc' });

    const filter = scrollCalls[0].filter as {
      must?: Array<{ key: string; match: { value: string } }>;
    };
    expect(filter?.must).toBeDefined();
    const tenantCondition = filter.must?.find((c) => c.key === 'tenant_id');
    expect(tenantCondition?.match.value).toBe('proj-abc');
  });

  it('retrieve by id uses scratchpad collection name', async () => {
    const tool = makeTool('proj-abc');
    const result = await tool.retrieve({
      collection: 'scratchpad',
      documentId: 'sp-doc-1',
      projectId: 'proj-abc',
    });

    expect(result.success).toBe(true);
    expect(retrieveCalls.length).toBeGreaterThan(0);
    expect(retrieveCalls[0].collection).toBe(COLLECTION_SCRATCHPAD);
  });

  it('returns error when scratchpad requested but project is unresolvable', async () => {
    const tool = makeTool(undefined);
    const result = await tool.retrieve({ collection: 'scratchpad' });

    expect(result.success).toBe(false);
    expect(result.documents).toHaveLength(0);
    expect(scrollCalls).toHaveLength(0);
  });

  it('RetrieveCollectionType includes scratchpad', () => {
    // Type-level test: RetrieveCollectionType must accept 'scratchpad'
    const t: RetrieveCollectionType = 'scratchpad';
    expect(t).toBe('scratchpad');
  });
});
