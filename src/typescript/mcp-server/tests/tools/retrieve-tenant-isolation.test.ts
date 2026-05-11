/**
 * Regression tests for F-002 (by-ID retrieve bypassed tenant ownership
 * check) and F-011 (project-scope retrieve without resolved projectId
 * scrolled arbitrary documents).
 *
 * These tests exercise the contract documented in `retrieve.ts`:
 * - `projects` collection requires a resolvable `tenant_id`. By-ID
 *   lookups verify the returned payload's `tenant_id` matches the
 *   caller; mismatches are reported as not-found, never as the foreign
 *   document. By-filter retrieves without a resolved tenant refuse to
 *   scroll.
 * - `libraries` collection requires `libraryName`. Without it, both
 *   by-id and by-filter refuse.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { RetrieveTool } from '../../src/tools/retrieve.js';
import type { ProjectDetector } from '../../src/utils/project-detector.js';

// Mock the Qdrant client so we can override its impl per test.
vi.mock('@qdrant/js-client-rest', () => ({
  QdrantClient: vi.fn().mockImplementation(() => ({
    retrieve: vi.fn().mockResolvedValue([]),
    scroll: vi.fn().mockResolvedValue({ points: [] }),
  })),
}));

function makeMockClient(returns: {
  retrieve?: ReturnType<typeof vi.fn>;
  scroll?: ReturnType<typeof vi.fn>;
}) {
  return async () => {
    const QdrantClientMock = await import('@qdrant/js-client-rest');
    const retrieveFn = returns.retrieve ?? vi.fn().mockResolvedValue([]);
    const scrollFn = returns.scroll ?? vi.fn().mockResolvedValue({ points: [] });
    vi.mocked(QdrantClientMock.QdrantClient).mockImplementationOnce(
      () =>
        ({
          retrieve: retrieveFn,
          scroll: scrollFn,
        }) as unknown as ReturnType<typeof QdrantClientMock.QdrantClient>
    );
    return { retrieveFn, scrollFn };
  };
}

function detectorReturning(projectId: string | undefined): ProjectDetector {
  return {
    findProjectRoot: vi.fn().mockReturnValue('/test/project'),
    getProjectInfo: vi.fn().mockResolvedValue(projectId ? { projectId } : null),
  } as unknown as ProjectDetector;
}

describe('RetrieveTool — F-002 by-ID ownership check', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('refuses to return a point whose tenant_id does not match the caller', async () => {
    const setupClient = makeMockClient({
      retrieve: vi.fn().mockResolvedValue([
        {
          id: 'foreign-doc',
          payload: { content: 'private to project B', tenant_id: 'project-b' },
        },
      ]),
    });
    const { retrieveFn } = await setupClient();

    const detector = detectorReturning('project-a');
    const tool = new RetrieveTool({ qdrantUrl: 'http://localhost:6333' }, detector);

    const result = await tool.retrieve({
      documentId: 'foreign-doc',
      collection: 'projects',
      projectId: 'project-a',
    });

    expect(retrieveFn).toHaveBeenCalled();
    expect(result.success).toBe(false);
    expect(result.documents).toHaveLength(0);
    expect(result.message).toContain('Document not found');
  });

  it('returns the document when tenant_id matches the caller', async () => {
    const setupClient = makeMockClient({
      retrieve: vi.fn().mockResolvedValue([
        {
          id: 'owned-doc',
          payload: { content: 'mine', tenant_id: 'project-a' },
        },
      ]),
    });
    await setupClient();

    const detector = detectorReturning('project-a');
    const tool = new RetrieveTool({ qdrantUrl: 'http://localhost:6333' }, detector);

    const result = await tool.retrieve({
      documentId: 'owned-doc',
      collection: 'projects',
      projectId: 'project-a',
    });

    expect(result.success).toBe(true);
    expect(result.documents).toHaveLength(1);
    expect(result.documents[0].id).toBe('owned-doc');
    expect(result.documents[0].content).toBe('mine');
  });

  it('refuses by-ID lookup when projects scope cannot resolve a tenant', async () => {
    const retrieveFn = vi
      .fn()
      .mockResolvedValue([{ id: 'x', payload: { content: 'whatever', tenant_id: 'project-a' } }]);
    const setupClient = makeMockClient({ retrieve: retrieveFn });
    await setupClient();

    const detector = detectorReturning(undefined);
    const tool = new RetrieveTool({ qdrantUrl: 'http://localhost:6333' }, detector);

    const result = await tool.retrieve({ documentId: 'x', collection: 'projects' });

    expect(retrieveFn).not.toHaveBeenCalled();
    expect(result.success).toBe(false);
    expect(result.message).toContain('scope');
  });

  it('library by-ID lookup requires libraryName and verifies it on the payload', async () => {
    const setupClient = makeMockClient({
      retrieve: vi.fn().mockResolvedValue([
        {
          id: 'lib-doc',
          payload: { content: 'lib content', library_name: 'foo-lib', tenant_id: 'foo-lib' },
        },
      ]),
    });
    await setupClient();

    const detector = detectorReturning('project-a');
    const tool = new RetrieveTool({ qdrantUrl: 'http://localhost:6333' }, detector);

    // Without libraryName: refused
    const refused = await tool.retrieve({
      documentId: 'lib-doc',
      collection: 'libraries',
    });
    expect(refused.success).toBe(false);
    expect(refused.message).toContain('scope');

    // With wrong libraryName: not-found
    const wrong = await tool.retrieve({
      documentId: 'lib-doc',
      collection: 'libraries',
      libraryName: 'other-lib',
    });
    expect(wrong.success).toBe(false);
    expect(wrong.message).toContain('Document not found');
  });

  it('does NOT enforce tenant check for rules collection (mixed-tenancy)', async () => {
    const setupClient = makeMockClient({
      retrieve: vi.fn().mockResolvedValue([
        {
          id: 'rule-1',
          payload: { content: 'global rule', label: 'foo' },
        },
      ]),
    });
    await setupClient();

    const detector = detectorReturning('project-a');
    const tool = new RetrieveTool({ qdrantUrl: 'http://localhost:6333' }, detector);

    const result = await tool.retrieve({
      documentId: 'rule-1',
      collection: 'rules',
    });

    expect(result.success).toBe(true);
    expect(result.documents).toHaveLength(1);
  });
});

describe('RetrieveTool — F-011 project-scope retrieve null projectId', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('returns empty error when scope=projects and tenant cannot be resolved (no scroll)', async () => {
    const scrollFn = vi.fn().mockResolvedValue({ points: [] });
    const setupClient = makeMockClient({ scroll: scrollFn });
    await setupClient();

    const detector = detectorReturning(undefined);
    const tool = new RetrieveTool({ qdrantUrl: 'http://localhost:6333' }, detector);

    const result = await tool.retrieve({ collection: 'projects' });

    expect(scrollFn).not.toHaveBeenCalled();
    expect(result.success).toBe(false);
    expect(result.documents).toHaveLength(0);
    expect(result.total).toBe(0);
    expect(result.hasMore).toBe(false);
    expect(result.message).toContain('scope');
  });

  it('scrolls with tenant filter when projects scope resolves', async () => {
    const scrollFn = vi.fn().mockResolvedValue({
      points: [{ id: '1', payload: { content: 'mine', tenant_id: 'project-a' } }],
    });
    const setupClient = makeMockClient({ scroll: scrollFn });
    await setupClient();

    const detector = detectorReturning('project-a');
    const tool = new RetrieveTool({ qdrantUrl: 'http://localhost:6333' }, detector);

    await tool.retrieve({ collection: 'projects' });

    expect(scrollFn).toHaveBeenCalledTimes(1);
    const [, request] = scrollFn.mock.calls[0];
    const filter = (request as Record<string, unknown>).filter as Record<string, unknown>;
    expect(filter).toBeDefined();
    const must = filter.must as Record<string, unknown>[];
    const tenantClause = must.find((c) => (c as Record<string, unknown>).key === 'tenant_id');
    expect(tenantClause).toEqual({ key: 'tenant_id', match: { value: 'project-a' } });
  });

  it('libraries scope without libraryName refuses to scroll', async () => {
    const scrollFn = vi.fn().mockResolvedValue({ points: [] });
    const setupClient = makeMockClient({ scroll: scrollFn });
    await setupClient();

    const detector = detectorReturning('project-a');
    const tool = new RetrieveTool({ qdrantUrl: 'http://localhost:6333' }, detector);

    const result = await tool.retrieve({ collection: 'libraries' });

    expect(scrollFn).not.toHaveBeenCalled();
    expect(result.success).toBe(false);
    expect(result.message).toContain('scope');
  });
});
