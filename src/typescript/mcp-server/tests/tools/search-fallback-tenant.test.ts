/**
 * Regression tests for F-001 — fallback Qdrant scroll MUST apply tenant
 * filtering or refuse to scroll. The pre-fix path scrolled every
 * collection unfiltered and did local substring matching on the first
 * page, leaking documents owned by foreign tenants.
 */

import { describe, it, expect, vi } from 'vitest';
import { fallbackSearch } from '../../src/tools/search-qdrant.js';
import {
  PROJECTS_COLLECTION,
  LIBRARIES_COLLECTION,
  type SearchOptions,
} from '../../src/tools/search-types.js';
import type { QdrantClient } from '@qdrant/js-client-rest';

function makeOptions(overrides: Partial<SearchOptions> = {}): SearchOptions {
  return {
    query: 'needle',
    scope: 'project',
    ...overrides,
  };
}

function makeMockClient(
  scrollImpl: (
    collection: string,
    request: Record<string, unknown>
  ) => Promise<{
    points: Array<{ id: string | number; payload: Record<string, unknown> }>;
  }>
): QdrantClient {
  return {
    scroll: vi.fn(scrollImpl),
  } as unknown as QdrantClient;
}

describe('fallbackSearch — tenant isolation (F-001)', () => {
  it('refuses to scroll when scope=project and tenant is unresolved', async () => {
    const calls: Array<{ collection: string; request: Record<string, unknown> }> = [];
    const client = makeMockClient(async (collection, request) => {
      calls.push({ collection, request });
      return { points: [] };
    });

    const response = await fallbackSearch(client, makeOptions(), [PROJECTS_COLLECTION], {
      currentProjectId: undefined,
      basePoints: undefined,
    });

    expect(calls).toHaveLength(0);
    expect(response.results).toEqual([]);
    expect(response.status).toBe('uncertain');
    expect(response.status_reason).toContain('project scope unresolved');
    expect(response.status_reason).toContain(PROJECTS_COLLECTION);
  });

  it('applies tenant filter to scroll when scope=project and tenant resolves', async () => {
    const calls: Array<{ collection: string; request: Record<string, unknown> }> = [];
    const client = makeMockClient(async (collection, request) => {
      calls.push({ collection, request });
      return {
        points: [
          {
            id: 'p1',
            payload: { content: 'a needle in haystack', tenant_id: 'project-a' },
          },
        ],
      };
    });

    const response = await fallbackSearch(
      client,
      makeOptions({ projectId: 'project-a' }),
      [PROJECTS_COLLECTION],
      { currentProjectId: 'project-a', basePoints: undefined }
    );

    expect(calls).toHaveLength(1);
    const filter = calls[0].request.filter as Record<string, unknown>;
    expect(filter).toBeDefined();
    const must = filter.must as Record<string, unknown>[];
    const tenantClause = must.find((c) => (c as Record<string, unknown>).key === 'tenant_id');
    expect(tenantClause).toEqual({ key: 'tenant_id', match: { value: 'project-a' } });
    expect(response.results).toHaveLength(1);
    expect(response.status).toBe('uncertain');
  });

  it('refuses every collection when project-scope unresolved across multi-collection request', async () => {
    const calls: string[] = [];
    const client = makeMockClient(async (collection) => {
      calls.push(collection);
      return { points: [] };
    });

    const response = await fallbackSearch(
      client,
      makeOptions(),
      [PROJECTS_COLLECTION, LIBRARIES_COLLECTION],
      { currentProjectId: undefined, basePoints: undefined }
    );

    expect(calls).toHaveLength(0);
    expect(response.status_reason).toContain(PROJECTS_COLLECTION);
    // libraries collection is only filtered when libraryName is set; with
    // no libraryName + no projectId, project-scope refuses libraries too.
    expect(response.status_reason).toContain(LIBRARIES_COLLECTION);
  });

  it('still substring-filters within the tenant-scoped page', async () => {
    const client = makeMockClient(async () => ({
      points: [
        { id: 'p1', payload: { content: 'has the needle', tenant_id: 'project-a' } },
        { id: 'p2', payload: { content: 'no match here', tenant_id: 'project-a' } },
      ],
    }));

    const response = await fallbackSearch(client, makeOptions(), [PROJECTS_COLLECTION], {
      currentProjectId: 'project-a',
      basePoints: undefined,
    });

    expect(response.results).toHaveLength(1);
    expect(response.results[0].id).toBe('p1');
  });

  it('caps the response to options.limit', async () => {
    const client = makeMockClient(async () => ({
      points: [
        { id: '1', payload: { content: 'needle 1', tenant_id: 'project-a' } },
        { id: '2', payload: { content: 'needle 2', tenant_id: 'project-a' } },
        { id: '3', payload: { content: 'needle 3', tenant_id: 'project-a' } },
      ],
    }));

    const response = await fallbackSearch(
      client,
      makeOptions({ limit: 2 }),
      [PROJECTS_COLLECTION],
      { currentProjectId: 'project-a', basePoints: undefined }
    );

    expect(response.results).toHaveLength(2);
  });
});
