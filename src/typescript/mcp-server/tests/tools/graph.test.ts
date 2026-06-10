/**
 * Tests for the `graph` MCP tool handler (code-relationship navigation).
 *
 * Tenant resolution mirrors `search`/`grep`/`list`: explicit `projectId`
 * wins; otherwise the caller's cwd is resolved via the ProjectDetector.
 * There is deliberately NO "first active project" fallback — see graph.ts.
 */

import { describe, it, expect, vi } from 'vitest';
import { createHash } from 'node:crypto';

import { handleGraph } from '../../src/tools/graph.js';
import type { DaemonClient } from '../../src/clients/daemon-client.js';
import type { ProjectDetector } from '../../src/utils/project-detector.js';

/** node_id matching Rust's compute_node_id (and the handler's). */
function computeNodeId(
  tenantId: string,
  filePath: string,
  symbolName: string,
  symbolType: string
): string {
  return createHash('sha256')
    .update(`${tenantId}|${filePath}|${symbolName}|${symbolType}`)
    .digest('hex')
    .slice(0, 32);
}

function mockClient(overrides: Partial<Record<string, ReturnType<typeof vi.fn>>> = {}): {
  client: DaemonClient;
  mocks: Record<string, ReturnType<typeof vi.fn>>;
} {
  const mocks = {
    getGraphStats: vi
      .fn()
      .mockResolvedValue({ total_nodes: 10, total_edges: 20, nodes_by_type: {}, edges_by_type: {} }),
    impactAnalysis: vi
      .fn()
      .mockResolvedValue({ impacted_nodes: [], total_impacted: 0, query_time_ms: 1 }),
    computePageRank: vi.fn().mockResolvedValue({ entries: [], total: 0, query_time_ms: 1 }),
    detectCommunities: vi
      .fn()
      .mockResolvedValue({ communities: [], total_communities: 0, query_time_ms: 1 }),
    computeBetweenness: vi.fn().mockResolvedValue({ entries: [], total: 0, query_time_ms: 1 }),
    queryRelated: vi.fn().mockResolvedValue({ nodes: [], total: 0, query_time_ms: 1 }),
    ...overrides,
  };
  return { client: mocks as unknown as DaemonClient, mocks };
}

/** ProjectDetector stub: resolves the caller's cwd to `projectId` (or null). */
function mockDetector(projectId: string | null): {
  detector: ProjectDetector;
  getProjectInfo: ReturnType<typeof vi.fn>;
} {
  const getProjectInfo = vi
    .fn()
    .mockResolvedValue(projectId === null ? null : { projectId });
  return { detector: { getProjectInfo } as unknown as ProjectDetector, getProjectInfo };
}

describe('handleGraph', () => {
  it('rejects when no daemon client is connected', async () => {
    const { detector } = mockDetector('auto-tenant');
    await expect(handleGraph({ action: 'stats' }, undefined, detector)).rejects.toThrow(
      /daemon client/i
    );
  });

  it("defaults to 'stats' and resolves the tenant from the caller's cwd project", async () => {
    const { client, mocks } = mockClient();
    const { detector, getProjectInfo } = mockDetector('auto-tenant');
    const result = (await handleGraph({}, client, detector)) as Record<string, unknown>;
    // Same resolution contract as search/grep/list: cwd → project, with the
    // sole-project convenience fallback.
    expect(getProjectInfo).toHaveBeenCalledWith(expect.any(String), false, {
      fallbackToSoleProject: true,
    });
    expect(mocks['getGraphStats']).toHaveBeenCalledWith({ tenant_id: 'auto-tenant' });
    expect(result['action']).toBe('stats');
    expect(result['tenant_id']).toBe('auto-tenant');
    expect(result['total_nodes']).toBe(10);
  });

  it('prefers an explicit projectId over auto-detection', async () => {
    const { client, mocks } = mockClient();
    const { detector, getProjectInfo } = mockDetector('auto-tenant');
    await handleGraph({ action: 'stats', projectId: 'explicit-tenant' }, client, detector);
    expect(getProjectInfo).not.toHaveBeenCalled();
    expect(mocks['getGraphStats']).toHaveBeenCalledWith({ tenant_id: 'explicit-tenant' });
  });

  it('impact forwards the symbol and optional file_path', async () => {
    const { client, mocks } = mockClient();
    const { detector } = mockDetector(null);
    await handleGraph(
      { action: 'impact', symbol: 'authenticate', filePath: 'src/auth.rs', projectId: 't1' },
      client,
      detector
    );
    expect(mocks['impactAnalysis']).toHaveBeenCalledWith({
      tenant_id: 't1',
      symbol_name: 'authenticate',
      file_path: 'src/auth.rs',
    });
  });

  it('usages forwards to ImpactAnalysis (find-usages framing)', async () => {
    const { client, mocks } = mockClient();
    const { detector } = mockDetector(null);
    const result = (await handleGraph(
      { action: 'usages', symbol: 'parse', projectId: 't1' },
      client,
      detector
    )) as Record<string, unknown>;
    expect(mocks['impactAnalysis']).toHaveBeenCalledWith({
      tenant_id: 't1',
      symbol_name: 'parse',
    });
    expect(result['action']).toBe('usages');
  });

  it('impact requires a symbol', async () => {
    const { client } = mockClient();
    const { detector } = mockDetector(null);
    await expect(
      handleGraph({ action: 'impact', projectId: 't1' }, client, detector)
    ).rejects.toThrow(/requires `symbol`/);
  });

  it('hotspots passes top_k (default 20)', async () => {
    const { client, mocks } = mockClient();
    const { detector } = mockDetector(null);
    await handleGraph({ action: 'hotspots', projectId: 't1' }, client, detector);
    expect(mocks['computePageRank']).toHaveBeenCalledWith({ tenant_id: 't1', top_k: 20 });
    await handleGraph({ action: 'hotspots', projectId: 't1', topK: 5 }, client, detector);
    expect(mocks['computePageRank']).toHaveBeenLastCalledWith({ tenant_id: 't1', top_k: 5 });
  });

  it('relations computes the Rust-compatible node_id and passes maxHops', async () => {
    const { client, mocks } = mockClient();
    const { detector } = mockDetector(null);
    await handleGraph(
      {
        action: 'relations',
        symbol: 'authenticate',
        filePath: 'src/auth.rs',
        symbolType: 'function',
        maxHops: 2,
        projectId: 't1',
      },
      client,
      detector
    );
    const expectedId = computeNodeId('t1', 'src/auth.rs', 'authenticate', 'function');
    expect(mocks['queryRelated']).toHaveBeenCalledWith({
      tenant_id: 't1',
      node_id: expectedId,
      max_hops: 2,
    });
  });

  it('relations requires symbol and filePath', async () => {
    const { client } = mockClient();
    const { detector } = mockDetector(null);
    await expect(
      handleGraph({ action: 'relations', symbol: 'x', projectId: 't1' }, client, detector)
    ).rejects.toThrow(/requires `symbol` and `filePath`/);
  });

  it('forwards edgeTypes filter when provided', async () => {
    const { client, mocks } = mockClient();
    const { detector } = mockDetector(null);
    await handleGraph(
      { action: 'hotspots', projectId: 't1', edgeTypes: ['CALLS', 'IMPORTS'] },
      client,
      detector
    );
    expect(mocks['computePageRank']).toHaveBeenCalledWith({
      tenant_id: 't1',
      top_k: 20,
      edge_types: ['CALLS', 'IMPORTS'],
    });
  });

  it('bridges (betweenness) passes top_k and optional max_samples', async () => {
    const { client, mocks } = mockClient();
    const { detector } = mockDetector(null);
    await handleGraph({ action: 'bridges', projectId: 't1' }, client, detector);
    expect(mocks['computeBetweenness']).toHaveBeenCalledWith({ tenant_id: 't1', top_k: 20 });
    await handleGraph(
      { action: 'bridges', projectId: 't1', topK: 5, maxSamples: 100 },
      client,
      detector
    );
    expect(mocks['computeBetweenness']).toHaveBeenLastCalledWith({
      tenant_id: 't1',
      top_k: 5,
      max_samples: 100,
    });
  });

  it('rejects an unknown action', async () => {
    const { client } = mockClient();
    const { detector } = mockDetector(null);
    await expect(
      handleGraph({ action: 'bogus', projectId: 't1' }, client, detector)
    ).rejects.toThrow(/Unknown graph action/);
  });

  it('errors clearly when the project cannot be resolved and no projectId is given', async () => {
    const { client } = mockClient();
    const { detector } = mockDetector(null);
    // No silent "first active project" fallback: unresolvable cwd must fail
    // loudly, pointing the caller at projectId/cwd.
    await expect(handleGraph({ action: 'stats' }, client, detector)).rejects.toThrow(
      /Could not resolve a project/
    );
  });
});
