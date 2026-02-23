/**
 * Tests for search graph context enrichment
 */

import { describe, it, expect, vi } from 'vitest';
import { expandGraphContext } from '../../src/tools/search-graph-context.js';
import type { SearchResult } from '../../src/tools/search-types.js';
import type { DaemonClient } from '../../src/clients/daemon-client.js';
import { createHash } from 'node:crypto';

/** Helper: compute node_id matching Rust algorithm */
function computeNodeId(tenantId: string, filePath: string, symbolName: string, symbolType: string): string {
  const input = `${tenantId}|${filePath}|${symbolName}|${symbolType}`;
  return createHash('sha256').update(input).digest('hex').slice(0, 32);
}

/** Helper: create a search result with code chunk metadata */
function codeResult(overrides?: Partial<SearchResult['metadata']>): SearchResult {
  return {
    id: 'point-1',
    score: 0.9,
    collection: 'projects',
    content: 'fn authenticate(user: &str) { ... }',
    metadata: {
      tenant_id: 'test-tenant',
      relative_path: 'src/auth.rs',
      file_path: 'src/auth.rs',
      chunk_symbol_name: 'authenticate',
      chunk_chunk_type: 'function',
      ...overrides,
    },
  };
}

/** Helper: create a non-code search result */
function textResult(): SearchResult {
  return {
    id: 'point-2',
    score: 0.8,
    collection: 'libraries',
    content: 'Some documentation text',
    metadata: {
      tenant_id: 'test-tenant',
      item_type: 'text',
    },
  };
}

function createMockDaemonClient(queryRelatedMock?: ReturnType<typeof vi.fn>): DaemonClient {
  return {
    queryRelated: queryRelatedMock ?? vi.fn().mockResolvedValue({ nodes: [], total: 0, query_time_ms: 0 }),
  } as unknown as DaemonClient;
}

describe('expandGraphContext', () => {
  it('should skip results without chunk_symbol_name', async () => {
    const client = createMockDaemonClient();
    const results = [textResult()];
    await expandGraphContext(client, results);
    expect(client.queryRelated).not.toHaveBeenCalled();
    expect(results[0].graph_context).toBeUndefined();
  });

  it('should skip results without tenant_id', async () => {
    const client = createMockDaemonClient();
    const results = [codeResult({ tenant_id: undefined })];
    await expandGraphContext(client, results);
    expect(client.queryRelated).not.toHaveBeenCalled();
  });

  it('should query graph for code chunk results', async () => {
    const queryRelated = vi.fn().mockResolvedValue({
      nodes: [
        {
          node_id: 'callee-1',
          symbol_name: 'hash_password',
          symbol_type: 'function',
          file_path: 'src/crypto.rs',
          edge_type: 'CALLS',
          depth: 1,
          path: 'authenticate -> hash_password',
        },
      ],
      total: 1,
      query_time_ms: 2,
    });
    const client = createMockDaemonClient(queryRelated);
    const results = [codeResult()];

    await expandGraphContext(client, results);

    expect(queryRelated).toHaveBeenCalledOnce();
    const call = queryRelated.mock.calls[0][0];
    expect(call.tenant_id).toBe('test-tenant');
    expect(call.node_id).toBe(computeNodeId('test-tenant', 'src/auth.rs', 'authenticate', 'function'));
    expect(call.max_hops).toBe(1);
  });

  it('should populate graph_context with callers and callees', async () => {
    const queryRelated = vi.fn().mockResolvedValue({
      nodes: [
        {
          node_id: 'callee-1',
          symbol_name: 'hash_password',
          symbol_type: 'function',
          file_path: 'src/crypto.rs',
          edge_type: 'CALLS',
          depth: 1,
          path: '',
        },
        {
          node_id: 'caller-1',
          symbol_name: 'AuthModule',
          symbol_type: 'struct',
          file_path: 'src/auth.rs',
          edge_type: 'CONTAINS',
          depth: 1,
          path: '',
        },
      ],
      total: 2,
      query_time_ms: 1,
    });

    const client = createMockDaemonClient(queryRelated);
    const results = [codeResult()];

    await expandGraphContext(client, results);

    expect(results[0].graph_context).toBeDefined();
    const ctx = results[0].graph_context!;
    expect(ctx.symbol).toBe('authenticate');
    expect(ctx.file_path).toBe('src/auth.rs');
    expect(ctx.callees).toHaveLength(1);
    expect(ctx.callees[0].symbol).toBe('hash_password');
    expect(ctx.callees[0].file_path).toBe('src/crypto.rs');
    expect(ctx.callers).toHaveLength(1);
    expect(ctx.callers[0].symbol).toBe('AuthModule');
  });

  it('should handle graph query failure gracefully', async () => {
    const queryRelated = vi.fn().mockRejectedValue(new Error('daemon unavailable'));
    const client = createMockDaemonClient(queryRelated);
    const results = [codeResult()];

    await expandGraphContext(client, results);

    expect(results[0].graph_context).toBeUndefined();
  });

  it('should handle graph query timeout gracefully', async () => {
    const queryRelated = vi.fn().mockImplementation(
      () => new Promise((resolve) => setTimeout(() => resolve({ nodes: [], total: 0, query_time_ms: 0 }), 500)),
    );
    const client = createMockDaemonClient(queryRelated);
    const results = [codeResult()];

    await expandGraphContext(client, results);

    // Should timeout (200ms) before the 500ms resolve
    expect(results[0].graph_context).toBeUndefined();
  });

  it('should process multiple results in parallel', async () => {
    const queryRelated = vi.fn().mockResolvedValue({
      nodes: [{ node_id: 'n1', symbol_name: 'helper', symbol_type: 'function', file_path: 'src/util.rs', edge_type: 'CALLS', depth: 1, path: '' }],
      total: 1,
      query_time_ms: 1,
    });
    const client = createMockDaemonClient(queryRelated);
    const results = [
      codeResult(),
      codeResult({ chunk_symbol_name: 'login', relative_path: 'src/routes.rs', file_path: 'src/routes.rs' }),
    ];

    await expandGraphContext(client, results);

    expect(queryRelated).toHaveBeenCalledTimes(2);
    expect(results[0].graph_context).toBeDefined();
    expect(results[1].graph_context).toBeDefined();
  });

  it('should skip non-code chunk types', async () => {
    const client = createMockDaemonClient();
    const results = [codeResult({ chunk_chunk_type: 'text' })];

    await expandGraphContext(client, results);

    expect(client.queryRelated).not.toHaveBeenCalled();
    expect(results[0].graph_context).toBeUndefined();
  });

  it('should not set graph_context when no nodes returned', async () => {
    const queryRelated = vi.fn().mockResolvedValue({ nodes: [], total: 0, query_time_ms: 0 });
    const client = createMockDaemonClient(queryRelated);
    const results = [codeResult()];

    await expandGraphContext(client, results);

    expect(queryRelated).toHaveBeenCalledOnce();
    expect(results[0].graph_context).toBeUndefined();
  });
});
