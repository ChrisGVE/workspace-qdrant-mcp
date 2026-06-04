/**
 * Tests for the idle-session reaper that reclaims leaked stateful sessions.
 *
 * A client that loses its SSE stream re-initializes with a NEW session id and
 * never DELETEs the old one, so without eviction each reconnect would leak a
 * transport + MCP server instance (and its daemon gRPC connection).
 */

import { describe, it, expect, vi } from 'vitest';
import type { StreamableHTTPServerTransport } from '@modelcontextprotocol/sdk/server/streamableHttp.js';
import { reapIdleSessions } from '../src/mcp-http-server.js';

function fakeTransport(): StreamableHTTPServerTransport {
  return { close: vi.fn().mockResolvedValue(undefined) } as unknown as StreamableHTTPServerTransport;
}

const NOW = 1_000_000_000_000;
const TTL = 10 * 60_000; // 10 min

describe('reapIdleSessions', () => {
  it('evicts only sessions idle longer than the TTL', () => {
    const a = fakeTransport(); // fresh
    const b = fakeTransport(); // stale
    const c = fakeTransport(); // borderline-fresh
    const transports = new Map([
      ['a', a],
      ['b', b],
      ['c', c],
    ]);
    const lastSeen = new Map([
      ['a', NOW - 1 * 60_000],
      ['b', NOW - 20 * 60_000],
      ['c', NOW - 9 * 60_000],
    ]);

    const evicted = reapIdleSessions(transports, lastSeen, NOW, TTL);

    expect(evicted).toEqual(['b']);
    expect(b.close).toHaveBeenCalledTimes(1);
    expect(a.close).not.toHaveBeenCalled();
    expect(c.close).not.toHaveBeenCalled();
  });

  it('treats a session with no recorded activity as infinitely idle', () => {
    const orphan = fakeTransport();
    const transports = new Map([['orphan', orphan]]);
    const lastSeen = new Map<string, number>(); // no entry for "orphan"

    const evicted = reapIdleSessions(transports, lastSeen, NOW, TTL);

    expect(evicted).toEqual(['orphan']);
    expect(orphan.close).toHaveBeenCalledTimes(1);
  });

  it('evicts nothing when all sessions are within the TTL', () => {
    const a = fakeTransport();
    const transports = new Map([['a', a]]);
    const lastSeen = new Map([['a', NOW - 5 * 60_000]]);

    const evicted = reapIdleSessions(transports, lastSeen, NOW, TTL);

    expect(evicted).toEqual([]);
    expect(a.close).not.toHaveBeenCalled();
  });
});
