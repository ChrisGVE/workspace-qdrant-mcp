/**
 * Token-economy byte computation for the three non-search MCP tools
 * (grep, retrieve, list).
 *
 * Spec: docs/specs/20-token-economy-instrumentation.md §3.2-§3.4.
 *
 * These tests cover the *pure* per-tool helpers — the gRPC wiring and
 * fire-and-forget instrumentation are tested separately via the mock
 * SqliteStateManager / DaemonClient in the existing tool-suite tests.
 */

import { describe, it, expect } from 'vitest';
import {
  computeGrepEconomy,
  GREP_BYTES_IN_PER_FILE_PROXY,
  type GrepMatch,
} from '../../src/tools/grep.js';
import { computeRetrieveEconomy } from '../../src/tools/retrieve.js';
import type { RetrievedDocument } from '../../src/tools/retrieve-types.js';
import {
  computeListEconomy,
  LIST_BYTES_IN_PER_FILE_PROXY,
} from '../../src/tools/list-files/index.js';

// ───────────────────── grep ─────────────────────

function grepMatch(overrides: Partial<GrepMatch> = {}): GrepMatch {
  return {
    file: 'src/foo.ts',
    line: 1,
    content: 'hit body',
    context_before: [],
    context_after: [],
    ...overrides,
  };
}

describe('computeGrepEconomy', () => {
  it('returns zero for an empty match set', () => {
    const { bytesIn, bytesOut } = computeGrepEconomy([]);
    expect(bytesOut).toBe(0);
    expect(bytesIn).toBe(0);
  });

  it('sums content + context_before + context_after for bytesOut', () => {
    const matches: GrepMatch[] = [
      grepMatch({
        content: 'aaaa', // 4
        context_before: ['bb', 'cc'], // 2 + 2 = 4
        context_after: ['ddd'], // 3
      }),
    ];
    const { bytesOut } = computeGrepEconomy(matches);
    expect(bytesOut).toBe(4 + 4 + 3);
  });

  it('uses unique_files * PROXY for bytesIn when files dominate', () => {
    // Two matches in different files, small content. bytesIn should be
    // dominated by the file-count proxy (2 × 8192).
    const matches: GrepMatch[] = [
      grepMatch({ file: 'a.ts', content: 'x' }),
      grepMatch({ file: 'b.ts', content: 'y' }),
    ];
    const { bytesIn } = computeGrepEconomy(matches);
    expect(bytesIn).toBe(2 * GREP_BYTES_IN_PER_FILE_PROXY);
  });

  it('counts each unique file only once regardless of match count', () => {
    const matches: GrepMatch[] = [
      grepMatch({ file: 'a.ts', line: 1 }),
      grepMatch({ file: 'a.ts', line: 5 }),
      grepMatch({ file: 'a.ts', line: 9 }),
    ];
    const { bytesIn } = computeGrepEconomy(matches);
    expect(bytesIn).toBe(1 * GREP_BYTES_IN_PER_FILE_PROXY);
  });

  it('falls back to bytesOut when summed-content exceeds the file proxy', () => {
    // One match in one file but with an enormous content. bytesIn must
    // never report less than what we actually shipped out.
    const matches: GrepMatch[] = [
      grepMatch({ file: 'a.ts', content: 'q'.repeat(20_000) }),
    ];
    const { bytesIn, bytesOut } = computeGrepEconomy(matches);
    expect(bytesOut).toBe(20_000);
    expect(bytesIn).toBeGreaterThanOrEqual(bytesOut);
  });

  describe('file-size probe (search.db v7)', () => {
    it('uses the reported file_size in place of the proxy', () => {
      // 12 KiB file — bigger than the 8 KiB proxy, smaller than two
      // proxies. The result must match the reported size, proving the
      // proxy was bypassed.
      const matches: GrepMatch[] = [
        grepMatch({ file: 'a.ts', content: 'x', file_size: 12_000 }),
      ];
      const { bytesIn } = computeGrepEconomy(matches);
      expect(bytesIn).toBe(12_000);
    });

    it('mixes real sizes with proxy fallback per-file', () => {
      // Two files: one carries a real size, the other does not. The
      // proxy MUST apply only to the size-less file.
      const matches: GrepMatch[] = [
        grepMatch({ file: 'a.ts', content: 'x', file_size: 50_000 }),
        grepMatch({ file: 'b.ts', content: 'y' }),
      ];
      const { bytesIn } = computeGrepEconomy(matches);
      expect(bytesIn).toBe(50_000 + GREP_BYTES_IN_PER_FILE_PROXY);
    });

    it('treats file_size === 0 as missing and falls back to the proxy', () => {
      // Proto3 defaults unset non-optional int64 to 0 — we must not
      // treat that as "this file is empty".
      const matches: GrepMatch[] = [
        grepMatch({ file: 'a.ts', content: 'x', file_size: 0 }),
      ];
      const { bytesIn } = computeGrepEconomy(matches);
      expect(bytesIn).toBe(GREP_BYTES_IN_PER_FILE_PROXY);
    });

    it('uses each unique file_size exactly once across matches in that file', () => {
      const matches: GrepMatch[] = [
        grepMatch({ file: 'a.ts', line: 1, file_size: 4_000 }),
        grepMatch({ file: 'a.ts', line: 9, file_size: 4_000 }),
      ];
      const { bytesIn } = computeGrepEconomy(matches);
      // 4_000 bytes counted once, NOT 8_000. bytesOut floor still
      // applies but content is small here.
      expect(bytesIn).toBe(4_000);
    });

    it('still floors bytesIn at bytesOut even when file_size is small', () => {
      // Real file_size is 100 B but the search shipped 20_000 B of
      // content (e.g. expanded context). The honest bytes_in must not
      // claim savings for content that was actually delivered.
      const matches: GrepMatch[] = [
        grepMatch({ file: 'a.ts', content: 'q'.repeat(20_000), file_size: 100 }),
      ];
      const { bytesIn, bytesOut } = computeGrepEconomy(matches);
      expect(bytesIn).toBe(bytesOut);
    });
  });
});

// ───────────────────── retrieve ─────────────────────

function doc(content: string, id = 'd1'): RetrievedDocument {
  return { id, content, metadata: {} };
}

describe('computeRetrieveEconomy', () => {
  it('returns zero for an empty document set', () => {
    const { bytesIn, bytesOut } = computeRetrieveEconomy([]);
    expect(bytesOut).toBe(0);
    expect(bytesIn).toBe(0);
  });

  it('sums content lengths', () => {
    const docs = [doc('abc'), doc('defgh'), doc('')];
    const { bytesOut } = computeRetrieveEconomy(docs);
    expect(bytesOut).toBe(8);
  });

  it('reports bytesIn === bytesOut for full-document retrieve (savings 0%)', () => {
    // Spec §3.3: current implementation always returns full docs, so
    // there is no shaving — savings_ratio should land at 0%. The row
    // remains useful for escalation tracking.
    const docs = [doc('alpha'), doc('beta')];
    const { bytesIn, bytesOut } = computeRetrieveEconomy(docs);
    expect(bytesIn).toBe(bytesOut);
  });
});

// ───────────────────── list ─────────────────────

describe('computeListEconomy', () => {
  it('reports rendered listing length as bytesOut', () => {
    const listing = 'src/\n  foo.ts\n  bar.ts\n';
    const { bytesOut } = computeListEconomy(listing, 2);
    expect(bytesOut).toBe(listing.length);
  });

  it('uses totalMatching * PROXY for bytesIn when files dominate', () => {
    // Compact listing but many matching files — bytesIn dominated by
    // the per-path proxy.
    const { bytesIn } = computeListEconomy('truncated\n', 1000);
    expect(bytesIn).toBe(1000 * LIST_BYTES_IN_PER_FILE_PROXY);
  });

  it('falls back to bytesOut when the listing exceeds the path proxy', () => {
    // Verbose listing (e.g. summary format) on a small project — bytesIn
    // must not understate what we shipped.
    const listing = 'x'.repeat(50_000);
    const { bytesIn, bytesOut } = computeListEconomy(listing, 5);
    expect(bytesOut).toBe(50_000);
    expect(bytesIn).toBeGreaterThanOrEqual(bytesOut);
  });

  it('handles zero matching files gracefully', () => {
    const { bytesIn, bytesOut } = computeListEconomy('', 0);
    expect(bytesOut).toBe(0);
    expect(bytesIn).toBe(0);
  });
});
