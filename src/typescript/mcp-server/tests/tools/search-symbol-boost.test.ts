/**
 * Unit tests for the definition-site symbol-match predicate behind
 * WQM_SYMBOL_MATCH_BOOST: a result whose chunk symbol is NAMED by the query
 * gets boosted past files that merely mention or test the symbol.
 */

import { describe, it, expect } from 'vitest';
import { symbolNamedInQuery } from '../../src/tools/search-helpers.js';

/** Mirror of the query-side tokenization relevant here (camelCase + _ split). */
function tokens(query: string): Set<string> {
  return new Set(
    query
      .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
      .toLowerCase()
      .split(/[^a-z0-9]+/)
      .filter((t) => t.length >= 3)
  );
}

describe('symbolNamedInQuery', () => {
  it('matches a camelCase symbol named verbatim in the query', () => {
    expect(symbolNamedInQuery(tokens('applyRRFFusion implementation'), 'applyRRFFusion')).toBe(true);
  });

  it('matches a SCREAMING_SNAKE constant named in the query', () => {
    expect(
      symbolNamedInQuery(tokens('SPARSE_ONLY_WEIGHT sparse-only demotion in fusion'), 'SPARSE_ONLY_WEIGHT')
    ).toBe(true);
  });

  it('matches a snake_case symbol whose full identifier appears in the query', () => {
    expect(
      symbolNamedInQuery(tokens('recover_stale_unified_leases startup recovery'), 'recover_stale_unified_leases')
    ).toBe(true);
  });

  it('does not match when a symbol token is missing from the query', () => {
    expect(symbolNamedInQuery(tokens('recover stale leases'), 'recover_stale_unified_leases')).toBe(false);
  });

  it('requires distinctive (>=8 char) single-token symbols — generic module/function names never boost', () => {
    expect(symbolNamedInQuery(tokens('create a new thing'), 'new')).toBe(false);
    // Common one-word symbols exist in dozens of files; boosting them on
    // ordinary queries measurably tanked top-1 (36.4 -> 20.5 on the 44-query
    // benchmark) before this guard.
    expect(symbolNamedInQuery(tokens('where is queue throughput measured'), 'queue')).toBe(false);
    expect(symbolNamedInQuery(tokens('how does search choose modes'), 'search')).toBe(false);
    expect(symbolNamedInQuery(tokens('where is the debouncer configured'), 'debouncer')).toBe(true);
  });

  it('never matches pseudo-symbols from text-fallback chunks', () => {
    expect(symbolNamedInQuery(tokens('text fallback chunks'), '_text')).toBe(false);
    expect(symbolNamedInQuery(tokens('file preamble docs'), '_preamble')).toBe(false);
  });

  it('never matches an empty symbol', () => {
    expect(symbolNamedInQuery(tokens('anything at all'), '')).toBe(false);
  });
});
