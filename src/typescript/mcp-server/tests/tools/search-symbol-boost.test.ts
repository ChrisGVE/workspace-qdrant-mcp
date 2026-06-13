/**
 * Unit tests for the definition-site symbol-match predicate behind
 * WQM_SYMBOL_MATCH_BOOST: a result whose chunk symbol is NAMED by the query
 * gets boosted past files that merely mention or test the symbol.
 */

import { describe, it, expect } from 'vitest';
import { extractSupplementalNeedles, symbolNamedInQuery } from '../../src/tools/search-helpers.js';

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

describe('extractSupplementalNeedles', () => {
  it('extracts explicit distinctive code identifiers for supplemental lookup', () => {
    expect(extractSupplementalNeedles('applyRRFFusion implementation')).toEqual([
      'applyRRFFusion',
      'applyrrffusion',
    ]);
    expect(extractSupplementalNeedles('SPARSE_ONLY_WEIGHT sparse-only demotion')).toEqual([
      'SPARSE_ONLY_WEIGHT',
      'sparse-only-weight',
    ]);
    expect(extractSupplementalNeedles('recover_stale_unified_leases startup recovery')).toEqual([
      'recover_stale_unified_leases',
      'recover-stale-unified-leases',
    ]);
  });

  it('does not trigger on common acronyms in natural-language queries', () => {
    expect(extractSupplementalNeedles('How does the MCP server resolve project scope?')).toEqual([]);
    expect(extractSupplementalNeedles('Where is the BM25 tokenizer implemented?')).toEqual([]);
    expect(extractSupplementalNeedles('Where is the FTS5 exact query built?')).toEqual([]);
    expect(extractSupplementalNeedles('Where does the daemon use FastEmbed ONNX?')).toEqual([
      'FastEmbed',
      'fastembed',
    ]);
  });

  it('adds high-precision conceptual path hints for known search concepts', () => {
    expect(extractSupplementalNeedles('reciprocal rank fusion dense sparse')).toEqual([
      'applyRRFFusion',
      'search-qdrant.ts',
    ]);
    expect(extractSupplementalNeedles('Where are search results filtered by git branch?')).toEqual([
      'search-filters.ts',
    ]);
    expect(
      extractSupplementalNeedles('Como as regras com escopo de projeto são filtradas por tenant?')
    ).toEqual(['tools/rules.ts']);
    expect(extractSupplementalNeedles('Where is queue throughput measured in the daemon?')).toEqual([
      'unified_queue_processor/metrics.rs',
    ]);
  });
});
