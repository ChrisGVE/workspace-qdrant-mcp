/**
 * Tests for branch auto-detection defaulting in buildSearchOptions and buildListOptions.
 */

import { describe, it, expect } from 'vitest';
import { buildSearchOptions } from '../../src/tool-builders/search.js';
import { buildListOptions } from '../../src/tool-builders/list.js';

// ── buildSearchOptions branch defaulting ────────────────────────────────────

describe('buildSearchOptions — branch defaulting', () => {
  it('uses the explicit branch when provided', () => {
    const opts = buildSearchOptions({ query: 'test', branch: 'feature/x' }, 'main');
    expect(opts.branch).toBe('feature/x');
  });

  it('applies defaultBranch when no branch arg is given', () => {
    const opts = buildSearchOptions({ query: 'test' }, 'main');
    expect(opts.branch).toBe('main');
  });

  it('skips defaultBranch when it is "default" (non-git sentinel)', () => {
    const opts = buildSearchOptions({ query: 'test' }, 'default');
    expect(opts.branch).toBeUndefined();
  });

  it('skips defaultBranch when it is null', () => {
    const opts = buildSearchOptions({ query: 'test' }, null);
    expect(opts.branch).toBeUndefined();
  });

  it('bypasses filter when branch is "*" regardless of defaultBranch', () => {
    const opts = buildSearchOptions({ query: 'test', branch: '*' }, 'main');
    expect(opts.branch).toBeUndefined();
  });

  it('does not set branch when no arg and no default', () => {
    const opts = buildSearchOptions({ query: 'test' });
    expect(opts.branch).toBeUndefined();
  });
});

// ── buildListOptions branch defaulting ──────────────────────────────────────

describe('buildListOptions — branch defaulting', () => {
  it('uses the explicit branch when provided', () => {
    const opts = buildListOptions({ branch: 'release/1.0' }, 'main');
    expect(opts.branch).toBe('release/1.0');
  });

  it('applies defaultBranch when no branch arg is given', () => {
    const opts = buildListOptions({}, 'main');
    expect(opts.branch).toBe('main');
  });

  it('skips defaultBranch when it is "default" (non-git sentinel)', () => {
    const opts = buildListOptions({}, 'default');
    expect(opts.branch).toBeUndefined();
  });

  it('skips defaultBranch when it is null', () => {
    const opts = buildListOptions({}, null);
    expect(opts.branch).toBeUndefined();
  });

  it('sets branch to "*" when explicitly passed', () => {
    const opts = buildListOptions({ branch: '*' }, 'main');
    // "*" is preserved in ListOptions to communicate cross-branch intent
    expect(opts.branch).toBe('*');
  });

  it('does not set branch when no arg and no default', () => {
    const opts = buildListOptions({});
    expect(opts.branch).toBeUndefined();
  });
});
