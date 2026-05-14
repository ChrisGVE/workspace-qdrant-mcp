/**
 * Cross-language validation parity tests for the canonical path abstraction.
 *
 * Validates that the TypeScript fromUserInput/fromValidated implementations
 * produce identical results (success or specific error kind) to the Rust
 * CanonicalPath::from_user_input implementation on a shared set of test
 * vectors.
 *
 * The Rust implementation lives in src/rust/common/src/paths/. These vectors
 * are derived from the Rust test suite in src/rust/common/src/paths/tests.rs.
 * Any discrepancy between Rust and TypeScript behavior must be investigated
 * and resolved — both implementations must agree on every input.
 *
 * Spec reference: docs/specs/16-path-abstraction.md §3.1, §4.2
 */

import { describe, it, expect } from 'vitest';
import { fromUserInput, fromValidated, PathError } from '../../src/common/paths.js';

// ---------------------------------------------------------------------------
// Shared test vector format
// ---------------------------------------------------------------------------

/**
 * A test vector shared with the Rust implementation.
 *
 * - `input`: raw string passed to fromUserInput / from_user_input
 * - `expect_ok`: if true, the call must succeed and `output` is the expected
 *   canonical string (tilde vectors use `null` to mean "starts with /")
 * - `expect_error`: if true, the call must throw a PathError
 * - `error_kind`: which PathErrorKind the TypeScript implementation must emit
 *   (must correspond to the Rust PathError variant)
 */
interface PathVector {
  label: string;
  input: string;
  expect_ok: boolean;
  output?: string | null; // null = dynamic (home-dir-based), string = literal
  expect_error: boolean;
  error_kind?: string;
}

// ---------------------------------------------------------------------------
// Rust-derived test vectors
// (sourced from src/rust/common/src/paths/tests.rs)
// ---------------------------------------------------------------------------

const SUCCESS_VECTORS: PathVector[] = [
  // rule_1_absolute_required (positive)
  {
    label: 'rule1: simple absolute',
    input: '/Users/chris',
    expect_ok: true,
    output: '/Users/chris',
    expect_error: false,
  },
  // rule_2_tilde_expansion (tilde → dynamic)
  {
    label: 'rule2: tilde+subdir',
    input: '~/project',
    expect_ok: true,
    output: null,
    expect_error: false,
  },
  // rule_3_dot_segments_removed
  {
    label: 'rule3: dot segments',
    input: '/Users/./chris/./dev',
    expect_ok: true,
    output: '/Users/chris/dev',
    expect_error: false,
  },
  // rule_5_duplicate_slash_collapsed
  {
    label: 'rule5: dup slashes',
    input: '/Users//chris///dev',
    expect_ok: true,
    output: '/Users/chris/dev',
    expect_error: false,
  },
  // rule_6_case_preserved
  {
    label: 'rule6: mixed case',
    input: '/Users/Chris/DevTools',
    expect_ok: true,
    output: '/Users/Chris/DevTools',
    expect_error: false,
  },
  // rule_7_no_symlink_resolution (non-existent path)
  {
    label: 'rule7: nonexistent path',
    input: '/definitely/does/not/exist/anywhere/on/this/machine/abc',
    expect_ok: true,
    output: '/definitely/does/not/exist/anywhere/on/this/machine/abc',
    expect_error: false,
  },
  // rule_8_no_fs_access (same pattern)
  {
    label: 'rule8: no fs access',
    input: '/nope/nope/nope',
    expect_ok: true,
    output: '/nope/nope/nope',
    expect_error: false,
  },
  // rule_9_utf8_required positive side
  {
    label: 'rule9: utf8 positive',
    input: '/utf8/ok',
    expect_ok: true,
    output: '/utf8/ok',
    expect_error: false,
  },
  // from_validated_accepts_canonical
  {
    label: 'from_validated: canonical',
    input: '/Users/chris/dev',
    expect_ok: true,
    output: '/Users/chris/dev',
    expect_error: false,
  },
];

const ERROR_VECTORS: PathVector[] = [
  // error_relative_input
  {
    label: 'error: relative path',
    input: 'relative/path',
    expect_ok: false,
    expect_error: true,
    error_kind: 'relative',
  },
  // error_relative_input_dot_prefix
  {
    label: 'error: dot-prefixed relative',
    input: './relative',
    expect_ok: false,
    expect_error: true,
    error_kind: 'relative',
  },
  // error_empty_path
  {
    label: 'error: empty string',
    input: '',
    expect_ok: false,
    expect_error: true,
    error_kind: 'empty',
  },
  // error_embedded_nul — Rust's InvalidNormalization ↔ TS 'nul-byte'
  {
    label: 'error: embedded NUL',
    input: '/Users/chris\0/dev',
    expect_ok: false,
    expect_error: true,
    error_kind: 'nul-byte',
  },
  // rule_4_parent_dir_rejected
  {
    label: 'rule4: dot-dot in middle',
    input: '/Users/chris/../other',
    expect_ok: false,
    expect_error: true,
    error_kind: 'dot-dot',
  },
  // error_parent_dir_at_end
  {
    label: 'error: dot-dot at end',
    input: '/a/b/..',
    expect_ok: false,
    expect_error: true,
    error_kind: 'dot-dot',
  },
  // error_parent_dir_at_start_after_root
  {
    label: 'error: dot-dot after root',
    input: '/..',
    expect_ok: false,
    expect_error: true,
    error_kind: 'dot-dot',
  },
  // from_validated_rejects_relative
  {
    label: 'from_validated: relative rejected',
    input: 'relative',
    expect_ok: false,
    expect_error: true,
    error_kind: 'relative',
  },
  // from_validated_rejects_parent_dir
  {
    label: 'from_validated: dot-dot rejected',
    input: '/a/../b',
    expect_ok: false,
    expect_error: true,
    error_kind: 'dot-dot',
  },
];

// ---------------------------------------------------------------------------
// Parity tests — fromUserInput
// ---------------------------------------------------------------------------

describe('Rust parity — fromUserInput success vectors', () => {
  for (const v of SUCCESS_VECTORS) {
    it(v.label, () => {
      const result = fromUserInput(v.input);
      if (v.output === null) {
        // Tilde-expanded: result must start with '/' and not contain '~'.
        expect(result).toMatch(/^\//);
        expect(result).not.toContain('~');
      } else if (v.output !== undefined) {
        expect(result).toBe(v.output);
      }
    });
  }
});

describe('Rust parity — fromUserInput error vectors', () => {
  for (const v of ERROR_VECTORS) {
    it(v.label, () => {
      let threw = false;
      try {
        fromUserInput(v.input);
      } catch (e) {
        threw = true;
        expect(e).toBeInstanceOf(PathError);
        if (v.error_kind !== undefined) {
          expect((e as PathError).kind).toBe(v.error_kind);
        }
      }
      expect(threw).toBe(true);
    });
  }
});

// ---------------------------------------------------------------------------
// Parity tests — fromValidated (same rules, no silent accept)
// ---------------------------------------------------------------------------

describe('Rust parity — fromValidated mirrors from_validated behavior', () => {
  it('accepts already-canonical path', () => {
    expect(fromValidated('/Users/chris/dev')).toBe('/Users/chris/dev');
  });

  it('rejects relative — same as from_validated_rejects_relative', () => {
    expect(() => fromValidated('relative')).toThrow(PathError);
    let kind: string | undefined;
    try {
      fromValidated('relative');
    } catch (e) {
      kind = (e as PathError).kind;
    }
    expect(kind).toBe('relative');
  });

  it('rejects dot-dot — same as from_validated_rejects_parent_dir', () => {
    expect(() => fromValidated('/a/../b')).toThrow(PathError);
    let kind: string | undefined;
    try {
      fromValidated('/a/../b');
    } catch (e) {
      kind = (e as PathError).kind;
    }
    expect(kind).toBe('dot-dot');
  });
});

// ---------------------------------------------------------------------------
// Additional normalization parity vectors
// (inputs where Rust and TypeScript must both succeed with the same output)
// ---------------------------------------------------------------------------

describe('Rust parity — additional normalization vectors', () => {
  const additionalVectors: Array<{ input: string; output: string }> = [
    { input: '/a/./b/./c', output: '/a/b/c' },
    { input: '/a//b////c', output: '/a/b/c' },
    { input: '/a/b/', output: '/a/b' },
    { input: '/', output: '/' },
    { input: '/.', output: '/' },
    { input: '///', output: '/' },
    { input: '/Users/chris/.config/file.json', output: '/Users/chris/.config/file.json' },
    { input: '/a/b/file.test.ts', output: '/a/b/file.test.ts' },
  ];

  for (const { input, output } of additionalVectors) {
    it(`normalizes ${JSON.stringify(input)} → ${JSON.stringify(output)}`, () => {
      expect(fromUserInput(input)).toBe(output);
    });
  }
});
