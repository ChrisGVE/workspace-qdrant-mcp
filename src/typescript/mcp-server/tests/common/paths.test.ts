/**
 * Unit tests for the canonical path abstraction module.
 *
 * Coverage: all nine normalization rules (spec §3.1), PathError variants,
 * MountMap construction/resolution, toLocal/toCanonical translation,
 * idempotency, and round-trip properties.
 */

import { describe, it, expect } from 'vitest';
import {
  fromUserInput,
  fromValidated,
  toLocal,
  toCanonical,
  asStdPath,
  MountMap,
  PathError,
  type LocalPath,
} from '../../src/common/paths.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Assert that the callback throws a PathError with the given kind. */
function expectPathError(kind: string, fn: () => unknown): void {
  let threw = false;
  try {
    fn();
  } catch (e) {
    threw = true;
    expect(e).toBeInstanceOf(PathError);
    expect((e as PathError).kind).toBe(kind);
  }
  if (!threw) {
    throw new Error(`Expected PathError(${kind}) but no error was thrown`);
  }
}

// ---------------------------------------------------------------------------
// §3.1 Rule 1 — absolute requirement
// ---------------------------------------------------------------------------

describe('fromUserInput — rule 1: absolute', () => {
  it('accepts a simple absolute path', () => {
    expect(fromUserInput('/Users/chris/dev')).toBe('/Users/chris/dev');
  });

  it('rejects a relative path', () => {
    expectPathError('relative', () => fromUserInput('relative/path'));
  });

  it('rejects a path with no leading slash', () => {
    expectPathError('relative', () => fromUserInput('foo'));
  });

  it('rejects an empty string', () => {
    expectPathError('empty', () => fromUserInput(''));
  });
});

// ---------------------------------------------------------------------------
// §3.1 Rule 2 — tilde expansion
// ---------------------------------------------------------------------------

describe('fromUserInput — rule 2: tilde expansion', () => {
  it('expands ~ to homedir', () => {
    const result = fromUserInput('~/projects/foo');
    expect(result).toMatch(/^\/.*\/projects\/foo$/);
    expect(result).not.toContain('~');
  });

  it('expands standalone ~ to homedir', () => {
    const result = fromUserInput('~');
    expect(result).toMatch(/^\//);
    expect(result).not.toContain('~');
  });

  it('does not expand ~ in the middle of a path', () => {
    // '/a/~/b' is absolute — ~ is treated as a literal segment name.
    const result = fromUserInput('/a/~/b');
    expect(result).toBe('/a/~/b');
  });
});

// ---------------------------------------------------------------------------
// §3.1 Rule 3 — remove '.' segments
// ---------------------------------------------------------------------------

describe('fromUserInput — rule 3: remove dot segments', () => {
  it('removes a single dot segment', () => {
    expect(fromUserInput('/a/./b')).toBe('/a/b');
  });

  it('removes multiple dot segments', () => {
    expect(fromUserInput('/a/./b/./c')).toBe('/a/b/c');
  });

  it('removes leading dot segment (after slash)', () => {
    expect(fromUserInput('/./a/b')).toBe('/a/b');
  });

  it('handles path with only dot segments after root', () => {
    // '/.' → '/'
    expect(fromUserInput('/.')).toBe('/');
  });
});

// ---------------------------------------------------------------------------
// §3.1 Rule 4 — reject '..'
// ---------------------------------------------------------------------------

describe('fromUserInput — rule 4: reject dot-dot', () => {
  it('rejects path with .. segment', () => {
    expectPathError('dot-dot', () => fromUserInput('/a/b/../c'));
  });

  it('rejects path with leading ..', () => {
    expectPathError('dot-dot', () => fromUserInput('/../../etc/passwd'));
  });

  it('rejects .. at path end', () => {
    expectPathError('dot-dot', () => fromUserInput('/a/b/..'));
  });
});

// ---------------------------------------------------------------------------
// §3.1 Rule 5 — collapse duplicate slashes
// ---------------------------------------------------------------------------

describe('fromUserInput — rule 5: collapse duplicate slashes', () => {
  it('collapses double slash', () => {
    expect(fromUserInput('/a//b')).toBe('/a/b');
  });

  it('collapses multiple consecutive slashes', () => {
    expect(fromUserInput('/a///b////c')).toBe('/a/b/c');
  });

  it('handles trailing slash', () => {
    expect(fromUserInput('/a/b/')).toBe('/a/b');
  });
});

// ---------------------------------------------------------------------------
// §3.1 Rule 6 — case preserved
// ---------------------------------------------------------------------------

describe('fromUserInput — rule 6: case preserved', () => {
  it('preserves uppercase letters', () => {
    expect(fromUserInput('/Users/Chris/Dev')).toBe('/Users/Chris/Dev');
  });

  it('preserves mixed case', () => {
    expect(fromUserInput('/MyProject/SRC/Main.ts')).toBe('/MyProject/SRC/Main.ts');
  });
});

// ---------------------------------------------------------------------------
// §3.1 Rule 7 — no symlink resolution / §3.1 Rule 8 — no fs access
// ---------------------------------------------------------------------------

describe('fromUserInput — rules 7 & 8: no symlink resolution, no fs access', () => {
  it('does not resolve symlinks (pure string operation)', () => {
    // /tmp on macOS is a symlink to /private/tmp — we must NOT follow it.
    const result = fromUserInput('/tmp/foo');
    expect(result).toBe('/tmp/foo');
  });

  it('accepts paths that do not exist on disk', () => {
    const result = fromUserInput('/nonexistent/path/that/cannot/exist');
    expect(result).toBe('/nonexistent/path/that/cannot/exist');
  });
});

// ---------------------------------------------------------------------------
// §3.1 Rule 9 — UTF-8 validity (lone surrogates)
// ---------------------------------------------------------------------------

describe('fromUserInput — rule 9: UTF-8 validity', () => {
  it('accepts valid ASCII path', () => {
    expect(fromUserInput('/valid/ascii/path')).toBe('/valid/ascii/path');
  });

  it('accepts valid Unicode path', () => {
    expect(fromUserInput('/Users/Ünîcödé/path')).toBe('/Users/Ünîcödé/path');
  });

  it('rejects path with lone surrogate', () => {
    // Construct a string containing a lone surrogate (high surrogate without
    // a following low surrogate). This is valid in JS (UTF-16) but not UTF-8.
    const withLoneSurrogate = '/a/' + String.fromCharCode(0xd800) + '/b';
    expectPathError('non-utf8', () => fromUserInput(withLoneSurrogate));
  });

  it('rejects NUL byte in path', () => {
    expectPathError('nul-byte', () => fromUserInput('/a/\0/b'));
  });
});

// ---------------------------------------------------------------------------
// fromValidated — same rules, no silent accept
// ---------------------------------------------------------------------------

describe('fromValidated', () => {
  it('accepts a canonical path', () => {
    expect(fromValidated('/Users/chris/dev')).toBe('/Users/chris/dev');
  });

  it('rejects relative input — never silently accepts', () => {
    expectPathError('relative', () => fromValidated('relative/path'));
  });

  it('rejects dot-dot', () => {
    expectPathError('dot-dot', () => fromValidated('/a/../b'));
  });

  it('normalizes dot segments (same behavior as fromUserInput)', () => {
    expect(fromValidated('/a/./b')).toBe('/a/b');
  });

  it('expands tilde same as fromUserInput', () => {
    const result = fromValidated('~/foo');
    expect(result).not.toContain('~');
  });
});

// ---------------------------------------------------------------------------
// Idempotency: fromUserInput(fromUserInput(x)) === fromUserInput(x)
// ---------------------------------------------------------------------------

describe('idempotency', () => {
  const paths = ['/Users/chris/dev', '/a//b///c', '/a/./b/./c', '/Volumes/External/books'];

  for (const p of paths) {
    it(`idempotent on: ${p}`, () => {
      const once = fromUserInput(p);
      const twice = fromUserInput(once);
      expect(twice).toBe(once);
    });
  }
});

// ---------------------------------------------------------------------------
// MountMap — construction
// ---------------------------------------------------------------------------

describe('MountMap construction', () => {
  it('creates an empty (identity) map', () => {
    const m = new MountMap([]);
    expect(m.isIdentity).toBe(true);
    expect(m.size).toBe(0);
  });

  it('creates a map with a single entry', () => {
    const m = new MountMap([{ host: '/Users/chris/dev', container: '/Users/chris/dev' }]);
    expect(m.isIdentity).toBe(false);
    expect(m.size).toBe(1);
  });

  it('rejects duplicate host prefix', () => {
    expect(
      () =>
        new MountMap([
          { host: '/Users/chris', container: '/mnt/a' },
          { host: '/Users/chris', container: '/mnt/b' },
        ])
    ).toThrow(PathError);
  });

  it('rejects duplicate container prefix', () => {
    expect(
      () =>
        new MountMap([
          { host: '/a', container: '/mnt/shared' },
          { host: '/b', container: '/mnt/shared' },
        ])
    ).toThrow(PathError);
  });

  it('allows overlapping mounts (longest-prefix-wins)', () => {
    const m = new MountMap([
      { host: '/Users/chris', container: '/mnt/home' },
      { host: '/Users/chris/dev', container: '/mnt/dev' },
    ]);
    expect(m.size).toBe(2);
  });

  it('validates and normalizes host/container entries', () => {
    // '~/dev' in host should be expanded and not throw.
    expect(() => new MountMap([{ host: '~/dev', container: '/mnt/dev' }])).not.toThrow();
  });

  it('rejects invalid host path (relative)', () => {
    expect(() => new MountMap([{ host: 'relative/path', container: '/mnt/a' }])).toThrow(PathError);
  });
});

// ---------------------------------------------------------------------------
// MountMap — longest-prefix-wins resolution
// ---------------------------------------------------------------------------

describe('MountMap resolution — longest-prefix-wins', () => {
  const m = new MountMap([
    { host: '/Users/chris', container: '/mnt/home' },
    { host: '/Users/chris/dev', container: '/mnt/dev' },
  ]);

  it('matches the longer prefix when both apply', () => {
    const entry = m.findForCanonical(fromUserInput('/Users/chris/dev/project'));
    expect(entry?.host).toBe('/Users/chris/dev');
    expect(entry?.container).toBe('/mnt/dev');
  });

  it('falls back to shorter prefix when longer does not apply', () => {
    const entry = m.findForCanonical(fromUserInput('/Users/chris/documents'));
    expect(entry?.host).toBe('/Users/chris');
  });

  it('returns undefined when no entry covers the path', () => {
    const entry = m.findForCanonical(fromUserInput('/Volumes/external'));
    expect(entry).toBeUndefined();
  });

  it('is component-aware: /a/b does not match /a/bc', () => {
    const m2 = new MountMap([{ host: '/a/b', container: '/mnt/b' }]);
    const entry = m2.findForCanonical(fromUserInput('/a/bc'));
    expect(entry).toBeUndefined();
  });

  it('matches exact path (host == path)', () => {
    const entry = m.findForCanonical(fromUserInput('/Users/chris'));
    expect(entry?.host).toBe('/Users/chris');
  });
});

// ---------------------------------------------------------------------------
// toLocal — identity map
// ---------------------------------------------------------------------------

describe('toLocal — identity map', () => {
  const identityMap = new MountMap([]);

  it('returns canonical as-is', () => {
    const c = fromUserInput('/Users/chris/dev');
    const l = toLocal(c, identityMap);
    expect(l).toBe('/Users/chris/dev');
  });
});

// ---------------------------------------------------------------------------
// toLocal — non-identity map
// ---------------------------------------------------------------------------

describe('toLocal — non-identity map', () => {
  const m = new MountMap([{ host: '/Volumes/External/books', container: '/mnt/books' }]);

  it('swaps host prefix for container prefix', () => {
    const c = fromUserInput('/Volumes/External/books/chapter1.pdf');
    const l = toLocal(c, m);
    expect(l).toBe('/mnt/books/chapter1.pdf');
  });

  it('handles exact match (path == host prefix)', () => {
    const c = fromUserInput('/Volumes/External/books');
    const l = toLocal(c, m);
    expect(l).toBe('/mnt/books');
  });

  it('throws no-mount-coverage when path is not covered', () => {
    const c = fromUserInput('/Users/chris/other');
    expectPathError('no-mount-coverage', () => toLocal(c, m));
  });
});

// ---------------------------------------------------------------------------
// toCanonical — identity map
// ---------------------------------------------------------------------------

describe('toCanonical — identity map', () => {
  const identityMap = new MountMap([]);

  it('re-canonicalizes the local string', () => {
    // Simulate a local path that came from a source already in canonical form.
    const local = '/Users/chris/dev' as LocalPath;
    const c = toCanonical(local, identityMap);
    expect(c).toBe('/Users/chris/dev');
  });
});

// ---------------------------------------------------------------------------
// toCanonical — non-identity map
// ---------------------------------------------------------------------------

describe('toCanonical — non-identity map', () => {
  const m = new MountMap([{ host: '/Volumes/External/books', container: '/mnt/books' }]);

  it('swaps container prefix for host prefix', () => {
    const local = '/mnt/books/chapter1.pdf' as LocalPath;
    const c = toCanonical(local, m);
    expect(c).toBe('/Volumes/External/books/chapter1.pdf');
  });

  it('throws no-mount-coverage when path is not covered', () => {
    const local = '/other/path/file.txt' as LocalPath;
    expectPathError('no-mount-coverage', () => toCanonical(local, m));
  });
});

// ---------------------------------------------------------------------------
// Round-trip: toCanonical(toLocal(c, m), m) === c
// ---------------------------------------------------------------------------

describe('round-trip property', () => {
  const cases: Array<{ label: string; mounts: MountMap; canonical: string }> = [
    {
      label: 'identity map',
      mounts: new MountMap([]),
      canonical: '/Users/chris/dev/project',
    },
    {
      label: 'mirror mount',
      mounts: new MountMap([{ host: '/Users/chris/dev', container: '/Users/chris/dev' }]),
      canonical: '/Users/chris/dev/project/src/main.ts',
    },
    {
      label: 'non-mirror mount',
      mounts: new MountMap([{ host: '/Volumes/books', container: '/mnt/books' }]),
      canonical: '/Volumes/books/chapter1.pdf',
    },
    {
      label: 'exact match on mount prefix',
      mounts: new MountMap([{ host: '/Volumes/books', container: '/mnt/books' }]),
      canonical: '/Volumes/books',
    },
  ];

  for (const { label, mounts, canonical } of cases) {
    it(`round-trip: ${label}`, () => {
      const c = fromUserInput(canonical);
      const l = toLocal(c, mounts);
      const c2 = toCanonical(l, mounts);
      expect(c2).toBe(c);
    });
  }
});

// ---------------------------------------------------------------------------
// asStdPath
// ---------------------------------------------------------------------------

describe('asStdPath', () => {
  it('returns the underlying string', () => {
    const m = new MountMap([]);
    const c = fromUserInput('/Users/chris/dev');
    const l = toLocal(c, m);
    expect(asStdPath(l)).toBe('/Users/chris/dev');
  });
});

// ---------------------------------------------------------------------------
// PathError structure
// ---------------------------------------------------------------------------

describe('PathError', () => {
  it('has the expected name', () => {
    const e = new PathError('empty', 'path is empty');
    expect(e.name).toBe('PathError');
  });

  it('is instanceof Error', () => {
    const e = new PathError('relative', 'relative path');
    expect(e).toBeInstanceOf(Error);
  });

  it('carries the kind discriminant', () => {
    const e = new PathError('dot-dot', 'contains ..');
    expect(e.kind).toBe('dot-dot');
  });
});

// ---------------------------------------------------------------------------
// Edge cases / additional negative tests
// ---------------------------------------------------------------------------

describe('edge cases', () => {
  it('accepts root path /', () => {
    expect(fromUserInput('/')).toBe('/');
  });

  it('accepts path with deeply nested segments', () => {
    expect(fromUserInput('/a/b/c/d/e/f')).toBe('/a/b/c/d/e/f');
  });

  it('accepts path with dots in filenames (not dot segments)', () => {
    expect(fromUserInput('/Users/chris/.config/file.json')).toBe('/Users/chris/.config/file.json');
  });

  it('accepts path with multiple dots in filename', () => {
    expect(fromUserInput('/a/b/file.test.ts')).toBe('/a/b/file.test.ts');
  });

  it('rejects path with embedded NUL', () => {
    expectPathError('nul-byte', () => fromUserInput('/a\0b'));
  });

  it('handles path that is only slashes', () => {
    // '///' → after normalization '/'
    expect(fromUserInput('///')).toBe('/');
  });
});
