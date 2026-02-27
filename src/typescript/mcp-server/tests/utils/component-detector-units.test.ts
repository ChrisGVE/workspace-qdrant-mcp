/**
 * Unit tests for component-detector: pathToComponentId, parseCargoMembers,
 * fileMatchesComponent, componentMatchesFilter, assignComponent
 */

import { describe, it, expect } from 'vitest';

import {
  parseCargoMembers,
  pathToComponentId,
  fileMatchesComponent,
  componentMatchesFilter,
  assignComponent,
  type ComponentInfo,
  type ComponentMap,
} from '../../src/utils/component-detector/index.js';

// ── pathToComponentId ────────────────────────────────────────────────────

describe('pathToComponentId', () => {
  it('should convert slashes to dots', () => {
    expect(pathToComponentId('daemon/core')).toBe('daemon.core');
  });

  it('should leave single segment unchanged', () => {
    expect(pathToComponentId('cli')).toBe('cli');
  });

  it('should handle deeply nested paths', () => {
    expect(pathToComponentId('src/typescript/mcp-server')).toBe('src.typescript.mcp-server');
  });

  it('should trim trailing slashes', () => {
    expect(pathToComponentId('daemon/core/')).toBe('daemon.core');
  });

  it('should trim leading slashes', () => {
    expect(pathToComponentId('/daemon/core')).toBe('daemon.core');
  });
});

// ── parseCargoMembers ────────────────────────────────────────────────────

describe('parseCargoMembers', () => {
  it('should parse multi-line members array', () => {
    const content = `
[workspace]
resolver = "2"
members = [
    "common",
    "cli",
    "daemon/core",
    "daemon/grpc",
]
`;
    const members = parseCargoMembers(content);
    expect(members).toEqual(['common', 'cli', 'daemon/core', 'daemon/grpc']);
  });

  it('should parse inline members array', () => {
    const content = `[workspace]\nmembers = ["a", "b/c"]`;
    expect(parseCargoMembers(content)).toEqual(['a', 'b/c']);
  });

  it('should return empty for non-workspace Cargo.toml', () => {
    const content = `[package]\nname = "my-crate"\nversion = "0.1.0"`;
    expect(parseCargoMembers(content)).toEqual([]);
  });

  it('should handle comments in members array', () => {
    const content = `
[workspace]
members = [
    "a",  # first member
    # "commented-out",
    "b",
]
`;
    expect(parseCargoMembers(content)).toEqual(['a', 'b']);
  });

  it('should handle single-quoted strings', () => {
    const content = `[workspace]\nmembers = ['a', 'b/c']`;
    expect(parseCargoMembers(content)).toEqual(['a', 'b/c']);
  });
});

// ── fileMatchesComponent ─────────────────────────────────────────────────

describe('fileMatchesComponent', () => {
  const component: ComponentInfo = {
    id: 'daemon.core',
    basePath: 'daemon/core',
    patterns: ['daemon/core/**'],
    source: 'cargo',
  };

  it('should match files inside component', () => {
    expect(fileMatchesComponent('daemon/core/src/main.rs', component)).toBe(true);
  });

  it('should match deeply nested files', () => {
    expect(fileMatchesComponent('daemon/core/src/tools/search.rs', component)).toBe(true);
  });

  it('should not match files in other components', () => {
    expect(fileMatchesComponent('daemon/grpc/src/lib.rs', component)).toBe(false);
  });

  it('should not match partial prefix', () => {
    expect(fileMatchesComponent('daemon/core_extra/file.rs', component)).toBe(false);
  });

  it('should match exact basePath', () => {
    expect(fileMatchesComponent('daemon/core', component)).toBe(true);
  });
});

// ── componentMatchesFilter ───────────────────────────────────────────────

describe('componentMatchesFilter', () => {
  it('should match exact component', () => {
    expect(componentMatchesFilter('daemon.core', 'daemon.core')).toBe(true);
  });

  it('should match prefix filter', () => {
    expect(componentMatchesFilter('daemon.core', 'daemon')).toBe(true);
  });

  it('should match deeper prefix', () => {
    expect(componentMatchesFilter('daemon.core.utils', 'daemon')).toBe(true);
    expect(componentMatchesFilter('daemon.core.utils', 'daemon.core')).toBe(true);
  });

  it('should not match unrelated component', () => {
    expect(componentMatchesFilter('cli', 'daemon')).toBe(false);
  });

  it('should not match partial name prefix', () => {
    // "daemon-extra" should NOT match filter "daemon"
    expect(componentMatchesFilter('daemon-extra', 'daemon')).toBe(false);
  });
});

// ── assignComponent ──────────────────────────────────────────────────────

describe('assignComponent', () => {
  const components: ComponentMap = new Map([
    ['daemon.core', { id: 'daemon.core', basePath: 'daemon/core', patterns: ['daemon/core/**'], source: 'cargo' }],
    ['daemon.grpc', { id: 'daemon.grpc', basePath: 'daemon/grpc', patterns: ['daemon/grpc/**'], source: 'cargo' }],
    ['cli', { id: 'cli', basePath: 'cli', patterns: ['cli/**'], source: 'cargo' }],
  ]);

  it('should assign file to correct component', () => {
    const result = assignComponent('daemon/core/src/main.rs', components);
    expect(result?.id).toBe('daemon.core');
  });

  it('should assign to most specific match', () => {
    // Add a broader "daemon" component
    const extended = new Map(components);
    extended.set('daemon', { id: 'daemon', basePath: 'daemon', patterns: ['daemon/**'], source: 'cargo' });

    const result = assignComponent('daemon/core/src/main.rs', extended);
    // Should match daemon.core (more specific) not daemon
    expect(result?.id).toBe('daemon.core');
  });

  it('should return undefined for unmatched files', () => {
    expect(assignComponent('README.md', components)).toBeUndefined();
  });
});
