/**
 * Tests for component auto-detection from workspace files.
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { mkdtempSync, rmSync, mkdirSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import {
  detectComponents,
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

// ── detectComponents (integration) ───────────────────────────────────────

describe('detectComponents', () => {
  let tempDir: string;

  beforeEach(() => {
    tempDir = mkdtempSync(join(tmpdir(), 'component-detect-'));
  });

  afterEach(() => {
    rmSync(tempDir, { recursive: true, force: true });
  });

  it('should detect Cargo workspace members', () => {
    writeFileSync(
      join(tempDir, 'Cargo.toml'),
      `[workspace]\nresolver = "2"\nmembers = [\n    "common",\n    "cli",\n    "daemon/core",\n    "daemon/grpc",\n]\n`,
    );

    const components = detectComponents(tempDir);

    expect(components.size).toBe(4);
    expect(components.get('common')?.basePath).toBe('common');
    expect(components.get('cli')?.basePath).toBe('cli');
    expect(components.get('daemon.core')?.basePath).toBe('daemon/core');
    expect(components.get('daemon.grpc')?.basePath).toBe('daemon/grpc');
    expect(components.get('daemon.core')?.source).toBe('cargo');
  });

  it('should detect Cargo.toml in src/rust/ subdirectory', () => {
    mkdirSync(join(tempDir, 'src', 'rust'), { recursive: true });
    writeFileSync(
      join(tempDir, 'src', 'rust', 'Cargo.toml'),
      `[workspace]\nmembers = ["core", "cli"]\n`,
    );

    const components = detectComponents(tempDir);

    expect(components.size).toBe(2);
    expect(components.get('core')?.basePath).toBe('src/rust/core');
    expect(components.get('cli')?.basePath).toBe('src/rust/cli');
  });

  it('should detect npm workspaces (array format)', () => {
    writeFileSync(
      join(tempDir, 'package.json'),
      JSON.stringify({ workspaces: ['packages/ui', 'packages/api'] }),
    );
    // Create the actual directories so they're detected
    mkdirSync(join(tempDir, 'packages', 'ui'), { recursive: true });
    mkdirSync(join(tempDir, 'packages', 'api'), { recursive: true });

    const components = detectComponents(tempDir);

    expect(components.size).toBe(2);
    expect(components.get('packages.ui')?.source).toBe('npm');
    expect(components.get('packages.api')?.source).toBe('npm');
  });

  it('should detect npm workspaces with glob pattern', () => {
    mkdirSync(join(tempDir, 'packages', 'lib-a'), { recursive: true });
    mkdirSync(join(tempDir, 'packages', 'lib-b'), { recursive: true });
    writeFileSync(
      join(tempDir, 'package.json'),
      JSON.stringify({ workspaces: ['packages/*'] }),
    );

    const components = detectComponents(tempDir);

    expect(components.size).toBe(2);
    expect(components.has('packages.lib-a')).toBe(true);
    expect(components.has('packages.lib-b')).toBe(true);
  });

  it('should prefer Cargo over npm for same component ID', () => {
    writeFileSync(
      join(tempDir, 'Cargo.toml'),
      `[workspace]\nmembers = ["common"]\n`,
    );
    writeFileSync(
      join(tempDir, 'package.json'),
      JSON.stringify({ workspaces: ['common'] }),
    );

    const components = detectComponents(tempDir);

    expect(components.get('common')?.source).toBe('cargo');
  });

  it('should fall back to directories when no workspace files', () => {
    mkdirSync(join(tempDir, 'src'));
    mkdirSync(join(tempDir, 'tests'));
    mkdirSync(join(tempDir, 'docs'));
    writeFileSync(join(tempDir, 'README.md'), '# Test');

    const components = detectComponents(tempDir);

    expect(components.size).toBe(3);
    expect(components.has('src')).toBe(true);
    expect(components.has('tests')).toBe(true);
    expect(components.has('docs')).toBe(true);
  });

  it('should skip ignored directories in fallback', () => {
    mkdirSync(join(tempDir, 'src'));
    mkdirSync(join(tempDir, 'node_modules'));
    mkdirSync(join(tempDir, '.git'));
    mkdirSync(join(tempDir, 'target'));

    const components = detectComponents(tempDir);

    expect(components.size).toBe(1);
    expect(components.has('src')).toBe(true);
    expect(components.has('node_modules')).toBe(false);
    expect(components.has('.git')).toBe(false);
  });

  it('should return empty map for empty directory', () => {
    const components = detectComponents(tempDir);
    expect(components.size).toBe(0);
  });

  it('should handle non-existent project path gracefully', () => {
    const components = detectComponents(join(tempDir, 'nonexistent'));
    expect(components.size).toBe(0);
  });
});
