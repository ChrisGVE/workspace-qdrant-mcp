/**
 * Integration tests for component-detector: detectComponents filesystem detection
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { mkdtempSync, rmSync, mkdirSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import {
  detectComponents,
} from '../../src/utils/component-detector/index.js';

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
