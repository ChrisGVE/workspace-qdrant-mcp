/**
 * Tests for list-files tool — tree building and tree rendering
 */

import { describe, it, expect } from 'vitest';

import type { TrackedFileEntry, SubmoduleEntry } from '../../src/clients/tracked-files-queries.js';
import {
  buildTree,
  renderTree,
} from '../../src/tools/list-files/index.js';

// ── Test helpers ─────────────────────────────────────────────────────────

function makeFile(
  relativePath: string,
  opts: Partial<TrackedFileEntry> = {},
): TrackedFileEntry {
  const ext = relativePath.split('.').pop() ?? null;
  return {
    relativePath,
    fileType: opts.fileType ?? 'code',
    language: opts.language ?? null,
    extension: opts.extension ?? ext,
    isTest: opts.isTest ?? false,
  };
}

function makeSubmodule(submodulePath: string, repoName: string): SubmoduleEntry {
  return { submodulePath, repoName };
}

// ── buildTree ────────────────────────────────────────────────────────────

describe('buildTree', () => {
  it('should build a basic tree from flat paths', () => {
    const files = [
      makeFile('src/main.rs'),
      makeFile('src/lib.rs'),
      makeFile('README.md'),
    ];

    const root = buildTree(files, [], '');

    expect(root.children.has('src')).toBe(true);
    expect(root.files).toHaveLength(1);
    expect(root.files[0].name).toBe('README.md');

    const src = root.children.get('src')!;
    expect(src.files).toHaveLength(2);
    expect(src.totalFiles).toBe(2);
  });

  it('should handle nested paths', () => {
    const files = [
      makeFile('src/tools/search.ts'),
      makeFile('src/tools/grep.ts'),
      makeFile('src/clients/daemon.ts'),
    ];

    const root = buildTree(files, [], '');
    const src = root.children.get('src')!;
    expect(src.children.has('tools')).toBe(true);
    expect(src.children.has('clients')).toBe(true);
    expect(src.children.get('tools')!.files).toHaveLength(2);
  });

  it('should strip basePath prefix', () => {
    const files = [
      makeFile('src/typescript/mcp-server/server.ts'),
      makeFile('src/typescript/mcp-server/index.ts'),
    ];

    const root = buildTree(files, [], 'src/typescript/mcp-server');
    // Files should be direct children, not nested under src/typescript/mcp-server
    expect(root.files).toHaveLength(2);
    expect(root.children.size).toBe(0);
  });

  it('should mark submodule folders and stop expansion', () => {
    const files = [
      makeFile('src/main.rs'),
      makeFile('vendor/lib-a/src/lib.rs'),
      makeFile('vendor/lib-a/Cargo.toml'),
    ];
    const submodules = [makeSubmodule('vendor/lib-a', 'lib-a')];

    const root = buildTree(files, submodules, '');

    const vendor = root.children.get('vendor')!;
    const libA = vendor.children.get('lib-a')!;
    expect(libA.submodule).toEqual({ repoName: 'lib-a' });
    // Files inside the submodule should NOT be added
    expect(libA.files).toHaveLength(0);
    expect(libA.children.size).toBe(0);
  });

  it('should compute totalFiles correctly', () => {
    const files = [
      makeFile('a/b/c/file1.rs'),
      makeFile('a/b/c/file2.rs'),
      makeFile('a/b/file3.rs'),
      makeFile('a/file4.rs'),
      makeFile('root.rs'),
    ];

    const root = buildTree(files, [], '');
    expect(root.totalFiles).toBe(5);

    const a = root.children.get('a')!;
    expect(a.totalFiles).toBe(4); // file4 + b/file3 + b/c/file1 + b/c/file2

    const b = a.children.get('b')!;
    expect(b.totalFiles).toBe(3); // file3 + c/file1 + c/file2
  });

  it('should handle empty file list', () => {
    const root = buildTree([], [], '');
    expect(root.files).toHaveLength(0);
    expect(root.children.size).toBe(0);
    expect(root.totalFiles).toBe(0);
  });
});

// ── renderTree ───────────────────────────────────────────────────────────

describe('renderTree', () => {
  it('should render indented tree with extensions', () => {
    const files = [
      makeFile('src/main.rs'),
      makeFile('src/lib.rs'),
      makeFile('README.md'),
    ];
    const root = buildTree(files, [], '');
    const { text } = renderTree(root, 5, 100);

    expect(text).toContain('src/');
    expect(text).toContain('  main.rs [rs]');
    expect(text).toContain('  lib.rs [rs]');
    expect(text).toContain('README.md [md]');
  });

  it('should show submodule markers', () => {
    const files = [makeFile('src/main.rs')];
    const submodules = [makeSubmodule('vendor/lib-x', 'lib-x')];
    // Add a file in submodule path to trigger folder creation
    files.push(makeFile('vendor/lib-x/ignored.rs'));
    const root = buildTree(files, submodules, '');
    const { text } = renderTree(root, 5, 100);

    expect(text).toContain('lib-x/ [submodule: lib-x]');
  });

  it('should collapse folders beyond max depth', () => {
    const files = [
      makeFile('a/b/c/d/deep.rs'),
      makeFile('a/b/c/d/deeper.rs'),
    ];
    const root = buildTree(files, [], '');
    const { text } = renderTree(root, 2, 100);

    // At depth 2: a/ is depth 1, b/ is depth 2, so b/ gets collapsed
    expect(text).toContain('b/ (2 files)');
    expect(text).not.toContain('deep.rs');
  });

  it('should respect limit', () => {
    const files = Array.from({ length: 20 }, (_, i) => makeFile(`file${i}.ts`));
    const root = buildTree(files, [], '');
    const { text, count } = renderTree(root, 5, 5);

    expect(count).toBe(5);
    const lines = text.split('\n').filter(Boolean);
    expect(lines).toHaveLength(5);
  });

  it('should sort folders and files alphabetically', () => {
    const files = [
      makeFile('zebra.ts'),
      makeFile('alpha.ts'),
      makeFile('b/beta.ts'),
      makeFile('a/alpha.ts'),
    ];
    const root = buildTree(files, [], '');
    const { text } = renderTree(root, 5, 100);
    const lines = text.split('\n');

    // Folders come first, then files
    expect(lines[0]).toContain('a/');
    expect(lines[2]).toContain('b/');
  });
});
