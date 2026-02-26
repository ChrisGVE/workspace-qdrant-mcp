/**
 * Tests for list-files tool — tree building, rendering, and formatting
 */

import { describe, it, expect } from 'vitest';

import type { TrackedFileEntry, SubmoduleEntry } from '../../src/clients/tracked-files-queries.js';
import type { FolderNode } from '../../src/tools/list-files-types.js';
import {
  buildTree,
  renderTree,
  renderSummary,
  renderFlat,
  globToRegex,
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

// ── renderSummary ────────────────────────────────────────────────────────

describe('renderSummary', () => {
  it('should show folders with file counts', () => {
    const files = [
      makeFile('src/main.rs', { extension: 'rs' }),
      makeFile('src/lib.rs', { extension: 'rs' }),
      makeFile('src/server.ts', { extension: 'ts' }),
    ];
    const root = buildTree(files, [], '');
    const { text } = renderSummary(root, 5, 100);

    expect(text).toContain('src/');
    expect(text).toContain('3 files');
    expect(text).toContain('2 rs');
    expect(text).toContain('1 ts');
  });

  it('should collapse single-child folder chains', () => {
    const files = [
      makeFile('src/typescript/mcp-server/server.ts'),
      makeFile('src/typescript/mcp-server/index.ts'),
    ];
    const root = buildTree(files, [], '');
    const { text } = renderSummary(root, 5, 100);

    // Should show collapsed chain, not three separate levels
    expect(text).toContain('src/typescript/mcp-server/');
  });

  it('should NOT collapse chains when there are files at intermediate levels', () => {
    const files = [
      makeFile('src/root-file.ts'),
      makeFile('src/deep/child.ts'),
    ];
    const root = buildTree(files, [], '');
    const { text } = renderSummary(root, 5, 100);

    // src/ has both a file and a child, so it should NOT be collapsed
    expect(text).toMatch(/^src\//m);
  });

  it('should show submodule markers in summary', () => {
    const files = [makeFile('vendor/lib-a/src/lib.rs')];
    const submodules = [makeSubmodule('vendor/lib-a', 'lib-a')];
    const root = buildTree(files, submodules, '');
    const { text } = renderSummary(root, 5, 100);

    expect(text).toContain('[submodule: lib-a]');
  });

  it('should show empty for folders with no files', () => {
    const root: FolderNode = {
      name: '.',
      children: new Map([
        ['empty', { name: 'empty', children: new Map(), files: [], totalFiles: 0 }],
      ]),
      files: [],
      totalFiles: 0,
    };
    const { text } = renderSummary(root, 5, 100);

    expect(text).toContain('(empty)');
  });

  it('should limit extension types shown', () => {
    const files = [
      makeFile('a/f1.rs', { extension: 'rs' }),
      makeFile('a/f2.ts', { extension: 'ts' }),
      makeFile('a/f3.py', { extension: 'py' }),
      makeFile('a/f4.go', { extension: 'go' }),
      makeFile('a/f5.lua', { extension: 'lua' }),
      makeFile('a/f6.rb', { extension: 'rb' }),
    ];
    const root = buildTree(files, [], '');
    const { text } = renderSummary(root, 5, 100);

    // Should show top 4 + "other" count
    expect(text).toContain('6 files');
    expect(text).toContain('other');
  });
});

// ── renderFlat ───────────────────────────────────────────────────────────

describe('renderFlat', () => {
  it('should list files one per line', () => {
    const files = [
      makeFile('src/main.rs'),
      makeFile('src/lib.rs'),
      makeFile('README.md'),
    ];
    const { text, count } = renderFlat(files, 100);

    expect(count).toBe(3);
    const lines = text.split('\n');
    expect(lines).toHaveLength(3);
    expect(lines[0]).toBe('src/main.rs');
  });

  it('should respect limit', () => {
    const files = Array.from({ length: 10 }, (_, i) => makeFile(`f${i}.ts`));
    const { count } = renderFlat(files, 3);
    expect(count).toBe(3);
  });
});

// ── globToRegex ──────────────────────────────────────────────────────────

describe('globToRegex', () => {
  it('should match exact filename', () => {
    const re = globToRegex('README.md');
    expect(re.test('README.md')).toBe(true);
    expect(re.test('src/README.md')).toBe(false);
  });

  it('should match * as non-slash wildcard', () => {
    const re = globToRegex('*.ts');
    expect(re.test('file.ts')).toBe(true);
    expect(re.test('src/file.ts')).toBe(false);
  });

  it('should match ** as any-depth wildcard', () => {
    const re = globToRegex('**/*.ts');
    expect(re.test('file.ts')).toBe(true);
    expect(re.test('src/file.ts')).toBe(true);
    expect(re.test('a/b/c/file.ts')).toBe(true);
    expect(re.test('a/b/c/file.rs')).toBe(false);
  });

  it('should match path prefix with **', () => {
    const re = globToRegex('src/**/*.test.ts');
    expect(re.test('src/tools/search.test.ts')).toBe(true);
    expect(re.test('src/search.test.ts')).toBe(true);
    expect(re.test('tests/search.test.ts')).toBe(false);
  });

  it('should escape regex special chars', () => {
    const re = globToRegex('file.test.ts');
    expect(re.test('file.test.ts')).toBe(true);
    expect(re.test('filextest.ts')).toBe(false); // . should not match arbitrary char
  });

  it('should match ? as single non-slash char', () => {
    const re = globToRegex('file?.ts');
    expect(re.test('file1.ts')).toBe(true);
    expect(re.test('fileAB.ts')).toBe(false);
  });
});
