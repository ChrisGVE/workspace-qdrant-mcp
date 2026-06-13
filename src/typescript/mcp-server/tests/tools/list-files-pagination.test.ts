/**
 * Tests for ListFilesTool pagination, filter pushdown, and accurate total_matches (F-016).
 *
 * Verifies that:
 * - glob filter is pushed into SQLite, not applied post-fetch
 * - component filter is pushed into SQLite
 * - total_matches reflects the full filtered count, not capped at 500
 * - cursor-based pagination via next_token works correctly
 * - 700-file project with glob filter returns accurate total and can be paged through
 */

import { describe, it, expect, vi } from 'vitest';
import { ListFilesTool } from '../../src/tools/list-files/index.js';
import type { SqliteStateManager } from '../../src/clients/sqlite-state-manager.js';
import type { ProjectDetector } from '../../src/utils/project-detector.js';
import type { ListTrackedFilesOptions } from '../../src/clients/tracked-files-queries/index.js';

vi.mock('../../src/utils/git-utils.js', () => ({
  getCurrentBranch: vi.fn().mockReturnValue('main'),
}));

// ── Helpers ───────────────────────────────────────────────────────────────

function makeFile(path: string, ext = 'rs', lang = 'rust') {
  return {
    relativePath: path,
    fileType: 'code',
    language: lang,
    extension: ext,
    isTest: false,
  };
}

/** Generate N files with paths like "src/file_0001.rs" */
function generateFiles(
  count: number,
  prefix = 'src/file_',
  ext = 'rs'
): ReturnType<typeof makeFile>[] {
  return Array.from({ length: count }, (_, i) =>
    makeFile(`${prefix}${String(i + 1).padStart(4, '0')}.${ext}`, ext)
  );
}

function makeStateManager(
  opts: {
    files?: ReturnType<typeof makeFile>[];
    total?: number;
    listTrackedFiles?: (
      o: ListTrackedFilesOptions
    ) => { status: 'ok'; data: ReturnType<typeof makeFile>[] } | { status: 'degraded' };
    countTrackedFiles?: (o: Omit<ListTrackedFilesOptions, 'limit'>) => number;
  } = {}
): SqliteStateManager {
  const allFiles = opts.files ?? [];
  const total = opts.total ?? allFiles.length;

  const listTrackedFiles =
    opts.listTrackedFiles ??
    ((o: ListTrackedFilesOptions) => {
      const limit = o.limit ?? 500;
      const afterPath = o.afterPath;
      let data = allFiles;
      if (afterPath) {
        const idx = data.findIndex((f) => f.relativePath > afterPath);
        data = idx === -1 ? [] : data.slice(idx);
      }
      return { status: 'ok' as const, data: data.slice(0, limit) };
    });

  const countTrackedFiles = opts.countTrackedFiles ?? (() => total);

  return {
    getWatchFolderIdByTenantId: vi.fn().mockReturnValue('watch-1'),
    getProjectById: vi.fn().mockReturnValue({ data: { project_path: '/proj' } }),
    listTrackedFiles: vi.fn().mockImplementation(listTrackedFiles),
    countTrackedFiles: vi.fn().mockImplementation(countTrackedFiles),
    listSubmodules: vi.fn().mockReturnValue({ data: [] }),
    listProjectComponents: vi.fn().mockReturnValue({ status: 'ok', data: [] }),
    // Token-economy instrumentation (spec 20) — fire-and-forget; stub
    // out so the tool's start/finish lifecycle doesn't blow up the mock.
    logSearchEvent: vi.fn(),
    updateSearchEvent: vi.fn(),
    updateSearchEventEconomy: vi.fn(),
  } as unknown as SqliteStateManager;
}

function makeProjectDetector(projectId = 'proj-123'): ProjectDetector {
  return {
    getProjectInfo: vi.fn().mockResolvedValue({
      projectId,
      projectPath: '/proj',
      name: 'proj',
    }),
  } as unknown as ProjectDetector;
}

// ── Tests ─────────────────────────────────────────────────────────────────

describe('ListFilesTool — filter pushdown and accurate total_matches (F-016)', () => {
  it('passes glob pattern to listTrackedFiles as glob option', async () => {
    const files = generateFiles(10, 'src/');
    const sm = makeStateManager({ files });
    const tool = new ListFilesTool(sm, makeProjectDetector());

    await tool.list({ format: 'flat', pattern: '*.rs' });

    expect(sm.listTrackedFiles).toHaveBeenCalled();
    const callArg = (sm.listTrackedFiles as ReturnType<typeof vi.fn>).mock
      .calls[0][0] as ListTrackedFilesOptions;
    expect(callArg.glob).toBe('*.rs');
  });

  it('passes component base paths to listTrackedFiles when component filter is set', async () => {
    const files = generateFiles(5, 'src/rust/daemon/');
    const sm = makeStateManager({ files });
    // Override listProjectComponents to return a component
    (sm.listProjectComponents as ReturnType<typeof vi.fn>).mockReturnValue({
      status: 'ok',
      data: [{ componentName: 'daemon', basePath: 'src/rust/daemon', source: 'cargo' }],
    });

    const tool = new ListFilesTool(sm, makeProjectDetector());
    await tool.list({ format: 'flat', component: 'daemon' });

    const callArg = (sm.listTrackedFiles as ReturnType<typeof vi.fn>).mock
      .calls[0][0] as ListTrackedFilesOptions;
    expect(callArg.componentBasePaths).toEqual(['src/rust/daemon']);
  });

  it('defaults project listings to the current git branch', async () => {
    const sm = makeStateManager({ files: generateFiles(5) });
    const tool = new ListFilesTool(sm, makeProjectDetector());

    await tool.list({ format: 'flat' });

    const callArg = (sm.listTrackedFiles as ReturnType<typeof vi.fn>).mock
      .calls[0][0] as ListTrackedFilesOptions;
    expect(callArg.branch).toBe('main');
  });

  it('omits branch filter when branch is wildcard', async () => {
    const sm = makeStateManager({ files: generateFiles(5) });
    const tool = new ListFilesTool(sm, makeProjectDetector());

    await tool.list({ format: 'flat', branch: '*' });

    const callArg = (sm.listTrackedFiles as ReturnType<typeof vi.fn>).mock
      .calls[0][0] as ListTrackedFilesOptions;
    expect(callArg.branch).toBeUndefined();
  });

  it('forwards explicit branch filters unchanged', async () => {
    const sm = makeStateManager({ files: generateFiles(5) });
    const tool = new ListFilesTool(sm, makeProjectDetector());

    await tool.list({ format: 'flat', branch: 'dev' });

    const callArg = (sm.listTrackedFiles as ReturnType<typeof vi.fn>).mock
      .calls[0][0] as ListTrackedFilesOptions;
    expect(callArg.branch).toBe('dev');
  });

  it('total_matches reflects countTrackedFiles result, not capped at 500', async () => {
    // Simulate 700 matching files but listTrackedFiles returns only 500
    const allFiles = generateFiles(700);
    const sm = makeStateManager({
      files: allFiles,
      total: 700,
      listTrackedFiles: (o) => ({
        status: 'ok',
        data: allFiles.slice(0, o.limit ?? 500),
      }),
      countTrackedFiles: () => 700,
    });

    const tool = new ListFilesTool(sm, makeProjectDetector());
    const result = await tool.list({ format: 'flat', limit: 500 });

    expect(result.success).toBe(true);
    expect(result.stats.totalMatching).toBe(700);
  });

  it('countTrackedFiles receives the same filters as listTrackedFiles (glob)', async () => {
    const files = generateFiles(50, 'src/', 'ts');
    const sm = makeStateManager({ files, total: 50 });
    const tool = new ListFilesTool(sm, makeProjectDetector());

    await tool.list({ format: 'flat', pattern: '*.ts' });

    const countArg = (sm.countTrackedFiles as ReturnType<typeof vi.fn>).mock.calls[0][0] as Omit<
      ListTrackedFilesOptions,
      'limit'
    >;
    expect(countArg.glob).toBe('*.ts');
  });
});

describe('ListFilesTool — cursor-based pagination (F-016)', () => {
  it('first page returns next_token when results reach pageSize', async () => {
    const allFiles = generateFiles(300);
    const sm = makeStateManager({ files: allFiles, total: 300 });
    const tool = new ListFilesTool(sm, makeProjectDetector());

    const result = await tool.list({ format: 'flat', pageSize: 100 });

    expect(result.success).toBe(true);
    expect(result.next_token).toBeDefined();
  });

  it('no next_token when all results fit in one page', async () => {
    const allFiles = generateFiles(50);
    const sm = makeStateManager({ files: allFiles, total: 50 });
    const tool = new ListFilesTool(sm, makeProjectDetector());

    const result = await tool.list({ format: 'flat', pageSize: 200 });

    expect(result.success).toBe(true);
    expect(result.next_token).toBeUndefined();
  });

  it('cursor decodes to last relativePath of previous page and is passed as afterPath', async () => {
    const allFiles = generateFiles(200);
    const sm = makeStateManager({ files: allFiles, total: 200 });
    const tool = new ListFilesTool(sm, makeProjectDetector());

    // First page
    const page1 = await tool.list({ format: 'flat', pageSize: 100 });
    expect(page1.next_token).toBeDefined();

    // Second page using the cursor
    await tool.list({ format: 'flat', pageSize: 100, cursor: page1.next_token });

    const calls = (sm.listTrackedFiles as ReturnType<typeof vi.fn>).mock.calls;
    const page2Arg = calls[calls.length - 1][0] as ListTrackedFilesOptions;
    expect(page2Arg.afterPath).toBeDefined();
    // The afterPath should be the last file from page 1 (decoded from base64)
    expect(page2Arg.afterPath).toBe(Buffer.from(page1.next_token!, 'base64').toString('utf8'));
  });

  it('paginating through 700 files with pageSize=500 yields accurate totals both pages', async () => {
    const allFiles = generateFiles(700);
    const sm = makeStateManager({ files: allFiles, total: 700 });
    const tool = new ListFilesTool(sm, makeProjectDetector());

    const page1 = await tool.list({ format: 'flat', pageSize: 500 });
    expect(page1.stats.totalMatching).toBe(700);
    expect(page1.next_token).toBeDefined();

    const page2 = await tool.list({ format: 'flat', pageSize: 500, cursor: page1.next_token });
    expect(page2.stats.totalMatching).toBe(700);
  });

  it('second page has no next_token when it is the last page', async () => {
    const allFiles = generateFiles(150);
    const sm = makeStateManager({ files: allFiles, total: 150 });
    const tool = new ListFilesTool(sm, makeProjectDetector());

    const page1 = await tool.list({ format: 'flat', pageSize: 100 });
    expect(page1.next_token).toBeDefined();

    const page2 = await tool.list({ format: 'flat', pageSize: 100, cursor: page1.next_token });
    // page2 has 50 files < pageSize(100), so no next_token
    expect(page2.next_token).toBeUndefined();
  });
});
