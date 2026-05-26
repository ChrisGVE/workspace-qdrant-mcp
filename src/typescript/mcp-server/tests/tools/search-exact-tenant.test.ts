/**
 * Regression tests for F-004 — project-scope exact (FTS5) search MUST
 * refuse to run when no tenant can be resolved. The pre-fix path
 * silently omitted `tenant_id` from the daemon request, and the daemon
 * query builder dropped the `fm.tenant_id = ?` clause, broadening the
 * search to every tenant in the FTS index.
 */

import { describe, it, expect, vi } from 'vitest';
import { searchExact } from '../../src/tools/search-exact.js';
import type { DaemonClient } from '../../src/clients/daemon-client.js';
import type { SqliteStateManager } from '../../src/clients/sqlite-state-manager.js';
import type { ProjectDetector } from '../../src/utils/project-detector.js';
import type { SearchOptions } from '../../src/tools/search-types.js';

function makeStateManager(): SqliteStateManager {
  return {
    logSearchEvent: vi.fn(),
    updateSearchEvent: vi.fn(),
    updateSearchEventEconomy: vi.fn(),
  } as unknown as SqliteStateManager;
}

function makeProjectDetector(projectId: string | undefined): ProjectDetector {
  return {
    findProjectRoot: vi.fn().mockReturnValue('/some/path'),
    getProjectInfo: vi.fn().mockResolvedValue(projectId ? { projectId } : null),
  } as unknown as ProjectDetector;
}

function makeDaemonClient(matches: Array<Record<string, unknown>> = []): DaemonClient {
  return {
    textSearch: vi.fn().mockResolvedValue({
      matches,
      total_matches: matches.length,
    }),
  } as unknown as DaemonClient;
}

function makeOptions(overrides: Partial<SearchOptions> = {}): SearchOptions {
  return {
    query: 'needle',
    scope: 'project',
    ...overrides,
  };
}

describe('searchExact — F-004 project-scope refuses unresolved tenant', () => {
  it('returns unresolved-project response when scope=project and no projectId can be found', async () => {
    const daemon = makeDaemonClient();
    const state = makeStateManager();
    const detector = makeProjectDetector(undefined);

    const response = await searchExact(daemon, state, detector, makeOptions());

    // Daemon MUST NOT be called — the pre-fix code did call it without
    // tenant_id and the daemon broadened to all tenants.
    expect(daemon.textSearch).not.toHaveBeenCalled();
    expect(response.results).toHaveLength(0);
    expect(response.total).toBe(0);
    expect(response.status).toBe('uncertain');
    expect(response.status_reason).toContain('project');
    expect(response.status_reason).toMatch(/resolved|scope/i);
  });

  it('passes tenant_id to daemon when projectId is explicit', async () => {
    const daemon = makeDaemonClient([
      {
        file_path: 'src/foo.ts',
        line_number: 10,
        content: 'has needle',
        tenant_id: 'project-a',
      },
    ]);
    const state = makeStateManager();
    const detector = makeProjectDetector(undefined); // detector won't be used

    const response = await searchExact(
      daemon,
      state,
      detector,
      makeOptions({ projectId: 'project-a' })
    );

    expect(daemon.textSearch).toHaveBeenCalledTimes(1);
    const request = (daemon.textSearch as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(request.tenant_id).toBe('project-a');
    expect(response.results).toHaveLength(1);
  });

  it('passes detector-resolved tenant_id to daemon when projectId is implicit', async () => {
    const daemon = makeDaemonClient([]);
    const state = makeStateManager();
    const detector = makeProjectDetector('project-a');

    await searchExact(daemon, state, detector, makeOptions());

    const request = (daemon.textSearch as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(request.tenant_id).toBe('project-a');
  });

  it('omits tenant_id when scope=all (intentional cross-tenant search)', async () => {
    const daemon = makeDaemonClient([]);
    const state = makeStateManager();
    // Detector returns nothing — scope=all overrides regardless.
    const detector = makeProjectDetector(undefined);

    await searchExact(daemon, state, detector, makeOptions({ scope: 'all' }));

    expect(daemon.textSearch).toHaveBeenCalledTimes(1);
    const request = (daemon.textSearch as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(request.tenant_id).toBeUndefined();
  });
});
