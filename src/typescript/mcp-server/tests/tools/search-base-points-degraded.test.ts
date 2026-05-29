/**
 * Regression tests for F-014 — base-point (worktree/instance) isolation.
 *
 * base_point narrowing only disambiguates multiple clones/worktrees that
 * share a single tenant_id. With one watch folder the tenant filter alone
 * isolates results, so `resolveProjectContext` MUST NOT enumerate base
 * points or flag degradation — doing so produced a false `status:
 * uncertain` on every project search (one base_point per tracked file
 * always blows past the cap on a real repo).
 *
 * Degradation is surfaced ONLY when there are genuinely 2+ clones AND the
 * active base-point set exceeds {@link BASE_POINTS_FILTER_CAP}.
 */

import { describe, it, expect, vi } from 'vitest';
import { resolveProjectContext } from '../../src/tools/search-helpers.js';
import type { SqliteStateManager } from '../../src/clients/sqlite-state-manager.js';
import type { ProjectDetector } from '../../src/utils/project-detector.js';

const TENANT = 'project-a';
const WATCH_FOLDER = 'watch-1';

function makeStateManager(activeCount: number, cloneCount: number): SqliteStateManager {
  const points = Array.from({ length: activeCount }, (_, i) => `bp-${i}`);
  return {
    getWatchFolderIdByTenantId: vi.fn().mockReturnValue(WATCH_FOLDER),
    countWatchFoldersByTenantId: vi.fn().mockReturnValue(cloneCount),
    getActiveBasePoints: vi.fn().mockReturnValue(points),
  } as unknown as SqliteStateManager;
}

function makeProjectDetector(): ProjectDetector {
  return {
    findProjectRoot: vi.fn().mockReturnValue('/some/path'),
    getProjectInfo: vi.fn().mockResolvedValue({ projectId: TENANT }),
  } as unknown as ProjectDetector;
}

describe('resolveProjectContext — single-clone projects never degrade', () => {
  it('does not enumerate base points or degrade for a single clone (within cap)', async () => {
    const state = makeStateManager(50, 1);
    const result = await resolveProjectContext(TENANT, 'project', makeProjectDetector(), state);

    expect(result.currentProjectId).toBe(TENANT);
    // Single clone → tenant filter isolates; base_point narrowing skipped.
    expect(result.basePoints).toBeUndefined();
    expect(result.basePointsDegraded).toBeFalsy();
    expect(state.getActiveBasePoints).not.toHaveBeenCalled();
  });

  it('does not degrade for a single clone even far above the cap', async () => {
    const state = makeStateManager(2027, 1);
    const result = await resolveProjectContext(TENANT, 'project', makeProjectDetector(), state);

    // The common case: a real repo with thousands of files, one clone.
    // Pre-fix this reported status: uncertain on every search.
    expect(result.basePoints).toBeUndefined();
    expect(result.basePointsDegraded).toBeFalsy();
    expect(state.getActiveBasePoints).not.toHaveBeenCalled();
  });
});

describe('resolveProjectContext — multi-clone isolation', () => {
  it('attaches base points when 2+ clones share a tenant and count is within the cap', async () => {
    const state = makeStateManager(50, 2);
    const result = await resolveProjectContext(TENANT, 'project', makeProjectDetector(), state);

    expect(result.basePoints).toHaveLength(50);
    expect(result.basePointsDegraded).toBeFalsy();
  });

  it('surfaces degradation when 2+ clones exceed the 500 cap', async () => {
    const state = makeStateManager(501, 2);
    const result = await resolveProjectContext(TENANT, 'project', makeProjectDetector(), state);

    // Genuine instance ambiguity we cannot enumerate → explicit signal.
    expect(result.basePoints).toBeUndefined();
    expect(result.basePointsDegraded).toBe(true);
    expect(result.basePointsActiveCount).toBe(501);
  });

  it('does not flag degradation when multi-clone but no base points exist', async () => {
    const state = makeStateManager(0, 2);
    const result = await resolveProjectContext(TENANT, 'project', makeProjectDetector(), state);

    expect(result.basePoints).toBeUndefined();
    expect(result.basePointsDegraded).toBeFalsy();
  });
});

describe('resolveProjectContext — scope guard', () => {
  it('does not flag degradation when scope is not project', async () => {
    const state = makeStateManager(501, 2);
    const result = await resolveProjectContext(TENANT, 'all', makeProjectDetector(), state);

    expect(result.basePoints).toBeUndefined();
    expect(result.basePointsDegraded).toBeFalsy();
  });
});
