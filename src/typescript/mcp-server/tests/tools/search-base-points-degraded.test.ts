/**
 * Regression tests for F-014 — when a project has more than 500 active
 * base points, the pre-fix code path silently dropped the
 * `base_points` filter (worktree/instance isolation broadens to the
 * whole tenant). `resolveProjectContext` MUST surface this degradation
 * via an explicit flag instead of silently broadening.
 */

import { describe, it, expect, vi } from 'vitest';
import { resolveProjectContext } from '../../src/tools/search-helpers.js';
import type { SqliteStateManager } from '../../src/clients/sqlite-state-manager.js';
import type { ProjectDetector } from '../../src/utils/project-detector.js';

const TENANT = 'project-a';
const WATCH_FOLDER = 'watch-1';

function makeStateManager(activeCount: number): SqliteStateManager {
  const points = Array.from({ length: activeCount }, (_, i) => `bp-${i}`);
  return {
    getWatchFolderIdByTenantId: vi.fn().mockReturnValue(WATCH_FOLDER),
    getActiveBasePoints: vi.fn().mockReturnValue(points),
  } as unknown as SqliteStateManager;
}

function makeProjectDetector(): ProjectDetector {
  return {
    findProjectRoot: vi.fn().mockReturnValue('/some/path'),
    getProjectInfo: vi.fn().mockResolvedValue({ projectId: TENANT }),
  } as unknown as ProjectDetector;
}

describe('resolveProjectContext — F-014 base-point cutoff is surfaced', () => {
  it('returns basePoints when active count is within budget (<= 500)', async () => {
    const state = makeStateManager(50);
    const detector = makeProjectDetector();

    const result = await resolveProjectContext(TENANT, 'project', detector, state);

    expect(result.currentProjectId).toBe(TENANT);
    expect(result.basePoints).toHaveLength(50);
    expect(result.basePointsDegraded).toBeFalsy();
  });

  it('surfaces basePointsDegraded when active count exceeds the 500 cap', async () => {
    const state = makeStateManager(501);
    const detector = makeProjectDetector();

    const result = await resolveProjectContext(TENANT, 'project', detector, state);

    expect(result.currentProjectId).toBe(TENANT);
    // base_points filter dropped — tenant filter still applies.
    expect(result.basePoints).toBeUndefined();
    // F-014: degradation MUST be explicit so the caller can surface it
    // in the response status_reason instead of silently broadening.
    expect(result.basePointsDegraded).toBe(true);
    expect(result.basePointsActiveCount).toBe(501);
  });

  it('does not flag degradation when no base points exist', async () => {
    const state = makeStateManager(0);
    const detector = makeProjectDetector();

    const result = await resolveProjectContext(TENANT, 'project', detector, state);

    expect(result.basePoints).toBeUndefined();
    expect(result.basePointsDegraded).toBeFalsy();
  });

  it('does not flag degradation when scope is not project', async () => {
    const state = makeStateManager(501);
    const detector = makeProjectDetector();

    const result = await resolveProjectContext(TENANT, 'all', detector, state);

    expect(result.basePoints).toBeUndefined();
    expect(result.basePointsDegraded).toBeFalsy();
  });
});
