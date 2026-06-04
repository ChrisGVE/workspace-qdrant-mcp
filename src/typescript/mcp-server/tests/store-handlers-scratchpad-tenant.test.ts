/**
 * Tests for storeScratchpad tenant resolution.
 *
 * A scratchpad note must carry the current project's tenant_id so the
 * tenant-filtered recall lane can surface it on project-scoped search. The
 * resolution order is: active session project → project detected from the
 * effective cwd → global tenant.
 */

import { describe, it, expect, vi } from 'vitest';
import { storeScratchpad } from '../src/store-handlers.js';
import type { SqliteStateManager } from '../src/clients/sqlite-state-manager.js';
import type { ProjectDetector } from '../src/utils/project-detector.js';
import { TENANT_GLOBAL } from '../src/constants/tenants.js';

function mockStateManager(): SqliteStateManager {
  return {
    enqueueUnified: vi.fn().mockResolvedValue({ status: 'ok', data: { queueId: 'q-1' } }),
    upsertScratchpadMirror: vi.fn(),
  } as unknown as SqliteStateManager;
}

function mockDetector(projectId: string | null): ProjectDetector {
  return {
    getProjectInfo: vi.fn().mockResolvedValue(projectId ? { projectId } : null),
  } as unknown as ProjectDetector;
}

/** The tenant_id is the 3rd positional arg of enqueueUnified. */
function tenantOf(sm: SqliteStateManager): unknown {
  return (sm.enqueueUnified as unknown as ReturnType<typeof vi.fn>).mock.calls[0][2];
}

describe('storeScratchpad — tenant resolution', () => {
  it('prefers the active session project (no detection)', async () => {
    const sm = mockStateManager();
    const detector = mockDetector('detected-xyz');

    const res = await storeScratchpad({ content: 'note' }, sm, detector, {
      projectId: 'session-abc',
    });

    expect(res.success).toBe(true);
    expect(detector.getProjectInfo).not.toHaveBeenCalled();
    expect(tenantOf(sm)).toBe('session-abc');
  });

  it('detects the project from cwd when no session project is set', async () => {
    const sm = mockStateManager();
    const detector = mockDetector('detected-xyz');

    await storeScratchpad({ content: 'note' }, sm, detector, { projectId: undefined });

    expect(detector.getProjectInfo).toHaveBeenCalled();
    expect(tenantOf(sm)).toBe('detected-xyz');
  });

  it('falls back to the global tenant when nothing resolves', async () => {
    const sm = mockStateManager();
    const detector = mockDetector(null);

    await storeScratchpad({ content: 'note' }, sm, detector, { projectId: undefined });

    expect(tenantOf(sm)).toBe(TENANT_GLOBAL);
  });

  it('falls back to global when project detection throws', async () => {
    const sm = mockStateManager();
    const detector = {
      getProjectInfo: vi.fn().mockRejectedValue(new Error('boom')),
    } as unknown as ProjectDetector;

    await storeScratchpad({ content: 'note' }, sm, detector, { projectId: undefined });

    expect(tenantOf(sm)).toBe(TENANT_GLOBAL);
  });
});
