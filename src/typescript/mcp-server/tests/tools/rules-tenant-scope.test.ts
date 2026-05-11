/**
 * Regression tests for F-015 — same-label rules across two projects.
 *
 * Labels are not globally unique. Project A having a rule labeled
 * "prefer-uv" and project B having a different rule with the same
 * label is a legitimate state. The daemon's F-005 fix scoped Qdrant
 * delete/update by (label, tenant_id), but the TypeScript client
 * still had to surface the correct tenant_id:
 *
 * 1. `findSimilarRules` searched the entire RULES_COLLECTION, so a
 *    project A add was rejected as a duplicate of a project B rule.
 * 2. `persistUpdateRule` hardcoded TENANT_GLOBAL, so project-scope
 *    updates landed in the wrong tenant on the daemon side.
 * 3. `removeRule` did not accept scope/projectId, so the queued
 *    payload had no tenant — the daemon then defaulted to global.
 */

import { beforeEach, describe, it, expect, vi } from 'vitest';
import { RulesTool } from '../../src/tools/rules.js';
import type { DaemonClient } from '../../src/clients/daemon-client.js';
import type { SqliteStateManager } from '../../src/clients/sqlite-state-manager.js';
import type { ProjectDetector } from '../../src/utils/project-detector.js';
import { TENANT_GLOBAL } from '../../src/constants/tenants.js';

const PROJECT_A = 'project-a';

let qdrantSearchCalls: Array<{ collection: string; request: Record<string, unknown> }> = [];

/** Recursively find a `{ key: "project_id", match: { value } }` clause
 * anywhere in a Qdrant filter tree. Returns the value or `undefined`. */
function findProjectIdMatch(node: unknown): unknown {
  if (!node || typeof node !== 'object') return undefined;
  const obj = node as Record<string, unknown>;
  if (obj.key === 'project_id' && obj.match && typeof obj.match === 'object') {
    return (obj.match as Record<string, unknown>).value;
  }
  for (const v of Object.values(obj)) {
    if (Array.isArray(v)) {
      for (const item of v) {
        const found = findProjectIdMatch(item);
        if (found !== undefined) return found;
      }
    } else if (v && typeof v === 'object') {
      const found = findProjectIdMatch(v);
      if (found !== undefined) return found;
    }
  }
  return undefined;
}

vi.mock('@qdrant/js-client-rest', () => ({
  QdrantClient: vi.fn().mockImplementation(() => ({
    search: vi.fn((collection: string, request: Record<string, unknown>) => {
      qdrantSearchCalls.push({ collection, request });
      // Scoped (project_id in filter tree) → no foreign-tenant hit.
      const scoped = findProjectIdMatch(request.filter) !== undefined;
      if (scoped) return Promise.resolve([]);
      return Promise.resolve([
        {
          id: 'rule-from-project-b',
          score: 0.95,
          payload: { label: 'prefer-uv', content: 'foreign rule' },
        },
      ]);
    }),
    scroll: vi.fn().mockResolvedValue({ points: [] }),
  })),
}));

function makeDaemon(): DaemonClient {
  return {
    isConnected: vi.fn().mockReturnValue(true),
    ingestText: vi
      .fn()
      .mockResolvedValue({ success: true, document_id: 'doc-1', chunks_created: 1 }),
    embedText: vi.fn().mockResolvedValue({ success: true, embedding: [0.1, 0.2, 0.3] }),
    upsertRuleMirror: vi.fn().mockResolvedValue({}),
    deleteRuleMirror: vi.fn().mockResolvedValue({}),
    generateSparseVector: vi.fn(),
    connect: vi.fn(),
    close: vi.fn(),
    healthCheck: vi.fn(),
  } as unknown as DaemonClient;
}

function makeState(): SqliteStateManager {
  return {
    enqueueUnified: vi
      .fn()
      .mockResolvedValue({ status: 'ok', data: { queueId: 'q-1', isNew: true } }),
    upsertRulesMirror: vi.fn(),
    deleteRulesMirror: vi.fn(),
    listRulesMirror: vi.fn().mockReturnValue([]),
  } as unknown as SqliteStateManager;
}

function makeDetector(projectId?: string): ProjectDetector {
  return {
    findProjectRoot: vi.fn().mockReturnValue('/some/path'),
    getProjectInfo: vi.fn().mockResolvedValue(projectId ? { projectId } : null),
  } as unknown as ProjectDetector;
}

function makeTool(detector: ProjectDetector, daemon: DaemonClient, state: SqliteStateManager) {
  return new RulesTool({ qdrantUrl: 'http://localhost:6333' }, daemon, state, detector);
}

beforeEach(() => {
  qdrantSearchCalls = [];
});

describe('Rules — F-015 same-label across projects is correctly scoped', () => {
  it('scopes duplicate detection to the active project so cross-project labels do not collide', async () => {
    const daemon = makeDaemon();
    const state = makeState();
    const detector = makeDetector(PROJECT_A);
    const tool = makeTool(detector, daemon, state);

    const response = await tool.execute({
      action: 'add',
      label: 'prefer-uv',
      content: 'use uv in project A',
      scope: 'project',
      projectId: PROJECT_A,
    });

    // Project A's add MUST succeed even though project B has a rule
    // with the same label. The mock returns no hits when the filter
    // tree contains a project_id match clause.
    expect(response.success).toBe(true);
    expect(qdrantSearchCalls).toHaveLength(1);
    const projectIdInFilter = findProjectIdMatch(qdrantSearchCalls[0].request.filter);
    expect(projectIdInFilter).toBe(PROJECT_A);
  });

  it('passes project tenant_id to daemon on update so the right rule is updated', async () => {
    const daemon = makeDaemon();
    const state = makeState();
    const detector = makeDetector(PROJECT_A);
    const tool = makeTool(detector, daemon, state);

    const response = await tool.execute({
      action: 'update',
      label: 'prefer-uv',
      content: 'updated content for project A',
      scope: 'project',
      projectId: PROJECT_A,
    });

    expect(response.success).toBe(true);
    const ingestArgs = (daemon.ingestText as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(ingestArgs.tenant_id).toBe(PROJECT_A);
    expect(ingestArgs.tenant_id).not.toBe(TENANT_GLOBAL);
  });

  it('passes project tenant_id when queueing a remove so the daemon scopes deletion correctly', async () => {
    const daemon = makeDaemon();
    const state = makeState();
    const detector = makeDetector(PROJECT_A);
    const tool = makeTool(detector, daemon, state);

    const response = await tool.execute({
      action: 'remove',
      label: 'prefer-uv',
      scope: 'project',
      projectId: PROJECT_A,
    });

    expect(response.success).toBe(true);
    const enqueueArgs = (state.enqueueUnified as ReturnType<typeof vi.fn>).mock.calls[0];
    // Signature: (item_type, op, tenant_id, collection, payload, priority, branch, meta)
    expect(enqueueArgs[2]).toBe(PROJECT_A);
    const payload = enqueueArgs[4] as Record<string, unknown>;
    expect(payload['project_id']).toBe(PROJECT_A);
    expect(payload.scope).toBe('project');
  });

  it('falls back to TENANT_GLOBAL for global rule updates', async () => {
    const daemon = makeDaemon();
    const state = makeState();
    const detector = makeDetector(PROJECT_A);
    const tool = makeTool(detector, daemon, state);

    const response = await tool.execute({
      action: 'update',
      label: 'prefer-uv',
      content: 'global update',
      scope: 'global',
    });

    expect(response.success).toBe(true);
    const ingestArgs = (daemon.ingestText as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(ingestArgs.tenant_id).toBe(TENANT_GLOBAL);
  });
});
