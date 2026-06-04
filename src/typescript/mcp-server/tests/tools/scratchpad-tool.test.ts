/**
 * Tests for the ScratchpadTool (list / update / delete).
 *
 * Mutations enqueue to the unified queue (item_type "text") scoped to the
 * resolved tenant; the entry is identified by its current content. list scrolls
 * the scratchpad Qdrant collection filtered by tenant.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { ScratchpadTool } from '../../src/tools/scratchpad.js';
import type { SqliteStateManager } from '../../src/clients/sqlite-state-manager.js';
import type { ProjectDetector } from '../../src/utils/project-detector.js';

let lastScrollFilter: unknown;

vi.mock('@qdrant/js-client-rest', () => ({
  QdrantClient: vi.fn().mockImplementation(() => ({
    scroll: vi.fn().mockImplementation((_coll: string, req: { filter?: unknown }) => {
      lastScrollFilter = req.filter;
      return Promise.resolve({
        points: [
          {
            id: 'pt-1',
            payload: {
              content: 'a project note',
              title: 'T',
              tags: ['x'],
              created_at: '2026-06-04T00:00:00Z',
            },
          },
        ],
      });
    }),
  })),
}));

function mockStateManager(): SqliteStateManager {
  return {
    enqueueUnified: vi.fn().mockResolvedValue({ status: 'ok', data: { queueId: 'q-1' } }),
    upsertScratchpadMirror: vi.fn(),
  } as unknown as SqliteStateManager;
}

function mockProjectDetector(): ProjectDetector {
  return {
    getProjectInfo: vi.fn().mockResolvedValue({ projectId: 'detected' }),
  } as unknown as ProjectDetector;
}

function enqueueCall(sm: SqliteStateManager): unknown[] {
  return (sm.enqueueUnified as unknown as ReturnType<typeof vi.fn>).mock.calls[0];
}

describe('ScratchpadTool', () => {
  let sm: SqliteStateManager;
  let detector: ProjectDetector;
  let tool: ScratchpadTool;

  beforeEach(() => {
    vi.clearAllMocks();
    sm = mockStateManager();
    detector = mockProjectDetector();
    tool = new ScratchpadTool({ qdrantUrl: 'http://localhost:6333' }, sm, detector);
  });

  it('delete enqueues a tenant-scoped delete op identified by content', async () => {
    const res = await tool.execute({ action: 'delete', content: 'a note', projectId: 't1' });

    expect(res.success).toBe(true);
    const call = enqueueCall(sm);
    expect(call[0]).toBe('text'); // item_type
    expect(call[1]).toBe('delete'); // op
    expect(call[2]).toBe('t1'); // tenant
    expect(call[4]).toMatchObject({ content: 'a note', source_type: 'scratchpad' });
  });

  it('delete without content is rejected (no destructive guess)', async () => {
    const res = await tool.execute({ action: 'delete', projectId: 't1' });
    expect(res.success).toBe(false);
    expect(sm.enqueueUnified).not.toHaveBeenCalled();
  });

  it('update enqueues new content + old_content and refreshes the mirror', async () => {
    const res = await tool.execute({
      action: 'update',
      content: 'old text',
      newContent: 'new text',
      title: 'Title',
      tags: ['t'],
      projectId: 't1',
    });

    expect(res.success).toBe(true);
    const call = enqueueCall(sm);
    expect(call[1]).toBe('update');
    expect(call[2]).toBe('t1');
    expect(call[4]).toMatchObject({
      content: 'new text',
      old_content: 'old text',
      source_type: 'scratchpad',
      title: 'Title',
      tags: ['t'],
    });
    expect(sm.upsertScratchpadMirror).toHaveBeenCalledTimes(1);
  });

  it('update without newContent is rejected', async () => {
    const res = await tool.execute({ action: 'update', content: 'old', projectId: 't1' });
    expect(res.success).toBe(false);
    expect(sm.enqueueUnified).not.toHaveBeenCalled();
  });

  it('list scrolls the scratchpad collection filtered by tenant', async () => {
    const res = await tool.execute({ action: 'list', projectId: 't1', limit: 10 });

    expect(res.success).toBe(true);
    expect(res.count).toBe(1);
    expect(res.entries?.[0]).toMatchObject({ id: 'pt-1', content: 'a project note', title: 'T' });
    // tenant filter applied
    expect(JSON.stringify(lastScrollFilter)).toContain('t1');
  });

  it('resolves the tenant from the project detector when no projectId is given', async () => {
    await tool.execute({ action: 'delete', content: 'x' });
    expect(detector.getProjectInfo).toHaveBeenCalled();
    expect(enqueueCall(sm)[2]).toBe('detected');
  });
});
