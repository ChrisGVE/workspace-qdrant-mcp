import { describe, it, expect, vi } from 'vitest';

import { handleWorkspaceIndex } from '../../src/tools/workspace-index.js';
import type { DaemonClient } from '../../src/clients/daemon-client.js';
import type { ProjectDetector } from '../../src/utils/project-detector.js';
import { runWithRequestContext } from '../../src/utils/request-context.js';

function makeStatus(projectId: string) {
  return {
    found: true,
    project_id: projectId,
    project_name: 'workspace-qdrant-mcp',
    project_root: '/home/alkmimm/respositorios/workspace-qdrant-mcp',
    is_active: true,
    pending_count: 0,
    in_progress_count: 0,
    failed_count: 0,
    done_count: 1933,
    total_count: 1933,
    percent_complete: 100,
  };
}

function makeDaemon(projectId = '367157a01d98'): {
  daemon: DaemonClient;
  getProjectStatus: ReturnType<typeof vi.fn>;
  listProjects: ReturnType<typeof vi.fn>;
} {
  const getProjectStatus = vi.fn().mockResolvedValue(makeStatus(projectId));
  const listProjects = vi.fn().mockResolvedValue({
    projects: [{ project_id: 'fallback-active' }],
  });
  return {
    daemon: { getProjectStatus, listProjects } as unknown as DaemonClient,
    getProjectStatus,
    listProjects,
  };
}

function makeDetector(projectId: string | null): {
  detector: ProjectDetector;
  getProjectInfo: ReturnType<typeof vi.fn>;
} {
  const getProjectInfo = vi.fn().mockResolvedValue(
    projectId
      ? {
          projectId,
          projectPath: '/home/alkmimm/respositorios/workspace-qdrant-mcp',
          isActive: true,
        }
      : null
  );
  return { detector: { getProjectInfo } as unknown as ProjectDetector, getProjectInfo };
}

describe('workspace_index status resolution', () => {
  it('project_status resolves the current repo from request cwd when projectId is omitted', async () => {
    const { daemon, getProjectStatus, listProjects } = makeDaemon();
    const { detector, getProjectInfo } = makeDetector('367157a01d98');

    const result = (await runWithRequestContext(
      { hostCwd: '/home/alkmimm/respositorios/workspace-qdrant-mcp' },
      () => handleWorkspaceIndex({ action: 'project_status' }, daemon, detector)
    )) as Record<string, unknown>;

    expect(getProjectInfo).toHaveBeenCalledWith(
      '/home/alkmimm/respositorios/workspace-qdrant-mcp',
      false,
      { fallbackToSoleProject: true }
    );
    expect(listProjects).not.toHaveBeenCalled();
    expect(getProjectStatus).toHaveBeenCalledWith({ project_id: '367157a01d98' });
    expect(result).toMatchObject({
      success: true,
      action: 'project_status',
      project_id: '367157a01d98',
      project_name: 'workspace-qdrant-mcp',
    });
  });

  it('indexing_status resolves cwd explicitly and reports indexing progress', async () => {
    const { daemon, getProjectStatus, listProjects } = makeDaemon();
    const { detector, getProjectInfo } = makeDetector('367157a01d98');

    const result = (await handleWorkspaceIndex(
      {
        action: 'indexing_status',
        cwd: '/home/alkmimm/respositorios/workspace-qdrant-mcp/src/typescript/mcp-server',
      },
      daemon,
      detector
    )) as Record<string, unknown>;

    expect(getProjectInfo).toHaveBeenCalledWith(
      '/home/alkmimm/respositorios/workspace-qdrant-mcp/src/typescript/mcp-server',
      false,
      { fallbackToSoleProject: true }
    );
    expect(listProjects).not.toHaveBeenCalled();
    expect(getProjectStatus).toHaveBeenCalledWith({ project_id: '367157a01d98' });
    expect(result).toMatchObject({
      success: true,
      action: 'indexing_status',
      project_id: '367157a01d98',
      indexing: {
        pending: 0,
        in_progress: 0,
        failed: 0,
        done: 1933,
        total: 1933,
        percent: 100,
      },
    });
  });

  it('falls back to the first active project when cwd cannot be resolved', async () => {
    const { daemon, getProjectStatus, listProjects } = makeDaemon('fallback-active');
    const { detector } = makeDetector(null);

    await handleWorkspaceIndex({ action: 'indexing_status', cwd: '/unregistered' }, daemon, detector);

    expect(listProjects).toHaveBeenCalledWith({ active_only: true });
    expect(getProjectStatus).toHaveBeenCalledWith({ project_id: 'fallback-active' });
  });
});
