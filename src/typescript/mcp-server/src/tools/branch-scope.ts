/**
 * Shared branch scoping helpers for project-aware tools.
 *
 * Default project-scoped reads to the caller's current Git branch so indexed
 * feature/worktree branches do not bleed into ordinary results. `branch: "*"`
 * remains the explicit opt-out for cross-branch reads.
 */

import type { ProjectDetector, ProjectInfo } from '../utils/project-detector.js';
import { getCurrentBranch } from '../utils/git-utils.js';
import { getEffectiveCwd } from '../utils/request-context.js';

export interface ProjectIdentity {
  projectId: string | undefined;
  projectPath: string | undefined;
}

export async function resolveProjectIdentity(
  projectDetector: ProjectDetector,
  explicitProjectId: string | undefined,
  fallbackToSoleProject = true
): Promise<ProjectIdentity> {
  if (explicitProjectId) return { projectId: explicitProjectId, projectPath: undefined };
  const projectInfo: ProjectInfo | null = await projectDetector.getProjectInfo(
    getEffectiveCwd(),
    false,
    { fallbackToSoleProject }
  );
  return {
    projectId: projectInfo?.projectId,
    projectPath: projectInfo?.projectPath,
  };
}

export function resolveEffectiveBranch(params: {
  explicitBranch: string | undefined;
  scope: string;
  projectId: string | undefined;
  projectPath: string | undefined;
}): string | undefined {
  if (params.explicitBranch !== undefined) return params.explicitBranch;
  if (params.scope !== 'project' || !params.projectId) return undefined;
  if (!params.projectPath) return undefined;
  const branch = getCurrentBranch(params.projectPath);
  return branch && branch !== 'HEAD' ? branch : undefined;
}

export function applyEffectiveBranch<T extends { branch?: string }>(
  options: T,
  effectiveBranch: string | undefined
): T {
  if (effectiveBranch === undefined || effectiveBranch === options.branch) return options;
  return { ...options, branch: effectiveBranch };
}

export function concreteBranchFilter(branch: string | undefined): string | undefined {
  return branch && branch !== '*' ? branch : undefined;
}
