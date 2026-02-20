/**
 * Memory list operation — query rules by scope from Qdrant with mirror fallback.
 */

import type { QdrantClient } from '@qdrant/js-client-rest';
import type { SqliteStateManager } from '../clients/sqlite-state-manager.js';
import type { ProjectDetector } from '../utils/project-detector.js';
import type { MemoryOptions, MemoryResponse, MemoryRule, MemoryScope } from './memory-types.js';
import { MEMORY_COLLECTION } from './memory-types.js';

/** Build Qdrant filter for list query based on scope. */
function buildListFilter(
  scope: MemoryScope,
  projectId?: string,
): Record<string, unknown> | undefined {
  const mustConditions: Record<string, unknown>[] = [];

  if (scope === 'global') {
    mustConditions.push({ key: 'scope', match: { value: 'global' } });
  } else if (scope === 'project' && projectId) {
    mustConditions.push({ key: 'scope', match: { value: 'project' } });
    mustConditions.push({ key: 'project_id', match: { value: projectId } });
  }

  return mustConditions.length > 0 ? { must: mustConditions } : undefined;
}

/** List rules by scope from Qdrant, with memory_mirror fallback. */
export async function listRules(
  qdrantClient: QdrantClient,
  stateManager: SqliteStateManager,
  projectDetector: ProjectDetector,
  options: MemoryOptions,
): Promise<MemoryResponse> {
  const { scope = 'project', projectId, limit = 50 } = options;

  let resolvedProjectId = projectId;
  if (scope === 'project' && !resolvedProjectId) {
    const cwd = process.cwd();
    const projectInfo = await projectDetector.getProjectInfo(cwd, false);
    resolvedProjectId = projectInfo?.projectId;
  }

  try {
    const filter = buildListFilter(scope, resolvedProjectId);
    const scrollRequest: {
      limit: number;
      with_payload: boolean;
      filter?: Record<string, unknown>;
    } = { limit, with_payload: true };
    if (filter) scrollRequest.filter = filter;

    const scrollResult = await qdrantClient.scroll(MEMORY_COLLECTION, scrollRequest);

    const rules: MemoryRule[] = scrollResult.points.map((point) => {
      const rule: MemoryRule = {
        id: String(point.id),
        content: (point.payload?.['content'] as string) ?? '',
        scope: (point.payload?.['scope'] as MemoryScope) ?? 'global',
      };

      const label = point.payload?.['label'] as string | undefined;
      if (label) rule.label = label;

      const pid = point.payload?.['project_id'] as string | undefined;
      if (pid) rule.projectId = pid;

      const title = point.payload?.['title'] as string | undefined;
      if (title) rule.title = title;

      const tagsStr = point.payload?.['tags'] as string | undefined;
      if (tagsStr) rule.tags = tagsStr.split(',');

      const priorityRaw = point.payload?.['priority'];
      if (priorityRaw !== undefined && priorityRaw !== null) {
        rule.priority = Number(priorityRaw);
      }

      const createdAt = point.payload?.['created_at'] as string | undefined;
      if (createdAt) rule.createdAt = createdAt;

      const updatedAt = point.payload?.['updated_at'] as string | undefined;
      if (updatedAt) rule.updatedAt = updatedAt;

      return rule;
    });

    return {
      success: true,
      action: 'list',
      rules,
      message: `Found ${rules.length} rule(s)`,
    };
  } catch (error) {
    // Qdrant unavailable — fall back to memory_mirror
    try {
      const mirrorRows = stateManager.listMemoryMirror(scope, resolvedProjectId, limit);
      if (mirrorRows.length > 0) {
        const rules: MemoryRule[] = mirrorRows.map((row) => {
          const rule: MemoryRule = {
            id: row.memoryId,
            content: row.ruleText,
            scope: (row.scope as MemoryScope) ?? 'global',
            createdAt: row.createdAt,
            updatedAt: row.updatedAt,
          };
          if (row.tenantId) rule.projectId = row.tenantId;
          return rule;
        });

        return {
          success: true,
          action: 'list',
          rules,
          message: `Found ${rules.length} rule(s) from local mirror (Qdrant unavailable)`,
        };
      }
    } catch {
      // memory_mirror also unavailable
    }

    return {
      success: false,
      action: 'list',
      rules: [],
      message: `Failed to list rules: ${error instanceof Error ? error.message : 'unknown error'}`,
    };
  }
}
