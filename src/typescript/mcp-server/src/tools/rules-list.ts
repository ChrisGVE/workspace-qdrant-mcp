/**
 * Rules list operation — query rules by scope from Qdrant with mirror fallback.
 */

import type { QdrantClient } from '@qdrant/js-client-rest';
import type { SqliteStateManager } from '../clients/sqlite-state-manager.js';
import type { ProjectDetector } from '../utils/project-detector.js';
import { getEffectiveCwd } from '../utils/request-context.js';
import type { RuleOptions, RuleResponse, Rule, RuleScope } from './rules-types.js';
import { RULES_COLLECTION } from './rules-types.js';
import { FIELD_PROJECT_ID, FIELD_CONTENT, FIELD_TITLE } from '../common/native-bridge.js';
import { TENANT_GLOBAL } from '../constants/tenants.js';

/** Build Qdrant filter for list query based on scope. */
function buildListFilter(
  scope: RuleScope,
  projectId?: string
): Record<string, unknown> | undefined {
  const mustConditions: Record<string, unknown>[] = [];

  if (scope === TENANT_GLOBAL) {
    mustConditions.push({ key: 'scope', match: { value: TENANT_GLOBAL } });
  } else if (scope === 'project' && projectId) {
    mustConditions.push({ key: 'scope', match: { value: 'project' } });
    mustConditions.push({ key: FIELD_PROJECT_ID, match: { value: projectId } });
  }

  return mustConditions.length > 0 ? { must: mustConditions } : undefined;
}

/** Map a Qdrant point payload to a Rule object. */
function pointToRule(point: {
  id: string | number;
  payload?: Record<string, unknown> | null;
}): Rule {
  const rule: Rule = {
    id: String(point.id),
    content: (point.payload?.[FIELD_CONTENT] as string) ?? '',
    scope: (point.payload?.['scope'] as RuleScope) ?? TENANT_GLOBAL,
  };
  const label = point.payload?.['label'] as string | undefined;
  if (label) rule.label = label;
  const pid = point.payload?.[FIELD_PROJECT_ID] as string | undefined;
  if (pid) rule.projectId = pid;
  // Always surface the owner so a multi-tenant listing is unambiguous:
  // project rules → owning tenant_id; global rules → "global".
  rule.owner = rule.scope === 'project' ? (pid ?? 'unknown-project') : TENANT_GLOBAL;
  const title = point.payload?.[FIELD_TITLE] as string | undefined;
  if (title) rule.title = title;
  const tagsStr = point.payload?.['tags'] as string | undefined;
  if (tagsStr) rule.tags = tagsStr.split(',');
  const priorityRaw = point.payload?.['priority'];
  if (priorityRaw !== undefined && priorityRaw !== null) rule.priority = Number(priorityRaw);
  const createdAt = point.payload?.['created_at'] as string | undefined;
  if (createdAt) rule.createdAt = createdAt;
  const updatedAt = point.payload?.['updated_at'] as string | undefined;
  if (updatedAt) rule.updatedAt = updatedAt;
  return rule;
}

/** Build a scroll request for the rules collection. */
function buildScrollRequest(
  limit: number,
  filter: Record<string, unknown> | undefined
): { limit: number; with_payload: boolean; filter?: Record<string, unknown> } {
  const req: { limit: number; with_payload: boolean; filter?: Record<string, unknown> } = {
    limit,
    with_payload: true,
  };
  if (filter) req.filter = filter;
  return req;
}

/** Attempt to read rules from the local mirror as fallback. */
function readRulesFromMirror(
  stateManager: SqliteStateManager,
  scope: RuleScope,
  resolvedProjectId: string | undefined,
  limit: number
): RuleResponse | null {
  try {
    const mirrorRows = stateManager.listRulesMirror(scope, resolvedProjectId, limit);
    if (mirrorRows.length === 0) return null;
    const rules: Rule[] = mirrorRows.map((row) => {
      const scope = (row.scope as RuleScope) ?? TENANT_GLOBAL;
      const rule: Rule = {
        id: row.ruleId,
        content: row.ruleText,
        scope,
        owner: scope === 'project' ? (row.tenantId ?? 'unknown-project') : TENANT_GLOBAL,
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
  } catch {
    return null;
  }
}

/** List rules by scope from Qdrant, with rules_mirror fallback. */
export async function listRules(
  qdrantClient: QdrantClient,
  stateManager: SqliteStateManager,
  projectDetector: ProjectDetector,
  options: RuleOptions
): Promise<RuleResponse> {
  const { scope = 'project', projectId, limit = 50 } = options;

  let resolvedProjectId = projectId;
  if (scope === 'project' && !resolvedProjectId) {
    const projectInfo = await projectDetector.getProjectInfo(getEffectiveCwd(), false, {
      fallbackToSoleProject: true,
    });
    resolvedProjectId = projectInfo?.projectId;
  }

  // When a project-scoped list can't resolve a tenant, buildListFilter yields
  // no filter and the scroll spans every project's rules. Surface that so the
  // agent knows the listing is multi-tenant and reads each rule's `owner`.
  const unresolvedProjectScope = scope === 'project' && !resolvedProjectId;

  try {
    const filter = buildListFilter(scope, resolvedProjectId);
    const scrollResult = await qdrantClient.scroll(
      RULES_COLLECTION,
      buildScrollRequest(limit, filter)
    );
    const rules: Rule[] = scrollResult.points.map(pointToRule);
    const message = unresolvedProjectScope
      ? `Found ${rules.length} rule(s) across ALL projects — the current project could not be detected, so this listing is not scoped. Each rule's "owner" field identifies its project (or "global"). Pass cwd or projectId to scope to one project.`
      : `Found ${rules.length} rule(s)`;
    return { success: true, action: 'list', rules, message };
  } catch (error) {
    const mirror = readRulesFromMirror(stateManager, scope, resolvedProjectId, limit);
    if (mirror) return mirror;
    return {
      success: false,
      action: 'list',
      rules: [],
      message: `Failed to list rules: ${error instanceof Error ? error.message : 'unknown error'}`,
    };
  }
}
