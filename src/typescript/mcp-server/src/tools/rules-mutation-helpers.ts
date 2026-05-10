/**
 * Shared helpers for rules mutation operations.
 */

import type { DaemonClient } from '../clients/daemon-client.js';
import type { SqliteStateManager } from '../clients/sqlite-state-manager.js';
import type { ProjectDetector } from '../utils/project-detector.js';
import { randomUUID } from 'node:crypto';
import { utcNow } from '../utils/timestamps.js';
import {
  PRIORITY_HIGH,
  FIELD_SOURCE_TYPE,
  FIELD_PROJECT_ID,
  FIELD_CONTENT,
  FIELD_TITLE,
} from '../common/native-bridge.js';
import type { RuleAction, RuleResponse, RuleScope } from './rules-types.js';
import { RULES_BASENAME, RULES_COLLECTION } from './rules-types.js';
import { TENANT_GLOBAL } from '../constants/tenants.js';

export function isConnectivityError(err: unknown): boolean {
  const msg = err instanceof Error ? err.message : String(err);
  const code = (err as { code?: string })?.code;
  return (
    code === 'UNAVAILABLE' ||
    code === 'DEADLINE_EXCEEDED' ||
    code === 'ECONNREFUSED' ||
    msg.includes('UNAVAILABLE') ||
    msg.includes('DEADLINE_EXCEEDED') ||
    msg.includes('ECONNREFUSED') ||
    msg.includes('connect ECONNREFUSED')
  );
}

function ruleActionToQueueOp(action: RuleAction): 'add' | 'update' | 'delete' {
  if (action === 'update') return 'update';
  if (action === 'remove') return 'delete';
  return 'add';
}

function buildRulePayload(operation: {
  action: RuleAction;
  label?: string;
  content?: string;
  scope?: RuleScope;
  projectId?: string;
  title?: string;
  tags?: string[];
  priority?: number;
}): Record<string, unknown> {
  const payload: Record<string, unknown> = {
    label: operation.label,
    action: operation.action,
    [FIELD_SOURCE_TYPE]: 'rule',
  };
  if (operation.content) payload[FIELD_CONTENT] = operation.content;
  if (operation.scope) payload['scope'] = operation.scope;
  if (operation.projectId) payload[FIELD_PROJECT_ID] = operation.projectId;
  if (operation.title) payload[FIELD_TITLE] = operation.title;
  if (operation.tags) payload['tags'] = operation.tags;
  if (operation.priority !== undefined) payload['priority'] = operation.priority;
  return payload;
}

export async function queueRuleOperation(
  stateManager: SqliteStateManager,
  operation: {
    action: RuleAction;
    label?: string;
    content?: string;
    scope?: RuleScope;
    projectId?: string;
    title?: string;
    tags?: string[];
    priority?: number;
  }
): Promise<{ queueId: string }> {
  const payload = buildRulePayload(operation);
  const op = ruleActionToQueueOp(operation.action);

  const result = await stateManager.enqueueUnified(
    'text',
    op,
    operation.projectId ?? TENANT_GLOBAL,
    RULES_COLLECTION,
    payload,
    PRIORITY_HIGH,
    'main',
    { source: 'mcp_rules_tool' }
  );

  if (result.status !== 'ok' || !result.data) {
    throw new Error(result.message ?? 'Failed to enqueue operation');
  }
  return { queueId: result.data.queueId };
}

export function upsertMirror(
  stateManager: SqliteStateManager,
  label: string,
  content: string,
  scope: RuleScope | null,
  tenantId: string | null
): void {
  const now = utcNow();
  stateManager.upsertRulesMirror({
    ruleId: label,
    ruleText: content,
    scope,
    tenantId,
    createdAt: now,
    updatedAt: now,
  });
}

export async function resolveProjectScopeId(
  scope: RuleScope,
  projectId: string | undefined,
  projectDetector: ProjectDetector
): Promise<{ resolvedProjectId: string | undefined; error?: RuleResponse }> {
  let resolvedProjectId = projectId;
  if (scope === 'project' && !resolvedProjectId) {
    const projectInfo = await projectDetector.getProjectInfo(process.cwd(), false);
    resolvedProjectId = projectInfo?.projectId;
  }
  if (scope === 'project' && !resolvedProjectId) {
    return {
      resolvedProjectId: undefined,
      error: {
        success: false,
        action: 'add',
        message:
          'Project-scoped rule requested but the current directory is not a registered project. ' +
          'Run `wqm project watch <path>` first, or pass `projectId` explicitly, or set `scope: "global"`.',
      },
    };
  }
  return { resolvedProjectId };
}

function buildAddMetadata(
  label: string,
  scope: RuleScope,
  resolvedProjectId: string | undefined,
  title: string | undefined,
  tags: string[] | undefined,
  priority: number | undefined
): Record<string, string> {
  const m: Record<string, string> = { scope, rule_type: 'behavioral', label };
  if (resolvedProjectId) m[FIELD_PROJECT_ID] = resolvedProjectId;
  if (title) m[FIELD_TITLE] = title;
  if (tags && tags.length > 0) m['tags'] = tags.join(',');
  if (priority !== undefined) m['priority'] = String(priority);
  return m;
}

function buildAddQueueOp(
  label: string,
  content: string,
  scope: RuleScope,
  resolvedProjectId: string | undefined,
  title: string | undefined,
  tags: string[] | undefined,
  priority: number | undefined
): Parameters<typeof queueRuleOperation>[1] {
  const op: Parameters<typeof queueRuleOperation>[1] = { action: 'add', label, content, scope };
  if (resolvedProjectId) op.projectId = resolvedProjectId;
  if (title) op.title = title;
  if (tags) op.tags = tags;
  if (priority !== undefined) op.priority = priority;
  return op;
}

export async function persistAddRule(
  daemonClient: DaemonClient,
  stateManager: SqliteStateManager,
  label: string,
  content: string,
  scope: RuleScope,
  resolvedProjectId: string | undefined,
  title: string | undefined,
  tags: string[] | undefined,
  priority: number | undefined
): Promise<RuleResponse> {
  const metadata = buildAddMetadata(label, scope, resolvedProjectId, title, tags, priority);
  try {
    const response = await daemonClient.ingestText({
      content,
      collection_basename: RULES_BASENAME,
      tenant_id: resolvedProjectId ?? TENANT_GLOBAL,
      document_id: randomUUID(),
      metadata,
    });
    if (response.success) {
      upsertMirror(stateManager, label, content, scope ?? null, resolvedProjectId ?? null);
      return { success: true, action: 'add', label, message: 'Rule added successfully' };
    }
  } catch (err: unknown) {
    if (!isConnectivityError(err)) throw err;
  }

  const queueResult = await queueRuleOperation(
    stateManager,
    buildAddQueueOp(label, content, scope, resolvedProjectId, title, tags, priority)
  );
  upsertMirror(stateManager, label, content, scope ?? null, resolvedProjectId ?? null);
  return {
    success: true,
    action: 'add',
    label,
    message: 'Rule queued for processing',
    fallback_mode: 'unified_queue',
    queue_id: queueResult.queueId,
  };
}

function buildUpdateMetadata(
  label: string,
  title: string | undefined,
  tags: string[] | undefined,
  priority: number | undefined
): Record<string, string> {
  const m: Record<string, string> = { label };
  if (title) m[FIELD_TITLE] = title;
  if (tags && tags.length > 0) m['tags'] = tags.join(',');
  if (priority !== undefined) m['priority'] = String(priority);
  return m;
}

function buildUpdateQueueOp(
  label: string,
  content: string,
  title: string | undefined,
  tags: string[] | undefined,
  priority: number | undefined
): Parameters<typeof queueRuleOperation>[1] {
  const op: Parameters<typeof queueRuleOperation>[1] = { action: 'update', label, content };
  if (title) op.title = title;
  if (tags) op.tags = tags;
  if (priority !== undefined) op.priority = priority;
  return op;
}

export async function persistUpdateRule(
  daemonClient: DaemonClient,
  stateManager: SqliteStateManager,
  label: string,
  content: string,
  title: string | undefined,
  tags: string[] | undefined,
  priority: number | undefined
): Promise<RuleResponse> {
  const metadata = buildUpdateMetadata(label, title, tags, priority);
  try {
    const response = await daemonClient.ingestText({
      content,
      collection_basename: RULES_BASENAME,
      tenant_id: TENANT_GLOBAL,
      document_id: label,
      metadata,
    });
    if (response.success) {
      upsertMirror(stateManager, label, content, null, null);
      return { success: true, action: 'update', label, message: 'Rule updated successfully' };
    }
  } catch (err: unknown) {
    if (!isConnectivityError(err)) throw err;
  }

  const queueResult = await queueRuleOperation(
    stateManager,
    buildUpdateQueueOp(label, content, title, tags, priority)
  );
  upsertMirror(stateManager, label, content, null, null);
  return {
    success: true,
    action: 'update',
    label,
    message: 'Rule update queued for processing',
    fallback_mode: 'unified_queue',
    queue_id: queueResult.queueId,
  };
}
