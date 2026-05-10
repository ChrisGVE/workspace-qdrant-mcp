/**
 * Rules mutation operations: add, update, remove rules.
 * Uses daemon gRPC with unified_queue fallback.
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
import type { RuleAction, RuleOptions, RuleResponse, RuleScope } from './rules-types.js';
import { RULES_BASENAME, RULES_COLLECTION } from './rules-types.js';
import { TENANT_GLOBAL } from '../constants/tenants.js';

/** Check if an error is a connectivity error (daemon unavailable). */
function isConnectivityError(err: unknown): boolean {
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

/** Map a RuleAction to a queue operation string. */
function ruleActionToQueueOp(action: RuleAction): 'add' | 'update' | 'delete' {
  if (action === 'update') return 'update';
  if (action === 'remove') return 'delete';
  return 'add';
}

/** Build the queue payload for a rule operation. */
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

/** Queue a rule operation for daemon processing via unified_queue. */
async function queueRuleOperation(
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

/** Upsert a rule into the local SQLite mirror. */
function upsertMirror(
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

/** Resolve the project ID for a project-scoped rule. Returns an error response if unresolvable. */
async function resolveProjectScopeId(
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

/** Build metadata for a new rule ingest. */
function buildAddRuleMetadata(
  label: string,
  scope: RuleScope,
  resolvedProjectId: string | undefined,
  title: string | undefined,
  tags: string[] | undefined,
  priority: number | undefined
): Record<string, string> {
  const metadata: Record<string, string> = { scope, rule_type: 'behavioral', label };
  if (resolvedProjectId) metadata[FIELD_PROJECT_ID] = resolvedProjectId;
  if (title) metadata[FIELD_TITLE] = title;
  if (tags && tags.length > 0) metadata['tags'] = tags.join(',');
  if (priority !== undefined) metadata['priority'] = String(priority);
  return metadata;
}

/** Build the queue operation object for adding a rule. */
function buildAddRuleQueueOp(
  label: string,
  content: string,
  scope: RuleScope,
  resolvedProjectId: string | undefined,
  title: string | undefined,
  tags: string[] | undefined,
  priority: number | undefined
): {
  action: RuleAction;
  label: string;
  content: string;
  scope: RuleScope;
  projectId?: string;
  title?: string;
  tags?: string[];
  priority?: number;
} {
  const op: {
    action: RuleAction;
    label: string;
    content: string;
    scope: RuleScope;
    projectId?: string;
    title?: string;
    tags?: string[];
    priority?: number;
  } = { action: 'add', label, content, scope };
  if (resolvedProjectId) op.projectId = resolvedProjectId;
  if (title) op.title = title;
  if (tags) op.tags = tags;
  if (priority !== undefined) op.priority = priority;
  return op;
}

/** Add a new rule. */
export async function addRule(
  daemonClient: DaemonClient,
  stateManager: SqliteStateManager,
  projectDetector: ProjectDetector,
  options: RuleOptions
): Promise<RuleResponse> {
  const { content, scope = 'project', projectId, title, tags, priority } = options;

  if (!content?.trim()) {
    return { success: false, action: 'add', message: 'Content is required for adding a rule' };
  }
  if (!options.label?.trim()) {
    return {
      success: false,
      action: 'add',
      message:
        'Label is required for adding a rule (max 15 chars, format: word-word-word, e.g. "prefer-uv", "use-pytest")',
    };
  }

  const { resolvedProjectId, error } = await resolveProjectScopeId(
    scope,
    projectId,
    projectDetector
  );
  if (error) return error;

  const label = options.label.trim();
  const metadata = buildAddRuleMetadata(label, scope, resolvedProjectId, title, tags, priority);

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

  const queueOp = buildAddRuleQueueOp(
    label,
    content,
    scope,
    resolvedProjectId,
    title,
    tags,
    priority
  );
  const queueResult = await queueRuleOperation(stateManager, queueOp);
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

/** Build metadata for a rule update. */
function buildUpdateRuleMetadata(
  label: string,
  title: string | undefined,
  tags: string[] | undefined,
  priority: number | undefined
): Record<string, string> {
  const metadata: Record<string, string> = { label };
  if (title) metadata[FIELD_TITLE] = title;
  if (tags && tags.length > 0) metadata['tags'] = tags.join(',');
  if (priority !== undefined) metadata['priority'] = String(priority);
  return metadata;
}

/** Build the queue operation object for updating a rule. */
function buildUpdateRuleQueueOp(
  label: string,
  content: string,
  title: string | undefined,
  tags: string[] | undefined,
  priority: number | undefined
): {
  action: RuleAction;
  label: string;
  content: string;
  title?: string;
  tags?: string[];
  priority?: number;
} {
  const op: {
    action: RuleAction;
    label: string;
    content: string;
    title?: string;
    tags?: string[];
    priority?: number;
  } = { action: 'update', label, content };
  if (title) op.title = title;
  if (tags) op.tags = tags;
  if (priority !== undefined) op.priority = priority;
  return op;
}

/** Update an existing rule. */
export async function updateRule(
  daemonClient: DaemonClient,
  stateManager: SqliteStateManager,
  options: RuleOptions
): Promise<RuleResponse> {
  const { label, content, title, tags, priority } = options;

  if (!label) {
    return { success: false, action: 'update', message: 'Label is required for updating' };
  }
  if (!content?.trim()) {
    return { success: false, action: 'update', message: 'Content is required for updating a rule' };
  }

  const metadata = buildUpdateRuleMetadata(label, title, tags, priority);

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
    buildUpdateRuleQueueOp(label, content, title, tags, priority)
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

/** Remove a rule. */
export async function removeRule(
  stateManager: SqliteStateManager,
  options: RuleOptions
): Promise<RuleResponse> {
  const { label } = options;
  if (!label) {
    return { success: false, action: 'remove', message: 'Label is required for removal' };
  }

  const queueResult = await queueRuleOperation(stateManager, { action: 'remove', label });
  stateManager.deleteRulesMirror(label);

  return {
    success: true,
    action: 'remove',
    label,
    message: 'Rule removal queued for processing',
    fallback_mode: 'unified_queue',
    queue_id: queueResult.queueId,
  };
}
