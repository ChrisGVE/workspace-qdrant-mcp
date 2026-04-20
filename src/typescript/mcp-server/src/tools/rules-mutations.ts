/**
 * Rules mutation operations: add, update, remove rules.
 * Uses daemon gRPC with unified_queue fallback.
 */

import type { DaemonClient } from '../clients/daemon-client.js';
import type { SqliteStateManager } from '../clients/sqlite-state-manager.js';
import type { ProjectDetector } from '../utils/project-detector.js';
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

  let op: 'add' | 'update' | 'delete';
  switch (operation.action) {
    case 'add':
      op = 'add';
      break;
    case 'update':
      op = 'update';
      break;
    case 'remove':
      op = 'delete';
      break;
    default:
      op = 'add';
  }

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

  let resolvedProjectId = projectId;
  if (scope === 'project' && !resolvedProjectId) {
    const cwd = process.cwd();
    const projectInfo = await projectDetector.getProjectInfo(cwd, false);
    resolvedProjectId = projectInfo?.projectId;
  }

  // Don't silently fall back to the global tenant when a project rule was
  // requested but the project isn't tracked — the rule would end up
  // misfiled and the caller would never know. Tell them instead.
  if (scope === 'project' && !resolvedProjectId) {
    return {
      success: false,
      action: 'add',
      message:
        'Project-scoped rule requested but the current directory is not a registered project. ' +
        'Run `wqm project watch <path>` first, or pass `projectId` explicitly, or set `scope: "global"`.',
    };
  }

  const label = options.label.trim();
  const metadata: Record<string, string> = { scope, rule_type: 'behavioral', label };
  if (resolvedProjectId) metadata[FIELD_PROJECT_ID] = resolvedProjectId;
  if (title) metadata[FIELD_TITLE] = title;
  if (tags && tags.length > 0) metadata['tags'] = tags.join(',');
  if (priority !== undefined) metadata['priority'] = String(priority);

  // Try daemon first
  try {
    const response = await daemonClient.ingestText({
      content,
      collection_basename: RULES_BASENAME,
      tenant_id: resolvedProjectId ?? TENANT_GLOBAL,
      document_id: label,
      metadata,
    });

    if (response.success) {
      const now = utcNow();
      stateManager.upsertRulesMirror({
        ruleId: response.document_id ?? label,
        ruleText: content,
        scope: scope ?? null,
        tenantId: resolvedProjectId ?? null,
        createdAt: now,
        updatedAt: now,
      });
      return {
        success: true,
        action: 'add',
        label: response.document_id,
        message: 'Rule added successfully',
      };
    }
  } catch (err: unknown) {
    // Only fall back to queue for connectivity errors; rethrow others
    if (!isConnectivityError(err)) throw err;
  }

  // Fallback: queue the operation
  const queueOp: {
    action: RuleAction;
    label: string;
    content: string;
    scope: RuleScope;
    projectId?: string;
    title?: string;
    tags?: string[];
    priority?: number;
  } = { action: 'add', label, content, scope };
  if (resolvedProjectId) queueOp.projectId = resolvedProjectId;
  if (title) queueOp.title = title;
  if (tags) queueOp.tags = tags;
  if (priority !== undefined) queueOp.priority = priority;

  const queueResult = await queueRuleOperation(stateManager, queueOp);

  const now = utcNow();
  stateManager.upsertRulesMirror({
    ruleId: label,
    ruleText: content,
    scope: scope ?? null,
    tenantId: resolvedProjectId ?? null,
    createdAt: now,
    updatedAt: now,
  });

  return {
    success: true,
    action: 'add',
    label,
    message: 'Rule queued for processing',
    fallback_mode: 'unified_queue',
    queue_id: queueResult.queueId,
  };
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

  const metadata: Record<string, string> = { label };
  if (title) metadata[FIELD_TITLE] = title;
  if (tags && tags.length > 0) metadata['tags'] = tags.join(',');
  if (priority !== undefined) metadata['priority'] = String(priority);

  try {
    const response = await daemonClient.ingestText({
      content,
      collection_basename: RULES_BASENAME,
      tenant_id: TENANT_GLOBAL,
      document_id: label,
      metadata,
    });

    if (response.success) {
      stateManager.upsertRulesMirror({
        ruleId: label,
        ruleText: content,
        scope: null,
        tenantId: null,
        createdAt: utcNow(),
        updatedAt: utcNow(),
      });
      return { success: true, action: 'update', label, message: 'Rule updated successfully' };
    }
  } catch (err: unknown) {
    // Only fall back to queue for connectivity errors; rethrow others
    if (!isConnectivityError(err)) throw err;
  }

  const updateOp: {
    action: RuleAction;
    label: string;
    content: string;
    title?: string;
    tags?: string[];
    priority?: number;
  } = { action: 'update', label, content };
  if (title) updateOp.title = title;
  if (tags) updateOp.tags = tags;
  if (priority !== undefined) updateOp.priority = priority;

  const queueResult = await queueRuleOperation(stateManager, updateOp);

  stateManager.upsertRulesMirror({
    ruleId: label,
    ruleText: content,
    scope: null,
    tenantId: null,
    createdAt: utcNow(),
    updatedAt: utcNow(),
  });

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
