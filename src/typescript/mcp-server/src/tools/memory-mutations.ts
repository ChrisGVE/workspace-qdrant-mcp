/**
 * Memory mutation operations: add, update, remove rules.
 * Uses daemon gRPC with unified_queue fallback.
 */

import type { DaemonClient } from '../clients/daemon-client.js';
import type { SqliteStateManager } from '../clients/sqlite-state-manager.js';
import type { ProjectDetector } from '../utils/project-detector.js';
import { utcNow } from '../utils/timestamps.js';
import { PRIORITY_HIGH } from '../common/native-bridge.js';
import type { MemoryAction, MemoryOptions, MemoryResponse, MemoryScope } from './memory-types.js';
import { MEMORY_BASENAME, MEMORY_COLLECTION } from './memory-types.js';

/** Queue a memory operation for daemon processing via unified_queue. */
function queueMemoryOperation(
  stateManager: SqliteStateManager,
  operation: {
    action: MemoryAction;
    label?: string;
    content?: string;
    scope?: MemoryScope;
    projectId?: string;
    title?: string;
    tags?: string[];
    priority?: number;
  },
): { queueId: string } {
  const payload: Record<string, unknown> = {
    label: operation.label,
    action: operation.action,
    source_type: 'memory_rule',
  };
  if (operation.content) payload['content'] = operation.content;
  if (operation.scope) payload['scope'] = operation.scope;
  if (operation.projectId) payload['project_id'] = operation.projectId;
  if (operation.title) payload['title'] = operation.title;
  if (operation.tags) payload['tags'] = operation.tags;
  if (operation.priority !== undefined) payload['priority'] = operation.priority;

  let op: 'add' | 'update' | 'delete';
  switch (operation.action) {
    case 'add': op = 'add'; break;
    case 'update': op = 'update'; break;
    case 'remove': op = 'delete'; break;
    default: op = 'add';
  }

  const result = stateManager.enqueueUnified(
    'text', op, operation.projectId ?? 'global',
    MEMORY_COLLECTION, payload, PRIORITY_HIGH, 'main',
    { source: 'mcp_memory_tool' },
  );

  if (result.status !== 'ok' || !result.data) {
    throw new Error(result.message ?? 'Failed to enqueue operation');
  }
  return { queueId: result.data.queueId };
}

/** Add a new memory rule. */
export async function addRule(
  daemonClient: DaemonClient,
  stateManager: SqliteStateManager,
  projectDetector: ProjectDetector,
  options: MemoryOptions,
): Promise<MemoryResponse> {
  const { content, scope = 'project', projectId, title, tags, priority } = options;

  if (!content?.trim()) {
    return { success: false, action: 'add', message: 'Content is required for adding a rule' };
  }
  if (!options.label?.trim()) {
    return {
      success: false, action: 'add',
      message: 'Label is required for adding a rule (max 15 chars, format: word-word-word, e.g. "prefer-uv", "use-pytest")',
    };
  }

  let resolvedProjectId = projectId;
  if (scope === 'project' && !resolvedProjectId) {
    const cwd = process.cwd();
    const projectInfo = await projectDetector.getProjectInfo(cwd, false);
    resolvedProjectId = projectInfo?.projectId;
  }

  const label = options.label.trim();
  const metadata: Record<string, string> = { scope, rule_type: 'behavioral', label };
  if (resolvedProjectId) metadata['project_id'] = resolvedProjectId;
  if (title) metadata['title'] = title;
  if (tags && tags.length > 0) metadata['tags'] = tags.join(',');
  if (priority !== undefined) metadata['priority'] = String(priority);

  // Try daemon first
  try {
    const response = await daemonClient.ingestText({
      content, collection_basename: MEMORY_BASENAME,
      tenant_id: resolvedProjectId ?? 'global', document_id: label, metadata,
    });

    if (response.success) {
      const now = utcNow();
      stateManager.upsertMemoryMirror({
        memoryId: response.document_id ?? label, ruleText: content,
        scope: scope ?? null, tenantId: resolvedProjectId ?? null,
        createdAt: now, updatedAt: now,
      });
      return { success: true, action: 'add', label: response.document_id, message: 'Rule added successfully' };
    }
  } catch {
    // Daemon unavailable - fall back to queue
  }

  // Fallback: queue the operation
  const queueOp: {
    action: MemoryAction; label: string; content: string; scope: MemoryScope;
    projectId?: string; title?: string; tags?: string[]; priority?: number;
  } = { action: 'add', label, content, scope };
  if (resolvedProjectId) queueOp.projectId = resolvedProjectId;
  if (title) queueOp.title = title;
  if (tags) queueOp.tags = tags;
  if (priority !== undefined) queueOp.priority = priority;

  const queueResult = queueMemoryOperation(stateManager, queueOp);

  const now = utcNow();
  stateManager.upsertMemoryMirror({
    memoryId: label, ruleText: content,
    scope: scope ?? null, tenantId: resolvedProjectId ?? null,
    createdAt: now, updatedAt: now,
  });

  return {
    success: true, action: 'add', label,
    message: 'Rule queued for processing', fallback_mode: 'unified_queue',
    queue_id: queueResult.queueId,
  };
}

/** Update an existing memory rule. */
export async function updateRule(
  daemonClient: DaemonClient,
  stateManager: SqliteStateManager,
  options: MemoryOptions,
): Promise<MemoryResponse> {
  const { label, content, title, tags, priority } = options;

  if (!label) {
    return { success: false, action: 'update', message: 'Label is required for updating' };
  }
  if (!content?.trim()) {
    return { success: false, action: 'update', message: 'Content is required for updating a rule' };
  }

  const metadata: Record<string, string> = { label };
  if (title) metadata['title'] = title;
  if (tags && tags.length > 0) metadata['tags'] = tags.join(',');
  if (priority !== undefined) metadata['priority'] = String(priority);

  try {
    const response = await daemonClient.ingestText({
      content, collection_basename: MEMORY_BASENAME,
      tenant_id: 'global', document_id: label, metadata,
    });

    if (response.success) {
      stateManager.upsertMemoryMirror({
        memoryId: label, ruleText: content, scope: null, tenantId: null,
        createdAt: utcNow(), updatedAt: utcNow(),
      });
      return { success: true, action: 'update', label, message: 'Rule updated successfully' };
    }
  } catch {
    // Daemon unavailable - fall back to queue
  }

  const updateOp: {
    action: MemoryAction; label: string; content: string;
    title?: string; tags?: string[]; priority?: number;
  } = { action: 'update', label, content };
  if (title) updateOp.title = title;
  if (tags) updateOp.tags = tags;
  if (priority !== undefined) updateOp.priority = priority;

  const queueResult = queueMemoryOperation(stateManager, updateOp);

  stateManager.upsertMemoryMirror({
    memoryId: label, ruleText: content, scope: null, tenantId: null,
    createdAt: utcNow(), updatedAt: utcNow(),
  });

  return {
    success: true, action: 'update', label,
    message: 'Rule update queued for processing', fallback_mode: 'unified_queue',
    queue_id: queueResult.queueId,
  };
}

/** Remove a memory rule. */
export function removeRule(
  stateManager: SqliteStateManager,
  options: MemoryOptions,
): MemoryResponse {
  const { label } = options;
  if (!label) {
    return { success: false, action: 'remove', message: 'Label is required for removal' };
  }

  const queueResult = queueMemoryOperation(stateManager, { action: 'remove', label });
  stateManager.deleteMemoryMirror(label);

  return {
    success: true, action: 'remove', label,
    message: 'Rule removal queued for processing', fallback_mode: 'unified_queue',
    queue_id: queueResult.queueId,
  };
}
