/**
 * Rules mutation operations: add, update, remove rules.
 * Uses daemon gRPC with unified_queue fallback.
 */

import type { DaemonClient } from '../clients/daemon-client.js';
import type { SqliteStateManager } from '../clients/sqlite-state-manager.js';
import type { ProjectDetector } from '../utils/project-detector.js';
import type { RuleOptions, RuleResponse } from './rules-types.js';
import {
  resolveProjectScopeId,
  persistAddRule,
  persistUpdateRule,
  queueRuleOperation,
} from './rules-mutation-helpers.js';

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
  return persistAddRule(
    daemonClient,
    stateManager,
    label,
    content,
    scope,
    resolvedProjectId,
    title,
    tags,
    priority
  );
}

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

  return persistUpdateRule(daemonClient, stateManager, label, content, title, tags, priority);
}

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
