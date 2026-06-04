/**
 * Shared helpers for rules mutation operations.
 */
import { randomUUID } from 'node:crypto';
import { utcNow } from '../utils/timestamps.js';
import { PRIORITY_HIGH, FIELD_SOURCE_TYPE, FIELD_PROJECT_ID, FIELD_CONTENT, FIELD_TITLE, } from '../common/native-bridge.js';
import { RULES_BASENAME, RULES_COLLECTION } from './rules-types.js';
import { TENANT_GLOBAL } from '../constants/tenants.js';
export function isConnectivityError(err) {
    const msg = err instanceof Error ? err.message : String(err);
    const code = err?.code;
    return (code === 'UNAVAILABLE' ||
        code === 'DEADLINE_EXCEEDED' ||
        code === 'ECONNREFUSED' ||
        msg.includes('UNAVAILABLE') ||
        msg.includes('DEADLINE_EXCEEDED') ||
        msg.includes('ECONNREFUSED') ||
        msg.includes('connect ECONNREFUSED'));
}
function ruleActionToQueueOp(action) {
    if (action === 'update')
        return 'update';
    if (action === 'remove')
        return 'delete';
    return 'add';
}
function buildRulePayload(operation) {
    const payload = {
        label: operation.label,
        action: operation.action,
        [FIELD_SOURCE_TYPE]: 'rule',
    };
    if (operation.content)
        payload[FIELD_CONTENT] = operation.content;
    if (operation.scope)
        payload['scope'] = operation.scope;
    if (operation.projectId)
        payload[FIELD_PROJECT_ID] = operation.projectId;
    if (operation.title)
        payload[FIELD_TITLE] = operation.title;
    if (operation.tags)
        payload['tags'] = operation.tags;
    if (operation.priority !== undefined)
        payload['priority'] = operation.priority;
    return payload;
}
export async function queueRuleOperation(stateManager, operation) {
    const payload = buildRulePayload(operation);
    const op = ruleActionToQueueOp(operation.action);
    const result = await stateManager.enqueueUnified('text', op, operation.projectId ?? TENANT_GLOBAL, RULES_COLLECTION, payload, PRIORITY_HIGH, 'main', { source: 'mcp_rules_tool' });
    if (result.status !== 'ok' || !result.data) {
        throw new Error(result.message ?? 'Failed to enqueue operation');
    }
    return { queueId: result.data.queueId };
}
export function upsertMirror(stateManager, label, content, scope, tenantId) {
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
export async function resolveProjectScopeId(scope, projectId, projectDetector) {
    let resolvedProjectId = projectId;
    if (scope === 'project' && !resolvedProjectId) {
        const projectInfo = await projectDetector.getProjectInfo(process.cwd(), false);
        resolvedProjectId = projectInfo?.projectId;
    }
    if (scope === 'project' && !resolvedProjectId) {
        // Caller decides which action label to report — the spread on the
        // returned response can override `action` (see updateRule /
        // removeRule). Default to 'add' for backward compat with addRule.
        return {
            resolvedProjectId: undefined,
            error: {
                success: false,
                action: 'add',
                message: 'Project-scoped rule requested but the current directory is not a registered project. ' +
                    'Run `wqm project watch <path>` first, or pass `projectId` explicitly, or set `scope: "global"`.',
            },
        };
    }
    return { resolvedProjectId };
}
function buildAddMetadata(label, scope, resolvedProjectId, title, tags, priority) {
    const m = { scope, rule_type: 'behavioral', label };
    if (resolvedProjectId)
        m[FIELD_PROJECT_ID] = resolvedProjectId;
    if (title)
        m[FIELD_TITLE] = title;
    if (tags && tags.length > 0)
        m['tags'] = tags.join(',');
    if (priority !== undefined)
        m['priority'] = String(priority);
    return m;
}
function buildAddQueueOp(label, content, scope, resolvedProjectId, title, tags, priority) {
    const op = { action: 'add', label, content, scope };
    if (resolvedProjectId)
        op.projectId = resolvedProjectId;
    if (title)
        op.title = title;
    if (tags)
        op.tags = tags;
    if (priority !== undefined)
        op.priority = priority;
    return op;
}
export async function persistAddRule(daemonClient, stateManager, label, content, scope, resolvedProjectId, title, tags, priority) {
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
    }
    catch (err) {
        if (!isConnectivityError(err))
            throw err;
    }
    const queueResult = await queueRuleOperation(stateManager, buildAddQueueOp(label, content, scope, resolvedProjectId, title, tags, priority));
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
function buildUpdateMetadata(label, scope, resolvedProjectId, title, tags, priority) {
    const m = { label, scope };
    if (resolvedProjectId)
        m[FIELD_PROJECT_ID] = resolvedProjectId;
    if (title)
        m[FIELD_TITLE] = title;
    if (tags && tags.length > 0)
        m['tags'] = tags.join(',');
    if (priority !== undefined)
        m['priority'] = String(priority);
    return m;
}
function buildUpdateQueueOp(label, content, scope, resolvedProjectId, title, tags, priority) {
    const op = {
        action: 'update',
        label,
        content,
        scope,
    };
    if (resolvedProjectId)
        op.projectId = resolvedProjectId;
    if (title)
        op.title = title;
    if (tags)
        op.tags = tags;
    if (priority !== undefined)
        op.priority = priority;
    return op;
}
export async function persistUpdateRule(daemonClient, stateManager, label, content, scope, resolvedProjectId, title, tags, priority) {
    // F-015: pass the resolved project tenant (or TENANT_GLOBAL for
    // global rules) so the daemon's (label, tenant_id) match targets
    // the correct rule.
    const tenantId = resolvedProjectId ?? TENANT_GLOBAL;
    const metadata = buildUpdateMetadata(label, scope, resolvedProjectId, title, tags, priority);
    try {
        const response = await daemonClient.ingestText({
            content,
            collection_basename: RULES_BASENAME,
            tenant_id: tenantId,
            document_id: label,
            metadata,
        });
        if (response.success) {
            upsertMirror(stateManager, label, content, scope, resolvedProjectId ?? null);
            return { success: true, action: 'update', label, message: 'Rule updated successfully' };
        }
    }
    catch (err) {
        if (!isConnectivityError(err))
            throw err;
    }
    const queueResult = await queueRuleOperation(stateManager, buildUpdateQueueOp(label, content, scope, resolvedProjectId, title, tags, priority));
    upsertMirror(stateManager, label, content, scope, resolvedProjectId ?? null);
    return {
        success: true,
        action: 'update',
        label,
        message: 'Rule update queued for processing',
        fallback_mode: 'unified_queue',
        queue_id: queueResult.queueId,
    };
}
//# sourceMappingURL=rules-mutation-helpers.js.map