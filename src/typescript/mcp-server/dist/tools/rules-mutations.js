/**
 * Rules mutation operations: add, update, remove rules.
 * Uses daemon gRPC with unified_queue fallback.
 */
import { resolveProjectScopeId, persistAddRule, persistUpdateRule, queueRuleOperation, } from './rules-mutation-helpers.js';
export async function addRule(daemonClient, stateManager, projectDetector, options) {
    const { content, scope = 'project', projectId, title, tags, priority } = options;
    if (!content?.trim()) {
        return { success: false, action: 'add', message: 'Content is required for adding a rule' };
    }
    if (!options.label?.trim()) {
        return {
            success: false,
            action: 'add',
            message: 'Label is required for adding a rule (max 15 chars, format: word-word-word, e.g. "prefer-uv", "use-pytest")',
        };
    }
    const { resolvedProjectId, error } = await resolveProjectScopeId(scope, projectId, projectDetector);
    if (error)
        return error;
    const label = options.label.trim();
    return persistAddRule(daemonClient, stateManager, label, content, scope, resolvedProjectId, title, tags, priority);
}
export async function updateRule(daemonClient, stateManager, projectDetector, options) {
    const { label, content, title, tags, priority, scope = 'project', projectId } = options;
    if (!label) {
        return { success: false, action: 'update', message: 'Label is required for updating' };
    }
    if (!content?.trim()) {
        return { success: false, action: 'update', message: 'Content is required for updating a rule' };
    }
    // F-015: project-scope updates must carry the project tenant_id so
    // the daemon's (label, tenant_id) match targets the right rule.
    // Pre-fix the call hardcoded TENANT_GLOBAL.
    const { resolvedProjectId, error } = await resolveProjectScopeId(scope, projectId, projectDetector);
    if (error)
        return { ...error, action: 'update' };
    return persistUpdateRule(daemonClient, stateManager, label, content, scope, resolvedProjectId, title, tags, priority);
}
export async function removeRule(stateManager, projectDetector, options) {
    const { label, scope = 'project', projectId } = options;
    if (!label) {
        return { success: false, action: 'remove', message: 'Label is required for removal' };
    }
    // F-015: same-label rules can exist in two projects. Resolve the
    // project tenant and pass it through so the daemon can scope the
    // Qdrant delete by (label, tenant_id) — not by label alone.
    const { resolvedProjectId, error } = await resolveProjectScopeId(scope, projectId, projectDetector);
    if (error)
        return { ...error, action: 'remove' };
    const queueResult = await queueRuleOperation(stateManager, {
        action: 'remove',
        label,
        scope,
        ...(resolvedProjectId !== undefined ? { projectId: resolvedProjectId } : {}),
    });
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
//# sourceMappingURL=rules-mutations.js.map