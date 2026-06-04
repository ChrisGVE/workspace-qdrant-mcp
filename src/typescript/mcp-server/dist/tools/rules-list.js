/**
 * Rules list operation — query rules by scope from Qdrant with mirror fallback.
 */
import { RULES_COLLECTION } from './rules-types.js';
import { FIELD_PROJECT_ID, FIELD_CONTENT, FIELD_TITLE } from '../common/native-bridge.js';
import { TENANT_GLOBAL } from '../constants/tenants.js';
/** Build Qdrant filter for list query based on scope. */
function buildListFilter(scope, projectId) {
    const mustConditions = [];
    if (scope === TENANT_GLOBAL) {
        mustConditions.push({ key: 'scope', match: { value: TENANT_GLOBAL } });
    }
    else if (scope === 'project' && projectId) {
        mustConditions.push({ key: 'scope', match: { value: 'project' } });
        mustConditions.push({ key: FIELD_PROJECT_ID, match: { value: projectId } });
    }
    return mustConditions.length > 0 ? { must: mustConditions } : undefined;
}
/** Map a Qdrant point payload to a Rule object. */
function pointToRule(point) {
    const rule = {
        id: String(point.id),
        content: point.payload?.[FIELD_CONTENT] ?? '',
        scope: point.payload?.['scope'] ?? TENANT_GLOBAL,
    };
    const label = point.payload?.['label'];
    if (label)
        rule.label = label;
    const pid = point.payload?.[FIELD_PROJECT_ID];
    if (pid)
        rule.projectId = pid;
    const title = point.payload?.[FIELD_TITLE];
    if (title)
        rule.title = title;
    const tagsStr = point.payload?.['tags'];
    if (tagsStr)
        rule.tags = tagsStr.split(',');
    const priorityRaw = point.payload?.['priority'];
    if (priorityRaw !== undefined && priorityRaw !== null)
        rule.priority = Number(priorityRaw);
    const createdAt = point.payload?.['created_at'];
    if (createdAt)
        rule.createdAt = createdAt;
    const updatedAt = point.payload?.['updated_at'];
    if (updatedAt)
        rule.updatedAt = updatedAt;
    return rule;
}
/** Build a scroll request for the rules collection. */
function buildScrollRequest(limit, filter) {
    const req = {
        limit,
        with_payload: true,
    };
    if (filter)
        req.filter = filter;
    return req;
}
/** Attempt to read rules from the local mirror as fallback. */
function readRulesFromMirror(stateManager, scope, resolvedProjectId, limit) {
    try {
        const mirrorRows = stateManager.listRulesMirror(scope, resolvedProjectId, limit);
        if (mirrorRows.length === 0)
            return null;
        const rules = mirrorRows.map((row) => {
            const rule = {
                id: row.ruleId,
                content: row.ruleText,
                scope: row.scope ?? TENANT_GLOBAL,
                createdAt: row.createdAt,
                updatedAt: row.updatedAt,
            };
            if (row.tenantId)
                rule.projectId = row.tenantId;
            return rule;
        });
        return {
            success: true,
            action: 'list',
            rules,
            message: `Found ${rules.length} rule(s) from local mirror (Qdrant unavailable)`,
        };
    }
    catch {
        return null;
    }
}
/** List rules by scope from Qdrant, with rules_mirror fallback. */
export async function listRules(qdrantClient, stateManager, projectDetector, options) {
    const { scope = 'project', projectId, limit = 50 } = options;
    let resolvedProjectId = projectId;
    if (scope === 'project' && !resolvedProjectId) {
        const projectInfo = await projectDetector.getProjectInfo(process.cwd(), false);
        resolvedProjectId = projectInfo?.projectId;
    }
    try {
        const filter = buildListFilter(scope, resolvedProjectId);
        const scrollResult = await qdrantClient.scroll(RULES_COLLECTION, buildScrollRequest(limit, filter));
        const rules = scrollResult.points.map(pointToRule);
        return { success: true, action: 'list', rules, message: `Found ${rules.length} rule(s)` };
    }
    catch (error) {
        const mirror = readRulesFromMirror(stateManager, scope, resolvedProjectId, limit);
        if (mirror)
            return mirror;
        return {
            success: false,
            action: 'list',
            rules: [],
            message: `Failed to list rules: ${error instanceof Error ? error.message : 'unknown error'}`,
        };
    }
}
//# sourceMappingURL=rules-list.js.map