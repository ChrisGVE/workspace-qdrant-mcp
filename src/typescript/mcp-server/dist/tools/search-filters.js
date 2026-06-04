/**
 * Search filter construction and collection determination.
 */
import { PROJECTS_COLLECTION, LIBRARIES_COLLECTION, SCRATCHPAD_COLLECTION, } from './search-types.js';
import { FIELD_TENANT_ID, FIELD_BASE_POINT, FIELD_BRANCHES, FIELD_FILE_TYPE, FIELD_LIBRARY_NAME, FIELD_LIBRARY_PATH, FIELD_CONCEPT_TAGS, FIELD_TAGS, FIELD_FILE_PATH, FIELD_DELETED, } from '../common/native-bridge.js';
/**
 * Extract the deterministic path prefix from a glob pattern.
 * Returns everything before the first glob metacharacter (* ? [ {).
 * Example: "src/**\/*.rs" → "src/", "**\/*.rs" → ""
 */
export function extractGlobPrefix(glob) {
    const metaChars = /[*?[{]/;
    const match = metaChars.exec(glob);
    if (!match) {
        // No glob chars — the whole string is a literal path
        return glob;
    }
    // Take everything up to the first metachar,
    // then trim to the last path separator for a clean prefix
    const beforeMeta = glob.slice(0, match.index);
    const lastSlash = beforeMeta.lastIndexOf('/');
    return lastSlash >= 0 ? beforeMeta.slice(0, lastSlash + 1) : '';
}
/**
 * Determine which collections to search based on scope.
 */
export function determineCollections(collection, scope, includeLibraries) {
    if (collection)
        return [collection];
    switch (scope) {
        case 'project':
        case 'group':
            return includeLibraries ? [PROJECTS_COLLECTION, LIBRARIES_COLLECTION] : [PROJECTS_COLLECTION];
        case 'all':
            return [PROJECTS_COLLECTION, LIBRARIES_COLLECTION, SCRATCHPAD_COLLECTION];
        default:
            return [PROJECTS_COLLECTION];
    }
}
// ── Filter condition builders ─────────────────────────────────────────────
function buildProjectCondition(params) {
    if (params.scope === 'group') {
        if (!params.groupTenantIds || params.groupTenantIds.length === 0) {
            throw new Error('Group scope requires non-empty tenant ID set');
        }
        return { key: FIELD_TENANT_ID, match: { any: params.groupTenantIds } };
    }
    if (params.scope !== 'project' || !params.projectId)
        return null;
    return { key: FIELD_TENANT_ID, match: { value: params.projectId } };
}
function buildBasePointCondition(params) {
    if (!params.basePoints || params.basePoints.length === 0)
        return null;
    return { key: FIELD_BASE_POINT, match: { any: params.basePoints } };
}
function buildBranchCondition(params) {
    if (!params.branch || params.branch === '*')
        return null;
    return { key: FIELD_BRANCHES, match: { value: params.branch } };
}
function buildFileTypeCondition(params) {
    if (!params.fileType)
        return null;
    return { key: FIELD_FILE_TYPE, match: { value: params.fileType } };
}
function buildLibraryNameCondition(params) {
    if (params.collection !== LIBRARIES_COLLECTION || !params.libraryName)
        return null;
    return { key: FIELD_LIBRARY_NAME, match: { value: params.libraryName } };
}
function buildLibraryPathCondition(params) {
    if (params.collection !== LIBRARIES_COLLECTION || !params.libraryPath)
        return null;
    return { key: FIELD_LIBRARY_PATH, match: { text: params.libraryPath } };
}
function buildTagConditions(params) {
    const conditions = [];
    if (params.tag) {
        // Match against both concept_tags (keyword-extracted) and tags (Tier 1 metadata)
        conditions.push({
            should: [
                { key: FIELD_CONCEPT_TAGS, match: { value: params.tag } },
                { key: FIELD_TAGS, match: { value: params.tag } },
            ],
        });
    }
    if (params.tags && params.tags.length > 0) {
        // For multi-tag filter, match any tag in either concept_tags or tags fields
        const tagShouldConditions = params.tags.flatMap((t) => [
            { key: FIELD_CONCEPT_TAGS, match: { value: t } },
            { key: FIELD_TAGS, match: { value: t } },
        ]);
        conditions.push({ should: tagShouldConditions });
    }
    return conditions;
}
function buildComponentCondition(params) {
    if (!params.component)
        return null;
    return {
        should: [
            { key: 'component_id', match: { value: params.component } },
            { key: 'component_id', match: { text: `${params.component}.` } },
        ],
    };
}
function buildPathGlobCondition(params) {
    if (!params.pathGlob)
        return null;
    const prefix = extractGlobPrefix(params.pathGlob);
    if (!prefix)
        return null;
    return { key: FIELD_FILE_PATH, match: { text: prefix } };
}
function buildMustConditions(params) {
    const conditions = [];
    const projectCond = buildProjectCondition(params);
    if (projectCond)
        conditions.push(projectCond);
    const basePointCond = buildBasePointCondition(params);
    if (basePointCond)
        conditions.push(basePointCond);
    const branchCond = buildBranchCondition(params);
    if (branchCond)
        conditions.push(branchCond);
    const fileTypeCond = buildFileTypeCondition(params);
    if (fileTypeCond)
        conditions.push(fileTypeCond);
    const libNameCond = buildLibraryNameCondition(params);
    if (libNameCond)
        conditions.push(libNameCond);
    const libPathCond = buildLibraryPathCondition(params);
    if (libPathCond)
        conditions.push(libPathCond);
    conditions.push(...buildTagConditions(params));
    const componentCond = buildComponentCondition(params);
    if (componentCond)
        conditions.push(componentCond);
    const pathGlobCond = buildPathGlobCondition(params);
    if (pathGlobCond)
        conditions.push(pathGlobCond);
    return conditions;
}
function buildMustNotConditions(params) {
    if (params.collection !== LIBRARIES_COLLECTION)
        return [];
    return [{ key: FIELD_DELETED, match: { value: true } }];
}
/**
 * Build Qdrant filter based on search parameters.
 */
export function buildFilter(params) {
    const mustConditions = buildMustConditions(params);
    const mustNotConditions = buildMustNotConditions(params);
    if (mustConditions.length === 0 && mustNotConditions.length === 0) {
        return null;
    }
    const filter = {};
    if (mustConditions.length > 0)
        filter.must = mustConditions;
    if (mustNotConditions.length > 0)
        filter.must_not = mustNotConditions;
    return filter;
}
//# sourceMappingURL=search-filters.js.map