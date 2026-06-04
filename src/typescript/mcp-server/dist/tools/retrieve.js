/**
 * Retrieve tool — direct document access from Qdrant collections.
 *
 * - retrieve-types.ts: Types, constants, helpers
 * - retrieve.ts (this): RetrieveTool class with byId and byFilter operations
 *
 * Tenant isolation invariants (F-002 / F-011):
 *
 * - `projects` collection accesses MUST resolve to a `tenant_id` before any
 *   Qdrant call. `retrieveById` verifies the returned point's `tenant_id`
 *   matches the caller; mismatches return an empty not-found response.
 * - `libraries` collection accesses MUST resolve to a `library_name` (or
 *   `tenant_id`, since the library collection keys both fields). The
 *   project detector is never used as a fall-back for libraries.
 * - `rules` is intentionally mixed-tenancy; no scope verification is
 *   performed for `retrieveById` against rules.
 * - Project-scope retrieve without a resolvable tenant returns an empty
 *   error response and does NOT scroll Qdrant (no broad reads).
 */
import { QdrantClient } from '@qdrant/js-client-rest';
import { FIELD_CONTENT, FIELD_TENANT_ID, FIELD_LIBRARY_NAME } from '../common/native-bridge.js';
import { getCollectionName, extractMetadata } from './retrieve-types.js';
/** Returned when the caller passes scope but it cannot be resolved. */
function unresolvedTenantResponse(collection) {
    return {
        success: false,
        documents: [],
        total: 0,
        hasMore: false,
        message: `Cannot retrieve from "${collection}" without a resolvable scope. ` +
            'Pass `projectId` (for projects) or `libraryName` (for libraries), ' +
            'or run from a registered project directory.',
    };
}
/**
 * Verify a retrieved point matches the caller's scope.
 * Returns `true` when the point belongs to the resolved tenant/library
 * (or when the collection is rules, which is intentionally mixed).
 */
function payloadMatchesScope(payload, collection, resolvedProjectId, libraryName) {
    if (!payload)
        return false;
    switch (collection) {
        case 'projects': {
            if (!resolvedProjectId)
                return false;
            return payload[FIELD_TENANT_ID] === resolvedProjectId;
        }
        case 'libraries': {
            if (!libraryName)
                return false;
            // The libraries collection stores `library_name` on every point;
            // some legacy paths also key by `tenant_id`. Accept either match.
            return (payload[FIELD_LIBRARY_NAME] === libraryName || payload[FIELD_TENANT_ID] === libraryName);
        }
        case 'rules':
            // Rules are explicitly mixed-tenancy. Direct-by-ID lookup is
            // therefore not gated on caller scope.
            return true;
        case 'scratchpad':
            // Scratchpad is project-scoped; verify tenant_id matches.
            if (!resolvedProjectId)
                return false;
            return payload[FIELD_TENANT_ID] === resolvedProjectId;
        default:
            return false;
    }
}
export class RetrieveTool {
    qdrantClient;
    projectDetector;
    constructor(config, projectDetector) {
        const clientConfig = {
            url: config.qdrantUrl,
            timeout: config.qdrantTimeout ?? 5000,
        };
        if (config.qdrantApiKey)
            clientConfig.apiKey = config.qdrantApiKey;
        this.qdrantClient = new QdrantClient(clientConfig);
        this.projectDetector = projectDetector;
    }
    async retrieve(options) {
        const { documentId, collection = 'projects', filter, limit = 10, offset = 0, projectId, libraryName, } = options;
        const collectionName = getCollectionName(collection);
        // F-002 / F-011: resolve the tenant context up front so that BOTH
        // by-id verification AND by-filter scoping share the same answer.
        const resolvedProjectId = collection === 'projects' || collection === 'scratchpad'
            ? (projectId ?? (await this.resolveProjectId()))
            : undefined;
        if (documentId) {
            return this.retrieveById(collectionName, collection, documentId, resolvedProjectId, libraryName);
        }
        // F-011: project-scope retrieve without a resolved tenant MUST refuse
        // to scroll. Same rule for libraries when no library_name is given.
        if (collection === 'projects' && !resolvedProjectId) {
            return unresolvedTenantResponse('projects');
        }
        if (collection === 'scratchpad' && !resolvedProjectId) {
            return unresolvedTenantResponse('scratchpad');
        }
        if (collection === 'libraries' && !libraryName) {
            return unresolvedTenantResponse('libraries');
        }
        const filterParams = { collectionName, collection, limit, offset };
        if (filter)
            filterParams.filter = filter;
        if (resolvedProjectId)
            filterParams.projectId = resolvedProjectId;
        if (libraryName)
            filterParams.libraryName = libraryName;
        return this.retrieveByFilter(filterParams);
    }
    /**
     * Direct point lookup by ID. Verifies the returned point matches the
     * caller's resolved scope (F-002). A mismatch is returned as a clean
     * "not found" rather than the foreign document, so the leak path
     * collapses to the same response as a genuinely missing ID.
     */
    async retrieveById(collectionName, collection, documentId, resolvedProjectId, libraryName) {
        // F-002: project-scope and library-scope lookups MUST resolve their
        // scope before reading. Without it, the verification step below
        // cannot distinguish "owned" from "foreign", so refuse the read.
        if (collection === 'projects' && !resolvedProjectId) {
            return unresolvedTenantResponse('projects');
        }
        if (collection === 'scratchpad' && !resolvedProjectId) {
            return unresolvedTenantResponse('scratchpad');
        }
        if (collection === 'libraries' && !libraryName) {
            return unresolvedTenantResponse('libraries');
        }
        try {
            const result = await this.qdrantClient.retrieve(collectionName, {
                ids: [documentId],
                with_payload: true,
                with_vector: false,
            });
            const point = result[0];
            if (!point) {
                return { success: false, documents: [], message: `Document not found: ${documentId}` };
            }
            // F-002: ownership check. A mismatch is reported as not-found —
            // we MUST NOT leak that the ID exists in a foreign tenant.
            if (!payloadMatchesScope(point.payload, collection, resolvedProjectId, libraryName)) {
                return { success: false, documents: [], message: `Document not found: ${documentId}` };
            }
            const document = {
                id: String(point.id),
                content: point.payload?.[FIELD_CONTENT] ?? '',
                metadata: extractMetadata(point.payload),
            };
            return { success: true, documents: [document], total: 1, hasMore: false };
        }
        catch (error) {
            return {
                success: false,
                documents: [],
                message: `Failed to retrieve document: ${error instanceof Error ? error.message : 'unknown error'}`,
            };
        }
    }
    async retrieveByFilter(params) {
        const { collectionName, collection, filter, limit, offset, projectId, libraryName } = params;
        try {
            const qdrantFilter = this.buildFilter(collection, filter, projectId, libraryName);
            const scrollRequest = { limit: limit + 1, with_payload: true, with_vector: false };
            if (offset > 0)
                scrollRequest.offset = offset;
            if (qdrantFilter)
                scrollRequest.filter = qdrantFilter;
            const result = await this.qdrantClient.scroll(collectionName, scrollRequest);
            const hasMore = result.points.length > limit;
            const points = hasMore ? result.points.slice(0, limit) : result.points;
            const documents = points.map((point) => ({
                id: String(point.id),
                content: point.payload?.[FIELD_CONTENT] ?? '',
                metadata: extractMetadata(point.payload),
            }));
            return { success: true, documents, total: documents.length, hasMore };
        }
        catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'unknown error';
            if (errorMessage.includes('not found') || errorMessage.includes("doesn't exist")) {
                return {
                    success: true,
                    documents: [],
                    total: 0,
                    hasMore: false,
                    message: 'Collection not found or empty',
                };
            }
            return {
                success: false,
                documents: [],
                message: `Failed to retrieve documents: ${errorMessage}`,
            };
        }
    }
    /**
     * Build the Qdrant filter for scroll-based retrieve.
     *
     * Callers MUST have already validated the scope (`projects` →
     * `projectId`; `libraries` → `libraryName`) — this helper trusts the
     * inputs and is no longer responsible for refusing on an unresolvable
     * tenant. That refusal happens in `retrieve()`.
     */
    buildFilter(collection, filter, projectId, libraryName) {
        const mustConditions = [];
        if ((collection === 'projects' || collection === 'scratchpad') && projectId) {
            mustConditions.push({ key: FIELD_TENANT_ID, match: { value: projectId } });
        }
        else if (collection === 'libraries' && libraryName) {
            mustConditions.push({ key: FIELD_TENANT_ID, match: { value: libraryName } });
        }
        if (filter) {
            for (const [key, value] of Object.entries(filter)) {
                mustConditions.push({ key, match: { value } });
            }
        }
        return mustConditions.length > 0 ? { must: mustConditions } : null;
    }
    async resolveProjectId() {
        const cwd = process.cwd();
        const projectInfo = await this.projectDetector.getProjectInfo(cwd, false);
        return projectInfo?.projectId;
    }
}
//# sourceMappingURL=retrieve.js.map