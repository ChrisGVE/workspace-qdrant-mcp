/**
 * Rules tool facade — delegates to domain-specific modules.
 *
 * - rules-types.ts: Types, interfaces, constants
 * - rules-mutations.ts: Add, update, remove operations
 * - rules-list.ts: List/query operations with mirror fallback
 */
import { QdrantClient } from '@qdrant/js-client-rest';
import { RULES_COLLECTION } from './rules-types.js';
import { TENANT_GLOBAL } from '../constants/tenants.js';
import { FIELD_CONTENT, FIELD_PROJECT_ID } from '../common/native-bridge.js';
import { addRule, updateRule, removeRule } from './rules-mutations.js';
import { listRules } from './rules-list.js';
import { resolveProjectScopeId } from './rules-mutation-helpers.js';
const DEFAULT_DUPLICATION_THRESHOLD = 0.7;
/**
 * Rules tool for behavioral rules management
 */
export class RulesTool {
    qdrantClient;
    daemonClient;
    stateManager;
    projectDetector;
    duplicationThreshold;
    constructor(config, daemonClient, stateManager, projectDetector) {
        const clientConfig = {
            url: config.qdrantUrl,
            timeout: config.qdrantTimeout ?? 5000,
        };
        if (config.qdrantApiKey) {
            clientConfig.apiKey = config.qdrantApiKey;
        }
        this.qdrantClient = new QdrantClient(clientConfig);
        this.daemonClient = daemonClient;
        this.stateManager = stateManager;
        this.projectDetector = projectDetector;
        this.duplicationThreshold = config.duplicationThreshold ?? DEFAULT_DUPLICATION_THRESHOLD;
    }
    async execute(options) {
        switch (options.action) {
            case 'add': {
                // Check for similar rules before inserting. F-015: scope the
                // duplicate-detection by (scope, tenant_id) so the same label
                // / similar content in another project does not block this
                // project's add.
                if (options.content?.trim()) {
                    const dupScope = options.scope ?? 'project';
                    const { resolvedProjectId } = await resolveProjectScopeId(dupScope, options.projectId, this.projectDetector);
                    const duplicates = await this.findSimilarRules(options.content, dupScope, resolvedProjectId);
                    if (duplicates.length > 0) {
                        return {
                            success: false,
                            action: 'add',
                            similar_rules: duplicates,
                            message: `Found ${duplicates.length} similar rule(s). Review before adding to avoid duplication.`,
                        };
                    }
                }
                return addRule(this.daemonClient, this.stateManager, this.projectDetector, options);
            }
            case 'update':
                return updateRule(this.daemonClient, this.stateManager, this.projectDetector, options);
            case 'remove':
                return removeRule(this.stateManager, this.projectDetector, options);
            case 'list':
                return listRules(this.qdrantClient, this.stateManager, this.projectDetector, options);
            default:
                return {
                    success: false,
                    action: options.action,
                    message: `Unknown action: ${options.action}`,
                };
        }
    }
    /**
     * Find existing rules similar to the given content using embedding similarity.
     * Returns rules with cosine similarity >= duplicationThreshold.
     *
     * F-015: results are scoped by (scope, tenant_id) so a project-scope
     * add is only matched against rules in the same project (plus global
     * rules), and a global-scope add is only matched against global
     * rules. Pre-fix the whole RULES_COLLECTION was scanned and the same
     * label / content in another project blocked the add.
     */
    async findSimilarRules(content, scope, projectId) {
        try {
            const embedResponse = await this.daemonClient.embedText({ text: content });
            if (!embedResponse.embedding?.length)
                return [];
            const searchRequest = {
                vector: embedResponse.embedding,
                limit: 5,
                score_threshold: this.duplicationThreshold,
                with_payload: true,
            };
            const filter = buildDuplicateScopeFilter(scope, projectId);
            if (filter)
                searchRequest.filter = filter;
            const searchResult = await this.qdrantClient.search(RULES_COLLECTION, searchRequest);
            return searchResult
                .filter((point) => point.score >= this.duplicationThreshold)
                .map((point) => ({
                id: String(point.id),
                content: point.payload?.[FIELD_CONTENT] ?? '',
                scope: point.payload?.['scope'] ?? TENANT_GLOBAL,
                label: point.payload?.['label'] ?? undefined,
                title: point.payload?.['title'] ?? undefined,
                similarity: Math.round(point.score * 1000) / 1000,
            }));
        }
        catch {
            // Embedding or search failed — allow the add to proceed
            return [];
        }
    }
}
/** F-015: Build a Qdrant filter that restricts duplicate detection to
 *  the active rule scope. Project-scope: match the current project_id
 *  (or any global rule too — global rules apply to every project and
 *  remain dup-detectable). Global-scope: only match global rules. */
function buildDuplicateScopeFilter(scope, projectId) {
    if (scope === 'project') {
        if (!projectId)
            return null; // unresolvable — let dup detection broaden rather than silently block
        return {
            must: [
                {
                    should: [
                        { key: FIELD_PROJECT_ID, match: { value: projectId } },
                        { key: 'scope', match: { value: 'global' } },
                    ],
                },
            ],
        };
    }
    // Global add — only check against other global rules.
    return { must: [{ key: 'scope', match: { value: 'global' } }] };
}
//# sourceMappingURL=rules.js.map