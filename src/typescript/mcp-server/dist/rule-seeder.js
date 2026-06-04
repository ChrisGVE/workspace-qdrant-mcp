/**
 * Default rule seeding for fresh installations.
 *
 * Extracted from WorkspaceQdrantMcpServer.seedDefaultRule to keep server.ts
 * within the 300-line file-size limit.
 */
import { logInfo, logDebug } from './utils/logger.js';
import { TENANT_GLOBAL } from './constants/tenants.js';
/**
 * Seed a default "search-first" rule if the rules collection is empty.
 * Only runs once per fresh installation; skipped if any rule already exists.
 */
export async function seedDefaultRule(rulesTool) {
    try {
        const listResult = await rulesTool.execute({ action: 'list', scope: TENANT_GLOBAL });
        if (!listResult.success || (listResult.rules && listResult.rules.length > 0)) {
            return; // Rules exist or list failed — skip seeding
        }
        const addResult = await rulesTool.execute({
            action: 'add',
            label: 'search-first',
            title: 'Always search before answering',
            content: [
                'When asked about the codebase, project structure, library documentation,',
                'or any topic that might be covered in the indexed knowledge base,',
                'ALWAYS use the workspace-qdrant search tool first.',
                'Do not rely on training data for project-specific questions.',
                'Use scope="project" for code questions and includeLibraries=true for broader knowledge queries.',
            ].join(' '),
            scope: TENANT_GLOBAL,
            priority: 100,
        });
        if (addResult.success) {
            logInfo('Created default search-first behavioral rule');
        }
    }
    catch (error) {
        logDebug('Skipped default rule seeding', { reason: String(error) });
    }
}
//# sourceMappingURL=rule-seeder.js.map