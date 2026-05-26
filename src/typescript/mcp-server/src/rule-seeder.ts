/**
 * Default rule seeding for fresh installations.
 *
 * Extracted from WorkspaceQdrantMcpServer.seedDefaultRule to keep server.ts
 * within the 300-line file-size limit.
 */

import { logInfo, logDebug } from './utils/logger.js';
import { TENANT_GLOBAL } from './constants/tenants.js';
import type { RulesTool } from './tools/index.js';

/**
 * Seed a default "search-first" rule if the rules collection is empty.
 * Only runs once per fresh installation; skipped if any rule already exists.
 */
export async function seedDefaultRule(rulesTool: RulesTool): Promise<void> {
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
        "For any question about this project's code, structure, or library docs,",
        'ALWAYS call the `search` tool before answering — do not rely on training data.',
        'Defaults: scope="project", limit=10.',
        'Widen to scope="all" or includeLibraries=true only after a project-scoped query comes back empty.',
        'Use mode="semantic" for concept queries, mode="keyword" or exact=true for known identifiers.',
      ].join(' '),
      scope: TENANT_GLOBAL,
      priority: 100,
    });

    if (addResult.success) {
      logInfo('Created default search-first behavioral rule');
    }
  } catch (error) {
    logDebug('Skipped default rule seeding', { reason: String(error) });
  }
}
