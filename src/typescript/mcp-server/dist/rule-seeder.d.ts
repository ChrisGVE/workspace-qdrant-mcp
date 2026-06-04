/**
 * Default rule seeding for fresh installations.
 *
 * Extracted from WorkspaceQdrantMcpServer.seedDefaultRule to keep server.ts
 * within the 300-line file-size limit.
 */
import type { RulesTool } from './tools/index.js';
/**
 * Seed a default "search-first" rule if the rules collection is empty.
 * Only runs once per fresh installation; skipped if any rule already exists.
 */
export declare function seedDefaultRule(rulesTool: RulesTool): Promise<void>;
//# sourceMappingURL=rule-seeder.d.ts.map