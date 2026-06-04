/**
 * Rules tool facade — delegates to domain-specific modules.
 *
 * - rules-types.ts: Types, interfaces, constants
 * - rules-mutations.ts: Add, update, remove operations
 * - rules-list.ts: List/query operations with mirror fallback
 */
import type { DaemonClient } from '../clients/daemon-client.js';
import type { SqliteStateManager } from '../clients/sqlite-state-manager.js';
import type { ProjectDetector } from '../utils/project-detector.js';
export type { RuleAction, RuleScope, Rule, RuleOptions, RuleResponse, RuleToolConfig, } from './rules-types.js';
import type { RuleOptions, RuleResponse, RuleToolConfig } from './rules-types.js';
/**
 * Rules tool for behavioral rules management
 */
export declare class RulesTool {
    private readonly qdrantClient;
    private readonly daemonClient;
    private readonly stateManager;
    private readonly projectDetector;
    private readonly duplicationThreshold;
    constructor(config: RuleToolConfig, daemonClient: DaemonClient, stateManager: SqliteStateManager, projectDetector: ProjectDetector);
    execute(options: RuleOptions): Promise<RuleResponse>;
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
    private findSimilarRules;
}
//# sourceMappingURL=rules.d.ts.map