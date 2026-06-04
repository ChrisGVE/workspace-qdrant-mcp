/**
 * Rules mutation operations: add, update, remove rules.
 * Uses daemon gRPC with unified_queue fallback.
 */
import type { DaemonClient } from '../clients/daemon-client.js';
import type { SqliteStateManager } from '../clients/sqlite-state-manager.js';
import type { ProjectDetector } from '../utils/project-detector.js';
import type { RuleOptions, RuleResponse } from './rules-types.js';
export declare function addRule(daemonClient: DaemonClient, stateManager: SqliteStateManager, projectDetector: ProjectDetector, options: RuleOptions): Promise<RuleResponse>;
export declare function updateRule(daemonClient: DaemonClient, stateManager: SqliteStateManager, projectDetector: ProjectDetector, options: RuleOptions): Promise<RuleResponse>;
export declare function removeRule(stateManager: SqliteStateManager, projectDetector: ProjectDetector, options: RuleOptions): Promise<RuleResponse>;
//# sourceMappingURL=rules-mutations.d.ts.map