/**
 * Shared helpers for rules mutation operations.
 */
import type { DaemonClient } from '../clients/daemon-client.js';
import type { SqliteStateManager } from '../clients/sqlite-state-manager.js';
import type { ProjectDetector } from '../utils/project-detector.js';
import type { RuleAction, RuleResponse, RuleScope } from './rules-types.js';
export declare function isConnectivityError(err: unknown): boolean;
export declare function queueRuleOperation(stateManager: SqliteStateManager, operation: {
    action: RuleAction;
    label?: string;
    content?: string;
    scope?: RuleScope;
    projectId?: string;
    title?: string;
    tags?: string[];
    priority?: number;
}): Promise<{
    queueId: string;
}>;
export declare function upsertMirror(stateManager: SqliteStateManager, label: string, content: string, scope: RuleScope | null, tenantId: string | null): void;
export declare function resolveProjectScopeId(scope: RuleScope, projectId: string | undefined, projectDetector: ProjectDetector): Promise<{
    resolvedProjectId: string | undefined;
    error?: RuleResponse;
}>;
export declare function persistAddRule(daemonClient: DaemonClient, stateManager: SqliteStateManager, label: string, content: string, scope: RuleScope, resolvedProjectId: string | undefined, title: string | undefined, tags: string[] | undefined, priority: number | undefined): Promise<RuleResponse>;
export declare function persistUpdateRule(daemonClient: DaemonClient, stateManager: SqliteStateManager, label: string, content: string, scope: RuleScope, resolvedProjectId: string | undefined, title: string | undefined, tags: string[] | undefined, priority: number | undefined): Promise<RuleResponse>;
//# sourceMappingURL=rules-mutation-helpers.d.ts.map