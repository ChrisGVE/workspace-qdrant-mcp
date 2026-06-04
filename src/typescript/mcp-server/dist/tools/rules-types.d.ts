/**
 * Rules tool types, interfaces, and constants.
 */
export declare const RULES_COLLECTION: string;
export declare const RULES_BASENAME = "rules";
export type RuleAction = 'add' | 'update' | 'remove' | 'list';
export type RuleScope = 'global' | 'project';
export interface Rule {
    id: string;
    label?: string;
    content: string;
    scope: RuleScope;
    projectId?: string;
    title?: string;
    tags?: string[];
    priority?: number;
    createdAt?: string;
    updatedAt?: string;
}
export interface RuleOptions {
    action: RuleAction;
    content?: string;
    label?: string;
    scope?: RuleScope;
    projectId?: string;
    title?: string;
    tags?: string[];
    priority?: number;
    limit?: number;
}
export interface RuleResponse {
    success: boolean;
    action: RuleAction;
    label?: string;
    rules?: Rule[];
    similar_rules?: Array<Rule & {
        similarity: number;
    }>;
    message?: string;
    fallback_mode?: 'unified_queue';
    queue_id?: string;
}
export interface RuleToolConfig {
    qdrantUrl: string;
    qdrantApiKey?: string;
    qdrantTimeout?: number;
    duplicationThreshold?: number;
}
//# sourceMappingURL=rules-types.d.ts.map