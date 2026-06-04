/**
 * Rules tool argument builder — parse raw MCP tool arguments into RuleOptions
 */
export type RuleOptions = {
    action: 'add' | 'update' | 'remove' | 'list';
    content?: string;
    label?: string;
    scope?: 'global' | 'project';
    projectId?: string;
    title?: string;
    tags?: string[];
    priority?: number;
    limit?: number;
};
/** Build rule options from raw tool arguments */
export declare function buildRuleOptions(args: Record<string, unknown> | undefined): RuleOptions;
//# sourceMappingURL=rules.d.ts.map