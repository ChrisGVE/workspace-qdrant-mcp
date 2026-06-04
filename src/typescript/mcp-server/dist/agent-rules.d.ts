/**
 * Rule fetching and formatting for Claude Agent SDK integration.
 */
import { loadConfig } from './config.js';
import { type Rule } from './tools/rules.js';
export type { Rule };
/**
 * Fetch rules from Qdrant via RulesTool.
 * Fetches both global and project-specific rules (if project detected).
 * Rules are sorted by priority (highest first) then by creation date (newest first).
 */
export declare function fetchRules(projectId: string | null, config: ReturnType<typeof loadConfig>): Promise<Rule[]>;
/** Format rules for system prompt injection. */
export declare function formatRulesForPrompt(rules: Rule[]): string;
//# sourceMappingURL=agent-rules.d.ts.map