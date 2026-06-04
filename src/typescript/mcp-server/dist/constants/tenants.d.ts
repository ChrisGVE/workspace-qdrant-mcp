/**
 * Tenant identifier constants shared across rules, scratchpad, and search paths.
 *
 * Mirrors the Rust constant `wqm_common::constants::TENANT_GLOBAL`. Use this
 * import wherever the literal `'global'` would be written as the `tenant_id`
 * sentinel for rules/scratchpad entries that apply across all projects.
 */
/**
 * Sentinel `tenant_id` for global-scope rules and scratchpad entries.
 *
 * Typed as a string literal via `as const` so it can be assigned to
 * positions typed `'global' | 'project'` or similar discriminated unions.
 */
export declare const TENANT_GLOBAL: "global";
export type TenantGlobal = typeof TENANT_GLOBAL;
//# sourceMappingURL=tenants.d.ts.map