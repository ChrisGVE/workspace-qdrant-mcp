/**
 * MCP tool input/output types for workspace-qdrant-mcp
 * Exactly 4 tools: search, retrieve, rules, store
 */
import { z } from 'zod';
export declare const SearchModeSchema: z.ZodEnum<{
    hybrid: "hybrid";
    semantic: "semantic";
    keyword: "keyword";
    retrieve: "retrieve";
}>;
export type SearchMode = z.infer<typeof SearchModeSchema>;
export declare const CollectionSchema: z.ZodEnum<{
    rules: "rules";
    projects: "projects";
    libraries: "libraries";
}>;
export type Collection = z.infer<typeof CollectionSchema>;
export declare const SearchScopeSchema: z.ZodEnum<{
    project: "project";
    all: "all";
    group: "group";
}>;
export type SearchScope = z.infer<typeof SearchScopeSchema>;
export declare const SearchInputSchema: z.ZodObject<{
    query: z.ZodString;
    collection: z.ZodEnum<{
        rules: "rules";
        projects: "projects";
        libraries: "libraries";
    }>;
    mode: z.ZodDefault<z.ZodEnum<{
        hybrid: "hybrid";
        semantic: "semantic";
        keyword: "keyword";
        retrieve: "retrieve";
    }>>;
    limit: z.ZodDefault<z.ZodNumber>;
    score_threshold: z.ZodDefault<z.ZodNumber>;
    scope: z.ZodOptional<z.ZodEnum<{
        project: "project";
        all: "all";
        group: "group";
    }>>;
    branch: z.ZodOptional<z.ZodString>;
    project_id: z.ZodOptional<z.ZodString>;
    library_name: z.ZodOptional<z.ZodString>;
}, z.core.$strip>;
export type SearchInput = z.infer<typeof SearchInputSchema>;
export interface SearchResult {
    id: string;
    score: number;
    content: string;
    metadata: Record<string, unknown>;
}
export interface SearchOutput {
    results: SearchResult[];
    status: 'healthy' | 'uncertain';
    reason?: string;
    message?: string;
}
export declare const RetrieveInputSchema: z.ZodObject<{
    document_id: z.ZodOptional<z.ZodString>;
    collection: z.ZodDefault<z.ZodEnum<{
        rules: "rules";
        projects: "projects";
        libraries: "libraries";
    }>>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    limit: z.ZodDefault<z.ZodNumber>;
    offset: z.ZodDefault<z.ZodNumber>;
}, z.core.$strip>;
export type RetrieveInput = z.infer<typeof RetrieveInputSchema>;
export interface RetrieveOutput {
    documents: SearchResult[];
    total: number;
    offset: number;
    status: 'healthy' | 'uncertain';
}
export declare const RuleActionSchema: z.ZodEnum<{
    add: "add";
    update: "update";
    remove: "remove";
    list: "list";
}>;
export type RuleAction = z.infer<typeof RuleActionSchema>;
export declare const RuleScopeSchema: z.ZodEnum<{
    global: "global";
    project: "project";
}>;
export type RuleScope = z.infer<typeof RuleScopeSchema>;
export declare const RuleInputSchema: z.ZodObject<{
    action: z.ZodEnum<{
        add: "add";
        update: "update";
        remove: "remove";
        list: "list";
    }>;
    label: z.ZodOptional<z.ZodString>;
    content: z.ZodOptional<z.ZodString>;
    scope: z.ZodDefault<z.ZodEnum<{
        global: "global";
        project: "project";
    }>>;
    project_id: z.ZodOptional<z.ZodString>;
}, z.core.$strip>;
export type RuleInput = z.infer<typeof RuleInputSchema>;
export interface RuleEntry {
    label: string;
    content: string;
    scope: RuleScope;
    project_id: string | null;
    created_at: string;
}
export interface RuleOutput {
    success: boolean;
    rules?: RuleEntry[];
    status?: 'queued' | 'completed';
    queue_id?: string;
    fallback_mode?: 'unified_queue';
    message?: string;
}
export declare const StoreSourceSchema: z.ZodEnum<{
    file: "file";
    user_input: "user_input";
    web: "web";
}>;
export type StoreSource = z.infer<typeof StoreSourceSchema>;
export declare const StoreInputSchema: z.ZodObject<{
    content: z.ZodString;
    library_name: z.ZodString;
    title: z.ZodOptional<z.ZodString>;
    source: z.ZodDefault<z.ZodEnum<{
        file: "file";
        user_input: "user_input";
        web: "web";
    }>>;
    url: z.ZodOptional<z.ZodString>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
}, z.core.$strip>;
export type StoreInput = z.infer<typeof StoreInputSchema>;
export interface StoreOutput {
    success: boolean;
    status: 'queued' | 'completed';
    queue_id?: string;
    fallback_mode?: 'unified_queue';
    message?: string;
}
export type HealthStatus = 'healthy' | 'uncertain' | 'degraded';
export interface SystemHealth {
    daemon: {
        connected: boolean;
        lastHeartbeat?: string;
    };
    qdrant: {
        connected: boolean;
        collectionsReady: boolean;
    };
    status: HealthStatus;
    reason?: string;
}
//# sourceMappingURL=mcp.d.ts.map