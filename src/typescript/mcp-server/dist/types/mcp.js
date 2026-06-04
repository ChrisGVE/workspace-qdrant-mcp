/**
 * MCP tool input/output types for workspace-qdrant-mcp
 * Exactly 4 tools: search, retrieve, rules, store
 */
import { z } from 'zod';
// ============================================================================
// Search Tool Types
// ============================================================================
export const SearchModeSchema = z.enum(['hybrid', 'semantic', 'keyword', 'retrieve']);
export const CollectionSchema = z.enum(['projects', 'libraries', 'rules']);
export const SearchScopeSchema = z.enum(['all', 'group', 'project']);
export const SearchInputSchema = z.object({
    query: z.string().min(1).describe('Search query'),
    collection: CollectionSchema.describe('Target collection: projects, libraries, or rules'),
    mode: SearchModeSchema.default('hybrid').describe('Search mode'),
    limit: z.number().int().min(1).max(100).default(10).describe('Maximum results'),
    score_threshold: z.number().min(0).max(1).default(0.3).describe('Minimum similarity score'),
    scope: SearchScopeSchema.optional().describe('Scope filter within collection'),
    branch: z.string().optional().describe('Branch filter for projects'),
    project_id: z.string().optional().describe('Specific project ID'),
    library_name: z.string().optional().describe('Specific library name'),
});
// ============================================================================
// Retrieve Tool Types
// ============================================================================
export const RetrieveInputSchema = z.object({
    document_id: z.string().optional().describe('Specific document ID'),
    collection: CollectionSchema.default('projects').describe('Target collection'),
    metadata: z.record(z.string(), z.unknown()).optional().describe('Metadata filters'),
    limit: z.number().int().min(1).max(100).default(10).describe('Maximum documents'),
    offset: z.number().int().min(0).default(0).describe('Pagination offset'),
});
// ============================================================================
// Rules Tool Types
// ============================================================================
export const RuleActionSchema = z.enum(['add', 'update', 'remove', 'list']);
export const RuleScopeSchema = z.enum(['global', 'project']);
export const RuleInputSchema = z.object({
    action: RuleActionSchema.describe('Rules operation'),
    label: z.string().optional().describe('Rule label (unique per scope)'),
    content: z.string().optional().describe('Rule content'),
    scope: RuleScopeSchema.default('project').describe('Rule scope (default: project)'),
    project_id: z.string().optional().describe('Project ID for project-scoped rules'),
});
// ============================================================================
// Store Tool Types
// ============================================================================
export const StoreSourceSchema = z.enum(['user_input', 'web', 'file']);
export const StoreInputSchema = z.object({
    content: z.string().min(1).describe('Text content to store'),
    library_name: z.string().min(1).describe('Library identifier'),
    title: z.string().optional().describe('Document title'),
    source: StoreSourceSchema.default('user_input').describe('Content source type'),
    url: z.string().url().optional().describe('Source URL for web content'),
    metadata: z.record(z.string(), z.unknown()).optional().describe('Additional metadata'),
});
//# sourceMappingURL=mcp.js.map