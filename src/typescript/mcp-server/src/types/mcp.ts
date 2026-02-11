/**
 * MCP tool input/output types for workspace-qdrant-mcp
 * Exactly 4 tools: search, retrieve, memory, store
 */

import { z } from 'zod';

// ============================================================================
// Search Tool Types
// ============================================================================

export const SearchModeSchema = z.enum(['hybrid', 'semantic', 'keyword', 'retrieve']);
export type SearchMode = z.infer<typeof SearchModeSchema>;

export const CollectionSchema = z.enum(['projects', 'libraries', 'memory']);
export type Collection = z.infer<typeof CollectionSchema>;

export const SearchScopeSchema = z.enum(['all', 'global', 'project', 'current', 'other']);
export type SearchScope = z.infer<typeof SearchScopeSchema>;

export const SearchInputSchema = z.object({
  query: z.string().min(1).describe('Search query'),
  collection: CollectionSchema.describe('Target collection: projects, libraries, or memory'),
  mode: SearchModeSchema.default('hybrid').describe('Search mode'),
  limit: z.number().int().min(1).max(100).default(10).describe('Maximum results'),
  score_threshold: z.number().min(0).max(1).default(0.3).describe('Minimum similarity score'),
  scope: SearchScopeSchema.optional().describe('Scope filter within collection'),
  branch: z.string().optional().describe('Branch filter for projects'),
  project_id: z.string().optional().describe('Specific project ID'),
  library_name: z.string().optional().describe('Specific library name'),
});

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

export type RetrieveInput = z.infer<typeof RetrieveInputSchema>;

export interface RetrieveOutput {
  documents: SearchResult[];
  total: number;
  offset: number;
  status: 'healthy' | 'uncertain';
}

// ============================================================================
// Memory Tool Types
// ============================================================================

export const MemoryActionSchema = z.enum(['add', 'update', 'remove', 'list']);
export type MemoryAction = z.infer<typeof MemoryActionSchema>;

export const MemoryScopeSchema = z.enum(['global', 'project']);
export type MemoryScope = z.infer<typeof MemoryScopeSchema>;

export const MemoryInputSchema = z.object({
  action: MemoryActionSchema.describe('Memory operation'),
  label: z.string().optional().describe('Rule label (unique per scope)'),
  content: z.string().optional().describe('Rule content'),
  scope: MemoryScopeSchema.default('project').describe('Rule scope (default: project)'),
  project_id: z.string().optional().describe('Project ID for project-scoped rules'),
});

export type MemoryInput = z.infer<typeof MemoryInputSchema>;

export interface MemoryRule {
  label: string;
  content: string;
  scope: MemoryScope;
  project_id: string | null;
  created_at: string;
}

export interface MemoryOutput {
  success: boolean;
  rules?: MemoryRule[];
  status?: 'queued' | 'completed';
  queue_id?: string;
  fallback_mode?: 'unified_queue';
  message?: string;
}

// ============================================================================
// Store Tool Types
// ============================================================================

export const StoreSourceSchema = z.enum(['user_input', 'web', 'file']);
export type StoreSource = z.infer<typeof StoreSourceSchema>;

export const StoreInputSchema = z.object({
  content: z.string().min(1).describe('Text content to store'),
  library_name: z.string().min(1).describe('Library identifier'),
  title: z.string().optional().describe('Document title'),
  source: StoreSourceSchema.default('user_input').describe('Content source type'),
  url: z.string().url().optional().describe('Source URL for web content'),
  metadata: z.record(z.string(), z.unknown()).optional().describe('Additional metadata'),
});

export type StoreInput = z.infer<typeof StoreInputSchema>;

export interface StoreOutput {
  success: boolean;
  status: 'queued' | 'completed';
  queue_id?: string;
  fallback_mode?: 'unified_queue';
  message?: string;
}

// ============================================================================
// Health Status Types
// ============================================================================

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
