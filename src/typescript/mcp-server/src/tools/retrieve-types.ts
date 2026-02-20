/**
 * Retrieve tool types and constants.
 */

// Canonical collection names from native bridge (single source of truth)
import { COLLECTION_PROJECTS, COLLECTION_LIBRARIES, COLLECTION_MEMORY } from '../common/native-bridge.js';
export const PROJECTS_COLLECTION = COLLECTION_PROJECTS;
export const LIBRARIES_COLLECTION = COLLECTION_LIBRARIES;
export const MEMORY_COLLECTION = COLLECTION_MEMORY;

export type RetrieveCollectionType = 'projects' | 'libraries' | 'memory';

export interface RetrieveOptions {
  documentId?: string;
  collection?: RetrieveCollectionType;
  filter?: Record<string, string>;
  limit?: number;
  offset?: number;
  projectId?: string;
  libraryName?: string;
}

export interface RetrievedDocument {
  id: string;
  content: string;
  metadata: Record<string, unknown>;
  score?: number;
}

export interface RetrieveResponse {
  success: boolean;
  documents: RetrievedDocument[];
  total?: number;
  hasMore?: boolean;
  message?: string;
}

export interface RetrieveToolConfig {
  qdrantUrl: string;
  qdrantApiKey?: string;
  qdrantTimeout?: number;
}

/** Map collection type to canonical Qdrant collection name. */
export function getCollectionName(collection: RetrieveCollectionType): string {
  switch (collection) {
    case 'projects': return PROJECTS_COLLECTION;
    case 'libraries': return LIBRARIES_COLLECTION;
    case 'memory': return MEMORY_COLLECTION;
    default: return PROJECTS_COLLECTION;
  }
}

/** Extract metadata from payload (excluding content and vector fields). */
export function extractMetadata(payload: Record<string, unknown> | null | undefined): Record<string, unknown> {
  if (!payload) return {};
  const metadata: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(payload)) {
    if (key === 'content' || key === 'dense_vector' || key === 'sparse_vector') continue;
    metadata[key] = value;
  }
  return metadata;
}
