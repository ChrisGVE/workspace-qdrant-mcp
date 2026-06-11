/**
 * Retrieve tool types and constants.
 */

// Canonical collection names from native bridge (single source of truth)
import {
  COLLECTION_PROJECTS,
  COLLECTION_LIBRARIES,
  COLLECTION_RULES,
  COLLECTION_SCRATCHPAD,
  FIELD_CONTENT,
} from '../common/native-bridge.js';
import { RANKING_AID_KEYS } from '../common/payload-noise.js';
export const PROJECTS_COLLECTION = COLLECTION_PROJECTS;
export const LIBRARIES_COLLECTION = COLLECTION_LIBRARIES;
export const RULES_COLLECTION = COLLECTION_RULES;
export const SCRATCHPAD_COLLECTION = COLLECTION_SCRATCHPAD;

export type RetrieveCollectionType = 'projects' | 'libraries' | 'rules' | 'scratchpad';

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
    case 'projects':
      return PROJECTS_COLLECTION;
    case 'libraries':
      return LIBRARIES_COLLECTION;
    case 'rules':
      return RULES_COLLECTION;
    case 'scratchpad':
      return SCRATCHPAD_COLLECTION;
    default:
      return PROJECTS_COLLECTION;
  }
}

/**
 * Extract metadata from payload, excluding content, vector fields, and the
 * daemon's ranking-aid fields ({@link RANKING_AID_KEYS} — keywords/baskets/tags
 * are indexing signal, ~1.5–2k tokens/hit, that a reading agent never consumes;
 * the `search` truncate path drops the same set).
 */
const RETRIEVE_DROP_KEYS: ReadonlySet<string> = new Set<string>([
  FIELD_CONTENT,
  'dense_vector',
  'sparse_vector',
  ...RANKING_AID_KEYS,
]);

export function extractMetadata(
  payload: Record<string, unknown> | null | undefined
): Record<string, unknown> {
  if (!payload) return {};
  const metadata: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(payload)) {
    if (RETRIEVE_DROP_KEYS.has(key)) continue;
    metadata[key] = value;
  }
  return metadata;
}
