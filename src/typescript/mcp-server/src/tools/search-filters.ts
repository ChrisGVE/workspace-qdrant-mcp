/**
 * Search filter construction and collection determination.
 */

import type { SearchScope, FilterParams } from './search-types.js';
import {
  PROJECTS_COLLECTION,
  LIBRARIES_COLLECTION,
  SCRATCHPAD_COLLECTION,
} from './search-types.js';

/**
 * Extract the deterministic path prefix from a glob pattern.
 * Returns everything before the first glob metacharacter (* ? [ {).
 * Example: "src/**\/*.rs" → "src/", "**\/*.rs" → ""
 */
export function extractGlobPrefix(glob: string): string {
  const metaChars = /[*?[{]/;
  const match = metaChars.exec(glob);
  if (!match) {
    // No glob chars — the whole string is a literal path
    return glob;
  }
  // Take everything up to the first metachar,
  // then trim to the last path separator for a clean prefix
  const beforeMeta = glob.slice(0, match.index);
  const lastSlash = beforeMeta.lastIndexOf('/');
  return lastSlash >= 0 ? beforeMeta.slice(0, lastSlash + 1) : '';
}

/**
 * Determine which collections to search based on scope.
 */
export function determineCollections(
  collection: string | undefined,
  scope: SearchScope,
  includeLibraries: boolean,
): string[] {
  if (collection) return [collection];

  switch (scope) {
    case 'project':
      return includeLibraries
        ? [PROJECTS_COLLECTION, LIBRARIES_COLLECTION]
        : [PROJECTS_COLLECTION];
    case 'global':
      return [PROJECTS_COLLECTION];
    case 'all':
      return [PROJECTS_COLLECTION, LIBRARIES_COLLECTION, SCRATCHPAD_COLLECTION];
    default:
      return [PROJECTS_COLLECTION];
  }
}

/**
 * Build Qdrant filter based on search parameters.
 */
export function buildFilter(params: FilterParams): Record<string, unknown> | null {
  const mustConditions: Record<string, unknown>[] = [];
  const mustNotConditions: Record<string, unknown>[] = [];

  // Project filter for project scope
  if (params.scope === 'project' && params.projectId) {
    mustConditions.push({
      key: 'tenant_id',
      match: { value: params.projectId },
    });
  }

  // Task 15: Instance-aware base_point filter
  if (params.basePoints && params.basePoints.length > 0) {
    mustConditions.push({
      key: 'base_point',
      match: { any: params.basePoints },
    });
  }

  // Branch filter
  if (params.branch && params.branch !== '*') {
    mustConditions.push({
      key: 'branch',
      match: { value: params.branch },
    });
  }

  // File type filter
  if (params.fileType) {
    mustConditions.push({
      key: 'file_type',
      match: { value: params.fileType },
    });
  }

  // Library name filter
  if (params.collection === LIBRARIES_COLLECTION && params.libraryName) {
    mustConditions.push({
      key: 'library_name',
      match: { value: params.libraryName },
    });
  }

  // Tag filter — single tag (exact match on concept_tags payload field)
  if (params.tag) {
    mustConditions.push({
      key: 'concept_tags',
      match: { value: params.tag },
    });
  }

  // Multi-tag filter (OR logic)
  if (params.tags && params.tags.length > 0) {
    const tagShouldConditions = params.tags.map((t) => ({
      key: 'concept_tags',
      match: { value: t },
    }));
    mustConditions.push({ should: tagShouldConditions });
  }

  // Path glob filter — extract deterministic prefix for Qdrant text match
  if (params.pathGlob) {
    const prefix = extractGlobPrefix(params.pathGlob);
    if (prefix) {
      mustConditions.push({
        key: 'file_path',
        match: { text: prefix },
      });
    }
  }

  // Exclude deleted libraries unless explicitly included
  if (params.collection === LIBRARIES_COLLECTION && !params.includeDeleted) {
    mustNotConditions.push({
      key: 'deleted',
      match: { value: true },
    });
  }

  if (mustConditions.length === 0 && mustNotConditions.length === 0) {
    return null;
  }

  const filter: Record<string, unknown> = {};
  if (mustConditions.length > 0) filter.must = mustConditions;
  if (mustNotConditions.length > 0) filter.must_not = mustNotConditions;
  return filter;
}
