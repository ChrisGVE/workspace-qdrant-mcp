/**
 * Search tool argument builder — parse raw MCP tool arguments into SearchOptions.
 *
 * The result type is the canonical {@link SearchOptions} from the search tool
 * itself — NOT a local copy. A local subset used to drift: a field added to the
 * canonical type / tool schema was silently dropped here until mapped by hand
 * (this is how `includeScratchpad`, `summary`, `maxBytesPerHit`, `expandContext`
 * and `rerank` all failed to take effect). Reusing the canonical type means the
 * extractors below must cover every arg-derived field, and adding a field there
 * surfaces here as the single source of truth.
 */

import type { SearchOptions } from '../tools/search-types.js';

export type { SearchOptions };

// ── Option group extractors ───────────────────────────────────────────────

function extractScopeOptions(
  args: Record<string, unknown> | undefined,
  options: SearchOptions
): void {
  const collection = args?.['collection'] as string | undefined;
  if (collection) options.collection = collection;

  const mode = args?.['mode'] as string | undefined;
  if (mode === 'hybrid' || mode === 'semantic' || mode === 'keyword') options.mode = mode;

  const scope = args?.['scope'] as string | undefined;
  if (scope === 'project' || scope === 'global' || scope === 'all') options.scope = scope;

  const limit = args?.['limit'] as number | undefined;
  if (limit !== undefined) options.limit = limit;

  const scoreThreshold = args?.['scoreThreshold'] as number | undefined;
  if (scoreThreshold !== undefined) options.scoreThreshold = scoreThreshold;
}

function extractIdentifierOptions(
  args: Record<string, unknown> | undefined,
  options: SearchOptions
): void {
  const projectId = args?.['projectId'] as string | undefined;
  if (projectId) options.projectId = projectId;

  const libraryName = args?.['libraryName'] as string | undefined;
  if (libraryName) options.libraryName = libraryName;

  const branch = args?.['branch'] as string | undefined;
  if (branch) options.branch = branch;

  const fileType = args?.['fileType'] as string | undefined;
  if (fileType) options.fileType = fileType;

  const includeLibraries = args?.['includeLibraries'] as boolean | undefined;
  if (includeLibraries !== undefined) options.includeLibraries = includeLibraries;

  const includeScratchpad = args?.['includeScratchpad'] as boolean | undefined;
  if (includeScratchpad !== undefined) options.includeScratchpad = includeScratchpad;
}

function extractFilterOptions(
  args: Record<string, unknown> | undefined,
  options: SearchOptions
): void {
  const tag = args?.['tag'] as string | undefined;
  if (tag) options.tag = tag;

  const tags = args?.['tags'] as string[] | undefined;
  if (tags && tags.length > 0) options.tags = tags;

  const pathGlob = args?.['pathGlob'] as string | undefined;
  if (pathGlob) options.pathGlob = pathGlob;

  const component = args?.['component'] as string | undefined;
  if (component) options.component = component;
}

function extractOutputOptions(
  args: Record<string, unknown> | undefined,
  options: SearchOptions
): void {
  const exact = args?.['exact'] as boolean | undefined;
  if (exact !== undefined) options.exact = exact;

  const contextLines = args?.['contextLines'] as number | undefined;
  if (contextLines !== undefined) options.contextLines = contextLines;

  const includeGraphContext = args?.['includeGraphContext'] as boolean | undefined;
  if (includeGraphContext !== undefined) options.includeGraphContext = includeGraphContext;

  const expandContext = args?.['expandContext'] as boolean | undefined;
  if (expandContext !== undefined) options.expandContext = expandContext;

  const rerank = args?.['rerank'] as boolean | undefined;
  if (rerank !== undefined) options.rerank = rerank;

  const maxBytesPerHit = args?.['maxBytesPerHit'] as number | undefined;
  if (maxBytesPerHit !== undefined) options.maxBytesPerHit = maxBytesPerHit;

  const summary = args?.['summary'] as boolean | undefined;
  if (summary !== undefined) options.summary = summary;
}

/** Build search options from raw tool arguments.
 *
 * When `defaults.branch` is provided and the caller did NOT pass an explicit
 * `branch` argument, the default fills in. This lets the dispatcher inject
 * the current git branch so the agent sees branch-scoped results by default
 * without every caller having to detect and pass it. Pass `branch: "*"` from
 * the agent side to opt out (treated as "any branch" by the search layer).
 */
export function buildSearchOptions(
  args: Record<string, unknown> | undefined,
  defaults?: { branch?: string | null }
): SearchOptions {
  const options: SearchOptions = { query: (args?.['query'] as string) ?? '' };

  extractScopeOptions(args, options);
  extractIdentifierOptions(args, options);
  extractFilterOptions(args, options);
  extractOutputOptions(args, options);

  if (options.branch === undefined && defaults?.branch) {
    options.branch = defaults.branch;
  }

  return options;
}
