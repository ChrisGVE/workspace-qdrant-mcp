/**
 * Search tool argument builder — parse raw MCP tool arguments into SearchOptions
 */

export type SearchOptions = {
  query: string;
  collection?: string;
  mode?: 'hybrid' | 'semantic' | 'keyword';
  scope?: 'project' | 'group' | 'all';
  limit?: number;
  scoreThreshold?: number;
  projectId?: string;
  libraryName?: string;
  libraryPath?: string;
  branch?: string;
  fileType?: string;
  includeLibraries?: boolean;
  tag?: string;
  tags?: string[];
  pathGlob?: string;
  component?: string;
  exact?: boolean;
  contextLines?: number;
  includeGraphContext?: boolean;
};

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
  if (scope === 'project' || scope === 'group' || scope === 'all') options.scope = scope;

  const limit = args?.['limit'] as number | undefined;
  if (limit !== undefined) options.limit = limit;

  const scoreThreshold = args?.['scoreThreshold'] as number | undefined;
  if (scoreThreshold !== undefined) options.scoreThreshold = scoreThreshold;
}

function extractIdentifierOptions(
  args: Record<string, unknown> | undefined,
  options: SearchOptions,
  defaultBranch: string | null | undefined
): void {
  const projectId = args?.['projectId'] as string | undefined;
  if (projectId) options.projectId = projectId;

  const libraryName = args?.['libraryName'] as string | undefined;
  if (libraryName) options.libraryName = libraryName;

  const libraryPath = args?.['libraryPath'] as string | undefined;
  if (libraryPath) options.libraryPath = libraryPath;

  const branch = args?.['branch'] as string | undefined;
  if (branch === '*') {
    // Explicit wildcard — cross-branch search, no filter applied
  } else if (branch) {
    options.branch = branch;
  } else if (defaultBranch && defaultBranch !== 'default') {
    // Fall back to the session's current branch when not explicitly provided.
    // Skip when the sentinel value "default" is set — that indicates the
    // session is not inside a git repository and no branch filter should apply.
    options.branch = defaultBranch;
  }

  const fileType = args?.['fileType'] as string | undefined;
  if (fileType) options.fileType = fileType;

  const includeLibraries = args?.['includeLibraries'] as boolean | undefined;
  if (includeLibraries !== undefined) options.includeLibraries = includeLibraries;
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
}

/**
 * Build search options from raw tool arguments.
 *
 * @param args           Raw MCP tool arguments.
 * @param defaultBranch  Session's current branch, used when the caller does
 *                       not explicitly pass a `branch` argument. Pass `null`
 *                       or omit to skip the default. Pass the string `"*"` as
 *                       the `branch` argument to bypass filtering entirely.
 */
export function buildSearchOptions(
  args: Record<string, unknown> | undefined,
  defaultBranch?: string | null
): SearchOptions {
  const options: SearchOptions = { query: (args?.['query'] as string) ?? '' };

  extractScopeOptions(args, options);
  extractIdentifierOptions(args, options, defaultBranch);
  extractFilterOptions(args, options);
  extractOutputOptions(args, options);

  return options;
}
