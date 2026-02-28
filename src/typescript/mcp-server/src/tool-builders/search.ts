/**
 * Search tool argument builder — parse raw MCP tool arguments into SearchOptions
 */

export type SearchOptions = {
  query: string;
  collection?: string;
  mode?: 'hybrid' | 'semantic' | 'keyword';
  scope?: 'project' | 'global' | 'all';
  limit?: number;
  projectId?: string;
  libraryName?: string;
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

/** Build search options from raw tool arguments */
export function buildSearchOptions(args: Record<string, unknown> | undefined): SearchOptions {
  const options: SearchOptions = { query: (args?.['query'] as string) ?? '' };

  const collection = args?.['collection'] as string | undefined;
  if (collection) options.collection = collection;

  const mode = args?.['mode'] as string | undefined;
  if (mode === 'hybrid' || mode === 'semantic' || mode === 'keyword') options.mode = mode;

  const scope = args?.['scope'] as string | undefined;
  if (scope === 'project' || scope === 'global' || scope === 'all') options.scope = scope;

  const limit = args?.['limit'] as number | undefined;
  if (limit !== undefined) options.limit = limit;

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

  const tag = args?.['tag'] as string | undefined;
  if (tag) options.tag = tag;

  const tags = args?.['tags'] as string[] | undefined;
  if (tags && tags.length > 0) options.tags = tags;

  const pathGlob = args?.['pathGlob'] as string | undefined;
  if (pathGlob) options.pathGlob = pathGlob;

  const component = args?.['component'] as string | undefined;
  if (component) options.component = component;

  const exact = args?.['exact'] as boolean | undefined;
  if (exact !== undefined) options.exact = exact;

  const contextLines = args?.['contextLines'] as number | undefined;
  if (contextLines !== undefined) options.contextLines = contextLines;

  const includeGraphContext = args?.['includeGraphContext'] as boolean | undefined;
  if (includeGraphContext !== undefined) options.includeGraphContext = includeGraphContext;

  return options;
}
