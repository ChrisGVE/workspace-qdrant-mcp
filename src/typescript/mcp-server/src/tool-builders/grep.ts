/**
 * Grep tool argument builder — parse raw MCP tool arguments into GrepOptions
 */

export type GrepOptions = {
  pattern: string;
  regex?: boolean;
  caseSensitive?: boolean;
  pathGlob?: string;
  scope?: 'project' | 'all';
  contextLines?: number;
  maxResults?: number;
  branch?: string;
  projectId?: string;
};

/** Build grep options from raw tool arguments */
export function buildGrepOptions(args: Record<string, unknown> | undefined): GrepOptions {
  const pattern = args?.['pattern'] as string;
  if (!pattern) {
    throw new Error('Pattern is required for grep operation');
  }

  const options: GrepOptions = { pattern };

  const regex = args?.['regex'] as boolean | undefined;
  if (regex !== undefined) options.regex = regex;

  const caseSensitive = args?.['caseSensitive'] as boolean | undefined;
  if (caseSensitive !== undefined) options.caseSensitive = caseSensitive;

  const pathGlob = args?.['pathGlob'] as string | undefined;
  if (pathGlob) options.pathGlob = pathGlob;

  const scope = args?.['scope'] as string | undefined;
  if (scope === 'project' || scope === 'all') options.scope = scope;

  const contextLines = args?.['contextLines'] as number | undefined;
  if (contextLines !== undefined) options.contextLines = contextLines;

  const maxResults = args?.['maxResults'] as number | undefined;
  if (maxResults !== undefined) options.maxResults = maxResults;

  const branch = args?.['branch'] as string | undefined;
  if (branch) options.branch = branch;

  const projectId = args?.['projectId'] as string | undefined;
  if (projectId) options.projectId = projectId;

  return options;
}
