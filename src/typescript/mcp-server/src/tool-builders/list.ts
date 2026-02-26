/**
 * List tool argument builder — parse raw MCP tool arguments into ListOptions
 */

import type { ListOptions } from '../tools/list-files-types.js';

export type { ListOptions };

/** Build list options from raw tool arguments */
export function buildListOptions(args: Record<string, unknown> | undefined): ListOptions {
  const options: ListOptions = {};

  const path = args?.['path'] as string | undefined;
  if (path) options.path = path;

  const depth = args?.['depth'] as number | undefined;
  if (depth !== undefined) options.depth = depth;

  const format = args?.['format'] as string | undefined;
  if (format === 'tree' || format === 'summary' || format === 'flat') options.format = format;

  const fileType = args?.['fileType'] as string | undefined;
  if (fileType) options.fileType = fileType;

  const language = args?.['language'] as string | undefined;
  if (language) options.language = language;

  const extension = args?.['extension'] as string | undefined;
  if (extension) options.extension = extension;

  const pattern = args?.['pattern'] as string | undefined;
  if (pattern) options.pattern = pattern;

  const includeTests = args?.['includeTests'] as boolean | undefined;
  if (includeTests !== undefined) options.includeTests = includeTests;

  const limit = args?.['limit'] as number | undefined;
  if (limit !== undefined) options.limit = limit;

  const projectId = args?.['projectId'] as string | undefined;
  if (projectId) options.projectId = projectId;

  const component = args?.['component'] as string | undefined;
  if (component) options.component = component;

  return options;
}
