/**
 * Retrieve tool argument builder — parse raw MCP tool arguments into RetrieveOptions
 */

export type RetrieveOptions = {
  documentId?: string;
  collection?: 'projects' | 'libraries' | 'rules';
  filter?: Record<string, string>;
  limit?: number;
  offset?: number;
  projectId?: string;
  libraryName?: string;
};

/** Build retrieve options from raw tool arguments */
export function buildRetrieveOptions(args: Record<string, unknown> | undefined): RetrieveOptions {
  const options: RetrieveOptions = {};

  const documentId = args?.['documentId'] as string | undefined;
  if (documentId) options.documentId = documentId;

  const collection = args?.['collection'] as string | undefined;
  if (collection === 'projects' || collection === 'libraries' || collection === 'rules') {
    options.collection = collection;
  }

  const filter = args?.['filter'] as Record<string, string> | undefined;
  if (filter) options.filter = filter;

  const limit = args?.['limit'] as number | undefined;
  if (limit !== undefined) options.limit = limit;

  const offset = args?.['offset'] as number | undefined;
  if (offset !== undefined) options.offset = offset;

  const projectId = args?.['projectId'] as string | undefined;
  if (projectId) options.projectId = projectId;

  const libraryName = args?.['libraryName'] as string | undefined;
  if (libraryName) options.libraryName = libraryName;

  return options;
}
