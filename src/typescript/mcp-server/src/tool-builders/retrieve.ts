/**
 * Retrieve tool argument builder — parse raw MCP tool arguments into RetrieveOptions
 */

import { RETRIEVE_ARG_KEYS } from '../tool-definitions/retrieve.js';

const KNOWN_ARG_KEYS: ReadonlySet<string> = new Set(RETRIEVE_ARG_KEYS);

export type RetrieveOptions = {
  documentId?: string;
  collection?: 'projects' | 'libraries' | 'rules' | 'scratchpad';
  filter?: Record<string, string>;
  limit?: number;
  offset?: number;
  projectId?: string;
  libraryName?: string;
  /**
   * Argument names the caller passed that retrieve does not accept. The
   * tool refuses such calls loudly instead of silently dropping the args
   * (e.g. `query` — a search parameter — would otherwise degrade into a
   * confusing unresolved-scope error).
   */
  unknownArgs?: string[];
};

/** Build retrieve options from raw tool arguments */
export function buildRetrieveOptions(args: Record<string, unknown> | undefined): RetrieveOptions {
  const options: RetrieveOptions = {};

  const documentId = args?.['documentId'] as string | undefined;
  if (documentId) options.documentId = documentId;

  const collection = args?.['collection'] as string | undefined;
  if (
    collection === 'projects' ||
    collection === 'libraries' ||
    collection === 'rules' ||
    collection === 'scratchpad'
  ) {
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

  const unknownArgs = args ? Object.keys(args).filter((key) => !KNOWN_ARG_KEYS.has(key)) : [];
  if (unknownArgs.length > 0) options.unknownArgs = unknownArgs;

  return options;
}
