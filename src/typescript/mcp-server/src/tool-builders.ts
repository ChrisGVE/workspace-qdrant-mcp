/**
 * Tool argument builders — parse raw MCP tool arguments into typed option objects
 */

import type { SessionState } from './server-types.js';

// ---- Return types ----

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

export type RetrieveOptions = {
  documentId?: string;
  collection?: 'projects' | 'libraries' | 'rules';
  filter?: Record<string, string>;
  limit?: number;
  offset?: number;
  projectId?: string;
  libraryName?: string;
};

export type RuleOptions = {
  action: 'add' | 'update' | 'remove' | 'list';
  content?: string;
  label?: string;
  scope?: 'global' | 'project';
  projectId?: string;
  title?: string;
  tags?: string[];
  priority?: number;
  limit?: number;
};

export type StoreOptions = {
  content: string;
  libraryName?: string;
  forProject?: boolean;
  projectId?: string;
  title?: string;
  url?: string;
  filePath?: string;
  sourceType?: 'user_input' | 'web' | 'file' | 'scratchbook' | 'note';
  metadata?: Record<string, string>;
};

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

export type { ListOptions } from './tools/list-files-types.js';

// ---- Builders ----

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

/** Build rule options from raw tool arguments */
export function buildRuleOptions(args: Record<string, unknown> | undefined): RuleOptions {
  const action = args?.['action'] as string;
  if (action !== 'add' && action !== 'update' && action !== 'remove' && action !== 'list') {
    throw new Error(`Invalid rules action: ${action}`);
  }

  const options: RuleOptions = { action };

  const content = args?.['content'] as string | undefined;
  if (content) options.content = content;

  const label = args?.['label'] as string | undefined;
  if (label) options.label = label;

  const scope = args?.['scope'] as string | undefined;
  if (scope === 'global' || scope === 'project') options.scope = scope;

  const projectId = args?.['projectId'] as string | undefined;
  if (projectId) options.projectId = projectId;

  const title = args?.['title'] as string | undefined;
  if (title) options.title = title;

  const tags = args?.['tags'] as string[] | undefined;
  if (tags) options.tags = tags;

  const priority = args?.['priority'] as number | undefined;
  if (priority !== undefined) options.priority = priority;

  const limit = args?.['limit'] as number | undefined;
  if (limit !== undefined) options.limit = limit;

  return options;
}

/**
 * Build store options from raw tool arguments.
 * Store tool is for libraries collection ONLY per spec.
 */
export function buildStoreOptions(
  args: Record<string, unknown> | undefined,
  sessionState: Pick<SessionState, 'projectId'>
): StoreOptions {
  const content = args?.['content'] as string;
  if (!content) {
    throw new Error('Content is required for store operation');
  }

  const forProject = args?.['forProject'] as boolean | undefined;
  const libraryName = args?.['libraryName'] as string | undefined;

  if (!forProject && !libraryName) {
    throw new Error(
      'libraryName is required - store tool is for libraries collection only. ' +
      'Use forProject: true to store to the current project\'s library.'
    );
  }

  const options: StoreOptions = { content };

  if (libraryName) options.libraryName = libraryName;
  if (forProject) {
    options.forProject = true;
    if (sessionState.projectId) options.projectId = sessionState.projectId;
  }

  const title = args?.['title'] as string | undefined;
  if (title) options.title = title;

  const url = args?.['url'] as string | undefined;
  if (url) options.url = url;

  const filePath = args?.['filePath'] as string | undefined;
  if (filePath) options.filePath = filePath;

  const sourceType = args?.['sourceType'] as string | undefined;
  if (
    sourceType === 'user_input' ||
    sourceType === 'web' ||
    sourceType === 'file' ||
    sourceType === 'scratchbook' ||
    sourceType === 'note'
  ) {
    options.sourceType = sourceType;
  }

  const metadata = args?.['metadata'] as Record<string, string> | undefined;
  if (metadata) options.metadata = metadata;

  return options;
}

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

/** Build list options from raw tool arguments */
export function buildListOptions(args: Record<string, unknown> | undefined): import('./tools/list-files-types.js').ListOptions {
  const options: import('./tools/list-files-types.js').ListOptions = {};

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
