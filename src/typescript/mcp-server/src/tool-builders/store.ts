/**
 * Store tool argument builder — parse raw MCP tool arguments into StoreOptions
 */

import type { SessionState } from '../server-types.js';

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

// ── Validation ────────────────────────────────────────────────────────────

function validateStoreArgs(args: Record<string, unknown> | undefined): string {
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

  return content;
}

// ── Option extractors ─────────────────────────────────────────────────────

function extractTargetOptions(
  args: Record<string, unknown> | undefined,
  options: StoreOptions,
  sessionState: Pick<SessionState, 'projectId'>,
): void {
  const libraryName = args?.['libraryName'] as string | undefined;
  if (libraryName) options.libraryName = libraryName;

  const forProject = args?.['forProject'] as boolean | undefined;
  if (forProject) {
    options.forProject = true;
    if (sessionState.projectId) options.projectId = sessionState.projectId;
  }
}

function extractMetadataOptions(
  args: Record<string, unknown> | undefined,
  options: StoreOptions,
): void {
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
}

/**
 * Build store options from raw tool arguments.
 * Store tool is for libraries collection ONLY per spec.
 */
export function buildStoreOptions(
  args: Record<string, unknown> | undefined,
  sessionState: Pick<SessionState, 'projectId'>
): StoreOptions {
  const content = validateStoreArgs(args);
  const options: StoreOptions = { content };

  extractTargetOptions(args, options, sessionState);
  extractMetadataOptions(args, options);

  return options;
}
