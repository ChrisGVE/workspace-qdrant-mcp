/**
 * Scratchpad tool argument builder — parse raw MCP tool arguments into the
 * canonical ScratchpadOptions. Maps every arg-derived field (a missing field
 * here means the option is silently dropped — see the buildSearchOptions drift).
 * `cwd` is intentionally not mapped: it is bound into the request context by the
 * HTTP layer and read via getEffectiveCwd() during tenant resolution.
 */

import type { ScratchpadOptions, ScratchpadAction } from '../tools/scratchpad.js';

export function buildScratchpadOptions(
  args: Record<string, unknown> | undefined
): ScratchpadOptions {
  const options: ScratchpadOptions = { action: args?.['action'] as ScratchpadAction };

  const content = args?.['content'] as string | undefined;
  if (content !== undefined) options.content = content;

  const newContent = args?.['newContent'] as string | undefined;
  if (newContent !== undefined) options.newContent = newContent;

  const title = args?.['title'] as string | undefined;
  if (title !== undefined) options.title = title;

  const tags = args?.['tags'] as string[] | undefined;
  if (tags !== undefined) options.tags = tags;

  const projectId = args?.['projectId'] as string | undefined;
  if (projectId !== undefined) options.projectId = projectId;

  const limit = args?.['limit'] as number | undefined;
  if (limit !== undefined) options.limit = limit;

  return options;
}
