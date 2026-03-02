/**
 * Store handler helpers for URL and scratchpad store types
 */

import type { SqliteStateManager } from './clients/sqlite-state-manager.js';
import type { SessionState } from './server-types.js';
import { COLLECTION_SCRATCHPAD, PRIORITY_HIGH } from './common/native-bridge.js';

type StoreResult = {
  success: boolean;
  message: string;
  queue_id?: string;
  collection: string;
};

/**
 * Store a URL for daemon-side fetch and ingestion.
 *
 * Queues the URL as item_type 'url' in the unified queue.
 * The daemon will fetch the page, extract text, generate embeddings,
 * and store in Qdrant.
 */
export async function storeUrl(
  args: Record<string, unknown> | undefined,
  stateManager: SqliteStateManager,
  sessionState: Pick<SessionState, 'projectId'>
): Promise<StoreResult> {
  const url = args?.['url'] as string;
  if (!url?.trim()) {
    return {
      success: false,
      message: 'url is required when type is "url"',
      collection: '',
    };
  }

  if (!url.startsWith('http://') && !url.startsWith('https://')) {
    return {
      success: false,
      message: 'url must start with http:// or https://',
      collection: '',
    };
  }

  const libraryName = args?.['libraryName'] as string | undefined;
  const title = args?.['title'] as string | undefined;
  const collection = libraryName ? 'libraries' : COLLECTION_SCRATCHPAD;
  const tenantId = libraryName?.trim() || sessionState.projectId || '_global_';

  const payload: Record<string, unknown> = {
    url: url.trim(),
    crawl: false,
    max_depth: 0,
    max_pages: 1,
  };
  if (libraryName) payload['library_name'] = libraryName.trim();
  if (title) payload['title'] = title;

  try {
    const result = stateManager.enqueueUnified(
      'url',
      'add',
      tenantId,
      collection,
      payload,
      PRIORITY_HIGH,
      'main',
      { source: 'mcp_store_url' }
    );

    if (result.status !== 'ok' || !result.data) {
      return {
        success: false,
        message: result.message ?? 'Failed to enqueue URL',
        collection,
      };
    }

    return {
      success: true,
      message: `URL queued for fetch and ingestion (${collection}/${tenantId})`,
      queue_id: result.data.queueId,
      collection,
    };
  } catch (error) {
    const msg = error instanceof Error ? error.message : String(error);
    return {
      success: false,
      message: `Failed to queue URL: ${msg}`,
      collection,
    };
  }
}

/**
 * Store content to scratchpad collection
 */
export async function storeScratchpad(
  args: Record<string, unknown> | undefined,
  stateManager: SqliteStateManager,
  sessionState: Pick<SessionState, 'projectId'>
): Promise<StoreResult> {
  const content = args?.['content'] as string;
  if (!content?.trim()) {
    return {
      success: false,
      message: 'content is required when type is "scratchpad"',
      collection: COLLECTION_SCRATCHPAD,
    };
  }

  const title = args?.['title'] as string | undefined;
  const tags = (args?.['tags'] as string[] | undefined) ?? [];
  const tenantId = sessionState.projectId || '_global_';

  const payload: Record<string, unknown> = {
    content: content.trim(),
    source_type: 'scratchpad',
  };
  if (title?.trim()) payload['title'] = title.trim();
  if (tags.length > 0) payload['tags'] = tags;

  try {
    const result = stateManager.enqueueUnified(
      'text',
      'add',
      tenantId,
      COLLECTION_SCRATCHPAD,
      payload,
      PRIORITY_HIGH,
      'main',
      { source: 'mcp_store_scratchpad' }
    );

    if (result.status !== 'ok' || !result.data) {
      return {
        success: false,
        message: result.message ?? 'Failed to enqueue scratchpad entry',
        collection: COLLECTION_SCRATCHPAD,
      };
    }

    return {
      success: true,
      message: `Scratchpad entry queued for processing (${tenantId})`,
      queue_id: result.data.queueId,
      collection: COLLECTION_SCRATCHPAD,
    };
  } catch (error) {
    const msg = error instanceof Error ? error.message : String(error);
    return {
      success: false,
      message: `Failed to queue scratchpad entry: ${msg}`,
      collection: COLLECTION_SCRATCHPAD,
    };
  }
}
