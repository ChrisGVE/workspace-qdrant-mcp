/**
 * Store handler helpers for URL and scratchpad store types
 */
import { randomUUID } from 'node:crypto';
import { COLLECTION_SCRATCHPAD, PRIORITY_HIGH } from './common/native-bridge.js';
import { TENANT_GLOBAL } from './constants/tenants.js';
import { utcNow } from './utils/timestamps.js';
/**
 * Pre-enqueue URL validation. Rejects malformed input, non-http(s) schemes,
 * and obviously bad hostnames so the daemon does not waste a queue cycle on
 * URLs it would reject at fetch time. Full SSRF policy (private-network
 * denylist, DNS rebinding defense, redirect re-validation) is enforced
 * daemon-side; this is a fast-fail surface for user-facing error messages.
 */
export function validateUrlInput(raw) {
    if (typeof raw !== 'string' || raw.trim().length === 0) {
        return { ok: false, message: 'url is required when type is "url"' };
    }
    const trimmed = raw.trim();
    let parsed;
    try {
        parsed = new URL(trimmed);
    }
    catch {
        return { ok: false, message: 'url is malformed (failed to parse)' };
    }
    if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
        return {
            ok: false,
            message: `url must use http:// or https:// (got ${parsed.protocol})`,
        };
    }
    const host = parsed.hostname;
    if (!host || host.length === 0) {
        return { ok: false, message: 'url has empty hostname' };
    }
    if (/^[.\s]+$/.test(host)) {
        return { ok: false, message: 'url has invalid hostname (dots/whitespace only)' };
    }
    return { ok: true };
}
/** Build the URL queue payload. */
function buildUrlPayload(url, libraryName, title) {
    const payload = {
        url: url.trim(),
        crawl: false,
        max_depth: 0,
        max_pages: 1,
    };
    if (libraryName)
        payload['library_name'] = libraryName.trim();
    if (title)
        payload['title'] = title;
    return payload;
}
/**
 * Store a URL for daemon-side fetch and ingestion.
 *
 * Queues the URL as item_type 'url' in the unified queue.
 * The daemon will fetch the page, extract text, generate embeddings,
 * and store in Qdrant.
 */
export async function storeUrl(args, stateManager, sessionState) {
    const url = args?.['url'];
    const validation = validateUrlInput(url);
    if (!validation.ok) {
        return { success: false, message: validation.message, collection: '' };
    }
    const libraryName = args?.['libraryName'];
    const title = args?.['title'];
    const collection = libraryName ? 'libraries' : COLLECTION_SCRATCHPAD;
    const tenantId = libraryName?.trim() || sessionState.projectId || TENANT_GLOBAL;
    const payload = buildUrlPayload(url, libraryName, title);
    try {
        const result = await stateManager.enqueueUnified('url', 'add', tenantId, collection, payload, PRIORITY_HIGH, 'main', { source: 'mcp_store_url' });
        if (result.status !== 'ok' || !result.data)
            return { success: false, message: result.message ?? 'Failed to enqueue URL', collection };
        return {
            success: true,
            message: `URL queued for fetch and ingestion (${collection}/${tenantId})`,
            queue_id: result.data.queueId,
            collection,
        };
    }
    catch (error) {
        return {
            success: false,
            message: `Failed to queue URL: ${error instanceof Error ? error.message : String(error)}`,
            collection,
        };
    }
}
/** Write a scratchpad entry to the local mirror (fire-and-forget). */
function writeScratchpadMirror(stateManager, content, title, tags, tenantId) {
    const now = utcNow();
    stateManager.upsertScratchpadMirror({
        scratchpadId: randomUUID(),
        title: title?.trim() ?? null,
        content: content.trim(),
        tags: JSON.stringify(tags),
        tenantId,
        createdAt: now,
        updatedAt: now,
    });
}
/**
 * Store content to scratchpad collection.
 *
 * After successful enqueue, also writes to scratchpad_mirror for rebuild
 * recovery. The mirror write is fire-and-forget (advisory).
 */
/** Build the scratchpad queue payload. */
function buildScratchpadPayload(content, title, tags) {
    const payload = { content: content.trim(), source_type: 'scratchpad' };
    if (title?.trim())
        payload['title'] = title.trim();
    if (tags.length > 0)
        payload['tags'] = tags;
    return payload;
}
export async function storeScratchpad(args, stateManager, sessionState) {
    const content = args?.['content'];
    if (!content?.trim())
        return {
            success: false,
            message: 'content is required when type is "scratchpad"',
            collection: COLLECTION_SCRATCHPAD,
        };
    const title = args?.['title'];
    const tags = args?.['tags'] ?? [];
    const tenantId = sessionState.projectId || TENANT_GLOBAL;
    const payload = buildScratchpadPayload(content, title, tags);
    return enqueueScratchpadEntry(stateManager, payload, tenantId, content, title, tags);
}
async function enqueueScratchpadEntry(stateManager, payload, tenantId, content, title, tags) {
    try {
        const result = await stateManager.enqueueUnified('text', 'add', tenantId, COLLECTION_SCRATCHPAD, payload, PRIORITY_HIGH, 'main', { source: 'mcp_store_scratchpad' });
        if (result.status !== 'ok' || !result.data)
            return {
                success: false,
                message: result.message ?? 'Failed to enqueue scratchpad entry',
                collection: COLLECTION_SCRATCHPAD,
            };
        writeScratchpadMirror(stateManager, content, title, tags, tenantId);
        return {
            success: true,
            message: `Scratchpad entry queued for processing (${tenantId})`,
            queue_id: result.data.queueId,
            collection: COLLECTION_SCRATCHPAD,
        };
    }
    catch (error) {
        return {
            success: false,
            message: `Failed to queue scratchpad entry: ${error instanceof Error ? error.message : String(error)}`,
            collection: COLLECTION_SCRATCHPAD,
        };
    }
}
//# sourceMappingURL=store-handlers.js.map