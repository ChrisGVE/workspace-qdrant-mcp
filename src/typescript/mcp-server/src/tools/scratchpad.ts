/**
 * Scratchpad management tool — list, update, and delete existing scratchpad
 * notes. (Creation stays in `store(type="scratchpad")`, which also wires the
 * project tenant + recall-lane behavior.)
 *
 * Identity is content-addressed: a note's Qdrant point id derives from
 * `hash(tenant_id, content)`, so an entry is identified by its current
 * `content` (obtained from a prior `search`/`scratchpad list`). Mutations are
 * enqueued to the unified queue (daemon-owned writes); the daemon removes the
 * point + its mirror row (delete) or upserts the new content and evicts the
 * superseded point (update). Reads (list) scroll Qdrant directly.
 */

import type { QdrantClient } from '@qdrant/js-client-rest';
import { getQdrantClient } from '../clients/qdrant-client-factory.js';
import type { SqliteStateManager } from '../clients/sqlite-state-manager.js';
import type { ProjectDetector } from '../utils/project-detector.js';
import { randomUUID } from 'node:crypto';
import { utcNow } from '../utils/timestamps.js';
import {
  COLLECTION_SCRATCHPAD,
  PRIORITY_HIGH,
  FIELD_TENANT_ID,
  FIELD_CONTENT,
  FIELD_TITLE,
} from '../common/native-bridge.js';
import { TENANT_GLOBAL } from '../constants/tenants.js';
import { resolveProjectIdentity } from './branch-scope.js';

export type ScratchpadAction = 'list' | 'update' | 'delete';

export interface ScratchpadOptions {
  action: ScratchpadAction;
  /** Current text of the target note — its identity for update/delete. */
  content?: string;
  /** Replacement text (update only). */
  newContent?: string;
  /** New title (update only). */
  title?: string;
  /** New tags (update only). */
  tags?: string[];
  /** Tenant the note belongs to (takes precedence over cwd). */
  projectId?: string;
  /** Max entries for list (default 50). */
  limit?: number;
}

export interface ScratchpadEntry {
  id: string;
  content: string;
  title?: string;
  tags?: string[];
  created_at?: string;
  updated_at?: string;
}

export interface ScratchpadResponse {
  success: boolean;
  action: ScratchpadAction;
  message?: string;
  entries?: ScratchpadEntry[];
  count?: number;
  queue_id?: string;
  tenant_id?: string;
}

export interface ScratchpadToolConfig {
  qdrantUrl: string;
  qdrantApiKey?: string;
  qdrantTimeout?: number;
}

export class ScratchpadTool {
  private readonly qdrantClient: QdrantClient;
  private readonly stateManager: SqliteStateManager;
  private readonly projectDetector: ProjectDetector;

  constructor(
    config: ScratchpadToolConfig,
    stateManager: SqliteStateManager,
    projectDetector: ProjectDetector
  ) {
    this.qdrantClient = getQdrantClient({
      url: config.qdrantUrl,
      apiKey: config.qdrantApiKey,
      timeout: config.qdrantTimeout ?? 5000,
    });
    this.stateManager = stateManager;
    this.projectDetector = projectDetector;
  }

  /**
   * Resolve the tenant whose notes the action targets. Mirrors the store path:
   * explicit projectId → project detected from the effective cwd → global. No
   * sessionState here (the tool is stateless); pass the tenant_id seen in a
   * search/list result as projectId to target a specific project's notes.
   */
  private async resolveTenant(projectId: string | undefined): Promise<string> {
    if (projectId && projectId.trim()) return projectId.trim();
    try {
      const identity = await resolveProjectIdentity(this.projectDetector, undefined);
      if (identity.projectId) return identity.projectId;
    } catch {
      // fall through to global
    }
    return TENANT_GLOBAL;
  }

  async execute(options: ScratchpadOptions): Promise<ScratchpadResponse> {
    const tenantId = await this.resolveTenant(options.projectId);
    switch (options.action) {
      case 'list':
        return this.list(tenantId, options.limit ?? 50);
      case 'delete':
        return this.delete(tenantId, options.content);
      case 'update':
        return this.update(tenantId, options);
      default:
        return {
          success: false,
          action: options.action,
          message: `Unknown action: ${options.action}`,
        };
    }
  }

  private async list(tenantId: string, limit: number): Promise<ScratchpadResponse> {
    try {
      const result = await this.qdrantClient.scroll(COLLECTION_SCRATCHPAD, {
        filter: { must: [{ key: FIELD_TENANT_ID, match: { value: tenantId } }] },
        limit,
        with_payload: true,
      });
      const entries: ScratchpadEntry[] = result.points.map((p) => {
        const payload = (p.payload ?? {}) as Record<string, unknown>;
        const entry: ScratchpadEntry = {
          id: String(p.id),
          content: (payload[FIELD_CONTENT] as string) ?? '',
        };
        const title = payload[FIELD_TITLE] as string | undefined;
        if (title) entry.title = title;
        const tags = payload['tags'] as string[] | undefined;
        if (Array.isArray(tags) && tags.length > 0) entry.tags = tags;
        const createdAt = payload['created_at'] as string | undefined;
        if (createdAt) entry.created_at = createdAt;
        const updatedAt = payload['updated_at'] as string | undefined;
        if (updatedAt) entry.updated_at = updatedAt;
        return entry;
      });
      return {
        success: true,
        action: 'list',
        entries,
        count: entries.length,
        tenant_id: tenantId,
        message: `Found ${entries.length} scratchpad entr${entries.length === 1 ? 'y' : 'ies'} for ${tenantId}`,
      };
    } catch (error) {
      return {
        success: false,
        action: 'list',
        message: `Failed to list scratchpad entries: ${error instanceof Error ? error.message : 'unknown error'}`,
      };
    }
  }

  /**
   * Does a scratchpad note with EXACTLY this content exist for the tenant?
   * update/delete are content-addressed (document_id = hash(tenant, content)),
   * so a near-miss (e.g. a truncated `search` hit) would otherwise silently
   * no-op. Fails OPEN on a Qdrant error — never blocks the mutation.
   */
  private async noteExists(tenantId: string, content: string): Promise<boolean> {
    try {
      const res = await this.qdrantClient.scroll(COLLECTION_SCRATCHPAD, {
        filter: {
          must: [
            { key: FIELD_TENANT_ID, match: { value: tenantId } },
            { key: FIELD_CONTENT, match: { value: content } },
          ],
        },
        limit: 1,
        with_payload: false,
      });
      return (res.points?.length ?? 0) > 0;
    } catch {
      return true; // fail open: a transient lookup error must not block the op
    }
  }

  /** Shared "exact content not found" message for update/delete. */
  private notFoundMessage(tenantId: string): string {
    return (
      `No scratchpad entry with that exact content was found for ${tenantId}. ` +
      'Entries are content-addressed, so the text must match the note VERBATIM — ' +
      'get it from `scratchpad list` (which returns full, untruncated content), ' +
      'not from a `search` hit (whose content may be truncated).'
    );
  }

  private async delete(tenantId: string, content: string | undefined): Promise<ScratchpadResponse> {
    if (!content?.trim()) {
      return {
        success: false,
        action: 'delete',
        message:
          'content is required for delete — the current text of the note to remove (from `scratchpad list`).',
      };
    }
    if (!(await this.noteExists(tenantId, content.trim()))) {
      return {
        success: false,
        action: 'delete',
        message: this.notFoundMessage(tenantId),
        tenant_id: tenantId,
      };
    }
    const result = await this.stateManager.enqueueUnified(
      'text',
      'delete',
      tenantId,
      COLLECTION_SCRATCHPAD,
      { content: content.trim(), source_type: 'scratchpad' },
      PRIORITY_HIGH,
      'main',
      { source: 'mcp_scratchpad_tool' }
    );
    if (result.status !== 'ok' || !result.data) {
      return {
        success: false,
        action: 'delete',
        message: result.message ?? 'Failed to enqueue scratchpad delete',
      };
    }
    return {
      success: true,
      action: 'delete',
      message: `Scratchpad entry deletion queued for processing (${tenantId})`,
      queue_id: result.data.queueId,
      tenant_id: tenantId,
    };
  }

  private async update(tenantId: string, options: ScratchpadOptions): Promise<ScratchpadResponse> {
    const oldContent = options.content;
    const newContent = options.newContent;
    if (!oldContent?.trim()) {
      return {
        success: false,
        action: 'update',
        message:
          'content is required for update — the CURRENT text of the note to edit (from a search/list result).',
      };
    }
    if (!newContent?.trim()) {
      return {
        success: false,
        action: 'update',
        message: 'newContent is required for update — the replacement text.',
      };
    }
    if (!(await this.noteExists(tenantId, oldContent.trim()))) {
      return {
        success: false,
        action: 'update',
        message: this.notFoundMessage(tenantId),
        tenant_id: tenantId,
      };
    }

    const payload: Record<string, unknown> = {
      content: newContent.trim(),
      old_content: oldContent.trim(),
      source_type: 'scratchpad',
    };
    if (options.title?.trim()) payload['title'] = options.title.trim();
    if (options.tags && options.tags.length > 0) payload['tags'] = options.tags;

    const result = await this.stateManager.enqueueUnified(
      'text',
      'update',
      tenantId,
      COLLECTION_SCRATCHPAD,
      payload,
      PRIORITY_HIGH,
      'main',
      { source: 'mcp_scratchpad_tool' }
    );
    if (result.status !== 'ok' || !result.data) {
      return {
        success: false,
        action: 'update',
        message: result.message ?? 'Failed to enqueue scratchpad update',
      };
    }

    // Refresh the advisory mirror with the new content (best-effort). The daemon
    // evicts the old mirror row by old content; this writes the new one so the
    // mirror stays usable for the Qdrant-down fallback before the next rebuild.
    const now = utcNow();
    this.stateManager.upsertScratchpadMirror({
      scratchpadId: randomUUID(),
      title: options.title?.trim() ?? null,
      content: newContent.trim(),
      tags: JSON.stringify(options.tags ?? []),
      tenantId,
      createdAt: now,
      updatedAt: now,
    });

    return {
      success: true,
      action: 'update',
      message: `Scratchpad entry update queued for processing (${tenantId})`,
      queue_id: result.data.queueId,
      tenant_id: tenantId,
    };
  }
}
