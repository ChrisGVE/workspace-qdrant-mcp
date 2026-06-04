/**
 * Retrieve tool — direct document access from Qdrant collections.
 *
 * - retrieve-types.ts: Types, constants, helpers
 * - retrieve.ts (this): RetrieveTool class with byId and byFilter operations
 *
 * Tenant isolation invariants (F-002 / F-011):
 *
 * - `projects` collection accesses MUST resolve to a `tenant_id` before any
 *   Qdrant call. `retrieveById` verifies the returned point's `tenant_id`
 *   matches the caller; mismatches return an empty not-found response.
 * - `libraries` collection accesses MUST resolve to a `library_name` (or
 *   `tenant_id`, since the library collection keys both fields). The
 *   project detector is never used as a fall-back for libraries.
 * - `rules` is intentionally mixed-tenancy; no scope verification is
 *   performed for `retrieveById` against rules.
 * - Project-scope retrieve without a resolvable tenant returns an empty
 *   error response and does NOT scroll Qdrant (no broad reads).
 */

import { randomUUID } from 'node:crypto';
import type { QdrantClient } from '@qdrant/js-client-rest';
import { getQdrantClient } from '../clients/qdrant-client-factory.js';
import type { DaemonClient } from '../clients/daemon-client.js';
import type { ProjectDetector } from '../utils/project-detector.js';
import { getEffectiveCwd } from '../utils/request-context.js';
import { FIELD_CONTENT, FIELD_TENANT_ID, FIELD_LIBRARY_NAME } from '../common/native-bridge.js';
import { finishToolEvent, logSearchEvent } from '../clients/search-event-queries.js';
import { SERVER_VERSION as MCP_SERVER_VERSION } from '../server-types.js';

// Re-export all types so existing imports from './retrieve.js' continue to work
export type {
  RetrieveCollectionType,
  RetrieveOptions,
  RetrievedDocument,
  RetrieveResponse,
  RetrieveToolConfig,
} from './retrieve-types.js';

import type {
  RetrieveCollectionType,
  RetrieveOptions,
  RetrievedDocument,
  RetrieveResponse,
  RetrieveToolConfig,
} from './retrieve-types.js';
import { getCollectionName, extractMetadata } from './retrieve-types.js';

/**
 * Pure helper: compute `bytes_out` / `bytes_in` for a retrieve result.
 * Spec `docs/specs/20-token-economy-instrumentation.md` §3.3: for the
 * current implementation (full-document retrieve only — no ranged
 * retrieve) `bytes_in == bytes_out`, so `savings_ratio` is 0%. The row
 * still matters: when `parent_event_id` is populated, the v38
 * `token_savings` view uses it to detect escalation (search → retrieve
 * of the same document).
 */
export function computeRetrieveEconomy(documents: RetrievedDocument[]): {
  bytesOut: number;
  bytesIn: number;
} {
  let bytesOut = 0;
  for (const d of documents) bytesOut += d.content.length;
  // No ranged retrieve yet — full doc cost === served cost.
  return { bytesOut, bytesIn: bytesOut };
}

/** Returned when the caller passes scope but it cannot be resolved. */
function unresolvedTenantResponse(collection: RetrieveCollectionType): RetrieveResponse {
  return {
    success: false,
    documents: [],
    total: 0,
    hasMore: false,
    message:
      `Cannot retrieve from "${collection}" without a resolvable scope. ` +
      'Pass `cwd` (to auto-detect the project) or `projectId` (for projects), ' +
      'or `libraryName` (for libraries).',
  };
}

/**
 * Verify a retrieved point matches the caller's scope.
 * Returns `true` when the point belongs to the resolved tenant/library
 * (or when the collection is rules, which is intentionally mixed).
 */
function payloadMatchesScope(
  payload: Record<string, unknown> | null | undefined,
  collection: RetrieveCollectionType,
  resolvedProjectId: string | undefined,
  libraryName: string | undefined
): boolean {
  if (!payload) return false;
  switch (collection) {
    case 'projects': {
      if (!resolvedProjectId) return false;
      return payload[FIELD_TENANT_ID] === resolvedProjectId;
    }
    case 'libraries': {
      if (!libraryName) return false;
      // The libraries collection stores `library_name` on every point;
      // some legacy paths also key by `tenant_id`. Accept either match.
      return (
        payload[FIELD_LIBRARY_NAME] === libraryName || payload[FIELD_TENANT_ID] === libraryName
      );
    }
    case 'rules':
      // Rules are explicitly mixed-tenancy. Direct-by-ID lookup is
      // therefore not gated on caller scope.
      return true;
    case 'scratchpad':
      // Scratchpad is project-scoped; verify tenant_id matches.
      if (!resolvedProjectId) return false;
      return payload[FIELD_TENANT_ID] === resolvedProjectId;
    default:
      return false;
  }
}

export class RetrieveTool {
  private readonly qdrantClient: QdrantClient;
  private readonly projectDetector: ProjectDetector;
  private readonly daemonClient: DaemonClient | null;

  constructor(
    config: RetrieveToolConfig,
    projectDetector: ProjectDetector,
    daemonClient?: DaemonClient
  ) {
    this.qdrantClient = getQdrantClient({
      url: config.qdrantUrl,
      apiKey: config.qdrantApiKey,
      timeout: config.qdrantTimeout ?? 5000,
    });
    this.projectDetector = projectDetector;
    this.daemonClient = daemonClient ?? null;
  }

  async retrieve(options: RetrieveOptions): Promise<RetrieveResponse> {
    const {
      documentId,
      collection = 'projects',
      filter,
      limit = 10,
      offset = 0,
      projectId,
      libraryName,
    } = options;

    const startTime = Date.now();
    const eventId = randomUUID();

    const collectionName = getCollectionName(collection);

    // Log start. queryText carries documentId for by-id lookups, or a
    // compact filter summary for by-filter scans, so retrieve events
    // remain self-describing under `wqm admin token-savings`.
    logSearchEvent(this.daemonClient, {
      id: eventId,
      actor: 'claude',
      tool: 'mcp_qdrant',
      op: 'retrieve',
      queryText: documentId ?? (filter ? JSON.stringify(filter).slice(0, 500) : `:${collection}`),
      topK: limit,
      projectId: projectId,
    });

    // F-002 / F-011: resolve the tenant context up front so that BOTH
    // by-id verification AND by-filter scoping share the same answer.
    const resolvedProjectId =
      collection === 'projects' || collection === 'scratchpad'
        ? (projectId ?? (await this.resolveProjectId()))
        : undefined;

    if (documentId) {
      const result = await this.retrieveById(
        collectionName,
        collection,
        documentId,
        resolvedProjectId,
        libraryName
      );
      this.finishRetrieve(eventId, result, startTime);
      return result;
    }

    // F-011: project-scope retrieve without a resolved tenant MUST refuse
    // to scroll. Same rule for libraries when no library_name is given.
    if (collection === 'projects' && !resolvedProjectId) {
      const result = unresolvedTenantResponse('projects');
      this.finishRetrieve(eventId, result, startTime, 'unresolved_tenant');
      return result;
    }
    if (collection === 'scratchpad' && !resolvedProjectId) {
      const result = unresolvedTenantResponse('scratchpad');
      this.finishRetrieve(eventId, result, startTime, 'unresolved_tenant');
      return result;
    }
    if (collection === 'libraries' && !libraryName) {
      const result = unresolvedTenantResponse('libraries');
      this.finishRetrieve(eventId, result, startTime, 'unresolved_tenant');
      return result;
    }

    const filterParams: {
      collectionName: string;
      collection: RetrieveCollectionType;
      filter?: Record<string, string>;
      limit: number;
      offset: number;
      projectId?: string;
      libraryName?: string;
    } = { collectionName, collection, limit, offset };
    if (filter) filterParams.filter = filter;
    if (resolvedProjectId) filterParams.projectId = resolvedProjectId;
    if (libraryName) filterParams.libraryName = libraryName;

    const result = await this.retrieveByFilter(filterParams);
    this.finishRetrieve(eventId, result, startTime);
    return result;
  }

  /** Record post-execution metrics for a retrieve call. */
  private finishRetrieve(
    eventId: string,
    response: RetrieveResponse,
    startTime: number,
    outcome?: string
  ): void {
    const economy = computeRetrieveEconomy(response.documents);
    const finish: import('../clients/search-event-queries.js').ToolEventFinish = {
      resultCount: response.documents.length,
      latencyMs: Date.now() - startTime,
      bytesIn: economy.bytesIn,
      bytesOut: economy.bytesOut,
      toolVersion: MCP_SERVER_VERSION,
    };
    if (outcome !== undefined) finish.outcome = outcome;
    finishToolEvent(this.daemonClient, eventId, finish);
  }

  /**
   * Direct point lookup by ID. Verifies the returned point matches the
   * caller's resolved scope (F-002). A mismatch is returned as a clean
   * "not found" rather than the foreign document, so the leak path
   * collapses to the same response as a genuinely missing ID.
   */
  private async retrieveById(
    collectionName: string,
    collection: RetrieveCollectionType,
    documentId: string,
    resolvedProjectId: string | undefined,
    libraryName: string | undefined
  ): Promise<RetrieveResponse> {
    // F-002: project-scope and library-scope lookups MUST resolve their
    // scope before reading. Without it, the verification step below
    // cannot distinguish "owned" from "foreign", so refuse the read.
    if (collection === 'projects' && !resolvedProjectId) {
      return unresolvedTenantResponse('projects');
    }
    if (collection === 'scratchpad' && !resolvedProjectId) {
      return unresolvedTenantResponse('scratchpad');
    }
    if (collection === 'libraries' && !libraryName) {
      return unresolvedTenantResponse('libraries');
    }

    try {
      const result = await this.qdrantClient.retrieve(collectionName, {
        ids: [documentId],
        with_payload: true,
        with_vector: false,
      });

      const point = result[0];
      if (!point) {
        return { success: false, documents: [], message: `Document not found: ${documentId}` };
      }

      // F-002: ownership check. A mismatch is reported as not-found —
      // we MUST NOT leak that the ID exists in a foreign tenant.
      if (!payloadMatchesScope(point.payload, collection, resolvedProjectId, libraryName)) {
        return { success: false, documents: [], message: `Document not found: ${documentId}` };
      }

      const document: RetrievedDocument = {
        id: String(point.id),
        content: (point.payload?.[FIELD_CONTENT] as string) ?? '',
        metadata: extractMetadata(point.payload),
      };

      return { success: true, documents: [document], total: 1, hasMore: false };
    } catch (error) {
      return {
        success: false,
        documents: [],
        message: `Failed to retrieve document: ${error instanceof Error ? error.message : 'unknown error'}`,
      };
    }
  }

  private async retrieveByFilter(params: {
    collectionName: string;
    collection: RetrieveCollectionType;
    filter?: Record<string, string>;
    limit: number;
    offset: number;
    projectId?: string;
    libraryName?: string;
  }): Promise<RetrieveResponse> {
    const { collectionName, collection, filter, limit, offset, projectId, libraryName } = params;

    try {
      const qdrantFilter = this.buildFilter(collection, filter, projectId, libraryName);
      const scrollRequest: {
        limit: number;
        offset?: number;
        with_payload: boolean;
        with_vector: boolean;
        filter?: Record<string, unknown>;
      } = { limit: limit + 1, with_payload: true, with_vector: false };
      if (offset > 0) scrollRequest.offset = offset;
      if (qdrantFilter) scrollRequest.filter = qdrantFilter;

      const result = await this.qdrantClient.scroll(collectionName, scrollRequest);

      const hasMore = result.points.length > limit;
      const points = hasMore ? result.points.slice(0, limit) : result.points;

      const documents: RetrievedDocument[] = points.map((point) => ({
        id: String(point.id),
        content: (point.payload?.[FIELD_CONTENT] as string) ?? '',
        metadata: extractMetadata(point.payload),
      }));

      return { success: true, documents, total: documents.length, hasMore };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'unknown error';
      if (errorMessage.includes('not found') || errorMessage.includes("doesn't exist")) {
        return {
          success: true,
          documents: [],
          total: 0,
          hasMore: false,
          message: 'Collection not found or empty',
        };
      }
      return {
        success: false,
        documents: [],
        message: `Failed to retrieve documents: ${errorMessage}`,
      };
    }
  }

  /**
   * Build the Qdrant filter for scroll-based retrieve.
   *
   * Callers MUST have already validated the scope (`projects` →
   * `projectId`; `libraries` → `libraryName`) — this helper trusts the
   * inputs and is no longer responsible for refusing on an unresolvable
   * tenant. That refusal happens in `retrieve()`.
   */
  private buildFilter(
    collection: RetrieveCollectionType,
    filter?: Record<string, string>,
    projectId?: string,
    libraryName?: string
  ): Record<string, unknown> | null {
    const mustConditions: Record<string, unknown>[] = [];

    if ((collection === 'projects' || collection === 'scratchpad') && projectId) {
      mustConditions.push({ key: FIELD_TENANT_ID, match: { value: projectId } });
    } else if (collection === 'libraries' && libraryName) {
      mustConditions.push({ key: FIELD_TENANT_ID, match: { value: libraryName } });
    }

    if (filter) {
      for (const [key, value] of Object.entries(filter)) {
        mustConditions.push({ key, match: { value } });
      }
    }

    return mustConditions.length > 0 ? { must: mustConditions } : null;
  }

  private async resolveProjectId(): Promise<string | undefined> {
    const cwd = getEffectiveCwd();
    const projectInfo = await this.projectDetector.getProjectInfo(cwd, false, {
      fallbackToSoleProject: true,
    });
    return projectInfo?.projectId;
  }
}
