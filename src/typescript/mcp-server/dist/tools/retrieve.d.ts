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
import type { ProjectDetector } from '../utils/project-detector.js';
export type { RetrieveCollectionType, RetrieveOptions, RetrievedDocument, RetrieveResponse, RetrieveToolConfig, } from './retrieve-types.js';
import type { RetrieveOptions, RetrieveResponse, RetrieveToolConfig } from './retrieve-types.js';
export declare class RetrieveTool {
    private readonly qdrantClient;
    private readonly projectDetector;
    constructor(config: RetrieveToolConfig, projectDetector: ProjectDetector);
    retrieve(options: RetrieveOptions): Promise<RetrieveResponse>;
    /**
     * Direct point lookup by ID. Verifies the returned point matches the
     * caller's resolved scope (F-002). A mismatch is returned as a clean
     * "not found" rather than the foreign document, so the leak path
     * collapses to the same response as a genuinely missing ID.
     */
    private retrieveById;
    private retrieveByFilter;
    /**
     * Build the Qdrant filter for scroll-based retrieve.
     *
     * Callers MUST have already validated the scope (`projects` →
     * `projectId`; `libraries` → `libraryName`) — this helper trusts the
     * inputs and is no longer responsible for refusing on an unresolvable
     * tenant. That refusal happens in `retrieve()`.
     */
    private buildFilter;
    private resolveProjectId;
}
//# sourceMappingURL=retrieve.d.ts.map