/**
 * Rules tool facade — delegates to domain-specific modules.
 *
 * - rules-types.ts: Types, interfaces, constants
 * - rules-mutations.ts: Add, update, remove operations
 * - rules-list.ts: List/query operations with mirror fallback
 */

import type { QdrantClient } from '@qdrant/js-client-rest';
import { getQdrantClient } from '../clients/qdrant-client-factory.js';
import type { DaemonClient } from '../clients/daemon-client.js';
import type { SqliteStateManager } from '../clients/sqlite-state-manager.js';
import type { ProjectDetector } from '../utils/project-detector.js';

// Re-export all types
export type {
  RuleAction,
  RuleScope,
  Rule,
  RuleOptions,
  RuleResponse,
  RuleToolConfig,
} from './rules-types.js';

import type { RuleOptions, RuleResponse, RuleToolConfig, Rule, RuleScope } from './rules-types.js';
import { RULES_COLLECTION } from './rules-types.js';
import { TENANT_GLOBAL } from '../constants/tenants.js';
import { FIELD_CONTENT, FIELD_PROJECT_ID } from '../common/native-bridge.js';
import { addRule, updateRule, removeRule } from './rules-mutations.js';
import { listRules } from './rules-list.js';
import { resolveProjectScopeId } from './rules-mutation-helpers.js';

const DEFAULT_DUPLICATION_THRESHOLD = 0.7;

/**
 * Rules tool for behavioral rules management
 */
export class RulesTool {
  private readonly qdrantClient: QdrantClient;
  private readonly daemonClient: DaemonClient;
  private readonly stateManager: SqliteStateManager;
  private readonly projectDetector: ProjectDetector;
  private readonly duplicationThreshold: number;

  constructor(
    config: RuleToolConfig,
    daemonClient: DaemonClient,
    stateManager: SqliteStateManager,
    projectDetector: ProjectDetector
  ) {
    this.qdrantClient = getQdrantClient({
      url: config.qdrantUrl,
      apiKey: config.qdrantApiKey,
      timeout: config.qdrantTimeout ?? 5000,
    });
    this.daemonClient = daemonClient;
    this.stateManager = stateManager;
    this.projectDetector = projectDetector;
    this.duplicationThreshold = config.duplicationThreshold ?? DEFAULT_DUPLICATION_THRESHOLD;
  }

  async execute(options: RuleOptions): Promise<RuleResponse> {
    switch (options.action) {
      case 'add': {
        // Check for similar rules before inserting. F-015: scope the
        // duplicate-detection by (scope, tenant_id) so the same label
        // / similar content in another project does not block this
        // project's add.
        if (options.content?.trim()) {
          const dupScope: RuleScope = options.scope ?? 'project';
          const { resolvedProjectId, error: scopeError } = await resolveProjectScopeId(
            dupScope,
            options.projectId,
            this.projectDetector
          );

          // Tenant hardening: do not run a broad duplicate search when a
          // project-scoped rule cannot resolve a project tenant. Returning
          // the explicit scope error here prevents misleading duplicate
          // errors and, more importantly, prevents accidental global fallback.
          if (scopeError) return scopeError;

          // Idempotent-by-content: an EXACT duplicate (same trimmed content in
          // the same scope/tenant, regardless of label) is a no-op, not a new
          // row. Deterministic and daemon-independent, it runs before the fuzzy
          // embedding-similarity warning below — so re-adding identical content
          // can never create the duplicate pairs the eval found.
          const exact = await this.findExactContentRule(
            options.content,
            dupScope,
            resolvedProjectId
          );
          if (exact) {
            const resolvedLabel = exact.label ?? options.label;
            return {
              success: true,
              action: 'add',
              ...(resolvedLabel ? { label: resolvedLabel } : {}),
              message: exact.label
                ? `Identical rule already exists (label "${exact.label}"); add is a no-op (idempotent by content).`
                : 'Identical rule already exists; add is a no-op (idempotent by content).',
            };
          }

          const duplicates = await this.findSimilarRules(
            options.content,
            dupScope,
            resolvedProjectId
          );
          if (duplicates.length > 0) {
            return {
              success: false,
              action: 'add',
              similar_rules: duplicates,
              message: `Found ${duplicates.length} similar rule(s). Review before adding to avoid duplication.`,
            };
          }
        }
        return addRule(this.daemonClient, this.stateManager, this.projectDetector, options);
      }
      case 'update':
        return updateRule(
          this.daemonClient,
          this.qdrantClient,
          this.stateManager,
          this.projectDetector,
          options
        );
      case 'remove':
        return removeRule(this.stateManager, this.projectDetector, options);
      case 'list':
        return listRules(this.qdrantClient, this.stateManager, this.projectDetector, options);
      default:
        return {
          success: false,
          action: options.action,
          message: `Unknown action: ${options.action}`,
        };
    }
  }

  /**
   * Find an existing rule whose content is byte-identical (after trimming) to
   * `content`, within the same scope/tenant. Drives idempotent `add`: re-adding
   * the same rule — even under a different label — is a no-op instead of a new
   * duplicate row.
   *
   * Deterministic (exact string compare, no embedding) and fail-open: any
   * Qdrant error returns null so the add proceeds rather than being blocked.
   */
  private async findExactContentRule(
    content: string,
    scope: RuleScope,
    projectId: string | undefined
  ): Promise<{ id: string; label?: string } | null> {
    const target = content.trim();
    if (!target) return null;
    try {
      const filter = buildExactScopeFilter(scope, projectId);
      const scrollResult = await this.qdrantClient.scroll(RULES_COLLECTION, {
        limit: 256,
        with_payload: true,
        ...(filter ? { filter } : {}),
      });
      for (const point of scrollResult.points) {
        const existing = (point.payload?.[FIELD_CONTENT] as string | undefined)?.trim();
        if (existing === target) {
          const label = point.payload?.['label'] as string | undefined;
          return { id: String(point.id), ...(label ? { label } : {}) };
        }
      }
      return null;
    } catch {
      // Qdrant unavailable — let the add proceed rather than block it.
      return null;
    }
  }

  /**
   * Find existing rules similar to the given content using embedding similarity.
   * Returns rules with cosine similarity >= duplicationThreshold.
   *
   * F-015: results are scoped by (scope, tenant_id) so a project-scope
   * add is only matched against rules in the same project (plus global
   * rules), and a global-scope add is only matched against global
   * rules. Pre-fix the whole RULES_COLLECTION was scanned and the same
   * label / content in another project blocked the add.
   */
  private async findSimilarRules(
    content: string,
    scope: RuleScope,
    projectId: string | undefined
  ): Promise<Array<Rule & { similarity: number }>> {
    try {
      const embedResponse = await this.daemonClient.embedText({ text: content });
      if (!embedResponse.embedding?.length) return [];

      const searchRequest: {
        vector: number[];
        limit: number;
        score_threshold: number;
        with_payload: boolean;
        filter?: Record<string, unknown>;
      } = {
        vector: embedResponse.embedding,
        limit: 5,
        score_threshold: this.duplicationThreshold,
        with_payload: true,
      };
      const filter = buildDuplicateScopeFilter(scope, projectId);
      if (filter) searchRequest.filter = filter;

      const searchResult = await this.qdrantClient.search(RULES_COLLECTION, searchRequest);

      return searchResult
        .filter((point) => point.score >= this.duplicationThreshold)
        .map((point) => ({
          id: String(point.id),
          content: (point.payload?.[FIELD_CONTENT] as string) ?? '',
          scope: (point.payload?.['scope'] as RuleScope) ?? TENANT_GLOBAL,
          label: (point.payload?.['label'] as string) ?? undefined,
          title: (point.payload?.['title'] as string) ?? undefined,
          similarity: Math.round(point.score * 1000) / 1000,
        }));
    } catch {
      // Embedding or search failed — allow the add to proceed
      return [];
    }
  }
}

/** F-015: Build a Qdrant filter that restricts duplicate detection to
 *  the active rule scope. Project-scope: match the current project_id
 *  (or any global rule too — global rules apply to every project and
 *  remain dup-detectable). Global-scope: only match global rules. */
function buildDuplicateScopeFilter(
  scope: RuleScope,
  projectId: string | undefined
): Record<string, unknown> | null {
  if (scope === 'project') {
    if (!projectId) return null; // unresolvable — let dup detection broaden rather than silently block
    return {
      must: [
        {
          should: [
            { key: FIELD_PROJECT_ID, match: { value: projectId } },
            { key: 'scope', match: { value: 'global' } },
          ],
        },
      ],
    };
  }
  // Global add — only check against other global rules.
  return { must: [{ key: 'scope', match: { value: 'global' } }] };
}

/** Build a Qdrant filter restricting the EXACT-content idempotency check to a
 *  single (scope, tenant) bucket. Unlike {@link buildDuplicateScopeFilter},
 *  global rules are NOT folded into a project-scope check: idempotency must be
 *  exact — a project rule and a global rule with the same text are distinct
 *  rows in distinct scopes. Returns null when a project scope has no tenant
 *  (caller already refuses that path), leaving the check a safe no-op. */
function buildExactScopeFilter(
  scope: RuleScope,
  projectId: string | undefined
): Record<string, unknown> | null {
  if (scope === 'project') {
    if (!projectId) return null;
    return {
      must: [
        { key: 'scope', match: { value: 'project' } },
        { key: FIELD_PROJECT_ID, match: { value: projectId } },
      ],
    };
  }
  return { must: [{ key: 'scope', match: { value: 'global' } }] };
}
