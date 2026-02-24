/**
 * Rules tool facade — delegates to domain-specific modules.
 *
 * - rules-types.ts: Types, interfaces, constants
 * - rules-mutations.ts: Add, update, remove operations
 * - rules-list.ts: List/query operations with mirror fallback
 */

import { QdrantClient } from '@qdrant/js-client-rest';
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
import { FIELD_CONTENT } from '../common/native-bridge.js';
import { addRule, updateRule, removeRule } from './rules-mutations.js';
import { listRules } from './rules-list.js';

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
    projectDetector: ProjectDetector,
  ) {
    const clientConfig: { url: string; apiKey?: string; timeout?: number } = {
      url: config.qdrantUrl,
      timeout: config.qdrantTimeout ?? 5000,
    };
    if (config.qdrantApiKey) {
      clientConfig.apiKey = config.qdrantApiKey;
    }
    this.qdrantClient = new QdrantClient(clientConfig);
    this.daemonClient = daemonClient;
    this.stateManager = stateManager;
    this.projectDetector = projectDetector;
    this.duplicationThreshold = config.duplicationThreshold ?? DEFAULT_DUPLICATION_THRESHOLD;
  }

  async execute(options: RuleOptions): Promise<RuleResponse> {
    switch (options.action) {
      case 'add': {
        // Check for similar rules before inserting
        if (options.content?.trim()) {
          const duplicates = await this.findSimilarRules(options.content);
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
        return updateRule(this.daemonClient, this.stateManager, options);
      case 'remove':
        return removeRule(this.stateManager, options);
      case 'list':
        return listRules(this.qdrantClient, this.stateManager, this.projectDetector, options);
      default:
        return { success: false, action: options.action, message: `Unknown action: ${options.action}` };
    }
  }

  /**
   * Find existing rules similar to the given content using embedding similarity.
   * Returns rules with cosine similarity >= duplicationThreshold.
   */
  private async findSimilarRules(content: string): Promise<Array<Rule & { similarity: number }>> {
    try {
      const embedResponse = await this.daemonClient.embedText({ text: content });
      if (!embedResponse.embedding?.length) return [];

      const searchResult = await this.qdrantClient.search(RULES_COLLECTION, {
        vector: embedResponse.embedding,
        limit: 5,
        score_threshold: this.duplicationThreshold,
        with_payload: true,
      });

      return searchResult
        .filter((point) => point.score >= this.duplicationThreshold)
        .map((point) => ({
          id: String(point.id),
          content: (point.payload?.[FIELD_CONTENT] as string) ?? '',
          scope: (point.payload?.['scope'] as RuleScope) ?? 'global',
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
