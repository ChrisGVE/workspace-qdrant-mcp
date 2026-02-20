/**
 * Memory tool facade — delegates to domain-specific modules.
 *
 * - memory-types.ts: Types, interfaces, constants
 * - memory-mutations.ts: Add, update, remove operations
 * - memory-list.ts: List/query operations with mirror fallback
 */

import { QdrantClient } from '@qdrant/js-client-rest';
import type { DaemonClient } from '../clients/daemon-client.js';
import type { SqliteStateManager } from '../clients/sqlite-state-manager.js';
import type { ProjectDetector } from '../utils/project-detector.js';

// Re-export all types so existing imports from './memory.js' continue to work
export type {
  MemoryAction,
  MemoryScope,
  MemoryRule,
  MemoryOptions,
  MemoryResponse,
  MemoryToolConfig,
} from './memory-types.js';

import type { MemoryOptions, MemoryResponse, MemoryToolConfig } from './memory-types.js';
import { addRule, updateRule, removeRule } from './memory-mutations.js';
import { listRules } from './memory-list.js';

/**
 * Memory tool for behavioral rules management
 */
export class MemoryTool {
  private readonly qdrantClient: QdrantClient;
  private readonly daemonClient: DaemonClient;
  private readonly stateManager: SqliteStateManager;
  private readonly projectDetector: ProjectDetector;

  constructor(
    config: MemoryToolConfig,
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
  }

  async execute(options: MemoryOptions): Promise<MemoryResponse> {
    switch (options.action) {
      case 'add':
        return addRule(this.daemonClient, this.stateManager, this.projectDetector, options);
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
}
