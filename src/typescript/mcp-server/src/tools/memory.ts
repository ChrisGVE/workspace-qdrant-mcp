/**
 * Memory tool implementation for rule management
 *
 * Provides behavioral rules management with:
 * - Add: Store new rules (global or project-scoped)
 * - Update: Modify existing rules
 * - Remove: Delete rules
 * - List: Query rules by scope
 *
 * Uses unified_queue fallback when daemon unavailable (per ADR-002)
 */

import { QdrantClient } from '@qdrant/js-client-rest';
import { randomUUID } from 'node:crypto';
import type { DaemonClient } from '../clients/daemon-client.js';
import type { SqliteStateManager } from '../clients/sqlite-state-manager.js';
import type { ProjectDetector } from '../utils/project-detector.js';

// Canonical memory collection name per ADR-001
const MEMORY_COLLECTION = 'memory';

// Collection basename for daemon ingestion
const MEMORY_BASENAME = 'memory';

export type MemoryAction = 'add' | 'update' | 'remove' | 'list';
export type MemoryScope = 'global' | 'project';

export interface MemoryRule {
  id: string;
  label?: string;
  content: string;
  scope: MemoryScope;
  projectId?: string;
  title?: string;
  tags?: string[];
  priority?: number;
  createdAt?: string;
  updatedAt?: string;
}

export interface MemoryOptions {
  action: MemoryAction;
  content?: string;
  label?: string;
  scope?: MemoryScope;
  projectId?: string;
  title?: string;
  tags?: string[];
  priority?: number;
  limit?: number;
}

export interface MemoryResponse {
  success: boolean;
  action: MemoryAction;
  label?: string;
  rules?: MemoryRule[];
  message?: string;
  fallback_mode?: 'unified_queue';
  queue_id?: string;
}

export interface MemoryToolConfig {
  qdrantUrl: string;
  qdrantApiKey?: string;
  qdrantTimeout?: number;
}

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
    projectDetector: ProjectDetector
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

  /**
   * Execute memory action
   */
  async execute(options: MemoryOptions): Promise<MemoryResponse> {
    const { action } = options;

    switch (action) {
      case 'add':
        return this.addRule(options);
      case 'update':
        return this.updateRule(options);
      case 'remove':
        return this.removeRule(options);
      case 'list':
        return this.listRules(options);
      default:
        return {
          success: false,
          action,
          message: `Unknown action: ${action}`,
        };
    }
  }

  /**
   * Add a new rule
   */
  private async addRule(options: MemoryOptions): Promise<MemoryResponse> {
    const {
      content,
      scope = 'project',
      projectId,
      title,
      tags,
      priority,
    } = options;

    if (!content?.trim()) {
      return {
        success: false,
        action: 'add',
        message: 'Content is required for adding a rule',
      };
    }

    // Resolve project ID for project-scoped rules
    let resolvedProjectId = projectId;
    if (scope === 'project' && !resolvedProjectId) {
      const cwd = process.cwd();
      const projectInfo = await this.projectDetector.getProjectInfo(cwd, false);
      resolvedProjectId = projectInfo?.projectId;
    }

    // Use provided label or generate a UUID-based one (LLM should provide meaningful labels)
    const label = options.label ?? randomUUID();
    const metadata: Record<string, string> = {
      scope,
      rule_type: 'behavioral',
      label,
    };

    if (resolvedProjectId) {
      metadata['project_id'] = resolvedProjectId;
    }
    if (title) {
      metadata['title'] = title;
    }
    if (tags && tags.length > 0) {
      metadata['tags'] = tags.join(',');
    }
    if (priority !== undefined) {
      metadata['priority'] = String(priority);
    }

    // Try daemon first
    try {
      const response = await this.daemonClient.ingestText({
        content,
        collection_basename: MEMORY_BASENAME,
        tenant_id: resolvedProjectId ?? 'global',
        document_id: label,
        metadata,
      });

      if (response.success) {
        return {
          success: true,
          action: 'add',
          label: response.document_id,
          message: 'Rule added successfully',
        };
      }
    } catch {
      // Daemon unavailable - fall back to queue
    }

    // Fallback: queue the operation
    const queueOperation: {
      action: MemoryAction;
      label: string;
      content: string;
      scope: MemoryScope;
      projectId?: string;
      title?: string;
      tags?: string[];
      priority?: number;
    } = {
      action: 'add',
      label,
      content,
      scope,
    };
    if (resolvedProjectId) queueOperation.projectId = resolvedProjectId;
    if (title) queueOperation.title = title;
    if (tags) queueOperation.tags = tags;
    if (priority !== undefined) queueOperation.priority = priority;

    const queueResult = this.queueMemoryOperation(queueOperation);

    return {
      success: true,
      action: 'add',
      label,
      message: 'Rule queued for processing',
      fallback_mode: 'unified_queue',
      queue_id: queueResult.queueId,
    };
  }

  /**
   * Update an existing rule
   */
  private async updateRule(options: MemoryOptions): Promise<MemoryResponse> {
    const { label, content, title, tags, priority } = options;

    if (!label) {
      return {
        success: false,
        action: 'update',
        message: 'Label is required for updating',
      };
    }

    if (!content?.trim()) {
      return {
        success: false,
        action: 'update',
        message: 'Content is required for updating a rule',
      };
    }

    const metadata: Record<string, string> = {
      label,
    };
    if (title) {
      metadata['title'] = title;
    }
    if (tags && tags.length > 0) {
      metadata['tags'] = tags.join(',');
    }
    if (priority !== undefined) {
      metadata['priority'] = String(priority);
    }

    // Try daemon first - use ingestText with same document_id to update
    try {
      const response = await this.daemonClient.ingestText({
        content,
        collection_basename: MEMORY_BASENAME,
        tenant_id: 'global', // Will be overwritten by existing document's tenant
        document_id: label,
        metadata,
      });

      if (response.success) {
        return {
          success: true,
          action: 'update',
          label,
          message: 'Rule updated successfully',
        };
      }
    } catch {
      // Daemon unavailable - fall back to queue
    }

    // Fallback: queue the operation
    const updateOperation: {
      action: MemoryAction;
      label: string;
      content: string;
      title?: string;
      tags?: string[];
      priority?: number;
    } = {
      action: 'update',
      label,
      content,
    };
    if (title) updateOperation.title = title;
    if (tags) updateOperation.tags = tags;
    if (priority !== undefined) updateOperation.priority = priority;

    const queueResult = this.queueMemoryOperation(updateOperation);

    return {
      success: true,
      action: 'update',
      label,
      message: 'Rule update queued for processing',
      fallback_mode: 'unified_queue',
      queue_id: queueResult.queueId,
    };
  }

  /**
   * Remove a rule
   */
  private async removeRule(options: MemoryOptions): Promise<MemoryResponse> {
    const { label } = options;

    if (!label) {
      return {
        success: false,
        action: 'remove',
        message: 'Label is required for removal',
      };
    }

    // Always queue remove operations (daemon doesn't expose delete via gRPC)
    const queueResult = this.queueMemoryOperation({
      action: 'remove',
      label,
    });

    return {
      success: true,
      action: 'remove',
      label,
      message: 'Rule removal queued for processing',
      fallback_mode: 'unified_queue',
      queue_id: queueResult.queueId,
    };
  }

  /**
   * List rules by scope
   */
  private async listRules(options: MemoryOptions): Promise<MemoryResponse> {
    const { scope = 'project', projectId, limit = 50 } = options;

    // Resolve project ID for project-scoped queries
    let resolvedProjectId = projectId;
    if (scope === 'project' && !resolvedProjectId) {
      const cwd = process.cwd();
      const projectInfo = await this.projectDetector.getProjectInfo(cwd, false);
      resolvedProjectId = projectInfo?.projectId;
    }

    try {
      // Build filter based on scope
      const filter = this.buildListFilter(scope, resolvedProjectId);

      // Build scroll request
      const scrollRequest: {
        limit: number;
        with_payload: boolean;
        filter?: Record<string, unknown>;
      } = {
        limit,
        with_payload: true,
      };
      if (filter) {
        scrollRequest.filter = filter;
      }

      // Query Qdrant directly for list
      const scrollResult = await this.qdrantClient.scroll(MEMORY_COLLECTION, scrollRequest);

      const rules: MemoryRule[] = scrollResult.points.map((point) => {
        const rule: MemoryRule = {
          id: String(point.id),
          content: (point.payload?.['content'] as string) ?? '',
          scope: (point.payload?.['scope'] as MemoryScope) ?? 'global',
        };

        const projectId = point.payload?.['project_id'] as string | undefined;
        if (projectId) rule.projectId = projectId;

        const title = point.payload?.['title'] as string | undefined;
        if (title) rule.title = title;

        const tagsStr = point.payload?.['tags'] as string | undefined;
        if (tagsStr) rule.tags = tagsStr.split(',');

        const priorityStr = point.payload?.['priority'] as string | undefined;
        if (priorityStr) rule.priority = Number(priorityStr);

        const createdAt = point.payload?.['created_at'] as string | undefined;
        if (createdAt) rule.createdAt = createdAt;

        const updatedAt = point.payload?.['updated_at'] as string | undefined;
        if (updatedAt) rule.updatedAt = updatedAt;

        return rule;
      });

      return {
        success: true,
        action: 'list',
        rules,
        message: `Found ${rules.length} rule(s)`,
      };
    } catch (error) {
      // Collection may not exist or other error
      return {
        success: false,
        action: 'list',
        rules: [],
        message: `Failed to list rules: ${error instanceof Error ? error.message : 'unknown error'}`,
      };
    }
  }

  /**
   * Build filter for list query
   */
  private buildListFilter(
    scope: MemoryScope,
    projectId?: string
  ): Record<string, unknown> | undefined {
    const mustConditions: Record<string, unknown>[] = [];

    if (scope === 'global') {
      // Global scope: scope=global (no project_id)
      mustConditions.push({
        key: 'scope',
        match: { value: 'global' },
      });
    } else if (scope === 'project' && projectId) {
      // Project scope: scope=project AND project_id matches
      mustConditions.push({
        key: 'scope',
        match: { value: 'project' },
      });
      mustConditions.push({
        key: 'project_id',
        match: { value: projectId },
      });
    }

    if (mustConditions.length === 0) {
      return undefined;
    }

    return { must: mustConditions };
  }

  /**
   * Queue memory operation for daemon processing
   */
  private queueMemoryOperation(operation: {
    action: MemoryAction;
    label?: string;
    content?: string;
    scope?: MemoryScope;
    projectId?: string;
    title?: string;
    tags?: string[];
    priority?: number;
  }): { queueId: string } {
    const payload: Record<string, unknown> = {
      label: operation.label,
      action: operation.action,
      // source_type is required by Rust ContentPayload parser
      source_type: 'memory_rule',
    };

    if (operation.content) {
      payload['content'] = operation.content;
    }
    if (operation.scope) {
      payload['scope'] = operation.scope;
    }
    if (operation.projectId) {
      payload['project_id'] = operation.projectId;
    }
    if (operation.title) {
      payload['title'] = operation.title;
    }
    if (operation.tags) {
      payload['tags'] = operation.tags;
    }
    if (operation.priority !== undefined) {
      payload['priority'] = operation.priority;
    }

    // Determine operation type for queue
    let op: 'ingest' | 'update' | 'delete';
    switch (operation.action) {
      case 'add':
        op = 'ingest';
        break;
      case 'update':
        op = 'update';
        break;
      case 'remove':
        op = 'delete';
        break;
      default:
        op = 'ingest';
    }

    // Use state manager to enqueue
    const result = this.stateManager.enqueueUnified(
      'content',
      op,
      operation.projectId ?? 'global',
      MEMORY_COLLECTION,
      payload,
      8, // High priority for memory operations
      'main',
      { source: 'mcp_memory_tool' }
    );

    if (result.status !== 'ok' || !result.data) {
      throw new Error(result.message ?? 'Failed to enqueue operation');
    }

    return { queueId: result.data.queueId };
  }
}
