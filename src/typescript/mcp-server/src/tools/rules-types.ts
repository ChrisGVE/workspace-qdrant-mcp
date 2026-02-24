/**
 * Rules tool types, interfaces, and constants.
 */

// Canonical rules collection name from native bridge (single source of truth)
import { COLLECTION_RULES } from '../common/native-bridge.js';
export const RULES_COLLECTION = COLLECTION_RULES;
export const RULES_BASENAME = 'rules';

export type RuleAction = 'add' | 'update' | 'remove' | 'list';
export type RuleScope = 'global' | 'project';

export interface Rule {
  id: string;
  label?: string;
  content: string;
  scope: RuleScope;
  projectId?: string;
  title?: string;
  tags?: string[];
  priority?: number;
  createdAt?: string;
  updatedAt?: string;
}

export interface RuleOptions {
  action: RuleAction;
  content?: string;
  label?: string;
  scope?: RuleScope;
  projectId?: string;
  title?: string;
  tags?: string[];
  priority?: number;
  limit?: number;
}

export interface RuleResponse {
  success: boolean;
  action: RuleAction;
  label?: string;
  rules?: Rule[];
  similar_rules?: Array<Rule & { similarity: number }>;
  message?: string;
  fallback_mode?: 'unified_queue';
  queue_id?: string;
}

export interface RuleToolConfig {
  qdrantUrl: string;
  qdrantApiKey?: string;
  qdrantTimeout?: number;
  duplicationThreshold?: number;
}
