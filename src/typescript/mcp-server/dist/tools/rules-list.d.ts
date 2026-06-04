/**
 * Rules list operation — query rules by scope from Qdrant with mirror fallback.
 */
import type { QdrantClient } from '@qdrant/js-client-rest';
import type { SqliteStateManager } from '../clients/sqlite-state-manager.js';
import type { ProjectDetector } from '../utils/project-detector.js';
import type { RuleOptions, RuleResponse } from './rules-types.js';
/** List rules by scope from Qdrant, with rules_mirror fallback. */
export declare function listRules(qdrantClient: QdrantClient, stateManager: SqliteStateManager, projectDetector: ProjectDetector, options: RuleOptions): Promise<RuleResponse>;
//# sourceMappingURL=rules-list.d.ts.map