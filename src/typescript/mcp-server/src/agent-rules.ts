/**
 * Rule fetching and formatting for Claude Agent SDK integration.
 */

import { loadConfig } from './config.js';
import { SqliteStateManager } from './clients/sqlite-state-manager.js';
import { DaemonClient } from './clients/daemon-client.js';
import { ProjectDetector } from './utils/project-detector.js';
import { RulesTool, type Rule } from './tools/rules.js';
import { TENANT_GLOBAL } from './constants/tenants.js';
import { DEFAULT_CONFIG } from './types/generated-defaults.js';

export type { Rule };

/** Build a RulesTool instance from config. */
function buildRulesTool(config: ReturnType<typeof loadConfig>): RulesTool {
  // See server-factory.ts for the rationale on the 30s floor (LSP startup).
  const daemonTimeoutMs = Number(process.env['WQM_DAEMON_TIMEOUT_MS'] ?? '30000');
  const daemonClient = new DaemonClient({
    port: config.daemon.grpcPort,
    timeoutMs:
      Number.isFinite(daemonTimeoutMs) && daemonTimeoutMs > 0 ? daemonTimeoutMs : 30000,
  });
  const stateManager = new SqliteStateManager({
    dbPath: config.database.path.replace('~', process.env['HOME'] ?? ''),
  });
  stateManager.setDaemonClient(daemonClient);
  stateManager.initialize();
  const projectDetector = new ProjectDetector();
  const rulesToolConfig = {
    qdrantUrl: config.qdrant?.url ?? DEFAULT_CONFIG.qdrant.url,
    qdrantTimeout: 5000,
  } as { qdrantUrl: string; qdrantApiKey?: string; qdrantTimeout?: number };
  if (config.qdrant?.apiKey) rulesToolConfig.qdrantApiKey = config.qdrant.apiKey;
  return new RulesTool(rulesToolConfig, daemonClient, stateManager, projectDetector);
}

/** Comparator: sort rules by priority desc then creation date desc. */
function ruleComparator(a: Rule, b: Rule): number {
  const priorityDiff = (b.priority ?? 0) - (a.priority ?? 0);
  if (priorityDiff !== 0) return priorityDiff;
  const aDate = a.createdAt ? new Date(a.createdAt).getTime() : 0;
  const bDate = b.createdAt ? new Date(b.createdAt).getTime() : 0;
  return bDate - aDate;
}

/**
 * Fetch rules from Qdrant via RulesTool.
 * Fetches both global and project-specific rules (if project detected).
 * Rules are sorted by priority (highest first) then by creation date (newest first).
 */
export async function fetchRules(
  projectId: string | null,
  config: ReturnType<typeof loadConfig>
): Promise<Rule[]> {
  const rules: Rule[] = [];
  try {
    const rulesTool = buildRulesTool(config);

    const globalResponse = await rulesTool.execute({
      action: 'list',
      scope: TENANT_GLOBAL,
      limit: 50,
    });
    if (globalResponse.success && globalResponse.rules) {
      rules.push(...globalResponse.rules);
      console.log(`[Agent] Fetched ${globalResponse.rules.length} global rule(s)`);
    }

    if (projectId) {
      const projectResponse = await rulesTool.execute({
        action: 'list',
        scope: 'project',
        projectId,
        limit: 50,
      });
      if (projectResponse.success && projectResponse.rules) {
        rules.push(...projectResponse.rules);
        console.log(
          `[Agent] Fetched ${projectResponse.rules.length} project rule(s) for ${projectId}`
        );
      }
    }

    rules.sort(ruleComparator);
    console.log(`[Agent] Total rules fetched: ${rules.length}`);
    return rules;
  } catch (error) {
    console.error('[Agent] Error fetching rules:', error);
    return rules;
  }
}

/** Format rules for system prompt injection. */
export function formatRulesForPrompt(rules: Rule[]): string {
  if (rules.length === 0) return '';

  const lines: string[] = [
    '# Behavioral Rules',
    '',
    'The following behavioral rules have been configured and should be followed:',
    '',
  ];

  const formatSection = (title: string, sectionRules: Rule[]) => {
    if (sectionRules.length === 0) return;
    lines.push(`## ${title}`, '');
    sectionRules.forEach((rule, index) => {
      const heading = rule.title ? `**${rule.title}**` : `Rule ${index + 1}`;
      const priority = rule.priority !== undefined ? ` [Priority: ${rule.priority}]` : '';
      lines.push(`### ${heading}${priority}`, '', rule.content, '');
    });
  };

  formatSection(
    'Global Rules',
    rules.filter((r) => r.scope === TENANT_GLOBAL)
  );
  formatSection(
    'Project-Specific Rules',
    rules.filter((r) => r.scope === 'project')
  );

  return lines.join('\n');
}
