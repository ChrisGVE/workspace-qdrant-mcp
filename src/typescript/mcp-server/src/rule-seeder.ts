/**
 * Default rule seeding for fresh installations.
 *
 * Extracted from WorkspaceQdrantMcpServer.seedDefaultRule to keep server.ts
 * within the 300-line file-size limit.
 */

import { logInfo, logDebug } from './utils/logger.js';
import { TENANT_GLOBAL } from './constants/tenants.js';
import type { RulesTool } from './tools/index.js';

interface DefaultRule {
  label: string;
  title: string;
  content: string;
  priority: number;
}

/**
 * The default global rule set: the workspace-qdrant MCP usage conventions
 * distilled into behavioral rules. Seeded once on a fresh install so every
 * project (current and future) inherits them as global defaults — these are
 * the same rules the server advertises in its MCP instructions.
 */
export const DEFAULT_RULES: ReadonlyArray<DefaultRule> = [
  {
    label: 'search-first',
    title: 'Always search before answering',
    content: [
      "For any question about this project's code, structure, or library docs,",
      'ALWAYS call the `search` tool before answering — do not rely on training data.',
      'Defaults: scope="project", limit=10.',
      'Widen to scope="all" or includeLibraries=true only after a project-scoped query comes back empty.',
      'Use mode="semantic" for concept queries; for a known identifier or exact string use exact=true or `grep`, NOT mode="keyword" (sparse BM25 — will not reliably match a verbatim token).',
    ].join(' '),
    priority: 100,
  },
  {
    label: 'grep-for-exact',
    title: 'Use grep for exact identifiers',
    content: [
      'For exact identifiers, regex, or known substrings, use the `grep` tool before `search` —',
      'it is faster and cheaper than `search` with exact=true.',
      'Reserve `search` for semantic discovery (concepts, intent, "what does the code do around X").',
    ].join(' '),
    priority: 90,
  },
  {
    label: 'scope-project',
    title: 'Start searches project-scoped',
    content: [
      'Start `search` calls with scope="project" and a small limit (default 10).',
      'Widen to scope="all" or includeLibraries=true only after the project-scoped query returns nothing useful.',
      'Never widen silently — the widened call is more expensive and noisier.',
    ].join(' '),
    priority: 90,
  },
  {
    label: 'branch-default',
    title: 'Search the current branch by default',
    content: [
      '`search` operates on the current branch of the current project by default.',
      'To compare or include other branches, pass branch="<name>" or branch="*" (all branches) explicitly.',
      'Never widen branch scope silently — results from stale branches will mislead reasoning.',
    ].join(' '),
    priority: 80,
  },
  {
    label: 'no-proj-write',
    title: 'Never write the projects collection',
    content: [
      'Never call `store` targeting the `projects` collection — that collection is daemon-owned via file watching.',
      'Use `scratchpad` for ad-hoc notes, `libraries` only when the user explicitly asks,',
      'or `store` with type="project" to register/activate a project directory.',
    ].join(' '),
    priority: 80,
  },
  {
    label: 'scratch-memory',
    title: 'Record durable knowledge to the scratchpad',
    content: [
      'As you work, proactively record durable project knowledge to the scratchpad via `store` with type="scratchpad":',
      'decisions and their rationale, non-obvious gotchas, conventions, and context worth recalling in a later session.',
      'Keep each note self-contained and specific.',
      'Notes are project-scoped and resurface automatically — the project-scoped `search` recall lane appends the most',
      'relevant ones after the code hits, so you need not query the scratchpad explicitly.',
      'To revise or remove a note, use the `scratchpad` tool (update/delete) instead of creating near-duplicates.',
    ].join(' '),
    priority: 85,
  },
  {
    label: 'graph-impact',
    title: 'Use the graph for impact and relationships',
    content: [
      'For code-relationship questions ("what calls X", "what breaks if I change Y") and BEFORE refactoring or',
      'renaming a widely-used symbol, use the `graph` tool: `impact` to assess the blast radius, and relationship',
      'queries to find callers/dependencies that plain `search`/`grep` miss.',
      'Prefer it over guessing whenever a change is non-local.',
    ].join(' '),
    priority: 75,
  },
  {
    label: 'record-rules',
    title: 'Record durable conventions as rules',
    content: [
      'When you discover a durable project convention or preference (build/test commands, libraries to prefer,',
      'patterns to follow or avoid), record it with the `rules` tool (action="add") so it persists across sessions —',
      'scope it to the project unless it is clearly global. Keep rules short and imperative, and update an existing',
      'rule instead of adding a near-duplicate. Do NOT record one-off, task-specific details — those belong in the scratchpad.',
    ].join(' '),
    priority: 70,
  },
  {
    label: 'retrieve-by-id',
    title: 'Retrieve by ID instead of re-searching',
    content: [
      'When you already know a document\'s ID or metadata (e.g. from a prior `search`/`list` result), use the',
      '`retrieve` tool to fetch it directly instead of issuing another `search` — it is cheaper and exact.',
      'Reserve `search` for discovery by concept or content.',
    ].join(' '),
    priority: 65,
  },
  {
    label: 'confirm-mut',
    title: 'Confirm mutating index actions',
    content: [
      'Mutating `workspace_index` actions (add_project, start_agent_branch, finish_agent_branch,',
      'abandon_agent_branch, register_wqm, register_all_wqm, cleanup_orphans) require double opt-in:',
      'allowMutation:true in the call AND WQM_INDEX_MANAGER_ALLOW_MUTATION=1 in the server env.',
      'ALWAYS get explicit user confirmation before calling these — they affect persistent shared state across sessions.',
      '`sync_current_branch` is for git hooks only; agents must not call it directly.',
    ].join(' '),
    priority: 95,
  },
  {
    label: 'agent-branch',
    title: 'Register agent/feature branches',
    content: [
      'When creating an agent/feature branch (especially in a parallel worktree), register it via `workspace_index`',
      'with action="start_agent_branch", passing branchName, purpose, createdBy, and useWorktree=true if applicable.',
      'Close out with finish_agent_branch (merged) or abandon_agent_branch (discarded) — never leave it dangling in the registry.',
      'This is a mutating action and requires double opt-in (see [[confirm-mut]]).',
    ].join(' '),
    priority: 70,
  },
];

/**
 * Seed the default global rule set, idempotently PER LABEL.
 *
 * A fresh install gets all defaults; an install already holding some of them
 * gets only the missing labels backfilled. Dedup-by-label is the guard: it can
 * never create a duplicate of an existing default, even if invoked repeatedly or
 * after a partial earlier run (the failure mode that left duplicate rows in
 * production). If the list call fails we skip entirely — without the existing
 * set we cannot dedup safely.
 */
export async function seedDefaultRule(rulesTool: RulesTool): Promise<void> {
  try {
    const listResult = await rulesTool.execute({ action: 'list', scope: TENANT_GLOBAL });
    if (!listResult.success) {
      return; // list failed — skip seeding (cannot safely dedup)
    }
    const existingLabels = new Set(
      (listResult.rules ?? []).map((r) => r.label).filter((l): l is string => Boolean(l))
    );

    let seeded = 0;
    for (const rule of DEFAULT_RULES) {
      if (existingLabels.has(rule.label)) continue; // already present — never duplicate
      const addResult = await rulesTool.execute({
        action: 'add',
        label: rule.label,
        title: rule.title,
        content: rule.content,
        scope: TENANT_GLOBAL,
        priority: rule.priority,
      });
      if (addResult.success) seeded += 1;
    }

    if (seeded > 0) {
      logInfo('Seeded default behavioral rules', { count: seeded, of: DEFAULT_RULES.length });
    }
  } catch (error) {
    logDebug('Skipped default rule seeding', { reason: String(error) });
  }
}
