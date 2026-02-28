/**
 * Rules tool argument builder — parse raw MCP tool arguments into RuleOptions
 */

export type RuleOptions = {
  action: 'add' | 'update' | 'remove' | 'list';
  content?: string;
  label?: string;
  scope?: 'global' | 'project';
  projectId?: string;
  title?: string;
  tags?: string[];
  priority?: number;
  limit?: number;
};

/** Build rule options from raw tool arguments */
export function buildRuleOptions(args: Record<string, unknown> | undefined): RuleOptions {
  const action = args?.['action'] as string;
  if (action !== 'add' && action !== 'update' && action !== 'remove' && action !== 'list') {
    throw new Error(`Invalid rules action: ${action}`);
  }

  const options: RuleOptions = { action };

  const content = args?.['content'] as string | undefined;
  if (content) options.content = content;

  const label = args?.['label'] as string | undefined;
  if (label) options.label = label;

  const scope = args?.['scope'] as string | undefined;
  if (scope === 'global' || scope === 'project') options.scope = scope;

  const projectId = args?.['projectId'] as string | undefined;
  if (projectId) options.projectId = projectId;

  const title = args?.['title'] as string | undefined;
  if (title) options.title = title;

  const tags = args?.['tags'] as string[] | undefined;
  if (tags) options.tags = tags;

  const priority = args?.['priority'] as number | undefined;
  if (priority !== undefined) options.priority = priority;

  const limit = args?.['limit'] as number | undefined;
  if (limit !== undefined) options.limit = limit;

  return options;
}
