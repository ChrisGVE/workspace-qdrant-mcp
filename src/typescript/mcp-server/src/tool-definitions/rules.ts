/**
 * MCP tool schema definition for the 'rules' tool
 */

export const rulesToolDefinition = {
  name: 'rules',
  description:
    "Manage behavioral rules (add, update, remove, list). Check active rules at the start of each session to load the user's behavioral preferences. Rules persist across sessions and guide how you should work.",
  inputSchema: {
    type: 'object' as const,
    properties: {
      action: {
        type: 'string',
        enum: ['add', 'update', 'remove', 'list'],
        description: 'Action to perform',
      },
      content: {
        type: 'string',
        description: 'Rule content (required for add/update)',
      },
      label: {
        type: 'string',
        description:
          'Rule label (max 15 chars, format: word-word-word, e.g., "prefer-uv", "use-pytest"). Required for add/update/remove.',
      },
      scope: {
        type: 'string',
        enum: ['global', 'project'],
        description:
          'Rule scope (default: project). "project" ties the rule to the current project (resolved from cwd/projectId); "global" applies it across all projects. Pass scope:"global" explicitly for cross-project rules.',
      },
      projectId: {
        type: 'string',
        description: 'Project ID for project-scoped rules',
      },
      cwd: {
        type: 'string',
        description:
          'Absolute path of your current working directory. Pass this so the server can auto-detect the project over HTTP (it cannot otherwise observe your location). Ignored when projectId is provided.',
      },
      title: {
        type: 'string',
        description: 'Rule title (max 50 chars)',
      },
      tags: {
        type: 'array',
        items: { type: 'string' },
        description: 'Tags for categorization (max 5 tags, max 20 chars each)',
      },
      priority: {
        type: 'number',
        description: 'Rule priority (higher = more important)',
      },
      limit: {
        type: 'number',
        description: 'Max rules to return for list (default: 50)',
      },
    },
    required: ['action'],
  },
};
