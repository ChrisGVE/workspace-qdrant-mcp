/**
 * MCP tool schema definition for the 'workspace_index' tool.
 *
 * This tool manages the local registry of indexed projects and agent branches.
 * Read-only actions are allowed by default. Mutating actions require both:
 * - WQM_INDEX_MANAGER_ALLOW_MUTATION=1 in the MCP server environment
 * - allowMutation: true in the tool call
 */

export const workspaceIndexToolDefinition = {
  name: 'workspace_index',
  description:
    'Observe and manage indexed projects, agent-created branches, and worktrees for workspace-qdrant. Read-only by default; mutations require explicit double opt-in.',
  inputSchema: {
    type: 'object' as const,
    properties: {
      action: {
        type: 'string',
        enum: [
          'init',
          'list_projects',
          'project_status',
          'status_all',
          'list_branches',
          'agent_branch_status',
          'observe_project',
          'observe_all',
          'incremental_check',
          'incremental_check_all',
          'add_project',
          'start_agent_branch',
          'finish_agent_branch',
          'abandon_agent_branch',
          'register_wqm',
          'register_all_wqm',
          'cleanup_orphans',
        ],
        description: 'Workspace index action to execute.',
      },
      projectName: {
        type: 'string',
        description: 'Logical project name in .wqm-fork/indexed-projects.json.',
      },
      projectPath: {
        type: 'string',
        description: 'Absolute path to a project root when adding/registering a project.',
      },
      branchName: {
        type: 'string',
        description: 'Agent branch name, for example agent/auth-retry-20260523.',
      },
      baseBranch: {
        type: 'string',
        description: 'Base branch for an agent branch. Defaults to main in the helper script.',
      },
      returnBranch: {
        type: 'string',
        description: 'Branch to return to after the agent branch is complete.',
      },
      worktreePath: {
        type: 'string',
        description: 'Optional explicit worktree path for an agent branch.',
      },
      worktreeRoot: {
        type: 'string',
        description: 'Optional parent folder for generated worktree paths.',
      },
      useWorktree: {
        type: 'boolean',
        description: 'Create/use a parallel git worktree for the agent branch.',
      },
      purpose: {
        type: 'string',
        description: 'Human-readable purpose for an agent branch.',
      },
      createdBy: {
        type: 'string',
        description: 'Agent/user identifier recorded in the registry.',
      },
      repoDir: {
        type: 'string',
        description:
          'Optional absolute path to the workspace-qdrant-mcp repo. Defaults to WQM_REPO_DIR or process.cwd().',
      },
      allowMutation: {
        type: 'boolean',
        description:
          'Required for mutating actions, in addition to WQM_INDEX_MANAGER_ALLOW_MUTATION=1.',
      },
    },
    required: ['action'],
  },
};
