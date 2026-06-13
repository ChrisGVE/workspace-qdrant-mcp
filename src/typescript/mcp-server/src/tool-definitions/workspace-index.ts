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
          'sync_current_branch',
          'indexing_status',
        ],
        description: 'Workspace index action to execute.',
      },
      projectName: {
        type: 'string',
        description: 'Logical project name in .wqm-fork/indexed-projects.json.',
      },
      projectId: {
        type: 'string',
        description:
          'Project tenant ID from workspace-qdrant. Also accepted inside payload. For project_status/indexing_status, omit it to resolve from cwd/repoDir/projectPath.',
      },
      projectPath: {
        type: 'string',
        description: 'Absolute path to a project root when adding/registering a project.',
      },
      branchName: {
        type: 'string',
        description: 'Agent branch name, for example agent/auth-retry-20260523.',
      },
      branch: {
        type: 'string',
        description: 'Alias for branchName, useful in Codex context envelopes.',
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
      worktree: {
        type: 'string',
        description: 'Alias for worktreePath, useful in Codex context envelopes.',
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
          'Path to the workspace-qdrant-mcp repo for registry-backed actions. Also used by project_status/indexing_status to resolve the current project when projectId is omitted. For sync_current_branch: absolute path to the target repo whose branch is being synced (required).',
      },
      cwd: {
        type: 'string',
        description:
          'Caller working directory used by project_status/indexing_status when projectId is omitted and the transport cannot provide host cwd metadata.',
      },
      currentBranch: {
        type: 'string',
        description:
          'For sync_current_branch: current branch name reported by `git rev-parse --abbrev-ref HEAD`.',
      },
      commitHash: {
        type: 'string',
        description: 'For sync_current_branch: HEAD commit SHA reported by `git rev-parse HEAD`.',
      },
      isWorktree: {
        type: 'boolean',
        description:
          'For sync_current_branch: true when .git in the target repo is a file (linked worktree). The hook detects this by `test -f $REPO/.git`.',
      },
      gitRemote: {
        type: 'string',
        description:
          'For sync_current_branch: remote.origin.url from `git config --get remote.origin.url`, used by the daemon for tenant_id calculation.',
      },
      hookName: {
        type: 'string',
        description:
          'For sync_current_branch: name of the git hook that fired (post-checkout, post-commit, etc.). Recorded for observability only.',
      },
      allowMutation: {
        type: 'boolean',
        description:
          'Required for mutating actions, in addition to WQM_INDEX_MANAGER_ALLOW_MUTATION=1.',
      },
      payload: {
        type: 'object',
        description:
          'Optional action-specific arguments. Top-level arguments take precedence; payload may contain projectId, branch, worktree, baseBranch, returnBranch, useWorktree, purpose, createdBy, projectName, projectPath, cwd, worktreePath, worktreeRoot, or repoDir.',
        additionalProperties: true,
      },
    },
    required: ['action'],
  },
};
