/**
 * Session lifecycle management: initialization, project registration,
 * heartbeat, and cleanup.
 */

import { randomUUID } from 'node:crypto';

import type { DaemonClient } from './clients/daemon-client.js';
import type { SqliteStateManager } from './clients/sqlite-state-manager.js';
import type { ProjectDetector } from './utils/project-detector.js';
import { getGitRemoteUrl } from './utils/project-detector.js';
import { findGitRoot } from './utils/git-utils.js';
import { detectCurrentBranch } from './utils/git-branch.js';
import type { HealthMonitor } from './utils/health-monitor.js';
import { logInfo, logError, logDebug, logSessionEvent, logDaemonStatus } from './utils/logger.js';
import { recordDaemonFallback } from './telemetry/metrics.js';
import { HEARTBEAT_INTERVAL_MS } from './server-types.js';
import type { SessionState } from './server-types.js';

/**
 * Initialize session: detect project, connect daemon, start heartbeat.
 * Mutates sessionState in place.
 */
/** Detect and assign project info to session state. */
async function detectProjectForSession(
  sessionState: SessionState,
  projectDetector: ProjectDetector
): Promise<void> {
  const cwd = process.cwd();
  const projectRoot = projectDetector.findProjectRoot(cwd) ?? cwd;
  const projectInfo = await projectDetector.getProjectInfo(projectRoot, true);
  if (projectInfo) {
    sessionState.projectPath = projectInfo.projectPath;
    sessionState.projectId = projectInfo.projectId;
    logDebug('Project detected', {
      project_path: projectInfo.projectPath,
      project_id: projectInfo.projectId,
    });
  } else {
    sessionState.projectPath = projectRoot;
    logDebug('Project root detected but not registered', { projectRoot, cwd });
  }

  // Detect current git branch for use as the default filter
  const branch = detectCurrentBranch(projectRoot);
  sessionState.currentBranch = branch;
  logDebug('Branch detected', { branch });
}

export async function initializeSession(
  sessionState: SessionState,
  daemonClient: DaemonClient,
  projectDetector: ProjectDetector,
  startHeartbeatFn: () => void
): Promise<void> {
  sessionState.sessionId = randomUUID();
  const { setSessionId } = await import('./utils/logger.js');
  setSessionId(sessionState.sessionId);
  logSessionEvent('start', { session_id: sessionState.sessionId });

  await detectProjectForSession(sessionState, projectDetector);

  try {
    await daemonClient.connect();
    sessionState.daemonConnected = true;
    logDaemonStatus(true);
    if (sessionState.projectPath) await registerProject(sessionState, daemonClient);
    startHeartbeatFn();
  } catch (error) {
    sessionState.daemonConnected = false;
    logDaemonStatus(false, { reason: 'connection_failed' });
    logError('Daemon connection error', error);
    recordDaemonFallback('session', 'connection_failed');
  }
}

/**
 * Register/activate the current project with the daemon.
 *
 * Only re-activates EXISTING projects (register_if_new: false).
 * New projects must be registered explicitly via CLI or other means.
 */
/** Apply daemon registration response to session state. */
function applyRegistrationResponse(
  sessionState: SessionState,
  response: { project_id: string; is_worktree?: boolean; watch_path?: string }
): void {
  if (response.project_id && !sessionState.projectId) {
    sessionState.projectId = response.project_id;
    logDebug('Project ID assigned by daemon', { project_id: response.project_id });
  }
  if (response.is_worktree) {
    sessionState.isWorktree = true;
    sessionState.watchPath = response.watch_path ?? sessionState.projectPath ?? null;
    logInfo('Registered as worktree', {
      project_path: sessionState.projectPath,
      project_id: response.project_id,
      watch_path: sessionState.watchPath,
    });
  }
}

export async function registerProject(
  sessionState: SessionState,
  daemonClient: DaemonClient
): Promise<void> {
  if (!sessionState.projectPath) return;

  try {
    const gitRemote = getGitRemoteUrl(sessionState.projectPath);
    const response = await daemonClient.registerProject({
      path: sessionState.projectPath,
      project_id: sessionState.projectId ?? '',
      name: sessionState.projectPath.split('/').pop() ?? 'unknown',
      register_if_new: false,
      priority: 'high',
      ...(gitRemote ? { git_remote: gitRemote } : {}),
    });

    if (!response.is_active && !response.created) {
      logInfo('Project not registered with daemon, skipping activation', {
        project_path: sessionState.projectPath,
        project_id: response.project_id,
      });
      return;
    }

    applyRegistrationResponse(sessionState, response);
    logSessionEvent('register', {
      project_id: response.project_id,
      project_path: sessionState.projectPath,
      created: response.created,
      priority: response.priority,
      is_active: response.is_active,
      newly_registered: response.newly_registered,
      is_worktree: response.is_worktree,
      watch_path: response.watch_path,
    });
  } catch (error) {
    logError('Failed to register project', error, { project_path: sessionState.projectPath });
  }
}

/**
 * Register a project via the store tool (type: "project").
 *
 * Unlike session-start registration (register_if_new: false), this explicitly
 * registers new projects with register_if_new: true.
 */
/** Call the daemon to register a project and log the event. */
async function callRegisterProject(
  daemonClient: DaemonClient,
  resolvedPath: string,
  name: string,
  gitRemote: string | null
): Promise<{
  project_id: string;
  created: boolean;
  newly_registered: boolean;
  is_active: boolean;
  priority: string;
}> {
  const response = await daemonClient.registerProject({
    path: resolvedPath,
    project_id: '',
    name,
    register_if_new: true,
    priority: 'high',
    ...(gitRemote ? { git_remote: gitRemote } : {}),
  });
  logSessionEvent('register', {
    project_id: response.project_id,
    project_path: resolvedPath,
    created: response.created,
    priority: response.priority,
    is_active: response.is_active,
    newly_registered: response.newly_registered,
  });
  return response;
}

export async function registerProjectFromTool(
  args: Record<string, unknown> | undefined,
  sessionState: SessionState,
  daemonClient: DaemonClient
): Promise<{
  success: boolean;
  project_id: string;
  created: boolean;
  is_active: boolean;
  message: string;
}> {
  const path = args?.['path'] as string;
  if (!path) throw new Error('path is required for store type "project"');
  if (!sessionState.daemonConnected)
    throw new Error('Daemon is not connected — cannot register project');

  const resolvedPath = findGitRoot(path) ?? path;
  const name = (args?.['name'] as string) ?? resolvedPath.split('/').pop() ?? 'unknown';
  const gitRemote = getGitRemoteUrl(resolvedPath);

  const response = await callRegisterProject(daemonClient, resolvedPath, name, gitRemote);

  return {
    success: true,
    project_id: response.project_id,
    created: response.newly_registered,
    is_active: response.is_active,
    message: response.newly_registered
      ? `Project registered and activated: ${resolvedPath}`
      : `Project already registered and activated: ${resolvedPath}`,
  };
}

/**
 * Start the heartbeat interval. Mutates sessionState in place.
 */
export function startHeartbeat(sessionState: SessionState, sendHeartbeatFn: () => void): void {
  if (sessionState.heartbeatInterval) {
    clearInterval(sessionState.heartbeatInterval);
  }

  sendHeartbeatFn();

  sessionState.heartbeatInterval = setInterval(() => {
    sendHeartbeatFn();
  }, HEARTBEAT_INTERVAL_MS);

  logDebug('Heartbeat started', { interval_secs: HEARTBEAT_INTERVAL_MS / 1000 });
}

/**
 * Send a heartbeat to the daemon (fire-and-forget safe).
 */
export async function sendHeartbeat(
  sessionState: SessionState,
  daemonClient: DaemonClient
): Promise<void> {
  if (!sessionState.projectId || !sessionState.daemonConnected) {
    return;
  }

  try {
    const response = await daemonClient.heartbeat({
      project_id: sessionState.projectId,
    });

    if (response.acknowledged) {
      logSessionEvent('heartbeat', {
        project_id: sessionState.projectId,
        acknowledged: true,
      });
    }
  } catch (error) {
    logError('Heartbeat failed', error, { project_id: sessionState.projectId });
    sessionState.daemonConnected = false;
    logDaemonStatus(false, { reason: 'heartbeat_failed' });
    recordDaemonFallback('session', 'heartbeat_failed');
  }
}

/**
 * Clean up session resources: stop health monitor, heartbeat, deprioritize project,
 * close daemon and state manager.
 */
export async function cleanup(
  sessionState: SessionState,
  daemonClient: DaemonClient,
  stateManager: SqliteStateManager,
  healthMonitor: HealthMonitor
): Promise<void> {
  healthMonitor.stop();
  logDebug('Health monitoring stopped');

  if (sessionState.heartbeatInterval) {
    clearInterval(sessionState.heartbeatInterval);
    sessionState.heartbeatInterval = null;
    logDebug('Heartbeat stopped');
  }

  if (sessionState.projectId && sessionState.daemonConnected) {
    try {
      const projectId = sessionState.projectId!;
      const response = await daemonClient.deprioritizeProject({
        project_id: projectId,
        ...(sessionState.isWorktree && sessionState.watchPath
          ? { watch_path: sessionState.watchPath }
          : {}),
      });
      logSessionEvent('deprioritize', {
        project_id: sessionState.projectId,
        is_active: response.is_active,
        new_priority: response.new_priority,
      });
    } catch (error) {
      logError('Failed to deprioritize project', error, {
        project_id: sessionState.projectId,
      });
    }
  }

  daemonClient.close();
  stateManager.close();

  logSessionEvent('end', { session_id: sessionState.sessionId });
}
