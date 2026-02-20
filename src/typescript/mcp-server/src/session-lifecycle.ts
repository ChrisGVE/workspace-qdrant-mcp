/**
 * Session lifecycle management: initialization, project registration,
 * heartbeat, and cleanup.
 */

import { randomUUID } from 'node:crypto';

import type { DaemonClient } from './clients/daemon-client.js';
import type { SqliteStateManager } from './clients/sqlite-state-manager.js';
import type { ProjectDetector } from './utils/project-detector.js';
import type { HealthMonitor } from './utils/health-monitor.js';
import {
  logInfo,
  logError,
  logDebug,
  logSessionEvent,
  logDaemonStatus,
} from './utils/logger.js';
import { HEARTBEAT_INTERVAL_MS } from './server-types.js';
import type { SessionState } from './server-types.js';

/**
 * Initialize session: detect project, connect daemon, start heartbeat.
 * Mutates sessionState in place.
 */
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

  const cwd = process.cwd();
  const projectInfo = await projectDetector.getProjectInfo(cwd, true);

  if (projectInfo) {
    sessionState.projectPath = projectInfo.projectPath;
    sessionState.projectId = projectInfo.projectId;
    logDebug('Project detected', {
      project_path: projectInfo.projectPath,
      project_id: projectInfo.projectId,
    });
  } else {
    logDebug('No project detected from cwd', { cwd });
  }

  try {
    await daemonClient.connect();
    sessionState.daemonConnected = true;
    logDaemonStatus(true);

    if (sessionState.projectPath) {
      await registerProject(sessionState, daemonClient);
    }

    startHeartbeatFn();
  } catch (error) {
    sessionState.daemonConnected = false;
    logDaemonStatus(false, { reason: 'connection_failed' });
    logError('Daemon connection error', error);
  }
}

/**
 * Register/activate the current project with the daemon.
 *
 * Only re-activates EXISTING projects (register_if_new: false).
 * New projects must be registered explicitly via CLI or other means.
 */
export async function registerProject(
  sessionState: SessionState,
  daemonClient: DaemonClient
): Promise<void> {
  if (!sessionState.projectPath) {
    return;
  }

  try {
    const response = await daemonClient.registerProject({
      path: sessionState.projectPath,
      project_id: sessionState.projectId ?? '',
      name: sessionState.projectPath.split('/').pop() ?? 'unknown',
      register_if_new: false,
      priority: 'high',
    });

    if (!response.is_active && !response.created) {
      logInfo('Project not registered with daemon, skipping activation', {
        project_path: sessionState.projectPath,
        project_id: response.project_id,
      });
      return;
    }

    if (response.project_id && !sessionState.projectId) {
      sessionState.projectId = response.project_id;
      logDebug('Project ID assigned by daemon', { project_id: response.project_id });
    }

    logSessionEvent('register', {
      project_id: response.project_id,
      project_path: sessionState.projectPath,
      created: response.created,
      priority: response.priority,
      is_active: response.is_active,
      newly_registered: response.newly_registered,
    });
  } catch (error) {
    logError('Failed to register project', error, {
      project_path: sessionState.projectPath,
    });
  }
}

/**
 * Register a project via the store tool (type: "project").
 *
 * Unlike session-start registration (register_if_new: false), this explicitly
 * registers new projects with register_if_new: true.
 */
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
  if (!path) {
    throw new Error('path is required for store type "project"');
  }

  if (!sessionState.daemonConnected) {
    throw new Error('Daemon is not connected — cannot register project');
  }

  const name = (args?.['name'] as string) ?? path.split('/').pop() ?? 'unknown';

  const response = await daemonClient.registerProject({
    path,
    project_id: '',
    name,
    register_if_new: true,
    priority: 'high',
  });

  logSessionEvent('register', {
    project_id: response.project_id,
    project_path: path,
    created: response.created,
    priority: response.priority,
    is_active: response.is_active,
    newly_registered: response.newly_registered,
  });

  return {
    success: true,
    project_id: response.project_id,
    created: response.newly_registered,
    is_active: response.is_active,
    message: response.newly_registered
      ? `Project registered and activated: ${path}`
      : `Project already registered and activated: ${path}`,
  };
}

/**
 * Start the heartbeat interval. Mutates sessionState in place.
 */
export function startHeartbeat(
  sessionState: SessionState,
  sendHeartbeatFn: () => void
): void {
  if (sessionState.heartbeatInterval) {
    clearInterval(sessionState.heartbeatInterval);
  }

  sendHeartbeatFn();

  sessionState.heartbeatInterval = setInterval(() => {
    sendHeartbeatFn();
  }, HEARTBEAT_INTERVAL_MS);

  logDebug('Heartbeat started', { interval_minutes: HEARTBEAT_INTERVAL_MS / 1000 / 60 });
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
      const response = await daemonClient.deprioritizeProject({
        project_id: sessionState.projectId,
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
