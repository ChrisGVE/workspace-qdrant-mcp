/**
 * Session lifecycle management: initialization, project registration,
 * heartbeat, and cleanup.
 */
import type { DaemonClient } from './clients/daemon-client.js';
import type { SqliteStateManager } from './clients/sqlite-state-manager.js';
import type { ProjectDetector } from './utils/project-detector.js';
import type { HealthMonitor } from './utils/health-monitor.js';
import type { SessionState } from './server-types.js';
export declare function initializeSession(sessionState: SessionState, daemonClient: DaemonClient, projectDetector: ProjectDetector, startHeartbeatFn: () => void): Promise<void>;
export declare function registerProject(sessionState: SessionState, daemonClient: DaemonClient): Promise<void>;
export declare function registerProjectFromTool(args: Record<string, unknown> | undefined, sessionState: SessionState, daemonClient: DaemonClient): Promise<{
    success: boolean;
    project_id: string;
    created: boolean;
    is_active: boolean;
    message: string;
}>;
/**
 * Start the heartbeat interval. Mutates sessionState in place.
 */
export declare function startHeartbeat(sessionState: SessionState, sendHeartbeatFn: () => void): void;
/**
 * Send a heartbeat to the daemon (fire-and-forget safe).
 */
export declare function sendHeartbeat(sessionState: SessionState, daemonClient: DaemonClient): Promise<void>;
/**
 * Clean up session resources: stop health monitor, heartbeat, deprioritize project,
 * close daemon and state manager.
 */
export declare function cleanup(sessionState: SessionState, daemonClient: DaemonClient, stateManager: SqliteStateManager, healthMonitor: HealthMonitor): Promise<void>;
//# sourceMappingURL=session-lifecycle.d.ts.map