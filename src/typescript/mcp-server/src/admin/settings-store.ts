/**
 * Settings persistence for the admin UI.
 *
 * Daemon owns the SQLite schema (ADR-003) so we keep admin settings in a
 * plain JSON file under `WQM_DATA_DIR`. Single-process writes are fine:
 * only the MCP server's admin routes touch this file.
 */

import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs';
import { join, dirname } from 'node:path';

import { getDataDirectory } from '../utils/paths.js';

export interface AdminSettings {
  /** Parent directory scanned for child repositories. Empty string = unset. */
  devRoot: string;
  /** Depth of the recursive `.git` scan from `devRoot`. 1 = direct children only. */
  scanDepth: number;
  /** ISO 8601 timestamp of the last scan; empty string if never scanned. */
  lastScanAt: string;
  /**
   * Approved candidate paths (canonical, host-absolute). Only these get
   * forwarded to the daemon on register. Discovery may surface more — they
   * stay as "candidates" until the user clicks Approve.
   */
  approvedProjects: string[];
}

const DEFAULTS: AdminSettings = {
  devRoot: '',
  scanDepth: 1,
  lastScanAt: '',
  approvedProjects: [],
};

const FILE_NAME = 'admin-settings.json';

function settingsPath(): string {
  // `getDataDirectory()` resolves the WQM_DATA_DIR / XDG path. Use the
  // same directory the daemon and CLI write into, so the admin file
  // lives next to state.db rather than in a sibling location nobody
  // expects.
  return join(getDataDirectory(), FILE_NAME);
}

function ensureDir(path: string): void {
  const dir = dirname(path);
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
}

/**
 * Read settings from disk. Returns the defaults when the file does not
 * exist or is unparseable — callers never see exceptions for missing
 * or corrupted state, only for IO failures of the active write.
 */
export function loadSettings(): AdminSettings {
  const path = settingsPath();
  if (!existsSync(path)) return { ...DEFAULTS };

  try {
    const raw = readFileSync(path, 'utf-8');
    const parsed = JSON.parse(raw) as Partial<AdminSettings>;
    return {
      devRoot: typeof parsed.devRoot === 'string' ? parsed.devRoot : DEFAULTS.devRoot,
      scanDepth:
        typeof parsed.scanDepth === 'number' && parsed.scanDepth >= 1 && parsed.scanDepth <= 5
          ? parsed.scanDepth
          : DEFAULTS.scanDepth,
      lastScanAt: typeof parsed.lastScanAt === 'string' ? parsed.lastScanAt : DEFAULTS.lastScanAt,
      approvedProjects: Array.isArray(parsed.approvedProjects)
        ? parsed.approvedProjects.filter((p): p is string => typeof p === 'string')
        : DEFAULTS.approvedProjects,
    };
  } catch {
    // Corrupt JSON or read failure — fall back to defaults rather than
    // crash the admin routes. The first successful PUT /settings will
    // overwrite the bad file.
    return { ...DEFAULTS };
  }
}

/**
 * Atomic write: serialize to a sibling `.tmp` file, then rename over the
 * target. Avoids partial writes if the process is killed mid-flush.
 */
export function saveSettings(settings: AdminSettings): void {
  const path = settingsPath();
  ensureDir(path);
  const tmpPath = `${path}.tmp`;
  writeFileSync(tmpPath, JSON.stringify(settings, null, 2), 'utf-8');
  renameSync(tmpPath, path);
}

/**
 * Merge an update into the current settings and persist. Returns the
 * resulting full settings object.
 */
export function updateSettings(patch: Partial<AdminSettings>): AdminSettings {
  const current = loadSettings();
  const next: AdminSettings = {
    devRoot: patch.devRoot ?? current.devRoot,
    scanDepth: patch.scanDepth ?? current.scanDepth,
    lastScanAt: patch.lastScanAt ?? current.lastScanAt,
    approvedProjects: patch.approvedProjects ?? current.approvedProjects,
  };
  saveSettings(next);
  return next;
}

/**
 * Toggle a single project path in `approvedProjects` and persist.
 * Returns the updated full settings.
 */
export function setProjectApproval(projectPath: string, approved: boolean): AdminSettings {
  const current = loadSettings();
  const set = new Set(current.approvedProjects);
  if (approved) set.add(projectPath);
  else set.delete(projectPath);
  return updateSettings({ approvedProjects: Array.from(set).sort() });
}
