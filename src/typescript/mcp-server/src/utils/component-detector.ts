/**
 * Component auto-detection from workspace definition files.
 *
 * Parses Cargo.toml [workspace] members and package.json workspaces
 * to derive dot-separated hierarchical component names.
 *
 * Examples:
 *   Cargo.toml member "daemon/core"  → component "daemon.core"
 *   Cargo.toml member "cli"          → component "cli"
 *   package.json workspace "packages/ui" → component "packages.ui"
 */

import { readFileSync, existsSync, readdirSync, statSync } from 'node:fs';
import { join, resolve } from 'node:path';

// ── Types ────────────────────────────────────────────────────────────────

export interface ComponentInfo {
  /** Dot-separated component ID, e.g. "daemon.core" */
  id: string;
  /** Base directory relative to project root, e.g. "daemon/core" */
  basePath: string;
  /** Glob patterns matching files in this component */
  patterns: string[];
  /** Detection source */
  source: 'cargo' | 'npm' | 'directory';
}

export type ComponentMap = Map<string, ComponentInfo>;

// ── Detection ────────────────────────────────────────────────────────────

/**
 * Detect project components from workspace definition files.
 *
 * Tries Cargo.toml first, then package.json, then falls back to
 * top-level directory heuristic.
 */
export function detectComponents(projectPath: string): ComponentMap {
  const components = new Map<string, ComponentInfo>();

  // Try Cargo workspace first (searches up to 2 levels for Cargo.toml with [workspace])
  const cargoComponents = detectCargoWorkspace(projectPath);
  if (cargoComponents.size > 0) {
    for (const [id, info] of cargoComponents) {
      components.set(id, info);
    }
  }

  // Try npm/yarn workspaces
  const npmComponents = detectNpmWorkspace(projectPath);
  if (npmComponents.size > 0) {
    for (const [id, info] of npmComponents) {
      // Don't overwrite Cargo-detected components
      if (!components.has(id)) {
        components.set(id, info);
      }
    }
  }

  // Fallback: top-level directories if nothing detected
  if (components.size === 0) {
    const dirComponents = detectFromDirectories(projectPath);
    for (const [id, info] of dirComponents) {
      components.set(id, info);
    }
  }

  return components;
}

// ── Cargo workspace ──────────────────────────────────────────────────────

/**
 * Parse Cargo.toml [workspace] members into components.
 *
 * Uses a simple line-based parser (no TOML dependency needed)
 * since workspace member arrays are straightforward.
 */
function detectCargoWorkspace(projectPath: string): ComponentMap {
  const components = new Map<string, ComponentInfo>();

  // Search for Cargo.toml in project root and common subdirectories
  const candidates = [
    join(projectPath, 'Cargo.toml'),
    join(projectPath, 'src', 'rust', 'Cargo.toml'),
  ];

  for (const cargoPath of candidates) {
    if (!existsSync(cargoPath)) continue;

    let content: string;
    try {
      content = readFileSync(cargoPath, 'utf-8');
    } catch {
      continue;
    }

    const members = parseCargoMembers(content);
    if (members.length === 0) continue;

    // Compute base directory relative to project root
    const cargoDir = resolve(cargoPath, '..');
    const relativeBase = cargoDir === projectPath
      ? ''
      : cargoDir.slice(projectPath.length + 1);

    for (const member of members) {
      const fullPath = relativeBase ? `${relativeBase}/${member}` : member;
      const id = pathToComponentId(member);

      components.set(id, {
        id,
        basePath: fullPath,
        patterns: [`${fullPath}/**`],
        source: 'cargo',
      });
    }

    // Found a workspace, stop searching
    if (members.length > 0) break;
  }

  return components;
}

/**
 * Extract workspace members from Cargo.toml content.
 *
 * Parses the `members = [...]` array from a [workspace] section.
 * Handles multi-line arrays and inline comments.
 */
export function parseCargoMembers(content: string): string[] {
  const members: string[] = [];

  // Find [workspace] section
  const workspaceIdx = content.indexOf('[workspace]');
  if (workspaceIdx === -1) return members;

  // Find members = [...] after [workspace]
  const afterWorkspace = content.slice(workspaceIdx);
  const membersMatch = afterWorkspace.match(/members\s*=\s*\[([^\]]*)\]/s);
  if (!membersMatch) return members;

  // Strip line comments before extracting strings
  const arrayContent = (membersMatch[1] ?? '')
    .split('\n')
    .map(line => line.replace(/#.*$/, ''))
    .join('\n');

  // Extract quoted strings
  const stringPattern = /"([^"]+)"|'([^']+)'/g;
  let match: RegExpExecArray | null;
  while ((match = stringPattern.exec(arrayContent)) !== null) {
    const value = match[1] ?? match[2];
    if (value) members.push(value);
  }

  return members;
}

// ── npm workspace ────────────────────────────────────────────────────────

/**
 * Parse package.json workspaces into components.
 *
 * Handles both array format (`"workspaces": ["packages/*"]`)
 * and object format (`"workspaces": { "packages": ["pkg-a"] }`).
 */
function detectNpmWorkspace(projectPath: string): ComponentMap {
  const components = new Map<string, ComponentInfo>();

  const pkgPath = join(projectPath, 'package.json');
  if (!existsSync(pkgPath)) return components;

  let pkg: Record<string, unknown>;
  try {
    pkg = JSON.parse(readFileSync(pkgPath, 'utf-8')) as Record<string, unknown>;
  } catch {
    return components;
  }

  let workspacePaths: string[] = [];

  if (Array.isArray(pkg['workspaces'])) {
    workspacePaths = pkg['workspaces'] as string[];
  } else if (pkg['workspaces'] && typeof pkg['workspaces'] === 'object') {
    const ws = pkg['workspaces'] as Record<string, unknown>;
    if (Array.isArray(ws['packages'])) {
      workspacePaths = ws['packages'] as string[];
    }
  }

  for (const wsPath of workspacePaths) {
    // Resolve globs: "packages/*" → list actual subdirectories
    if (wsPath.includes('*')) {
      const baseDir = wsPath.replace(/\/?\*.*$/, '');
      const fullBase = join(projectPath, baseDir);
      if (!existsSync(fullBase)) continue;

      try {
        const entries = readdirSync(fullBase);
        for (const entry of entries) {
          const entryPath = join(fullBase, entry);
          try {
            if (statSync(entryPath).isDirectory()) {
              const relPath = baseDir ? `${baseDir}/${entry}` : entry;
              const id = pathToComponentId(relPath);
              components.set(id, {
                id,
                basePath: relPath,
                patterns: [`${relPath}/**`],
                source: 'npm',
              });
            }
          } catch {
            // Skip entries we can't stat
          }
        }
      } catch {
        // Skip if we can't read directory
      }
    } else {
      // Direct path
      const id = pathToComponentId(wsPath);
      components.set(id, {
        id,
        basePath: wsPath,
        patterns: [`${wsPath}/**`],
        source: 'npm',
      });
    }
  }

  return components;
}

// ── Directory fallback ───────────────────────────────────────────────────

const IGNORED_DIRS = new Set([
  '.git', '.github', '.vscode', '.idea',
  'node_modules', 'target', 'dist', 'build',
  '.taskmaster', '.claude', '.serena', 'tmp',
]);

/**
 * Fallback: use top-level directories as components.
 *
 * Only includes directories that likely contain source code
 * (skips hidden dirs, build output, etc.).
 */
function detectFromDirectories(projectPath: string): ComponentMap {
  const components = new Map<string, ComponentInfo>();

  try {
    const entries = readdirSync(projectPath);
    for (const entry of entries) {
      if (entry.startsWith('.') || IGNORED_DIRS.has(entry)) continue;

      const fullPath = join(projectPath, entry);
      try {
        if (statSync(fullPath).isDirectory()) {
          components.set(entry, {
            id: entry,
            basePath: entry,
            patterns: [`${entry}/**`],
            source: 'directory',
          });
        }
      } catch {
        // Skip entries we can't stat
      }
    }
  } catch {
    // Can't read project directory
  }

  return components;
}

// ── Helpers ──────────────────────────────────────────────────────────────

/**
 * Convert a path to a dot-separated component ID.
 *
 * "daemon/core"        → "daemon.core"
 * "cli"                → "cli"
 * "src/typescript/mcp" → "src.typescript.mcp"
 */
export function pathToComponentId(path: string): string {
  return path
    .replace(/\/+$/, '')   // trim trailing slashes
    .replace(/^\/+/, '')   // trim leading slashes
    .replace(/\//g, '.');  // replace / with .
}

/**
 * Check if a relative file path matches a component.
 *
 * A file matches if its relativePath starts with the component's basePath + "/".
 */
export function fileMatchesComponent(
  relativePath: string,
  component: ComponentInfo,
): boolean {
  const base = component.basePath;
  return relativePath === base ||
    relativePath.startsWith(base + '/');
}

/**
 * Check if a component ID matches a filter (exact or prefix).
 *
 * "daemon.core" matches filter "daemon.core" (exact)
 * "daemon.core" matches filter "daemon" (prefix)
 * "daemon.core" does NOT match filter "cli" (no match)
 */
export function componentMatchesFilter(
  componentId: string,
  filter: string,
): boolean {
  return componentId === filter ||
    componentId.startsWith(filter + '.');
}

/**
 * Assign a component to a file based on its relative path.
 *
 * Returns the most specific (longest basePath) matching component,
 * or undefined if no component matches.
 */
export function assignComponent(
  relativePath: string,
  components: ComponentMap,
): ComponentInfo | undefined {
  let bestMatch: ComponentInfo | undefined;
  let bestLength = -1;

  for (const component of components.values()) {
    if (fileMatchesComponent(relativePath, component)) {
      if (component.basePath.length > bestLength) {
        bestMatch = component;
        bestLength = component.basePath.length;
      }
    }
  }

  return bestMatch;
}
