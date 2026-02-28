/**
 * Cargo workspace component detection.
 *
 * Parses Cargo.toml [workspace] members to derive dot-separated
 * hierarchical component names.
 *
 * Example:
 *   Cargo.toml member "daemon/core" → component "daemon.core"
 */

import { readFileSync, existsSync } from 'node:fs';
import { join, resolve } from 'node:path';
import type { ComponentMap } from './types.js';
import { pathToComponentId } from './helpers.js';

// ── Cargo workspace ──────────────────────────────────────────────────────

/**
 * Parse Cargo.toml [workspace] members into components.
 *
 * Uses a simple line-based parser (no TOML dependency needed)
 * since workspace member arrays are straightforward.
 */
/** Add workspace members as components, resolving paths relative to the Cargo.toml location. */
function addMemberComponents(
  members: string[], cargoPath: string, projectPath: string, components: ComponentMap,
): void {
  const cargoDir = resolve(cargoPath, '..');
  const relativeBase = cargoDir === projectPath ? '' : cargoDir.slice(projectPath.length + 1);

  for (const member of members) {
    const fullPath = relativeBase ? `${relativeBase}/${member}` : member;
    const id = pathToComponentId(member);
    components.set(id, { id, basePath: fullPath, patterns: [`${fullPath}/**`], source: 'cargo' });
  }
}

export function detectCargoWorkspace(projectPath: string): ComponentMap {
  const components: ComponentMap = new Map();
  const candidates = [
    join(projectPath, 'Cargo.toml'),
    join(projectPath, 'src', 'rust', 'Cargo.toml'),
  ];

  for (const cargoPath of candidates) {
    if (!existsSync(cargoPath)) continue;
    let content: string;
    try { content = readFileSync(cargoPath, 'utf-8'); } catch { continue; }

    const members = parseCargoMembers(content);
    if (members.length === 0) continue;

    addMemberComponents(members, cargoPath, projectPath, components);
    break; // Found a workspace, stop searching
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
