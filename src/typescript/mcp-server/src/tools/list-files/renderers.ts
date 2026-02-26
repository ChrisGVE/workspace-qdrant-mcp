/**
 * Rendering functions for the list tool's three output formats.
 *
 * Provides tree, summary, and flat renderers that convert a FolderNode
 * tree (or flat file list) into a human-readable string representation.
 */

import type { TrackedFileEntry } from '../../clients/tracked-files-queries/index.js';
import type { FolderNode } from '../list-files-types.js';

// ── Shared types ─────────────────────────────────────────────────────────

export interface RenderResult {
  text: string;
  count: number;
}

// ── Tree renderer ─────────────────────────────────────────────────────────

export function renderTree(
  root: FolderNode,
  maxDepth: number,
  limit: number,
): RenderResult {
  const lines: string[] = [];
  let count = 0;

  function walk(node: FolderNode, indent: number, currentDepth: number): boolean {
    const prefix = '  '.repeat(indent);

    // Render child folders first (sorted)
    const sortedChildren = Array.from(node.children.entries()).sort((a, b) =>
      a[0].localeCompare(b[0]),
    );

    for (const [, child] of sortedChildren) {
      if (count >= limit) return false;

      if (child.submodule) {
        lines.push(`${prefix}${child.name}/ [submodule: ${child.submodule.repoName}]`);
        count++;
        continue;
      }

      if (currentDepth >= maxDepth) {
        lines.push(`${prefix}${child.name}/ (${child.totalFiles} files)`);
        count++;
        continue;
      }

      lines.push(`${prefix}${child.name}/`);
      count++;
      if (!walk(child, indent + 1, currentDepth + 1)) return false;
    }

    // Render files (sorted)
    const sortedFiles = [...node.files].sort((a, b) => a.name.localeCompare(b.name));
    for (const file of sortedFiles) {
      if (count >= limit) return false;
      const tag = file.extension ? ` [${file.extension}]` : '';
      lines.push(`${prefix}${file.name}${tag}`);
      count++;
    }

    return true;
  }

  // Start rendering from root's children (skip the root node name itself)
  walk(root, 0, 1);

  return { text: lines.join('\n'), count };
}

// ── Summary renderer ─────────────────────────────────────────────────────

export function renderSummary(
  root: FolderNode,
  maxDepth: number,
  limit: number,
): RenderResult {
  const lines: string[] = [];
  let count = 0;

  function walk(node: FolderNode, indent: number, currentDepth: number, chainPrefix: string): boolean {
    // Get sorted children
    const sortedChildren = Array.from(node.children.entries()).sort((a, b) =>
      a[0].localeCompare(b[0]),
    );

    for (const [name, child] of sortedChildren) {
      if (count >= limit) return false;

      const childPath = chainPrefix ? `${chainPrefix}${name}` : name;

      if (child.submodule) {
        const prefix = '  '.repeat(indent);
        lines.push(`${prefix}${childPath}/ [submodule: ${child.submodule.repoName}]`);
        count++;
        continue;
      }

      // Single-child chain collapsing: if this folder has exactly one child folder
      // and no files, collapse into the child's display
      if (
        child.children.size === 1 &&
        child.files.length === 0 &&
        currentDepth < maxDepth
      ) {
        // Continue the chain
        if (!walk(child, indent, currentDepth + 1, `${childPath}/`)) return false;
        continue;
      }

      const prefix = '  '.repeat(indent);
      const extCounts = aggregateExtensions(child);
      const summary = formatExtensionSummary(child.totalFiles, extCounts);

      if (currentDepth >= maxDepth) {
        lines.push(`${prefix}${childPath}/ ${summary}`);
        count++;
        continue;
      }

      lines.push(`${prefix}${childPath}/ ${summary}`);
      count++;

      if (!walk(child, indent + 1, currentDepth + 1, '')) return false;
    }

    return true;
  }

  walk(root, 0, 1, '');

  return { text: lines.join('\n'), count };
}

function aggregateExtensions(node: FolderNode): Map<string, number> {
  const counts = new Map<string, number>();

  function collect(n: FolderNode): void {
    for (const file of n.files) {
      const key = file.extension ?? 'other';
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
    for (const child of n.children.values()) {
      if (!child.submodule) collect(child);
    }
  }

  collect(node);
  return counts;
}

function formatExtensionSummary(
  totalFiles: number,
  extCounts: Map<string, number>,
): string {
  if (totalFiles === 0) return '(empty)';

  // Sort by count descending, show top 4
  const sorted = Array.from(extCounts.entries()).sort((a, b) => b[1] - a[1]);
  const shown = sorted.slice(0, 4);
  const parts = shown.map(([ext, n]) => `${n} ${ext}`);

  if (sorted.length > 4) {
    const remaining = totalFiles - shown.reduce((sum, [, n]) => sum + n, 0);
    if (remaining > 0) parts.push(`${remaining} other`);
  }

  return `(${totalFiles} files: ${parts.join(', ')})`;
}

// ── Flat renderer ─────────────────────────────────────────────────────────

export function renderFlat(
  files: TrackedFileEntry[],
  limit: number,
): RenderResult {
  const lines: string[] = [];
  let count = 0;

  for (const file of files) {
    if (count >= limit) break;
    lines.push(file.relativePath);
    count++;
  }

  return { text: lines.join('\n'), count };
}
