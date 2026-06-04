/**
 * Rendering functions for the list tool's three output formats.
 *
 * Provides tree, summary, and flat renderers that convert a FolderNode
 * tree (or flat file list) into a human-readable string representation.
 */
function walkTree(node, indent, currentDepth, maxDepth, limit, state) {
    const prefix = '  '.repeat(indent);
    const sortedChildren = Array.from(node.children.entries())
        .sort((a, b) => a[0].localeCompare(b[0]));
    for (const [, child] of sortedChildren) {
        if (state.count >= limit)
            return false;
        if (child.submodule) {
            state.lines.push(`${prefix}${child.name}/ [submodule: ${child.submodule.repoName}]`);
            state.count++;
            continue;
        }
        if (currentDepth >= maxDepth) {
            state.lines.push(`${prefix}${child.name}/ (${child.totalFiles} files)`);
            state.count++;
            continue;
        }
        state.lines.push(`${prefix}${child.name}/`);
        state.count++;
        if (!walkTree(child, indent + 1, currentDepth + 1, maxDepth, limit, state))
            return false;
    }
    const sortedFiles = [...node.files].sort((a, b) => a.name.localeCompare(b.name));
    for (const file of sortedFiles) {
        if (state.count >= limit)
            return false;
        const tag = file.extension ? ` [${file.extension}]` : '';
        state.lines.push(`${prefix}${file.name}${tag}`);
        state.count++;
    }
    return true;
}
export function renderTree(root, maxDepth, limit) {
    const state = { lines: [], count: 0 };
    walkTree(root, 0, 1, maxDepth, limit, state);
    return { text: state.lines.join('\n'), count: state.count };
}
// ── Summary renderer ─────────────────────────────────────────────────────
function walkSummary(node, indent, currentDepth, chainPrefix, maxDepth, limit, state) {
    const sortedChildren = Array.from(node.children.entries())
        .sort((a, b) => a[0].localeCompare(b[0]));
    for (const [name, child] of sortedChildren) {
        if (state.count >= limit)
            return false;
        const childPath = chainPrefix ? `${chainPrefix}${name}` : name;
        if (child.submodule) {
            state.lines.push(`${'  '.repeat(indent)}${childPath}/ [submodule: ${child.submodule.repoName}]`);
            state.count++;
            continue;
        }
        // Single-child chain collapsing
        if (child.children.size === 1 && child.files.length === 0 && currentDepth < maxDepth) {
            if (!walkSummary(child, indent, currentDepth + 1, `${childPath}/`, maxDepth, limit, state))
                return false;
            continue;
        }
        const prefix = '  '.repeat(indent);
        const summary = formatExtensionSummary(child.totalFiles, aggregateExtensions(child));
        state.lines.push(`${prefix}${childPath}/ ${summary}`);
        state.count++;
        if (currentDepth < maxDepth) {
            if (!walkSummary(child, indent + 1, currentDepth + 1, '', maxDepth, limit, state))
                return false;
        }
    }
    return true;
}
export function renderSummary(root, maxDepth, limit) {
    const state = { lines: [], count: 0 };
    walkSummary(root, 0, 1, '', maxDepth, limit, state);
    return { text: state.lines.join('\n'), count: state.count };
}
function aggregateExtensions(node) {
    const counts = new Map();
    function collect(n) {
        for (const file of n.files) {
            const key = file.extension ?? 'other';
            counts.set(key, (counts.get(key) ?? 0) + 1);
        }
        for (const child of n.children.values()) {
            if (!child.submodule)
                collect(child);
        }
    }
    collect(node);
    return counts;
}
function formatExtensionSummary(totalFiles, extCounts) {
    if (totalFiles === 0)
        return '(empty)';
    // Sort by count descending, show top 4
    const sorted = Array.from(extCounts.entries()).sort((a, b) => b[1] - a[1]);
    const shown = sorted.slice(0, 4);
    const parts = shown.map(([ext, n]) => `${n} ${ext}`);
    if (sorted.length > 4) {
        const remaining = totalFiles - shown.reduce((sum, [, n]) => sum + n, 0);
        if (remaining > 0)
            parts.push(`${remaining} other`);
    }
    return `(${totalFiles} files: ${parts.join(', ')})`;
}
// ── Flat renderer ─────────────────────────────────────────────────────────
export function renderFlat(files, limit) {
    const lines = [];
    let count = 0;
    for (const file of files) {
        if (count >= limit)
            break;
        lines.push(file.relativePath);
        count++;
    }
    return { text: lines.join('\n'), count };
}
//# sourceMappingURL=renderers.js.map