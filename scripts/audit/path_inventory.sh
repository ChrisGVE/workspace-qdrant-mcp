#!/usr/bin/env bash
# path_inventory.sh — Grep-based path field inventory for CI validation.
#
# Verifies that every *_path / *_file / path / file field in the known
# schema-mirroring structs and SQL schema files is accounted for in the
# audit doc docs/specs/16-path-abstraction-audit.md.
#
# Usage: ./scripts/audit/path_inventory.sh [<project_root>]
#
# Exit codes:
#   0 — all pattern matches found in audit doc
#   1 — one or more matches NOT mentioned in audit doc (potential missed site)
#
# See docs/specs/16-path-abstraction.md §4.3 for the CI discipline.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${1:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

AUDIT_DOC="$ROOT/docs/specs/16-path-abstraction-audit.md"

if [[ ! -f "$AUDIT_DOC" ]]; then
    echo "ERROR: Audit doc not found: $AUDIT_DOC" >&2
    exit 1
fi

PASS=0
FAIL=0

check_field() {
    local desc="$1"
    local pattern="$2"
    if grep -q "$pattern" "$AUDIT_DOC"; then
        echo "  OK: $desc"
        PASS=$((PASS + 1))
    else
        echo "  MISS: $desc — '$pattern' not found in audit doc" >&2
        FAIL=$((FAIL + 1))
    fi
}

echo "=== Path Field Inventory Check ==="
echo "Root: $ROOT"
echo "Audit doc: $AUDIT_DOC"
echo ""

echo "--- SQL schema canonical fields ---"
check_field "watch_folders.path" "watch_folders.*path"
check_field "tracked_files.file_path" "tracked_files.*file_path"
check_field "file_metadata.file_path" "file_metadata.*file_path"
check_field "graph_nodes.file_path" "graph_nodes.*file_path"
check_field "graph_edges.source_file" "graph_edges.*source_file"
check_field "unified_queue.file_path" "unified_queue.*file_path"
check_field "ignore_file_mtimes.project_root" "ignore_file_mtimes.*project_root"
check_field "ignore_file_mtimes.file_path" "ignore_file_mtimes.*file_path"

echo ""
echo "--- SQL schema relative/suffix fields ---"
check_field "watch_folders.submodule_path" "watch_folders.*submodule_path"
check_field "watch_folders.disambiguation_path" "watch_folders.*disambiguation_path"
check_field "tracked_files.relative_path" "tracked_files.*relative_path"
check_field "file_metadata.relative_path" "file_metadata.*relative_path"

echo ""
echo "--- Qdrant serde payload fields ---"
check_field "FilePayload.file_path" "FilePayload.*file_path"
check_field "FilePayload.old_path" "FilePayload.*old_path"
check_field "FolderPayload.folder_path" "FolderPayload.*folder_path"
check_field "FolderPayload.old_path" "FolderPayload.*old_path"
check_field "LibraryDocumentPayload.document_path" "LibraryDocumentPayload.*document_path"
check_field "LibraryDocumentPayload.library_path" "LibraryDocumentPayload.*library_path"
check_field "ImageSearchResult.file_path" "ImageSearchResult.*file_path"

echo ""
echo "--- Proto canonical fields ---"
check_field "RegisterProjectRequest.path" "RegisterProjectRequest.*path"
check_field "RegisterProjectResponse.watch_path" "RegisterProjectResponse.*watch_path"
check_field "DeprioritizeProjectRequest.watch_path" "DeprioritizeProjectRequest.*watch_path"
check_field "GetProjectStatusResponse.project_root" "GetProjectStatusResponse.*project_root"
check_field "GetProjectStatusResponse.main_worktree_path" "GetProjectStatusResponse.*main_worktree_path"
check_field "ProjectInfo.project_root" "ProjectInfo.*project_root"
check_field "ServerStatusNotification.project_root" "ServerStatusNotification.*project_root"
check_field "SetIncrementalRequest.file_paths" "SetIncrementalRequest.*file_paths"
check_field "ImpactAnalysisRequest.file_path" "ImpactAnalysisRequest.*file_path"
check_field "TextSearchMatch.file_path" "TextSearchMatch.*file_path"
check_field "TraversalNodeProto.file_path" "TraversalNodeProto.*file_path"
check_field "ImpactNodeProto.file_path" "ImpactNodeProto.*file_path"
check_field "PageRankNodeProto.file_path" "PageRankNodeProto.*file_path"
check_field "CommunityMemberProto.file_path" "CommunityMemberProto.*file_path"
check_field "BetweennessNodeProto.file_path" "BetweennessNodeProto.*file_path"
check_field "CancelItemsResponse.project_path" "CancelItemsResponse.*project_path"
check_field "ProjectPayload.file_absolute_path" "file_absolute_path"
check_field "ProjectPayload.file_path (relative)" "ProjectPayload.*file_path"
check_field "SymbolReference.file_path" "SymbolReference.*file_path"
check_field "LibraryPayload(proto).source_file" "LibraryPayload.*source_file"

echo ""
echo "--- Process-local fields ---"
check_field "QueueConnectionConfig.database_path" "QueueConnectionConfig"
check_field "DaemonConfig.log_file" "DaemonConfig.*log_file"
check_field "DaemonConfig.project_path" "DaemonConfig.*project_path"
check_field "Config.database_path" "Config::database_path"
check_field "GraphDbManager.path" "GraphDbManager"
check_field "LadybugConfig.db_path" "LadybugConfig"
check_field "LoggingConfig.log_file_path" "LoggingConfig.*log_file_path"
check_field "TlsConfig.cert_path" "TlsConfig.*cert_path"

echo ""
echo "==================================="
echo "PASS: $PASS  FAIL: $FAIL"
echo ""

if [[ $FAIL -gt 0 ]]; then
    echo "RESULT: FAILED — $FAIL field(s) not accounted for in audit doc." >&2
    exit 1
fi

echo "RESULT: PASSED — all known path fields present in audit doc."
exit 0
