//! Component detection and gitattributes cache helpers for file ingestion.
//!
//! Handles component ID injection into Qdrant points, gitattributes cache
//! invalidation/loading, and workspace-definition-file detection.

use std::path::Path;

use sqlx::SqlitePool;
use tracing::debug;

use crate::context::ProcessingContext;
use crate::patterns::GitattributesOverrides;

/// Detect and inject component_id into point payloads.
///
/// Also invalidates the component cache when a workspace definition file
/// (Cargo.toml, package.json) is modified, and invalidates the gitattributes
/// cache when `.gitattributes` changes.
pub(crate) async fn inject_component(
    ctx: &ProcessingContext,
    pool: &SqlitePool,
    watch_folder_id: &str,
    base_path: &str,
    relative_path: &str,
    points: &mut [crate::storage::DocumentPoint],
) {
    if is_workspace_definition_file(relative_path) {
        let mut cache = ctx.component_cache.write().await;
        cache.remove(watch_folder_id);
        debug!(
            "Invalidated component cache for {} (workspace file changed: {})",
            watch_folder_id, relative_path
        );
    }

    // Invalidate gitattributes cache when .gitattributes is modified
    if is_gitattributes_file(relative_path) {
        let mut cache = ctx.gitattributes_cache.write().await;
        cache.remove(base_path);
        debug!(
            "Invalidated gitattributes cache for {} (.gitattributes changed)",
            base_path
        );
    }

    let component = resolve_component(ctx, pool, watch_folder_id, base_path, relative_path).await;

    if let Some(ref comp) = component {
        for point in points.iter_mut() {
            point
                .payload
                .insert("component_id".to_string(), serde_json::json!(comp));
        }
    }
}

/// Get or load `.gitattributes` overrides for a project root.
///
/// Uses the per-project cache in ProcessingContext. On cache miss, parses
/// the `.gitattributes` file from the project root and caches the result.
pub(super) async fn get_gitattributes(
    ctx: &ProcessingContext,
    base_path: &str,
) -> GitattributesOverrides {
    // Fast path: read lock
    {
        let cache = ctx.gitattributes_cache.read().await;
        if let Some(overrides) = cache.get(base_path) {
            return overrides.clone();
        }
    }

    // Slow path: parse and cache
    let overrides = GitattributesOverrides::load(Path::new(base_path));
    {
        let mut cache = ctx.gitattributes_cache.write().await;
        cache.insert(base_path.to_string(), overrides.clone());
    }
    overrides
}

/// Check if a file is a workspace definition file that triggers component re-detection.
fn is_workspace_definition_file(relative_path: &str) -> bool {
    let filename = relative_path.rsplit('/').next().unwrap_or(relative_path);
    filename == "Cargo.toml" || filename == "package.json"
}

/// Check if a file change should invalidate the gitattributes cache.
fn is_gitattributes_file(relative_path: &str) -> bool {
    relative_path == ".gitattributes"
}

/// Resolve the component for a file, using the per-watch-folder cache.
///
/// On cache miss: detects components from the project's workspace files,
/// persists them to `project_components`, and caches the result.
async fn resolve_component(
    ctx: &ProcessingContext,
    pool: &SqlitePool,
    watch_folder_id: &str,
    base_path: &str,
    relative_path: &str,
) -> Option<String> {
    use crate::component_detection;

    // Fast path: check cache
    {
        let cache = ctx.component_cache.read().await;
        if let Some(components) = cache.get(watch_folder_id) {
            return component_detection::assign_component(relative_path, components)
                .map(|c| c.id.clone());
        }
    }

    // Slow path: detect from filesystem, persist, and cache
    let project_path = Path::new(base_path);
    let components = component_detection::detect_components(project_path);

    if !components.is_empty() {
        if let Err(e) =
            component_detection::persist_components(pool, watch_folder_id, &components).await
        {
            debug!(
                "Failed to persist components for {}: {}",
                watch_folder_id, e
            );
        }
    }

    let result =
        component_detection::assign_component(relative_path, &components).map(|c| c.id.clone());

    // Cache even if empty (avoids re-detecting for projects with no workspace)
    {
        let mut cache = ctx.component_cache.write().await;
        cache.insert(watch_folder_id.to_string(), components);
    }

    result
}
