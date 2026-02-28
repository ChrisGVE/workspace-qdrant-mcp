//! FTS5 index management operations for search.db.

use tracing::{debug, info};

use super::SearchDbManager;
use super::types::SearchDbResult;

impl SearchDbManager {
    /// Rebuild the FTS5 index from the external content table.
    ///
    /// Must be called after batch inserts/updates/deletes to `code_lines`
    /// to synchronize the FTS index with the content table.
    pub async fn rebuild_fts(&self) -> SearchDbResult<()> {
        use crate::code_lines_schema::FTS5_REBUILD_SQL;
        debug!("Rebuilding FTS5 index");
        sqlx::query(FTS5_REBUILD_SQL)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    /// Optimize the FTS5 index by merging internal b-tree segments.
    ///
    /// Call after large batch operations or periodically during idle time
    /// for improved query performance.
    pub async fn optimize_fts(&self) -> SearchDbResult<()> {
        use crate::code_lines_schema::FTS5_OPTIMIZE_SQL;
        debug!("Optimizing FTS5 index");
        sqlx::query(FTS5_OPTIMIZE_SQL)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    /// Rebuild and optionally optimize the FTS5 index.
    ///
    /// If `lines_affected` exceeds `FTS5_OPTIMIZE_THRESHOLD`, runs optimize
    /// after rebuild for better query performance.
    pub async fn rebuild_and_maybe_optimize_fts(&self, lines_affected: usize) -> SearchDbResult<()> {
        use crate::code_lines_schema::FTS5_OPTIMIZE_THRESHOLD;
        self.rebuild_fts().await?;
        if lines_affected >= FTS5_OPTIMIZE_THRESHOLD {
            info!("Lines affected ({}) >= threshold ({}), optimizing FTS5 index",
                lines_affected, FTS5_OPTIMIZE_THRESHOLD);
            self.optimize_fts().await?;
        }
        Ok(())
    }
}
