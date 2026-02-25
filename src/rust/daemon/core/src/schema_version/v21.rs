//! Migration v21: Add git-tracking columns, rules_mirror, and submodule junction table.

use async_trait::async_trait;
use sqlx::SqlitePool;
use tracing::{debug, info};

use super::SchemaError;
use super::migration::Migration;

pub struct V21Migration;

#[async_trait]
impl Migration for V21Migration {
    async fn up(&self, pool: &SqlitePool) -> Result<(), SchemaError> {
        info!("Migration v21: Adding git-tracking columns, rules_mirror, and submodule junction table");

        use crate::watch_folders_schema::{
            MIGRATE_V21_WATCH_FOLDERS_SQL,
            CREATE_RULES_MIRROR_SQL,
            CREATE_WATCH_FOLDER_SUBMODULES_SQL,
            CREATE_WATCH_FOLDER_SUBMODULES_INDEXES_SQL,
            MIGRATE_V21_SUBMODULE_DATA_SQL,
        };

        // 1. Add git-tracking columns to watch_folders
        let has_last_commit_hash: bool = sqlx::query_scalar(
            "SELECT COUNT(*) > 0 FROM pragma_table_info('watch_folders') WHERE name = 'last_commit_hash'"
        )
        .fetch_one(pool).await?;

        if !has_last_commit_hash {
            for alter_sql in MIGRATE_V21_WATCH_FOLDERS_SQL {
                debug!("Running ALTER TABLE: {}", alter_sql);
                sqlx::query(alter_sql).execute(pool).await?;
            }

            let backfilled: u64 = sqlx::query(
                "UPDATE watch_folders SET is_git_tracked = 1 WHERE git_remote_url IS NOT NULL"
            )
            .execute(pool).await?
            .rows_affected();

            if backfilled > 0 {
                info!("Backfilled is_git_tracked for {} watch folders", backfilled);
            }
        } else {
            debug!("last_commit_hash column already exists, skipping ALTER TABLE");
        }

        // 2. Create rules_mirror table
        sqlx::query(CREATE_RULES_MIRROR_SQL)
            .execute(pool).await?;

        // 3. Create watch_folder_submodules junction table
        sqlx::query(CREATE_WATCH_FOLDER_SUBMODULES_SQL)
            .execute(pool).await?;

        for index_sql in CREATE_WATCH_FOLDER_SUBMODULES_INDEXES_SQL {
            sqlx::query(index_sql).execute(pool).await?;
        }

        // 4. Migrate existing submodule relationships
        let migrated: u64 = sqlx::query(MIGRATE_V21_SUBMODULE_DATA_SQL)
            .execute(pool).await?
            .rows_affected();

        if migrated > 0 {
            info!("Migrated {} submodule relationships to junction table", migrated);
        }

        info!("Migration v21 complete");
        Ok(())
    }

    fn version(&self) -> i32 { 21 }
    fn description(&self) -> &'static str { "Add git-tracking, rules_mirror, submodule junction" }
}
