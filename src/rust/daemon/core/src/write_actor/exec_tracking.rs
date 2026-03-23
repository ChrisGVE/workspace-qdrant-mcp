//! SQL execution for TrackingWriteService commands.

use super::actor::WriteActor;
use super::commands::*;
use wqm_common::timestamps;

impl WriteActor {
    pub(super) async fn exec_log_search_event(&self, data: LogSearchEventData) -> WriteResult<()> {
        let now = timestamps::now_utc();

        sqlx::query(
            "INSERT INTO search_events (
                id, ts, session_id, project_id, actor, tool, op,
                query_text, filters, top_k, result_count, latency_ms,
                top_result_refs, outcome, parent_event_id, created_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?2)",
        )
        .bind(&data.id)
        .bind(&now)
        .bind(&data.session_id)
        .bind(&data.project_id)
        .bind(&data.actor)
        .bind(&data.tool)
        .bind(&data.op)
        .bind(&data.query_text)
        .bind(&data.filters)
        .bind(data.top_k)
        .bind(data.result_count)
        .bind(data.latency_ms)
        .bind(&data.top_result_refs)
        .bind(&data.outcome)
        .bind(&data.parent_event_id)
        .execute(&self.pool)
        .await
        .map_err(|e| format!("failed to log search event: {}", e))?;

        Ok(())
    }

    pub(super) async fn exec_update_search_event(
        &self,
        data: UpdateSearchEventData,
    ) -> WriteResult<()> {
        sqlx::query(
            "UPDATE search_events \
             SET result_count = ?1, latency_ms = ?2, top_result_refs = ?3, outcome = ?4 \
             WHERE id = ?5",
        )
        .bind(data.result_count)
        .bind(data.latency_ms)
        .bind(&data.top_result_refs)
        .bind(&data.outcome)
        .bind(&data.event_id)
        .execute(&self.pool)
        .await
        .map_err(|e| format!("failed to update search event: {}", e))?;

        Ok(())
    }

    pub(super) async fn exec_upsert_rule_mirror(
        &self,
        data: UpsertRuleMirrorData,
    ) -> WriteResult<()> {
        sqlx::query(
            "INSERT INTO rules_mirror (rule_id, rule_text, scope, tenant_id, created_at, updated_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6) \
             ON CONFLICT(rule_id) DO UPDATE SET \
                 rule_text = excluded.rule_text, \
                 scope = excluded.scope, \
                 tenant_id = excluded.tenant_id, \
                 updated_at = excluded.updated_at",
        )
        .bind(&data.rule_id)
        .bind(&data.rule_text)
        .bind(&data.scope)
        .bind(&data.tenant_id)
        .bind(&data.created_at)
        .bind(&data.updated_at)
        .execute(&self.pool)
        .await
        .map_err(|e| format!("failed to upsert rules mirror: {}", e))?;

        Ok(())
    }

    pub(super) async fn exec_delete_rule_mirror(
        &self,
        data: DeleteRuleMirrorData,
    ) -> WriteResult<()> {
        sqlx::query("DELETE FROM rules_mirror WHERE rule_id = ?1")
            .bind(&data.rule_id)
            .execute(&self.pool)
            .await
            .map_err(|e| format!("failed to delete rules mirror: {}", e))?;

        Ok(())
    }

    pub(super) async fn exec_upsert_scratchpad_mirror(
        &self,
        data: UpsertScratchpadMirrorData,
    ) -> WriteResult<()> {
        sqlx::query(
            "INSERT INTO scratchpad_mirror \
             (scratchpad_id, title, content, tags, tenant_id, created_at, updated_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7) \
             ON CONFLICT(scratchpad_id) DO UPDATE SET \
                 title = excluded.title, \
                 content = excluded.content, \
                 tags = excluded.tags, \
                 tenant_id = excluded.tenant_id, \
                 updated_at = excluded.updated_at",
        )
        .bind(&data.scratchpad_id)
        .bind(&data.title)
        .bind(&data.content)
        .bind(&data.tags)
        .bind(&data.tenant_id)
        .bind(&data.created_at)
        .bind(&data.updated_at)
        .execute(&self.pool)
        .await
        .map_err(|e| format!("failed to upsert scratchpad mirror: {}", e))?;

        Ok(())
    }

    pub(super) async fn exec_delete_scratchpad_mirror(
        &self,
        data: DeleteScratchpadMirrorData,
    ) -> WriteResult<()> {
        sqlx::query("DELETE FROM scratchpad_mirror WHERE scratchpad_id = ?1")
            .bind(&data.scratchpad_id)
            .execute(&self.pool)
            .await
            .map_err(|e| format!("failed to delete scratchpad mirror: {}", e))?;

        Ok(())
    }
}
