//! Code line insertion, rebalancing, and seq gap management for search.db.

use tracing::{debug, info};

use super::SearchDbManager;
use super::types::{InsertedLine, RebalanceResult, SearchDbResult};

impl SearchDbManager {
    /// Insert a single code line with the given seq value and line number.
    ///
    /// Returns the new `line_id`.
    async fn insert_code_line_raw(
        &self,
        file_id: i64,
        seq: f64,
        content: &str,
        line_number: i64,
    ) -> SearchDbResult<i64> {
        let result = sqlx::query(
            "INSERT INTO code_lines (file_id, seq, content, line_number) VALUES (?1, ?2, ?3, ?4)",
        )
        .bind(file_id)
        .bind(seq)
        .bind(content)
        .bind(line_number)
        .execute(&self.pool)
        .await?;

        Ok(result.last_insert_rowid())
    }

    /// Insert a code line between two adjacent seq values using midpoint insertion.
    ///
    /// Computes `new_seq = (before_seq + after_seq) / 2.0` and inserts there.
    /// If the resulting gap is below `MIN_SEQ_GAP`, triggers file-local rebalancing.
    ///
    /// Returns the inserted line's `line_id` and `seq`, plus whether rebalancing occurred.
    pub async fn insert_line_between(
        &self,
        file_id: i64,
        before_seq: f64,
        after_seq: f64,
        content: &str,
    ) -> SearchDbResult<(InsertedLine, bool)> {
        use crate::code_lines_schema::{midpoint_seq, MIN_SEQ_GAP};

        let new_seq = midpoint_seq(before_seq, after_seq);
        let gap = (after_seq - before_seq) / 2.0;

        // Temporary line_number=0; renumber_file_line_numbers called below
        let line_id = self.insert_code_line_raw(file_id, new_seq, content, 0).await?;

        // Check if rebalancing is needed
        let rebalanced = if gap < MIN_SEQ_GAP {
            debug!(
                "Gap {:.6} < MIN_SEQ_GAP {:.6} for file_id={}, triggering rebalance",
                gap, MIN_SEQ_GAP, file_id
            );
            self.rebalance_file_seqs(file_id).await?;
            true
        } else {
            false
        };

        // If rebalanced, the seq may have changed -- look up the new value
        let final_seq = if rebalanced {
            sqlx::query_scalar("SELECT seq FROM code_lines WHERE line_id = ?1")
                .bind(line_id)
                .fetch_one(&self.pool)
                .await?
        } else {
            new_seq
        };

        // Renumber all line_numbers for this file after insertion
        self.renumber_file_line_numbers(file_id).await?;

        Ok((InsertedLine { line_id, seq: final_seq }, rebalanced))
    }

    /// Insert a code line at the start of a file (before all existing lines).
    ///
    /// If the file has existing lines, the new line gets `seq = first_seq / 2.0`.
    /// If the file is empty, the new line gets `seq = INITIAL_SEQ_GAP`.
    /// Triggers rebalance if the new seq is below `MIN_SEQ_GAP`.
    pub async fn insert_line_at_start(
        &self,
        file_id: i64,
        content: &str,
    ) -> SearchDbResult<(InsertedLine, bool)> {
        use crate::code_lines_schema::{INITIAL_SEQ_GAP, MIN_SEQ_GAP};

        let first_seq: Option<f64> = sqlx::query_scalar(
            "SELECT MIN(seq) FROM code_lines WHERE file_id = ?1",
        )
        .bind(file_id)
        .fetch_optional(&self.pool)
        .await?
        .flatten();

        let new_seq = match first_seq {
            Some(first) => first / 2.0,
            None => INITIAL_SEQ_GAP,
        };

        // Temporary line_number=0; renumber below
        let line_id = self.insert_code_line_raw(file_id, new_seq, content, 0).await?;

        let rebalanced = if first_seq.is_some() && new_seq < MIN_SEQ_GAP {
            debug!(
                "Start-of-file seq {:.6} < MIN_SEQ_GAP for file_id={}, triggering rebalance",
                new_seq, file_id
            );
            self.rebalance_file_seqs(file_id).await?;
            true
        } else {
            false
        };

        let final_seq = if rebalanced {
            sqlx::query_scalar("SELECT seq FROM code_lines WHERE line_id = ?1")
                .bind(line_id)
                .fetch_one(&self.pool)
                .await?
        } else {
            new_seq
        };

        // Renumber all line_numbers for this file after insertion
        self.renumber_file_line_numbers(file_id).await?;

        Ok((InsertedLine { line_id, seq: final_seq }, rebalanced))
    }

    /// Insert a code line at the end of a file (after all existing lines).
    ///
    /// New line gets `seq = last_seq + INITIAL_SEQ_GAP`.
    /// If the file is empty, gets `seq = INITIAL_SEQ_GAP`.
    /// Appending never triggers rebalance since the gap is always `INITIAL_SEQ_GAP`.
    pub async fn insert_line_at_end(
        &self,
        file_id: i64,
        content: &str,
    ) -> SearchDbResult<InsertedLine> {
        use crate::code_lines_schema::INITIAL_SEQ_GAP;

        // Get max seq and current line count for this file
        let row: Option<(f64, i32)> = sqlx::query_as(
            "SELECT MAX(seq), COUNT(*) FROM code_lines WHERE file_id = ?1",
        )
        .bind(file_id)
        .fetch_optional(&self.pool)
        .await?;

        let (new_seq, line_number) = match row {
            Some((max_seq, count)) if count > 0 => {
                (max_seq + INITIAL_SEQ_GAP, (count + 1) as i64)
            }
            _ => (INITIAL_SEQ_GAP, 1),
        };

        let line_id = self.insert_code_line_raw(file_id, new_seq, content, line_number).await?;

        Ok(InsertedLine { line_id, seq: new_seq })
    }

    /// Rebalance all seq values for a file.
    ///
    /// Reads all lines ordered by current seq, then reassigns seq values
    /// starting at `INITIAL_SEQ_GAP` with `INITIAL_SEQ_GAP` increments
    /// (1000.0, 2000.0, 3000.0, ...).
    ///
    /// Uses a two-phase update to avoid UNIQUE constraint violations:
    /// 1. Set all seqs to negative (`-line_id`) -- guaranteed unique
    /// 2. Assign final seq values
    ///
    /// This is file-local: only affects lines with the given `file_id`.
    /// The FTS5 index does NOT need rebuilding after rebalance because
    /// only `seq` changes -- `line_id` and `content` remain the same.
    ///
    /// Returns the number of lines rebalanced and the new gap.
    pub async fn rebalance_file_seqs(&self, file_id: i64) -> SearchDbResult<RebalanceResult> {
        use crate::code_lines_schema::INITIAL_SEQ_GAP;

        // Read all line_ids in current seq order
        let line_ids: Vec<i64> = sqlx::query_scalar(
            "SELECT line_id FROM code_lines WHERE file_id = ?1 ORDER BY seq",
        )
        .bind(file_id)
        .fetch_all(&self.pool)
        .await?;

        if line_ids.is_empty() {
            return Ok(RebalanceResult {
                lines_rebalanced: 0,
                new_gap: INITIAL_SEQ_GAP,
            });
        }

        let mut tx = self.pool.begin().await?;

        // Phase 1: Set all seqs to negative line_id (unique, avoids constraint violations)
        for line_id in &line_ids {
            sqlx::query("UPDATE code_lines SET seq = ?1 WHERE line_id = ?2")
                .bind(-(*line_id as f64))
                .bind(*line_id)
                .execute(&mut *tx)
                .await?;
        }

        // Phase 2: Assign final seq values
        for (i, line_id) in line_ids.iter().enumerate() {
            let new_seq = (i as f64 + 1.0) * INITIAL_SEQ_GAP;
            sqlx::query("UPDATE code_lines SET seq = ?1 WHERE line_id = ?2")
                .bind(new_seq)
                .bind(*line_id)
                .execute(&mut *tx)
                .await?;
        }

        tx.commit().await?;

        let count = line_ids.len();
        info!("Rebalanced {} lines for file_id={}", count, file_id);

        Ok(RebalanceResult {
            lines_rebalanced: count,
            new_gap: INITIAL_SEQ_GAP,
        })
    }

    /// Renumber all `line_number` values for a file based on `seq` ordering.
    ///
    /// Assigns sequential 1-based line numbers by reading line_ids in seq order
    /// and updating each row. Called after insert_line_between/insert_line_at_start
    /// or after diff operations that change the line count.
    pub async fn renumber_file_line_numbers(&self, file_id: i64) -> SearchDbResult<()> {
        let line_ids: Vec<i64> = sqlx::query_scalar(
            "SELECT line_id FROM code_lines WHERE file_id = ?1 ORDER BY seq",
        )
        .bind(file_id)
        .fetch_all(&self.pool)
        .await?;

        if line_ids.is_empty() {
            return Ok(());
        }

        let mut tx = self.pool.begin().await?;
        for (i, line_id) in line_ids.iter().enumerate() {
            sqlx::query("UPDATE code_lines SET line_number = ?1 WHERE line_id = ?2")
                .bind((i + 1) as i64)
                .bind(*line_id)
                .execute(&mut *tx)
                .await?;
        }
        tx.commit().await?;

        Ok(())
    }

    /// Get the seq values of lines adjacent to a given seq in a file.
    ///
    /// Returns `(before_seq, after_seq)` where either may be `None`
    /// if the given seq is at the start or end of the file.
    pub async fn get_adjacent_seqs(
        &self,
        file_id: i64,
        target_seq: f64,
    ) -> SearchDbResult<(Option<f64>, Option<f64>)> {
        let before: Option<f64> = sqlx::query_scalar(
            "SELECT MAX(seq) FROM code_lines WHERE file_id = ?1 AND seq < ?2",
        )
        .bind(file_id)
        .bind(target_seq)
        .fetch_optional(&self.pool)
        .await?
        .flatten();

        let after: Option<f64> = sqlx::query_scalar(
            "SELECT MIN(seq) FROM code_lines WHERE file_id = ?1 AND seq > ?2",
        )
        .bind(file_id)
        .bind(target_seq)
        .fetch_optional(&self.pool)
        .await?
        .flatten();

        Ok((before, after))
    }

    /// Check if a gap between two seq values requires rebalancing.
    pub fn needs_rebalance(gap: f64) -> bool {
        use crate::code_lines_schema::MIN_SEQ_GAP;
        gap < MIN_SEQ_GAP
    }

    /// Get the minimum gap between adjacent seq values for a file.
    ///
    /// Returns `None` if the file has fewer than 2 lines.
    pub async fn min_seq_gap(&self, file_id: i64) -> SearchDbResult<Option<f64>> {
        let gap: Option<f64> = sqlx::query_scalar(
            r#"
            SELECT MIN(next_seq - seq) FROM (
                SELECT seq, LEAD(seq) OVER (ORDER BY seq) AS next_seq
                FROM code_lines
                WHERE file_id = ?1
            ) WHERE next_seq IS NOT NULL
            "#,
        )
        .bind(file_id)
        .fetch_optional(&self.pool)
        .await?
        .flatten();

        Ok(gap)
    }
}
