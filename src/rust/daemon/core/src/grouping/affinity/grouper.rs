/// AffinityGrouper: main entry point for computing and storing affinity groups.

use sqlx::{Row, SqlitePool};
use tracing::{debug, info};

use super::computation::{
    affinity_group_id, build_affinity_groups, compute_group_mean_similarity,
    compute_pairwise_affinities, group_mean_embedding,
};
use super::config::AffinityConfig;
use super::storage::{load_all_project_embeddings, store_affinity_label};
use crate::grouping::schema;
use crate::tagging::Tier2Tagger;

// ---- AffinityGroupInfo -----------------------------------------------------

/// Info about an affinity group a project belongs to.
#[derive(Debug, Clone)]
pub struct AffinityGroupInfo {
    pub group_id: String,
    pub confidence: f64,
    pub label: Option<String>,
    pub category: Option<String>,
    pub label_score: Option<f64>,
}

// ---- AffinityGrouper -------------------------------------------------------

/// Main affinity grouper: computes and stores project affinity groups.
pub struct AffinityGrouper {
    pool: SqlitePool,
    config: AffinityConfig,
}

impl AffinityGrouper {
    pub fn new(pool: SqlitePool, config: AffinityConfig) -> Self {
        Self { pool, config }
    }

    /// Compute and store affinity groups for all projects.
    ///
    /// 1. Load all project aggregate embeddings
    /// 2. Compute pairwise cosine similarities
    /// 3. Build connected components
    /// 4. Store in project_groups with group_type = "affinity"
    ///
    /// Returns the number of groups created.
    pub async fn compute_affinity_groups(&self) -> Result<usize, sqlx::Error> {
        // Clear existing affinity groups
        sqlx::query("DELETE FROM project_groups WHERE group_type = 'affinity'")
            .execute(&self.pool)
            .await?;

        // Also clear stale labels
        sqlx::query("DELETE FROM affinity_labels WHERE 1=1")
            .execute(&self.pool)
            .await?;

        let embeddings = load_all_project_embeddings(&self.pool).await?;
        if embeddings.len() < 2 {
            info!(
                projects = embeddings.len(),
                "Not enough projects for affinity grouping"
            );
            return Ok(0);
        }

        let affinities =
            compute_pairwise_affinities(&embeddings, self.config.similarity_threshold);

        if affinities.is_empty() {
            info!("No project pairs exceed affinity threshold");
            return Ok(0);
        }

        let groups = build_affinity_groups(&affinities);
        let mut groups_created = 0;

        for members in &groups {
            if members.len() < 2 {
                continue;
            }

            let group_id = affinity_group_id(members);
            let mean_sim = compute_group_mean_similarity(members, &affinities);

            for tenant_id in members {
                schema::add_to_group(
                    &self.pool,
                    &group_id,
                    tenant_id,
                    "affinity",
                    mean_sim,
                )
                .await?;
            }

            debug!(
                group_id = group_id.as_str(),
                members = members.len(),
                mean_similarity = mean_sim,
                "Created affinity group"
            );
            groups_created += 1;
        }

        info!(
            projects = embeddings.len(),
            pairs = affinities.len(),
            groups = groups_created,
            threshold = self.config.similarity_threshold,
            "Affinity group computation complete"
        );

        Ok(groups_created)
    }

    /// Compute affinity groups and label them using taxonomy classification.
    ///
    /// Uses `Tier2Tagger` to classify each group's mean embedding against
    /// the taxonomy, producing a human-readable group label.
    pub async fn compute_and_label_groups(
        &self,
        tagger: &Tier2Tagger,
    ) -> Result<usize, sqlx::Error> {
        let groups_created = self.compute_affinity_groups().await?;

        if groups_created == 0 {
            return Ok(0);
        }

        // Load all embeddings again for label computation
        let embeddings = load_all_project_embeddings(&self.pool).await?;

        // Load the groups we just created
        let group_rows = sqlx::query(
            r#"
            SELECT DISTINCT group_id
            FROM project_groups
            WHERE group_type = 'affinity'
            ORDER BY group_id
            "#,
        )
        .fetch_all(&self.pool)
        .await?;

        for row in &group_rows {
            let group_id: String = row.get("group_id");

            // Get members of this group
            let member_rows = sqlx::query(
                "SELECT tenant_id FROM project_groups WHERE group_id = ? ORDER BY tenant_id",
            )
            .bind(&group_id)
            .fetch_all(&self.pool)
            .await?;

            let members: Vec<String> = member_rows
                .iter()
                .map(|r| r.get::<String, _>("tenant_id"))
                .collect();

            // Compute group mean embedding and classify
            if let Some(group_embedding) = group_mean_embedding(&members, &embeddings) {
                let matches = tagger.classify(&group_embedding);
                if let Some(best) = matches.first() {
                    store_affinity_label(
                        &self.pool,
                        &group_id,
                        &best.term,
                        &best.category,
                        best.score,
                    )
                    .await?;

                    debug!(
                        group_id = group_id.as_str(),
                        label = best.term.as_str(),
                        category = best.category.as_str(),
                        score = best.score,
                        "Labeled affinity group"
                    );
                }
            }
        }

        Ok(groups_created)
    }

    /// Get the affinity group(s) for a specific project.
    ///
    /// Returns group_ids and their labels (if available).
    pub async fn get_project_affinity_groups(
        &self,
        tenant_id: &str,
    ) -> Result<Vec<AffinityGroupInfo>, sqlx::Error> {
        let rows = sqlx::query(
            r#"
            SELECT pg.group_id, pg.confidence, al.label, al.category, al.score
            FROM project_groups pg
            LEFT JOIN affinity_labels al ON pg.group_id = al.group_id
            WHERE pg.tenant_id = ? AND pg.group_type = 'affinity'
            ORDER BY pg.confidence DESC
            "#,
        )
        .bind(tenant_id)
        .fetch_all(&self.pool)
        .await?;

        let groups: Vec<AffinityGroupInfo> = rows
            .iter()
            .map(|r| AffinityGroupInfo {
                group_id: r.get("group_id"),
                confidence: r.get("confidence"),
                label: r.get("label"),
                category: r.get("category"),
                label_score: r.get("score"),
            })
            .collect();

        Ok(groups)
    }
}
