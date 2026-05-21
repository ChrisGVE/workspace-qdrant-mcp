/// Tag-based affinity grouping via Jaccard similarity (T24).
///
/// Aggregates per-project tag frequency profiles from the `tags` SQLite table,
/// computes pairwise Jaccard similarity, and groups projects that exceed the
/// threshold into `project_groups` entries with `group_type = "tag_affinity"`.
use std::collections::{HashMap, HashSet};

use sqlx::{Row, SqlitePool};
use tracing::{debug, info};

use crate::grouping::schema;

/// Default Jaccard similarity threshold for tag-based grouping.
const DEFAULT_TAG_SIMILARITY_THRESHOLD: f64 = 0.25;

/// Group type stored in `project_groups`.
const TAG_AFFINITY_GROUP_TYPE: &str = "tag_affinity";

/// Configuration for tag-based affinity grouping.
#[derive(Debug, Clone)]
pub struct TagAffinityConfig {
    /// Minimum Jaccard similarity to group two projects (default: 0.25).
    pub similarity_threshold: f64,
}

impl Default for TagAffinityConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: DEFAULT_TAG_SIMILARITY_THRESHOLD,
        }
    }
}

/// A computed tag affinity between two projects.
#[derive(Debug, Clone)]
pub struct TagAffinity {
    pub tenant_a: String,
    pub tenant_b: String,
    pub similarity: f64,
}

/// Load per-project tag sets from the `tags` table.
///
/// Returns a map from `tenant_id` to the set of distinct tag names. Both
/// `concept` and `structural` tag types are included.
pub async fn load_project_tag_profiles(
    pool: &SqlitePool,
) -> Result<HashMap<String, HashSet<String>>, sqlx::Error> {
    let rows = sqlx::query("SELECT DISTINCT tenant_id, tag FROM tags ORDER BY tenant_id, tag")
        .fetch_all(pool)
        .await?;

    let mut profiles: HashMap<String, HashSet<String>> = HashMap::new();
    for row in rows {
        let tenant: String = row.get("tenant_id");
        let tag: String = row.get("tag");
        profiles.entry(tenant).or_default().insert(tag);
    }

    Ok(profiles)
}

/// Compute Jaccard similarity between two tag sets.
///
/// Returns `|A intersect B| / |A union B|`, or 0.0 when both sets are empty.
pub fn tag_jaccard_similarity(a: &HashSet<String>, b: &HashSet<String>) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    let intersection = a.intersection(b).count();
    let union = a.union(b).count();
    if union == 0 {
        return 0.0;
    }
    intersection as f64 / union as f64
}

/// Compute pairwise tag Jaccard similarities and return pairs above threshold.
pub fn compute_tag_affinities(
    profiles: &HashMap<String, HashSet<String>>,
    threshold: f64,
) -> Vec<TagAffinity> {
    let tenants: Vec<&String> = profiles.keys().collect();
    let mut affinities = Vec::new();

    for i in 0..tenants.len() {
        for j in (i + 1)..tenants.len() {
            let a = &profiles[tenants[i]];
            let b = &profiles[tenants[j]];
            let sim = tag_jaccard_similarity(a, b);

            if sim >= threshold {
                affinities.push(TagAffinity {
                    tenant_a: tenants[i].clone(),
                    tenant_b: tenants[j].clone(),
                    similarity: sim,
                });
            }
        }
    }

    affinities
}

/// Build connected-component groups from pairwise affinities.
///
/// If A~B and B~C both exceed the threshold, all three form one group.
pub fn build_tag_affinity_groups(affinities: &[TagAffinity]) -> Vec<Vec<String>> {
    if affinities.is_empty() {
        return Vec::new();
    }

    // Build adjacency list
    let mut adj: HashMap<&str, HashSet<&str>> = HashMap::new();
    for a in affinities {
        adj.entry(&a.tenant_a).or_default().insert(&a.tenant_b);
        adj.entry(&a.tenant_b).or_default().insert(&a.tenant_a);
    }

    // BFS to find connected components
    let mut visited: HashSet<&str> = HashSet::new();
    let mut groups: Vec<Vec<String>> = Vec::new();

    for start in adj.keys() {
        if visited.contains(*start) {
            continue;
        }

        let mut component = Vec::new();
        let mut queue = vec![*start];

        while let Some(node) = queue.pop() {
            if !visited.insert(node) {
                continue;
            }
            component.push(node.to_string());
            if let Some(neighbors) = adj.get(node) {
                for &neighbor in neighbors {
                    if !visited.contains(neighbor) {
                        queue.push(neighbor);
                    }
                }
            }
        }

        component.sort();
        groups.push(component);
    }

    groups.sort_by(|a, b| a[0].cmp(&b[0]));
    groups
}

/// Generate a deterministic group_id for a tag-affinity group.
///
/// Uses sorted tenant_ids to produce a stable hash prefix.
pub fn tag_affinity_group_id(members: &[String]) -> String {
    use sha2::{Digest, Sha256};

    let mut sorted = members.to_vec();
    sorted.sort();
    let input = sorted.join("|");
    let hash = Sha256::digest(input.as_bytes());
    format!("tag_aff:{:x}", hash)[..28].to_string()
}

/// Compute the mean Jaccard similarity among all pairs within a group.
pub fn compute_group_mean_jaccard(members: &[String], affinities: &[TagAffinity]) -> f64 {
    let mut total = 0.0;
    let mut count = 0;

    for a in affinities {
        let a_in = members.contains(&a.tenant_a);
        let b_in = members.contains(&a.tenant_b);
        if a_in && b_in {
            total += a.similarity;
            count += 1;
        }
    }

    if count == 0 {
        0.0
    } else {
        total / count as f64
    }
}

/// Recompute tag-affinity groups for all projects.
///
/// 1. Load per-project tag profiles from the `tags` table
/// 2. Compute pairwise Jaccard similarities
/// 3. Build connected components from pairs above threshold
/// 4. Store in `project_groups` with `group_type = "tag_affinity"`
///
/// Returns the number of groups created.
pub async fn compute_tag_affinity_groups(
    pool: &SqlitePool,
    config: &TagAffinityConfig,
) -> Result<usize, sqlx::Error> {
    // Clear existing tag_affinity groups
    sqlx::query("DELETE FROM project_groups WHERE group_type = ?")
        .bind(TAG_AFFINITY_GROUP_TYPE)
        .execute(pool)
        .await?;

    let profiles = load_project_tag_profiles(pool).await?;
    if profiles.len() < 2 {
        info!(
            projects = profiles.len(),
            "Not enough projects for tag affinity grouping"
        );
        return Ok(0);
    }

    let affinities = compute_tag_affinities(&profiles, config.similarity_threshold);

    if affinities.is_empty() {
        info!("No project pairs exceed tag affinity threshold");
        return Ok(0);
    }

    let groups = build_tag_affinity_groups(&affinities);
    let mut groups_created = 0;

    for members in &groups {
        if members.len() < 2 {
            continue;
        }

        let group_id = tag_affinity_group_id(members);
        let mean_sim = compute_group_mean_jaccard(members, &affinities);

        for tenant_id in members {
            schema::add_to_group(
                pool,
                &group_id,
                tenant_id,
                TAG_AFFINITY_GROUP_TYPE,
                mean_sim,
            )
            .await?;
        }

        debug!(
            group_id = group_id.as_str(),
            members = members.len(),
            mean_jaccard = mean_sim,
            "Created tag affinity group"
        );
        groups_created += 1;
    }

    info!(
        projects = profiles.len(),
        pairs = affinities.len(),
        groups = groups_created,
        threshold = config.similarity_threshold,
        "Tag affinity group computation complete"
    );

    Ok(groups_created)
}
