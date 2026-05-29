//! Phase 4b: narrative graph extraction during file ingestion.
//!
//! Builds the per-tenant symbol automaton, runs the ordered narrative
//! extractor pipeline, and returns the resulting nodes/edges so they can be
//! merged into the file's single `reingest_file` transaction (alongside the
//! code- and concept-graph layers). This module performs NO graph writes: a
//! separate `reingest_file`/`insert_edges` call here would delete the code and
//! concept edges already prepared for this file.
//!
//! Failure isolation: callers wrap [`run`] so any extractor error yields an
//! empty result and a warning — narrative extraction must never abort ingest.

use std::path::Path;

use tracing::debug;

use crate::context::ProcessingContext;
use crate::graph::{
    compute_node_id_for_type, EdgeType, GraphEdge, GraphNode, NodeIdFields, NodeType,
};
use crate::narrative::comments::CommentExtractor;
use crate::narrative::depth::estimate_depth;
use crate::narrative::explains::ExplainsExtractor;
use crate::narrative::references::ReferencesExtractor;
use crate::narrative::sections::{SectionExtractor, SectionSpan};
use crate::narrative::{run_narrative_pipeline, NarrativeExtractionResult, NarrativeExtractor};
use crate::tagging::Tier2Tagger;

/// Run narrative extraction for a file and return its nodes/edges (plus any
/// COVERS_TOPIC concept nodes/edges) for merging into the graph reingest.
///
/// Returns an empty result when narrative extraction is disabled, no graph
/// store is configured, or no narrative content is found.
///
/// For narrative files, section-granular COVERS_TOPIC edges are produced from
/// each DocumentSection node by classifying the overlap-weighted mean of the
/// chunk vectors that fall within the section's line range (reusing existing
/// chunk vectors — no extra embedding). They share the reingest transaction.
pub(super) async fn run(
    ctx: &ProcessingContext,
    tenant_id: &str,
    file_path: &Path,
    relative_path: &str,
    content: &str,
    language: Option<&str>,
    branch: &str,
    points: &[crate::storage::DocumentPoint],
    chunk_records: &[super::chunk_embed::ChunkRecord],
) -> NarrativeExtractionResult {
    let cfg = &ctx.narrative_config;
    if !cfg.enabled {
        return NarrativeExtractionResult::default();
    }

    let Some(ref graph_store) = ctx.graph_store else {
        return NarrativeExtractionResult::default();
    };

    // Build (or reuse) the tenant's symbol automaton from REAL code-graph
    // symbols. This runs after Phase 4 has written this file's own code nodes,
    // so within-file references resolve.
    let automaton = ctx
        .automaton_cache
        .get_or_build(graph_store, tenant_id, cfg.explains_min_symbol_length)
        .await;

    // SectionExtractor is the single source of section identity; capture its
    // spans so ExplainsExtractor attaches edges to the same section node ids.
    let section_extractor = SectionExtractor::new();
    let section_spans = section_extractor.section_spans(tenant_id, relative_path, content);

    // Ordered extractors: sections first (already captured), then explains
    // (context-aware), then comments + references (context-free).
    let extractors: Vec<Box<dyn NarrativeExtractor>> = vec![
        Box::new(SectionExtractor::new()),
        Box::new(ExplainsExtractor::with_context(
            section_spans.clone(),
            automaton,
            (**cfg).clone(),
        )),
        Box::new(CommentExtractor),
        Box::new(ReferencesExtractor::new()),
    ];

    let mut result = run_narrative_pipeline(
        tenant_id,
        Path::new(relative_path),
        content,
        language,
        branch,
        &extractors,
    )
    .await;

    // Section-granular COVERS_TOPIC edges for narrative files (shares the
    // reingest transaction). Requires the Tier-2 tagger; classifies each
    // section's overlap-weighted chunk mean against the taxonomy.
    if is_narrative_file(file_path) && !section_spans.is_empty() {
        if let Some(ref tagger) = ctx.tier2_tagger {
            let covers = build_covers_topic_edges(
                tagger,
                &ctx.concept_config,
                tenant_id,
                relative_path,
                branch,
                content,
                &section_spans,
                points,
                chunk_records,
            );
            result.merge(covers);
        }
    }

    debug!(
        "narrative_phase: {} nodes, {} edges for {} (branch {})",
        result.nodes.len(),
        result.edges.len(),
        relative_path,
        branch
    );
    result
}

/// Build section-granular COVERS_TOPIC nodes/edges.
///
/// For each section span, classify the overlap-weighted mean of the chunk
/// vectors falling within the section's line range against the Tier-2 taxonomy
/// and emit a weighted COVERS_TOPIC edge from the **DocumentSection** node (not
/// the File node) to each matching ConceptNode, carrying section-depth metadata.
/// Reuses existing chunk vectors (no extra embedding). A section overlapping no
/// chunks produces no edge.
#[allow(clippy::too_many_arguments)]
fn build_covers_topic_edges(
    tagger: &Tier2Tagger,
    concept_config: &crate::config::ConceptConfig,
    tenant_id: &str,
    relative_path: &str,
    branch: &str,
    content: &str,
    section_spans: &[SectionSpan],
    points: &[crate::storage::DocumentPoint],
    chunk_records: &[super::chunk_embed::ChunkRecord],
) -> NarrativeExtractionResult {
    use std::collections::HashMap;

    let lines: Vec<&str> = content.lines().collect();
    let mut concept_nodes: HashMap<String, GraphNode> = HashMap::new();
    let mut edges = Vec::new();

    for span in section_spans {
        // Overlap-weighted mean of chunks intersecting [start_line, end_line].
        let mut weighted_sum: Vec<f32> = Vec::new();
        let mut total_weight: f32 = 0.0;
        for (point, record) in points.iter().zip(chunk_records.iter()) {
            let (Some(cs), Some(ce)) = (record.start_line, record.end_line) else {
                continue;
            };
            let cs = cs.max(0) as u32;
            let ce = ce.max(0) as u32;
            let ov_start = span.start_line.max(cs);
            let ov_end = span.end_line.min(ce);
            if ov_end < ov_start || point.dense_vector.is_empty() {
                continue;
            }
            let overlap = (ov_end - ov_start + 1) as f32;
            if weighted_sum.is_empty() {
                weighted_sum = vec![0.0; point.dense_vector.len()];
            }
            if weighted_sum.len() != point.dense_vector.len() {
                continue; // dimension mismatch — skip defensively
            }
            for (acc, v) in weighted_sum.iter_mut().zip(point.dense_vector.iter()) {
                *acc += overlap * v;
            }
            total_weight += overlap;
        }
        if total_weight <= 0.0 || weighted_sum.is_empty() {
            continue; // section overlaps no chunks → no edge
        }
        for v in weighted_sum.iter_mut() {
            *v /= total_weight;
        }

        let mut matches = tagger.classify(&weighted_sum);
        matches.retain(|m| m.score >= concept_config.min_confidence);
        matches.truncate(concept_config.max_per_unit);
        if matches.is_empty() {
            continue;
        }

        // Estimate depth from the section's own text.
        let s = span.start_line.saturating_sub(1) as usize;
        let e = (span.end_line as usize).min(lines.len());
        let section_text = if s < e {
            lines[s..e].join("\n")
        } else {
            String::new()
        };
        let depth_level = estimate_depth(&section_text, 0, false);

        for m in matches {
            let concept_label = m.category.as_str();
            let concept_id = compute_node_id_for_type(&NodeIdFields::new(
                "__global__",
                "",
                concept_label,
                NodeType::ConceptNode,
            ));
            concept_nodes.entry(concept_id.clone()).or_insert_with(|| {
                let mut node =
                    GraphNode::new("__global__", "", concept_label, NodeType::ConceptNode);
                node.node_id = concept_id.clone();
                node
            });

            let mut edge = GraphEdge::new(
                tenant_id,
                span.node_id.clone(),
                concept_id.clone(),
                EdgeType::CoversTopic,
                relative_path,
            )
            .with_depth(depth_level)
            .with_branch(branch);
            edge.weight = m.score;
            edges.push(edge);
        }
    }

    NarrativeExtractionResult {
        nodes: concept_nodes.into_values().collect(),
        edges,
    }
}

/// Check whether a file is a narrative document (markdown / text / rst).
fn is_narrative_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map_or(false, |ext| {
            let lower = ext.to_ascii_lowercase();
            lower == "md" || lower == "markdown" || lower == "txt" || lower == "rst"
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::storage::DocumentPoint;
    use crate::tagging::{TaxonomyEntry, Tier2Config};
    use std::collections::HashMap as StdHashMap;

    fn mk_point(vec: Vec<f32>, start_line: i32, end_line: i32) -> DocumentPoint {
        let _ = (start_line, end_line);
        DocumentPoint {
            id: "p".to_string(),
            dense_vector: vec,
            sparse_vector: None,
            payload: StdHashMap::new(),
        }
    }

    fn mk_record(start_line: i32, end_line: i32) -> super::super::chunk_embed::ChunkRecord {
        super::super::chunk_embed::ChunkRecord {
            point_id: "p".to_string(),
            chunk_index: 0,
            content_hash: "h".to_string(),
            chunk_type: None,
            symbol_name: None,
            start_line: Some(start_line),
            end_line: Some(end_line),
        }
    }

    fn mk_tagger() -> Tier2Tagger {
        Tier2Tagger::from_precomputed(
            vec![TaxonomyEntry {
                term: "databases".to_string(),
                category: "databases".to_string(),
            }],
            vec![vec![1.0_f32, 0.0, 0.0]],
            Tier2Config::default(),
        )
    }

    #[test]
    fn covers_topic_source_is_section_with_overlap_weighted_classification() {
        let tagger = mk_tagger();
        let cfg = crate::config::ConceptConfig::default();
        let spans = vec![SectionSpan {
            node_id: "sec-node-1".to_string(),
            start_line: 1,
            end_line: 10,
        }];
        // Chunk overlapping the section, matching the taxonomy term (cosine 1.0).
        let points = vec![mk_point(vec![1.0, 0.0, 0.0], 1, 8)];
        let records = vec![mk_record(1, 8)];

        let result = build_covers_topic_edges(
            &tagger,
            &cfg,
            "t1",
            "docs/readme.md",
            "dev",
            "# Title\nintro about databases\n",
            &spans,
            &points,
            &records,
        );

        assert_eq!(result.edges.len(), 1);
        let edge = &result.edges[0];
        // Source is the DocumentSection node, not a File node.
        assert_eq!(edge.source_node_id, "sec-node-1");
        assert_eq!(edge.edge_type, EdgeType::CoversTopic);
        assert_eq!(edge.branch.as_deref(), Some("dev"));
        assert_eq!(edge.source_file, "docs/readme.md");
        assert!(edge.metadata_json.is_some(), "depth metadata expected");
        assert!((edge.weight - 1.0).abs() < 1e-6);
        assert_eq!(result.nodes.len(), 1);
        assert_eq!(result.nodes[0].tenant_id, "__global__");
    }

    #[test]
    fn covers_topic_section_with_no_overlapping_chunks_yields_no_edge() {
        let tagger = mk_tagger();
        let cfg = crate::config::ConceptConfig::default();
        let spans = vec![SectionSpan {
            node_id: "sec-node-1".to_string(),
            start_line: 100,
            end_line: 110,
        }];
        let points = vec![mk_point(vec![1.0, 0.0, 0.0], 1, 8)];
        let records = vec![mk_record(1, 8)];

        let result = build_covers_topic_edges(
            &tagger,
            &cfg,
            "t1",
            "docs/readme.md",
            "dev",
            "# Title\n",
            &spans,
            &points,
            &records,
        );
        assert!(result.edges.is_empty());
        assert!(result.nodes.is_empty());
    }

    #[test]
    fn is_narrative_file_detection() {
        assert!(is_narrative_file(Path::new("a.md")));
        assert!(is_narrative_file(Path::new("a.rst")));
        assert!(is_narrative_file(Path::new("a.txt")));
        assert!(!is_narrative_file(Path::new("a.rs")));
    }
}
