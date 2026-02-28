//! Tests for node/edge ID determinism, constructors, and enum round-trips.

use super::*;

// -- Node ID determinism --

#[test]
fn test_node_id_deterministic() {
    let id1 = compute_node_id("t1", "src/main.rs", "main", NodeType::Function);
    let id2 = compute_node_id("t1", "src/main.rs", "main", NodeType::Function);
    assert_eq!(id1, id2);
    assert_eq!(id1.len(), 32); // 16 bytes = 32 hex chars
}

#[test]
fn test_node_id_differs_by_type() {
    let fn_id = compute_node_id("t1", "src/lib.rs", "Foo", NodeType::Function);
    let struct_id = compute_node_id("t1", "src/lib.rs", "Foo", NodeType::Struct);
    assert_ne!(fn_id, struct_id);
}

#[test]
fn test_node_id_differs_by_tenant() {
    let id1 = compute_node_id("tenant-a", "f.rs", "x", NodeType::Function);
    let id2 = compute_node_id("tenant-b", "f.rs", "x", NodeType::Function);
    assert_ne!(id1, id2);
}

#[test]
fn test_edge_id_deterministic() {
    let id1 = compute_edge_id("node-a", "node-b", EdgeType::Calls);
    let id2 = compute_edge_id("node-a", "node-b", EdgeType::Calls);
    assert_eq!(id1, id2);
    assert_eq!(id1.len(), 32);
}

#[test]
fn test_edge_id_differs_by_type() {
    let calls = compute_edge_id("a", "b", EdgeType::Calls);
    let imports = compute_edge_id("a", "b", EdgeType::Imports);
    assert_ne!(calls, imports);
}

// -- GraphNode constructors --

#[test]
fn test_graph_node_new() {
    let node = GraphNode::new(TENANT, "src/main.rs", "main", NodeType::Function);
    assert!(!node.node_id.is_empty());
    assert_eq!(node.tenant_id, TENANT);
    assert_eq!(node.symbol_name, "main");
    assert_eq!(node.file_path, "src/main.rs");
    assert!(node.start_line.is_none());
}

#[test]
fn test_graph_node_stub() {
    let stub = GraphNode::stub(TENANT, "HashMap", NodeType::Struct);
    assert!(!stub.node_id.is_empty());
    assert_eq!(stub.file_path, ""); // stub has empty path
}

// -- Enum round-trip serialization --

#[test]
fn test_edge_type_round_trip() {
    for et in [
        EdgeType::Calls,
        EdgeType::Contains,
        EdgeType::Imports,
        EdgeType::UsesType,
        EdgeType::Extends,
        EdgeType::Implements,
    ] {
        let s = et.as_str();
        let parsed = EdgeType::from_str(s).unwrap();
        assert_eq!(parsed, et);
    }
}

#[test]
fn test_node_type_round_trip() {
    for nt in [
        NodeType::File,
        NodeType::Function,
        NodeType::AsyncFunction,
        NodeType::Class,
        NodeType::Method,
        NodeType::Struct,
        NodeType::Trait,
        NodeType::Interface,
        NodeType::Enum,
        NodeType::Impl,
        NodeType::Module,
        NodeType::Constant,
        NodeType::TypeAlias,
        NodeType::Macro,
    ] {
        let s = nt.as_str();
        let parsed = NodeType::from_str(s).unwrap();
        assert_eq!(parsed, nt);
    }
}
