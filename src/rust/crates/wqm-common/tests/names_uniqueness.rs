//! N8 name-uniqueness tests. Within each vocabulary every canonical
//! name is distinct, so the registry never produces a colliding identifier.

use std::collections::HashSet;

use wqm_common::names::{Collection, Consumer, EnvVar, OpClass};

fn all_distinct(names: &[&'static str]) -> bool {
    names.iter().collect::<HashSet<_>>().len() == names.len()
}

#[test]
fn collection_names_are_unique() {
    let names: Vec<&str> = Collection::ALL.iter().map(|c| c.name()).collect();
    assert!(all_distinct(&names), "collection names collide: {names:?}");
}

#[test]
fn env_keys_are_unique() {
    let keys: Vec<&str> = EnvVar::ALL.iter().map(|e| e.key()).collect();
    assert!(all_distinct(&keys), "env keys collide: {keys:?}");
}

#[test]
fn op_class_names_are_unique() {
    let names: Vec<&str> = OpClass::ALL.iter().map(|o| o.name()).collect();
    assert!(all_distinct(&names), "op-class names collide: {names:?}");
}

#[test]
fn consumer_names_are_unique() {
    let names: Vec<&str> = Consumer::ALL.iter().map(|c| c.name()).collect();
    assert!(all_distinct(&names), "consumer names collide: {names:?}");
}
