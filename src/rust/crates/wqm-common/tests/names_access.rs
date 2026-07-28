//! N8 access-vocabulary tests: the N51 operation-class and consumer name sets.
//! One positive assertion per name plus the enumeration contracts.

use wqm_common::names::{Consumer, OpClass};

#[test]
fn op_class_read_name() {
    assert_eq!(OpClass::Read.name(), "read");
}

#[test]
fn op_class_proxied_read_name() {
    assert_eq!(OpClass::ProxiedRead.name(), "proxied_read");
}

#[test]
fn op_class_schedule_name() {
    assert_eq!(OpClass::Schedule.name(), "schedule");
}

#[test]
fn op_class_create_name() {
    assert_eq!(OpClass::Create.name(), "create");
}

#[test]
fn op_class_update_name() {
    assert_eq!(OpClass::Update.name(), "update");
}

#[test]
fn op_class_delete_name() {
    assert_eq!(OpClass::Delete.name(), "delete");
}

#[test]
fn op_class_all_enumerates_the_six_classes_in_order() {
    assert_eq!(
        OpClass::ALL,
        [
            OpClass::Read,
            OpClass::ProxiedRead,
            OpClass::Schedule,
            OpClass::Create,
            OpClass::Update,
            OpClass::Delete,
        ]
    );
}

#[test]
fn consumer_daemon_name() {
    assert_eq!(Consumer::Daemon.name(), "daemon");
}

#[test]
fn consumer_mcp_name() {
    assert_eq!(Consumer::Mcp.name(), "mcp");
}

#[test]
fn consumer_cli_name() {
    assert_eq!(Consumer::Cli.name(), "cli");
}

#[test]
fn consumer_tui_name() {
    assert_eq!(Consumer::Tui.name(), "tui");
}

#[test]
fn consumer_restore_name() {
    assert_eq!(Consumer::Restore.name(), "restore");
}

#[test]
fn consumer_all_enumerates_four_surfaces_plus_the_restore_binary_in_order() {
    assert_eq!(
        Consumer::ALL,
        [
            Consumer::Daemon,
            Consumer::Mcp,
            Consumer::Cli,
            Consumer::Tui,
            Consumer::Restore,
        ]
    );
}

/// ARCH rev15 closes the consumer set at "4 surfaces + the restore binary", and
/// the restore binary is explicitly NOT a surface -- it links no serving surface
/// and has no client seam (§3.4). Widening `Consumer` to carry N51's grant axis
/// must not silently promote it to a fifth surface.
#[test]
fn restore_is_the_only_consumer_that_is_not_a_surface() {
    let non_surfaces: Vec<Consumer> = Consumer::ALL
        .into_iter()
        .filter(|c| !c.is_surface())
        .collect();
    assert_eq!(non_surfaces, vec![Consumer::Restore]);
    assert_eq!(
        Consumer::ALL.iter().filter(|c| c.is_surface()).count(),
        4,
        "exactly four serving surfaces"
    );
}
