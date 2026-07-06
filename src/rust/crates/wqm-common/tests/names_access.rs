//! N8 access-vocabulary tests (PRD F-01): the N51 operation-class and consumer
//! name sets. One positive assertion per name plus the enumeration contracts.

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
fn consumer_all_enumerates_the_four_surfaces_in_order() {
    assert_eq!(
        Consumer::ALL,
        [
            Consumer::Daemon,
            Consumer::Mcp,
            Consumer::Cli,
            Consumer::Tui,
        ]
    );
}
