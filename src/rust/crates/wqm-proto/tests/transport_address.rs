//! `P04-GT002-WO079` — `C-ty-transport-address`.
//!
//! Vocabulary only: the type names where a daemon serves and a client dials, and
//! nothing here binds anything. What is worth asserting about a type with no
//! behavior is the shape its future enforcer depends on — so the first test
//! checks that the `Tcp` payload can answer "is this loopback?", which is the
//! question N48's serve loop will ask when it refuses a non-loopback bind. If the
//! payload ever regresses to a `String`, that question stops being answerable and
//! this test stops compiling, which is the point.
//!
//! The remaining tests pin already-landed formatting (`P04-GT001-WO011`) through
//! the rename, and pin the transition alias to the canonical type.

use std::any::TypeId;
use std::net::SocketAddr;
use std::path::PathBuf;

use wqm_proto::{Address, TransportAddress};

/// The fact N48 will act on is carried by the type, without the type acting on it.
///
/// A `SocketAddr` payload makes loopback a property that can be read off the
/// value; a `String` payload would make it a parse that can fail. The refusal
/// itself lives in N48's serve loop and is deliberately not exercised here.
#[test]
fn tcp_payload_discriminates_loopback() {
    let loopback: SocketAddr = "127.0.0.1:0".parse().expect("a literal loopback address");
    let wildcard: SocketAddr = "0.0.0.0:0".parse().expect("a literal wildcard address");

    let TransportAddress::Tcp(loopback) = TransportAddress::Tcp(loopback) else {
        panic!("constructed a Tcp variant");
    };
    let TransportAddress::Tcp(wildcard) = TransportAddress::Tcp(wildcard) else {
        panic!("constructed a Tcp variant");
    };

    assert!(loopback.ip().is_loopback(), "127.0.0.1 is loopback");
    assert!(!wildcard.ip().is_loopback(), "0.0.0.0 is not loopback");
    assert_ne!(
        loopback.ip().is_loopback(),
        wildcard.ip().is_loopback(),
        "the two bind cases N48 must tell apart are distinguishable on the payload"
    );
}

/// Exact strings, because both are read by a human operator and one is parsed by
/// tonic. A regression here would be silent at compile time.
#[test]
fn display_and_endpoint_uri_are_unchanged_by_the_rename() {
    let uds = TransportAddress::Uds(PathBuf::from("/var/run/wqm-v2/daemon.sock"));
    assert_eq!(uds.to_string(), "unix:/var/run/wqm-v2/daemon.sock");
    assert_eq!(uds.endpoint_uri(), "http://[::]:50051");

    let tcp = TransportAddress::Tcp("127.0.0.1:50051".parse().expect("a literal address"));
    assert_eq!(tcp.to_string(), "tcp:127.0.0.1:50051");
    assert_eq!(tcp.endpoint_uri(), "http://127.0.0.1:50051");
}

/// The transition alias is the same type, not a parallel one.
///
/// It exists only so the concurrent `ui-module` workspace — which this group task
/// may not edit — keeps compiling across the rename. If it ever stopped being an
/// alias, that tree would still compile against a different type and the two would
/// diverge silently.
#[test]
fn address_is_an_alias_of_transport_address() {
    fn takes_canonical(address: TransportAddress) -> String {
        address.to_string()
    }

    let via_alias: Address = Address::Uds(PathBuf::from("/tmp/wqm-v2.sock"));
    assert_eq!(takes_canonical(via_alias), "unix:/tmp/wqm-v2.sock");
    assert_eq!(
        TypeId::of::<Address>(),
        TypeId::of::<TransportAddress>(),
        "`Address` is an alias, not a second transport type"
    );
}
