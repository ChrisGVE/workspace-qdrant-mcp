//! Cross-crate domain newtypes for the branch-storage model.
//!
//! Location: `wqm-common/src/domain.rs`. Logical context: the single home (arch
//! §8 domain-type nexus, AC-F2.2) of the identifier types shared by daemon-core,
//! the read crate (`wqm-storage`), the write crate (`wqm-storage-write`), and
//! `wqm-client`. Defining them once here means no component invents a private
//! `type TenantId = String` alias and the 9-table schema, the membership/dedup
//! ladders, and the Qdrant payloads all speak one vocabulary.
//!
//! Each is a thin newtype over the value the canonical `hashing` producers emit:
//! `TenantId`/`ContentKey`/`BranchId` wrap the `String` forms, `PointId` wraps
//! the `Uuid` that `hashing::point_id` returns. The newtypes add type-safety at
//! call boundaries without changing the wire/storage representation — serde flattens
//! each back to its inner scalar (transparent), so payloads and rows are unchanged.
//!
//! `FileIdentityId` is deliberately ABSENT — it is retired in this model (§5.4):
//! the new model mints `file_id` per `(branch_id, path)` with no cross-branch
//! identity.
//!
//! Neighbors: [`crate::hashing`] (the producers), [`crate::error::StorageError`].

use std::fmt;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Generate a string newtype: `Debug`/`Clone`/`Eq`/`Hash`/`Ord`, transparent
/// serde, `Display`, `From<String>`/`From<&str>`, `as_str`, and `into_inner`.
macro_rules! string_newtype {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(String);

        impl $name {
            /// Wrap an owned `String` as this id.
            pub fn new(value: impl Into<String>) -> Self {
                Self(value.into())
            }

            /// Borrow the inner string.
            pub fn as_str(&self) -> &str {
                &self.0
            }

            /// Consume the newtype, returning the inner `String`.
            pub fn into_inner(self) -> String {
                self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.0)
            }
        }

        impl From<String> for $name {
            fn from(value: String) -> Self {
                Self(value)
            }
        }

        impl From<&str> for $name {
            fn from(value: &str) -> Self {
                Self(value.to_string())
            }
        }

        impl AsRef<str> for $name {
            fn as_ref(&self) -> &str {
                &self.0
            }
        }
    };
}

string_newtype! {
    /// A tenant (project) identifier — stable across renames, the partition key
    /// for every store and every `tenant_id = ?` Qdrant filter.
    TenantId
}

string_newtype! {
    /// A branch identifier — `SHA256(lp(tenant_id) ‖ lp(location) ‖ lp(branch_name))`,
    /// produced by `wqm_common::hashing` (F4). Hex-encoded; unique per checkout path.
    BranchId
}

string_newtype! {
    /// A content-addressing dedup key — the hex `String` returned by
    /// `hashing::content_key(tenant_id, collection, identity, content_hash)`.
    /// Equal bytes in the same tenant always yield the same `ContentKey` (a dedup HIT).
    ContentKey
}

/// A Qdrant point identifier — the `UUIDv5` that `hashing::point_id(content_key,
/// chunk_index)` derives. Wrapping the `Uuid` keeps point ids from being confused
/// with arbitrary uuids at call boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PointId(Uuid);

impl PointId {
    /// Wrap a `Uuid` as a point id.
    pub fn new(value: Uuid) -> Self {
        Self(value)
    }

    /// Borrow the inner `Uuid`.
    pub fn as_uuid(&self) -> &Uuid {
        &self.0
    }

    /// Consume the newtype, returning the inner `Uuid`.
    pub fn into_inner(self) -> Uuid {
        self.0
    }
}

impl fmt::Display for PointId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

impl From<Uuid> for PointId {
    fn from(value: Uuid) -> Self {
        Self(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn string_newtype_construct_display_and_unwrap() {
        let t = TenantId::new("acme");
        assert_eq!(t.as_str(), "acme");
        assert_eq!(t.to_string(), "acme");
        assert_eq!(TenantId::from("acme"), t);
        assert_eq!(t.clone().into_inner(), "acme".to_string());
    }

    #[test]
    fn string_newtype_serde_is_transparent() {
        // Transparent serde: a BranchId serializes as a bare JSON string, not an object.
        let b = BranchId::new("deadbeef");
        let json = serde_json::to_string(&b).unwrap();
        assert_eq!(json, "\"deadbeef\"");
        let back: BranchId = serde_json::from_str(&json).unwrap();
        assert_eq!(back, b);
    }

    #[test]
    fn content_key_equality_is_byte_exact() {
        assert_eq!(ContentKey::from("abc"), ContentKey::from("abc"));
        assert_ne!(ContentKey::from("abc"), ContentKey::from("abd"));
    }

    #[test]
    fn point_id_wraps_uuid_transparently() {
        let u = Uuid::from_u128(0x1234_5678_9abc_def0_1234_5678_9abc_def0);
        let p = PointId::new(u);
        assert_eq!(p.as_uuid(), &u);
        assert_eq!(p.to_string(), u.to_string());
        let json = serde_json::to_string(&p).unwrap();
        assert_eq!(json, serde_json::to_string(&u).unwrap());
        let back: PointId = serde_json::from_str(&json).unwrap();
        assert_eq!(back, p);
    }
}
