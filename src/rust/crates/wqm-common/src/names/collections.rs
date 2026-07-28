//! Collection-name constants (N8).
//!
//! The four canonical collections are locked by ADR-001: every stored point lives
//! in exactly one of them. `images` is a *reserved* name -- claimed here so nothing
//! else may take it, but not yet an active [`Collection`] variant (the image
//! pipeline arrives in a later phase).

/// A canonical Qdrant collection. The set is closed (ADR-001); N35 (`P04-GT055`) attaches
/// the per-collection behavioural profile keyed by this discriminant, while N8 owns
/// only the name each variant is spelled with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Collection {
    /// Project-scoped code and documents, partitioned by `tenant_id`.
    Projects,
    /// Shared reference libraries, partitioned by `library_name`.
    Libraries,
    /// Behavioural rules auto-injected into agent context.
    Rules,
    /// Scratchbook notes and half-baked ideas.
    Scratchpad,
}

impl Collection {
    /// Every canonical collection, in declaration order. Lets consumers (and the
    /// uniqueness tests) enumerate the closed set without re-spelling it.
    pub const ALL: [Collection; 4] = [
        Collection::Projects,
        Collection::Libraries,
        Collection::Rules,
        Collection::Scratchpad,
    ];

    /// The canonical wire/storage name of this collection.
    pub const fn name(self) -> &'static str {
        match self {
            Collection::Projects => "projects",
            Collection::Libraries => "libraries",
            Collection::Rules => "rules",
            Collection::Scratchpad => "scratchpad",
        }
    }
}

/// Free-function form of [`Collection::name`], matching the N8 contract's
/// `collection_name(Collection)` accessor.
pub const fn collection_name(collection: Collection) -> &'static str {
    collection.name()
}

/// The reserved `images` collection name -- claimed by N8 so no other collection
/// may take it, ahead of the image pipeline that will make it an active
/// [`Collection`] variant.
pub const RESERVED_IMAGES_COLLECTION: &str = "images";
