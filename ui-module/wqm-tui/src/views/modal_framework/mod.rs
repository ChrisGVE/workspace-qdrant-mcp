//! The modal framework — the windows the 20:30 composition ruling describes, composed.
//!
//! [`crate::widgets::modal_frame`] owns the Container and the Decoration; this owns the third
//! component, the **view**, in its two kinds, and the machinery that stacks them.
//!
//! A window is `Container + Decoration + View`, and the three are composable: a library's
//! detail window holds a record view, pressing Enter on its queue field drills down to the
//! Queue's own table view pre-filtered to that library, and the same table reached from the
//! Queue tab itself is the same view under a different trail. What differs between the two is
//! the decoration wrapped around it, which is exactly why the decoration is a separate thing.

pub mod record;
