//! Full screens — the pantry's fourth tab.
//!
//! `tui-pantry` sorts ingredients into Widgets, Panes, **Views** and Styles, and a View is a
//! full-page layout. Everything in [`crate::widgets`] is one element judged in isolation;
//! this is where those elements are put on a screen together, at the storyboard's own
//! geometry, and judged as a composition.
//!
//! # Why the composition is its own deliverable
//!
//! r02 opens with a complaint that no single-widget frame can answer: an earlier design
//! *"read flat"*, because five roles had collected on one colour and the eye could not find
//! where it was. Flatness is a property of a whole screen — of how much of it is muted, how
//! many things compete, where the one accent lands. Every widget in this crate can be
//! individually correct and the screen they make can still be flat, so the screen has to be
//! rendered before that can be judged.
//!
//! # What a view is allowed to invent: nothing
//!
//! A view composes built widgets, screen chrome included ([`crate::widgets::chrome`]). If a
//! screen needs an element that does not exist, that element is a widget and belongs in
//! [`crate::widgets`] — a view that grows its own rendering is how two screens end up drawing
//! the same thing twice.

pub mod dashboard;
pub mod service;
pub mod top;
pub mod shell;
