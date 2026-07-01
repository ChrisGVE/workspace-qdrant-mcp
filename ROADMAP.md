# Roadmap

## v0.2.0 — Ground-Up Rebuild

workspace-qdrant-mcp is being rebuilt from the ground up for **v0.2.0**. The 0.1.x line
proved the idea; 0.2.0 rebuilds the foundation it stands on so the project can grow without
carrying the rough edges of the first implementation.

This page is the top-line view. It will be expanded with technical detail once the design is
locked (targeted **early July**), at which point we will also open the work to outside
contributors.

---

### The goals, in one breath

- **One coherent storage model** instead of overlapping paths that grew organically.
- **Better search quality** — recall, accuracy, and performance must beat 0.1.x, never regress.
- **Reliable file watching** — moves, renames, and branch switches never drop or mis-attribute content.
- **A robust pipeline** — transient errors don't stall ingestion or silently lose work.
- **A cleaner architecture** — clear boundaries, easy to reason about, easy to contribute to.
- **A painless upgrade** — migrate in place, reuse existing embeddings, no re-indexing.

---

## Development phases

### Phase 0 — Audit the current system *(complete)*

- **Goal:** map exactly how 0.1.x behaves today — every storage path, every quirk, every
  known failure — before changing a line.
- **Benefit:** the rebuild is grounded in evidence, not guesswork. Nothing that works today
  gets lost, and every past rough edge is on the record so we don't reintroduce it.

### Phase 1 — Design the foundation *(in progress)*

- **Goal:** lock the core design — the unified storage model, the change-detection /
  file-watching model, and the overall code architecture.
- **Benefit:** a single, well-specified base that the rest of the system can be built on with
  confidence. This is the phase that ends with a **locked design** and the call for
  contributors.

### Phase 2 — Design the interfaces and subsystems

- **Goal:** specify how the pieces fit — the boundaries between components, the processing
  pipeline, and the public-facing surfaces.
- **Benefit:** clear contracts between parts of the system, so each can be built and tested
  independently and so the project is genuinely open to outside help.

### Phase 3 — Build

- **Goal:** implement the locked design to completion, with the search engine held to a hard
  rule: **beat 0.1.x on recall, accuracy, and performance** — matching is only a floor.
- **Benefit:** measurably better results, fewer surprises when your working tree changes, and
  a system that recovers gracefully instead of getting stuck.

### Phase 4 — Migrate, document, and release

- **Goal:** ship a guided in-place migration, complete the documentation, and cut v0.2.0.
- **Benefit:** existing users upgrade without re-indexing, with a backup taken first so the
  upgrade is reversible — and newcomers arrive to docs that actually match the system.

---

## Migration for existing users

Upgrading from 0.1.x will **not** require re-indexing from scratch:

- **Embeddings are reused** — no re-embedding cost, no re-downloading models.
- **Every existing collection is migrated in place** to the new model.
- **A backup is taken before migration**, so the upgrade is reversible.
- **A guided migration flow** ships with the release.

---

## Contributing

Once the design is locked (**early July**), we'll publish the detailed design and open issues
suitable for external contributors. Until then the architecture is moving quickly — if you'd
like to follow along, **watch the repository** and the issue tracker. External help will be
**very welcome** once the foundations are set.

---

## Release date

It will be ready when it's ready. We're optimizing for getting the foundation right rather
than hitting a fixed date. Watch this page and the
[releases](https://github.com/ChrisGVE/workspace-qdrant-mcp/releases) for updates.
