# TESTING.md -- the wqm-0.2 test charter (project-level teeth)

This charter governs testing for the whole wqm-0.2 rebuild. It refines the global
coding discipline and, unlike advisory guidance, it has **teeth**: mechanical CI
guards and a governance rule enforce it. Adopted 2026-07-06.

## Two tiers

### Unit tier
- Tests live **alongside** the code they cover but in **separate files**, never
  inline in production source. (A `src/foo.rs` module is tested from
  `tests/foo.rs` within the crate, or a sibling `foo_tests.rs` included only under
  `#[cfg(test)]` -- never a `#[cfg(test)] mod tests` block inside the production
  file.)
- Every situation gets a **positive** test: the regular path and **each** border
  case. Aspire past 100% case-coverage -- one test per distinguishable behavior,
  not one per function.
- A unit test is committed together with the code it covers, when that code is
  complete.

### Functional tier
- Verifies multi-block behavior **intra- and inter-module**, against the **real**
  backend (an ephemeral test DB / endpoint) whenever one exists -- not a mock.
- Covers three classes: **vanilla** (the happy path), **edge** (boundaries), and
  **meant-to-break** (malformed / hostile / impossible input the system must
  recognize, refuse, or gracefully handle -- never silently mishandle).
- Drafted with the feature and **auditor-approved at draft**; run at least at
  **every module completion**.

## Anti-drift teeth

These are non-negotiable and enforced (see *Enforcement* below):

1. **Frozen tests.** A committed test is frozen. Modifying it requires the coder
   **plus one auditor** sign-off, with a recorded reason drawn from exactly one of:
   `contract-changed` | `new-border-case` | `test-was-wrong`. No silent test edits.
2. **No disabling.** Never disable, ignore, skip, comment-out, or delete a test to
   get a green suite. There is no `#[ignore]` escape hatch; a genuinely
   long-running test goes behind an explicit feature gate that CI still runs.
3. **Green gates progression.** A full green suite gates forward work. A red suite
   **halts all forward work** -- only a sustainable root-cause fix is allowed, never
   a test edit to force green.

## Enforcement

- **`ci/guard_no_skipped_tests.py`** (run by `ci/run_guards.sh`, gated in CI) fails
  the build on any `#[ignore]`, `#[cfg_attr(.., ignore)]`, or `#[cfg(ignore)]` --
  mechanizing tooth #2.
- **`cargo test --workspace`** green is a required CI job on every push --
  mechanizing tooth #3.
- **Frozen-test governance** (tooth #1) is enforced in review: a diff that changes
  an existing test without the coder + auditor sign-off and a recorded reason is
  rejected. The workspace-qdrant rule `wqm-0.2-tdd-teeth` carries the same law
  across sessions.

## Status

The global Claude harness test-hardening is being stood up separately and is not
final; this project-level charter is self-contained and does not depend on it. As
the harness lands, redundant project machinery may be retired -- but only by
amendment, never by silently dropping a tooth.
