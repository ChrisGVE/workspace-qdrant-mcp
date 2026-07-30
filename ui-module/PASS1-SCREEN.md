# Pass 1 — the ratatui ecosystem screen

Run 2026-07-30 against `CRATE-INVENTORY.md`'s two-pass plan. **The authoritative output is
`crate-screen.csv`**: one verdict, one reason, for every crate in the screened universe, so that
a later session can re-litigate a decision by disagreeing with a recorded reason rather than by
redoing the search.

| | |
|---|---|
| crates screened | **5285** |
| KEEP | **257** |
| DROP-VERSION | 2947 |
| DROP-APP | 1705 |
| DROP-DOMAIN | 376 |

The plan estimated 21 agents over 508 rows. What actually happened is that **most of pass 1 turned
out to be a data join rather than a judgement**, and the rows the plan would have screened were
about half the rows that needed screening. Both halves of that are below.

---

## 1. Question zero is a measurement, and it was being asked against the wrong crate

The plan called the ratatui 0.30.2 compatibility gate "question zero … a hard gate", to be
answered per candidate by an agent. It is answerable in bulk: crates.io's reverse-dependency
endpoint returns, for every dependent, the *version requirement* it declares — together with
`license`, `has_lib`, `bin_names`, `edition` and `rust_version`. Evaluating those requirements
against 0.30.2 settles DROP-VERSION for the whole ecosystem at once, and `has_lib` settles
DROP-APP in one direction.

**But the facade is not the ecosystem.** ratatui 0.30 split into `ratatui-core` +
`ratatui-widgets`, and `ratatui-core`'s own description instructs *"Widget libraries should use
this crate."* So the best-behaved widget crates — the ones following the org's published guidance
— declare no dependency on `ratatui` at all and are absent from its reverse-dependency list.
Judging against the facade alone scored the ratatui org's **own umbrella crate `tui-widgets` as
"does not depend on ratatui"**, along with `ansi-to-tui` (4.0M downloads) and the entire `rat-*`
family.

The gate is therefore evaluated against whichever sibling a candidate actually uses — the exact
set ratatui 0.30.2 pins: `ratatui 0.30.2`, `ratatui-core 0.1.2`, `ratatui-widgets 0.3.2`.
Correcting it moved 29 crates from "non-dependent" to gate-OK.

**One asymmetry, worth keeping.** `has_lib = False` proves a crate cannot be adopted;
`has_lib = True` proves nothing — `kanban-tui` is an application that also publishes a lib
target. Only the negative direction is a verdict.

**Verified, not assumed:** across all 338 rows the endpoint returned, `dep_version` equalled the
crate's own `max_version` in every case, so the requirement measured is the requirement of the
*current* release. Had that not held, the gate would have been a claim about stale versions.

## 2. The fourth partial set — the CSV was the top of a search, not the ecosystem

`CRATE-INVENTORY.md` documents three instances of a partial set standing in for a complete one
(the ten-result search cap, the first-1000 `depends_on_ratatui` computation, the thematic tables
read as a list) and closes with: *"it is worth assuming there is a third instance somewhere and
looking for it."*

There is a fourth, and it is the CSV itself. `crate-inventory.csv` is the **top 500 hits of one
download-ranked full-text search**, floored at 2,211 downloads. The 5083 dependents were fetched
only to compute a *column*; they were never added as *rows*. **4745 dependents were therefore
never screened.**

The tell was available inside the document without any new search: **twelve crates
`CRATE-INVENTORY.md` names in its own thematic tables do not appear in its own CSV** —
`ratatui-themekit`, `hjkl-holler`, `hjkl-tabs`, `ratatui-comfy-toaster`, `tui-theme-builder`,
`ratatui-style-presets`, `viuer`, `tui-splitflap`, and both ratatui-org placeholders
(`ratatui-theme`, `ratatui-toast`).

This matters more than a coverage statistic, because the floor is not neutral about *what* it
excludes. The document's own tiering says T3 is "low downloads, aimed squarely at something on
the list. **Highest uplift risk and highest potential saving**" — and a 2,211-download floor
removes T3 almost entirely. **130 of the 257 KEEPs live below that floor**: just over half the
candidate pool was invisible to the screen as planned.

Screened universe = every crate depending on ratatui or a split crate that (a) admits the 0.30.2
sibling set, (b) publishes a library, and (c) ships no binary. Everything outside that is
excluded by measurement, not opinion.

## 3. Two "no candidate exists" findings do not survive

The plan was explicit that bespoke work is only justified once the screen returns empty: *"Only
after the screen returns empty for an area is 'no candidate exists' a measurement rather than a
guess."* Two areas were held back on a guess.

**§3 — status bar, help modal, keybindings.** The document argued this was *"a real finding
rather than a search failure"*, reasoning that both surfaces are too coupled to an application's
key-dispatch model for anyone to ship one. Refuted; the sub-floor tier is full of them:

| crate | what it is |
|---|---|
| `hjkl-statusline-tui` | a **renderer-agnostic statusline model** plus a ratatui adapter — the exact model/renderer split `tokens.rs` already uses |
| `ratatui-which-key`, `hjkl-which-key-tui` | which-key popups — a **help modal derived from the keymap**, which is the §3 surface by another name |
| `tui_pane` | keymap, status bar and framework panes in one crate |
| `monitrs-tui` | rendering, reducer, **keymap** and layout engine |
| `gx-tui` | terminal shell with a **keybind engine** |
| `scarab-nav-protocol` | a protocol for keyboard-driven TUI navigation (above the floor, but missed) |

The document's *reasoning* still looks sound — the keymap is worth designing as an SSOT because
both surfaces derive from it. What was wrong was the conclusion that nobody had shipped one.

**§2 — "No crate found for tab *containers* or *permanent cross-tab headers*. Expect bespoke."**
Also refuted. `ratatui-zonekit` is *"named zones, plugin-owned panes"* — recognisably the zone
model VISUAL-LANGUAGE §6 specifies — and `panes` / `panes-ratatui` is a renderer-agnostic layout
engine with a ratatui adapter. Nine of the ten tab/container KEEPs are sub-floor crates.

**Breadcrumbs remain without a candidate.** That one survives the screen and is now a
measurement.

## 4. Areas the thematic sections had no heading for

These emerged from reading the ranked list rather than from vocabulary guesses, which is the
lesson `tachyonfx` already taught this document.

- **RB — render backends (19).** `soft_ratatui` (software rendering), `ratatui-wgpu`,
  `parley_ratatui`, `native-ascii-renderer` ("render text-based apps in a native window"),
  `mousefood`, `egui_ratatui`. **This is a live lead for config.claude#45**, which currently
  blocks pixel capture behind iTerm2 Automation permissions: a ratatui frame rendered to pixels
  by our own process needs no terminal, no window server and no granted permission. It would also
  retire the last of the `.mock` → `freeze` approximation §1 exists to remove.
- **CAP — capability probing (2).** `termprofile` ("detect and handle terminal color/styling
  support") and `ftui-core` ("terminal lifecycle, capabilities"). This is the unbuilt half of
  workspace-qdrant-mcp#249 and of handover §12's Source × Encoding split — **the Encoding axis is
  a capability that is probed, never chosen**, and these two probe it.
- **TAB / NAV / FORM / MD / LY** — large-data tables (`rat-ftable`), sectioned lists with
  hit-testing (`ratatui-sectioned-list`), schema-driven config (`jkconfig`, `schemaui`), markdown
  (`tui-markdown`, `limner`, and `tuika-mermaid` for terminal-native Mermaid), layout engines
  (`ratatui-hypertile`, `panes`).
- **ART — prior art (22).** Crates that are structurally what `wqm-tui` is: one product's shared
  component-plus-theme library, published. `vtcode-ui`, `vtcode-design`, `apiari-tui`, `oxi-tui`,
  `eddacraft-tui`, `gx-tui`, `resq-tui`. Worth reading for how they drew the seam, independent of
  whether any is adopted.

One theming find deserves singling out: **`karet-theme` does WCAG contrast checking.** Our own
§8.5 defect — `strong()` resolving *below* `normal` under `Theme` and `Indexed` — is a contrast
measurement that no code in this crate performs. Whether or not the crate is adopted, the
technique is the missing instrument.

## 5. What pass 2 should now be

The per-area agent table in `CRATE-INVENTORY.md` (A–I) still holds, with amendments:

- Agent **A (theming)** now has 33 candidates, not 9. It must still answer the
  `ratatui-theme` placeholder question first — is there a public design behind the reservation? —
  and should additionally judge candidates against the **11-rung ladder** and the
  **Source × Encoding split**, since that is the design the evaluation exists to serve.
- Agent **B (tabs)** takes the zone/container question that was written off, with `ratatui-zonekit`
  and `panes` as the leads.
- A new agent takes **ST**: statusline model, keymap SSOT and which-key, evaluated together
  because they share a source — and against `HEALTH-MONITORING.md`'s disqualifying criterion,
  that **rendering must never trigger work**.
- A new agent takes **RB**, scoped to one question: can a ratatui frame be rendered to a PNG
  deterministically, headless, with no terminal? A yes closes config.claude#45.
- Agent **E (textarea forks)** is partly answered already: `tui-textarea` (2.2M, the original) is
  **DROP-VERSION** on `^0.29.0`, while the ratatui-org fork `ratatui-textarea` and `tui-textarea-2`
  both pass. The fork question is now "why did the org fork, and is `edtui` still needed for vim
  modality" rather than "which of three".

Pass 2's six questions are unchanged, except that question 0 is settled for every row that
reaches it and question 6 (licence) is already answered in `crate-screen.csv`.

## 6. Standing methodology note

Four times now, in one document's history, a partial set has stood in for a complete one, and
each time the partial set looked like an answer. The instrument that caught this instance was not
a better search — it was **checking the document against its own contents**: crates named in the
prose were absent from the data the prose claimed to summarise. That check is cheap, it needs no
network, and it is the first thing to run on the next inventory.
