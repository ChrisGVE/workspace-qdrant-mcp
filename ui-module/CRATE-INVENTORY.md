# Ratatui ecosystem inventory — candidates for wqm-tui / tui-pantry components

Assembled 2026-07-30. **This is an inventory, not an evaluation.** Nothing below has been read,
built, or judged — download counts and self-descriptions are all that is recorded, because those
are the only facts a search returns. Every "?" is a question for an evaluator, not a gap in the
search.

> **Pass 1 has since run — read `PASS1-SCREEN.md` first.** It screened 5285 crates (not 508) and
> corrected two claims made below: §2's "expect bespoke" for tab containers and §3's "no
> library-grade candidate found" are both **refuted**, and the compatibility gate was being asked
> against the wrong crate. `crate-screen.csv` carries a verdict and a reason for every row. The
> tables below are left as written, because what they got wrong is the point of §"the fourth
> partial set" in that document.

## The enumeration, and why the first pass was wrong

The first pass used the `cratesio` MCP, which shows **ten** results per query and exposes no
page parameter. The thematic tables below were therefore the *top* of each list, not the list —
they read as an inventory while being a sample, which is the failure mode that makes a survey
worth less than no survey. Chris caught it.

The real enumeration is `crate-inventory.csv` beside this file, taken from the crates.io HTTP
API ranked by all-time downloads:

| | |
|---|---|
| full-text `ratatui` matches | **3094** (top 500 fetched) |
| full-text `pantry` matches | **15** (all fetched) |
| crates that actually *depend* on ratatui | **5083** (all fetched) |
| after dedup, minus never-published | **508 rows in the CSV** |
| download range | 166,137,327 … 50 |
| download floor of the 500th search hit | **2,211** |

Two consequences worth stating plainly:

- **"Drop the never-published" removes almost nothing.** Only **3** of 500 were placeholders or
  near-zero. Sorting by downloads had already excluded the dead; the filter is still correct,
  it just does not do the work one might expect it to.
- **The search set and the dependent set disagree.** **338** of the 508 rows actually depend on
  ratatui; **170 merely mention it** in their description. And the dependent list runs to 5083,
  so most dependents sit outside the top 500 entirely — neither list alone is the ecosystem. The
  CSV carries a `depends_on_ratatui` column so an evaluator can tell which kind of row it is
  looking at.

  That column was wrong in the first draft and is worth flagging as a trap: it was computed
  against only the first 1000 dependents, so `False` conflated "does not depend on ratatui" with
  "not in the sample" — and an agent screening for `DROP-APP` would have taken the conflation as
  evidence. All 5083 are now fetched, so `False` means what it says. A partial set silently
  standing in for a complete one is the same failure as the ten-result search above; it is worth
  assuming there is a third instance somewhere and looking for it.

  **There was, and it is this table.** The 5083 dependents were fetched to compute a *column*,
  never to add *rows* — so the CSV is the top 500 of one search, floored at 2,211 downloads, and
  **4745 dependents were never screened**. `False` in that column also still means "does not
  depend on the `ratatui` facade", which is not the same as "does not depend on ratatui": the org
  tells widget libraries to depend on `ratatui-core` instead. Both corrections, and what fell out
  of them, are in `PASS1-SCREEN.md` §§1–2.

Freshness across the 508: 305 last updated in 2026, 121 in 2025, 67 in 2024, 15 older. Roughly
one in six has not moved in over a year, which is an uplift-risk signal available before anyone
opens a repository.

The bar is deliberately low: a crate does not have to be good, only *upliftable*. A crate that
does 70% of a job with a clean seam is a better starting point than a blank file, and several
below are explicitly modelled the way `wqm-tui` already is — a renderer-agnostic core plus a
thin ratatui adapter.

## How to read the tiers

- **T1 — first-party.** Published by the ratatui org itself. Version-tracks ratatui, so the
  0.30 coupling that constrains everything here is a non-issue.
- **T2 — established third-party.** Six figures of downloads, an obvious maintainer, a real
  repository.
- **T3 — small but on-point.** Low downloads, aimed squarely at something on the list. Highest
  uplift risk and highest potential saving.
- **T4 — adjacent frameworks.** Not ratatui. Recorded so the evaluation can say *why not*
  rather than leaving them unexamined.

## The constraint every candidate is judged against

`wqm-tui` is on **ratatui 0.30.2**, and `tui-pantry 0.4` re-exports ratatui, so a widget crate
built against 0.29 or earlier presents an incompatible `Widget` trait and cannot be composed at
all. **Ratatui version is the first thing to check on every candidate** — it is a hard gate, not
a preference, and it will silently disqualify a large fraction of this list.

---

## 1. Theming — the live decision

Directly relevant to today's settled palette work (`handover.md` §7.1) and to
workspace-qdrant-mcp#249, which is open precisely because our degradation story is documented
but unimplemented.

| Crate | Ver | Downloads | Self-description | Tier |
|---|---|---|---|---|
| `ratatui-themekit` | 0.6.1 | 1.4K / 1.1K recent | "Semantic theme system — 11 themes, widget builders, full-screen canvas, zebra rows, state-aware styles, **NO_COLOR**" | T3 |
| `ratatui-themes` | 0.2.0 | 11.3K / 5.8K | "A collection of popular color themes" | T3 |
| `tui-theme-builder` | 0.2.2 | 1.6K / 217 | "Theme deserializer for Ratatui" | T3 |
| `ratatui-style-presets` | 0.1.0 | 17 | "CSS theme & utility presets — Tailwind utilities, widget defaults, Catppuccin/Nord/Dracula palettes" | T3 |
| `hjkl-theme-tui` | 0.39.0 | 1.0K | "Ratatui adapters for hjkl-theme: Color/StyleSpec/Modifiers conversions" | T3 |
| `ratatui-bubbletea-theme` | 0.2.0 | 820 | "Charm/Bubble Tea-inspired theme helpers" | T3 |
| `ratatui-style-ayu` | 0.4.0 | 199 | single theme | T3 |
| `tca-ratatui` | 0.7.0 | 373 | "TCA theme support" | T3 |
| **`ratatui-theme`** | **0.0.0** | 19 | **Placeholder reserved by the ratatui org** — "Reserved for Ratatui theme models, theme application, and styling conventions" | T1 |

**The `ratatui-theme` placeholder is the most consequential row in this table** and none of the
others matter as much: the ratatui org has reserved the name and stated the intent. Adopting a
third-party theme system now may mean migrating off it when first-party lands. The evaluation
must find out whether there is a public design or RFC behind that reservation before
recommending any of the others.

Questions for the evaluator: does any of these express an **11-rung neutral ladder**, or do
they all assume a flat named-colour map? Does any **derive from the terminal** (`OSC 11`/`OSC
10`) rather than shipping fixed palettes? Does `ratatui-themekit`'s NO_COLOR support actually
degrade *structurally* (weight/glyph) or merely drop colour? Can any be fed our
source-vs-encoding split, or do they conflate them the way our `Palette` enum did?

## 2. Tabs, containers, permanent headers

| Crate | Ver | Downloads | Self-description | Tier |
|---|---|---|---|---|
| `hjkl-tabs` | 0.39.0 | 599 | "**Renderer-agnostic** tab bar data model — `TabBar<Id>` + `Tab<Id>` with open/close/focus/cycle/**overflow**" | T3 |
| `hjkl-tabs-tui` | 0.39.0 | 494 | "Ratatui adapter — active-tab highlight, **dirty marker**, overflow indicators" | T3 |
| `tui-tabs` | 0.1.1 | 7.7K / 6.8K | "Tab navigation widget with bordered boxes and rounded corners" | T3 |

`hjkl-tabs` is the interesting one: a data model separate from its renderer is exactly the
shape `wqm-tui` already uses (`tokens.rs` has no widget dependencies for the same reason), so
the adapter could be replaced while the model is kept. `tui-tabs` describes borders and rounded
corners, which VISUAL-LANGUAGE §6 explicitly rejects ("**NO frame border**", zones divided by
rules, boxes only for modals) — so it is likely a style mismatch rather than a candidate.

~~**No crate found for tab *containers* or *permanent cross-tab headers*.** Expect bespoke.~~
**REFUTED by pass 1.** Nine of the ten tab/container candidates sit below this document's
download floor and were never searched. `ratatui-zonekit` is *"named zones, plugin-owned panes"*
— recognisably the zone model VISUAL-LANGUAGE §6 specifies — and `panes` + `panes-ratatui` is a
renderer-agnostic layout engine with a ratatui adapter, the same model/renderer split used here.
Also on the list: `ratatui-tabs`, `ratatui-comfy-tabs`, `hjkl-tabs-tui`, `tui_pane`, `tuicore`,
`turtletap`. See `PASS1-SCREEN.md` §3.

## 3. Status bar, help modals, keybindings

~~**No library-grade candidate found.** Every hit for "status bar keybindings help" was an
*application* with one (kanban-tui, zeph-tui, fpv, bridgio…), not a reusable widget.~~

~~This is a real finding rather than a search failure~~ — **it was a search failure, and pass 1
refutes it.** Six candidates exist, all but one below this document's download floor:
`hjkl-statusline-tui` (a *renderer-agnostic statusline model* plus a ratatui adapter — the exact
split used here), `ratatui-which-key` and `hjkl-which-key-tui` (a help popup derived from the
keymap, which is this section's second surface), `tui_pane` (keymap + status bar + panes),
`monitrs-tui` (reducer + keymap + layout), `gx-tui` (a keybind engine), and above the floor
`scarab-nav-protocol`. See `PASS1-SCREEN.md` §3.

The *reasoning* below survives even though the conclusion did not — both surfaces derive from one
key table, so: (a) the keymap becomes an SSOT worth designing deliberately, since both surfaces
derive from it, and (b) `terminput` (§8) is the closest thing to a foundation.

### The status half now has a hard constraint

`../HEALTH-MONITORING.md` (commissioned by Chris, binding decision `CR-035`, owner `N49`) sets
the shape before this surface is built. Chris's requirement: **a stable value that can be
observed** — if it changes, alert; if it does not, nothing happens. v0.1 instead polls, driving
a pair of live Qdrant round-trips every ~3 s for as long as anyone is looking, and filling 84% of
the busiest minute's log with one unconditional success line.

Two evaluation criteria follow, and the first is disqualifying rather than advisory:

- **Rendering must never trigger work.** A status widget that probes, refreshes, or ticks on
  draw reproduces the v0.1 shape regardless of transport, and is out on that basis alone.
- **Settled transitions must coalesce.** A flapping value pushing one event per flap reproduces
  the volume this exists to remove — so a candidate needs debounce-to-settled, or a seam where
  we supply it.

The brief also notes the trap for the obvious fix: a watch changes *who initiates*, not *what a
read costs*, and with N subscribers a naive watch is **worse** than polling. That part is
daemon-side and not something a TUI crate can solve — worth knowing so an evaluator does not
credit a crate for fixing it.

**This adds an area the thematic sections missed: observable-state plumbing** — a watch channel
carrying the whole report, with debounce on settled change. Likely `tokio::sync::watch` plus a
debouncer rather than a TUI crate at all, but pass 1 should tag anything in that space, because
"alert on change" also makes this the **shared trigger with toasts (§7)** — the two surfaces
consume the same signal and should not be evaluated as if they were independent.

## 4. Modals, overlays, drill-down navigation, animation

| Crate | Ver | Downloads | Self-description | Tier |
|---|---|---|---|---|
| `tui-widgets` | 0.7.10 | 66.2K / 13.9K | "A collection of useful widgets" — ratatui org umbrella; **contains `tui-popup`, `tui-scrollview`, `tui-big-text`, `tui-prompts`** | T1 |
| `ratkit` | 0.2.18 | 677 / 294 | "resizable splits, tree views, markdown rendering, toast notifications, **dialogs**, terminal embedding" | T3 |
| `tui-lipan` | 0.1.0 | 40 | "component-based framework — reconciliation, layout engine, focus, **overlays**" | T3 |

**Correction — animation does have a candidate.** The first pass concluded "no candidate at
all" from a keyword search that returned five crates. The full enumeration contradicts it:

| Crate | Ver | Downloads | Self-description | Tier |
|---|---|---|---|---|
| **`tachyonfx`** | — | 268.0K | "A ratatui library for creating **shader-like effects** in TUIs" | T2 |
| `tui-skeleton` | 0.3.0 | 782 | "Animated skeleton loading widgets" | T3 |

A quarter-million downloads is not a fringe crate; it was invisible because "breadcrumb
navigation animation transition" is not how its author describes it. That is the general lesson
for this whole document — **keyword searches find crates that share your vocabulary, not crates
that solve your problem**, which is exactly why the evaluation runs over the ranked list rather
than over my thematic guesses.

The underlying difficulty still stands and the evaluator should confirm `tachyonfx` addresses
it: ratatui redraws a full frame each tick with no retained scene graph, so a directional slide
means owning an interpolation clock and rendering intermediate frames. An effects library that
assumes it drives the whole frame may not compose with a drill-down that has to keep the
surrounding chrome static. **Breadcrumbs themselves still have no candidate** — only the motion
half is covered.

## 5. Graph windows — sixel / kitty

| Crate | Ver | Downloads | Self-description | Tier |
|---|---|---|---|---|
| **`ratatui-image`** | 11.0.6 | 617.1K / 253.9K | "An image widget for ratatui, supporting **sixels, kitty, iterm2**, and unicode-halfblocks" | T1 |
| `viuer` | 0.11.0 | 1.1M / 166.2K | "Display images in the terminal" (not a ratatui widget) | T2 |
| `par-term-emu-core-rust` | 0.45.0 | 3.2K | VT100–VT520 + Sixel/iTerm2/Kitty graphics | T3 |

`ratatui-image` looks like a straight adoption rather than an uplift — ratatui org, version 11,
a quarter-million recent downloads, and it already covers exactly the three protocols named.
The open question is not the widget but what feeds it: rendering a graph to a raster inside the
TUI needs a layout pass first, which is a separate problem from displaying the result.

## 6. Editing — single-line and block, vim and not

| Crate | Ver | Downloads | Self-description | Tier |
|---|---|---|---|---|
| `tui-textarea` | 0.7.0 | 2.2M / 729.1K | the original; "simple yet powerful text editor widget for ratatui **and tui-rs**" | T2 |
| `ratatui-textarea` | 0.9.2 | 318.7K / 243.3K | ratatui-org fork of the above | T1 |
| `tui-textarea-2` | 0.12.1 | 85.0K / 58.4K | a second, higher-versioned fork | T2 |
| **`edtui`** | 0.11.6 | 228.2K / 95.1K | "A TUI based **vim inspired** editor" | T2 |

Three live forks of one widget is a maintenance signal the evaluation must untangle: which
tracks ratatui 0.30, which is actually maintained, and what the forks diverged over. **Pass 1
answers the first:** `tui-textarea` — the 2.2M-download original — is **DROP-VERSION**, its
current release pinned to `^0.29.0`; the ratatui-org fork `ratatui-textarea` and the
higher-versioned `tui-textarea-2` both pass the gate. So the surviving question is *why the org
forked*, not *which of three*. `edtui` is
the only one advertising vim modality, which the design needs in both editing surfaces (§3 of
VISUAL-LANGUAGE specifies insert `▏` and normal `[reverse]c` carets, so vim modality is already
a locked design commitment, not a preference).

The full enumeration adds two the keyword search missed:

| Crate | Ver | Downloads | Self-description | Tier |
|---|---|---|---|---|
| **`tui-input`** | — | 1.72M | "TUI input library supporting multiple backends" | T2 |
| **`tui-markdown`** | — | 384.7K | "converting markdown to a Ratatui `Text` value" | T2 |
| `tui-prompts` | — | 169.9K | "building interactive prompts for ratatui" (part of `tui-widgets`) | T1 |

`tui-input` at 1.7M downloads is very likely the answer for **single-line** editing, which is a
different problem from the multi-line textarea forks above and was being conflated with them.

**Markdown-aware block editing still has no direct candidate.** `tui-markdown` renders markdown
*to* ratatui text and `ratkit` claims rendering too — neither edits. The gap is real, but it is
now narrower than "no markdown support at all": rendering can be adopted and only the editing
half built.

## 7. Toasts

| Crate | Ver | Downloads | Self-description | Tier |
|---|---|---|---|---|
| `ratatui-toaster` | 0.1.4 | 6.0K / 3.4K | "extremely lightweight toast engine" | T3 |
| `ratatui-comfy-toaster` | 0.6.2 | 692 | "advanced toast engine — timed, sticky, left-click dismiss, right-click copy, presets" | T3 |
| `hjkl-holler` | 0.39.0 | 605 | "**renderer-agnostic** notification bus, severity-tagged toasts, ring-buffer history" | T3 |
| `hjkl-holler-tui` | 0.39.0 | 500 | ratatui adapter — top-right stack, severity-coloured borders | T3 |
| **`ratatui-toast`** | **0.0.0** | 19 | **Placeholder reserved by the ratatui org** | T1 |

Same first-party-reservation signal as theming. `hjkl-holler` again splits bus from renderer,
which suits a design where toasts are "not yet wired but will be" — the bus can be adopted and
fed before any rendering decision is made.

## 8. Scrolling, mouse, input, focus

| Crate | Ver | Downloads | Self-description | Tier |
|---|---|---|---|---|
| **`tui-scrollbar`** | 0.2.7 | 1.1M / 204.5K | "scrollbar widget with **fractional thumb** rendering"; part of `tui-widgets` | T1 |
| `terminput` | 0.5.15 | 223.9K / 168.8K | "TUI input parser/encoder and **abstraction over input backends**" | T2 |
| `ratatui-interact` | 0.5.3 | 33.9K / 27.5K | "Interactive components with **focus management and mouse support**" | T2 |
| `tui-framework-experiment` | 0.4.0 | 78.5K / 16.2K | "harmonious Ratatui widgets with a goal of building a proper widget framework" (joshka — a ratatui maintainer) | T2 |

`tui-scrollbar` at 1.1M downloads is effectively the default answer for scrolling. Mouse support
is the thinnest area relative to your stated interest: `ratatui-interact` is the only crate
naming focus *and* mouse together, and focus management is the harder half — it is the thing a
keyboard-first design needs anyway, with mouse falling out of it.

## 9. Other widgets worth recording

`tui-tree-widget` (939.1K) · `tui-term` pseudoterminal (1.1M) · `tui-logger` (2.2M) ·
`ansi-to-tui` (4.0M — could render `cargo pantry dump` output *inside* a TUI) ·
`throbber-widgets-tui` (785.0K) · `tui-big-text` (453.0K) · `tui-react` (397.4K) ·
`tui-widget-list` (225.0K) · `tuirealm` (214.4K, React/Elm-style framework) ·
`jkconfig` (135.9K, JSON-Schema-driven config components) · `tui-file-explorer` (7.8K) ·
`tui-splitflap` (28).

This section is illustrative only. **`crate-inventory.csv` is the authoritative list** — these
thematic tables were assembled from my guesses about vocabulary, and the `tachyonfx` correction
in §4 is the standing proof that such guesses miss things.

## 10. T4 — adjacent frameworks (record why not)

Not ratatui, so adoption means leaving the ecosystem, but each solves problems on this list and
should be examined for *ideas* even when rejected for use:

- **FrankenTUI** (`ftui-core`/`-render`/`-style`/`-layout`/`-text`/`-widgets`, ~21K each) —
  notably has a dedicated `ftui-style` for "style, theme, and color primitives" and
  `ftui-core` for "terminal **capabilities**", which is precisely the capability-probe our #249
  is missing.
- **Ansiq** (`ansiq-*`, 9 crates) — reactive primitives, framebuffer diffing, declarative
  `view!` macro.
- **r3bl_tui** (379.0K) — "React/Elm inspired, Flexbox, CSS, editor component".
- **cursive** / `cursive-tabs` (101.6K) — different retained-mode model.

`project-notes/tui-designer.md` already recorded that the framework trade is **DEAD** for the
design-tooling question. That verdict was about authoring tools, not widget libraries, so it
does not automatically carry — but the burden of proof sits with anyone proposing a move.

## 11. tui-pantry itself

`tui-pantry` 0.4.0 (470 downloads) and `tui-pantry-macros` 0.4.0, both from `taho-inc`. Our
three filed-but-unreported defects (`handover.md` §8) are against this crate. Searching
"pantry" surfaced **no alternative preview harness** — the nearest results were a recipe
manager and a shell-snippet tool. So there is no competitor to migrate to, and the three
defects are ours to either report upstream or work around.

Two neighbours by the same author as `tui-tabs` (`jharsono`) ship *as pantry-style components*
— `tui-skeleton`, `tui-splitflap` — which suggests a small convention already exists for
publishing widgets with previews attached. Worth confirming: is there a documented pattern for
shipping a widget *with* its pantry ingredients?

---

## Evaluation plan — two passes over the CSV, not over my tables

508 crates is too many for one deep evaluation each, and too few to sample. So it runs in two
passes, and the expensive pass only sees what survives the cheap one. The input is
`crate-inventory.csv` in rank order, **not** the thematic sections above — those exist to give
an evaluator context, and §4 records what happens when they are trusted as the list.

### Pass 1 — screen — **DONE 2026-07-30, see `PASS1-SCREEN.md`**

It did not need 21 agents. Most of the screen turned out to be a data join: crates.io's
reverse-dependency endpoint returns each dependent's *version requirement* plus `has_lib`,
`bin_names` and `license`, which settles DROP-VERSION and half of DROP-APP mechanically for the
whole ecosystem at once. The plan as written is preserved below; what it got wrong was the size
of the input, not the shape of the verdicts.

### Pass 1 as planned (21 agents × ~25 crates, metadata only)

Cheap and mechanical: crates.io metadata and docs.rs, no repository checkouts. Each crate gets
exactly one verdict, and the reason is recorded even for drops so the next person does not
re-litigate them:

- `DROP-APP` — an application, not a library (186 of the search hits never even depend on
  ratatui; most of those are here)
- `DROP-DOMAIN` — a library, but unrelated to any surface wqm needs
- `DROP-STALE` — unmaintained past the point of uplift (the 82 rows untouched since 2024 are
  candidates, but age alone is not the verdict — a finished small widget can be old and fine)
- `DROP-VERSION` — cannot build against **ratatui 0.30.2**. This is question zero and a hard
  gate: `tui-pantry 0.4` re-exports ratatui, so a widget on 0.29 presents an incompatible
  `Widget` trait and cannot be composed at all
- `KEEP` — plus one or more area tags from §§1–8, and a one-line reason

Batches are contiguous CSV rank ranges (1–25, 26–50, …) so coverage is provably complete and no
agent needs to know what another is doing.

### Pass 2 — deep evaluation (per area, only on `KEEP`)

Regrouped by area tag rather than by rank, so one agent sees all the competitors for a job at
once and can rank them against each other instead of judging each in isolation. Every agent
answers the same questions:

0. *(Settled in pass 1 — a crate that reaches pass 2 already compiles against 0.30.2.)*
1. Is it maintained (last release, open issues, bus factor)?
2. Does it match VISUAL-LANGUAGE r02, or does it impose a look (borders, boxes, its own
   palette) the design has already rejected?
3. Is the seam clean — can we take the model and replace the rendering?
4. Uplift cost if it is 70% right: hours, days, or a fork?
5. What does it pull in? (`wqm-tui` is a design instrument today but ships inside `wqm`.)
6. Licence.

| Agent | Scope |
|---|---|
| A | Theming (§1) — **plus the `ratatui-theme` placeholder question first** |
| B | Tabs and containers (§2) |
| C | Modals, overlays, `tui-widgets` umbrella (§4) |
| D | `ratatui-image` and the graph-render feed (§5) |
| E | The three textarea forks + `edtui` (§6) — untangle the fork situation |
| F | Toasts (§7) — plus the `ratatui-toast` placeholder question |
| G | Scrolling, input, focus, mouse (§8) |
| H | T4 frameworks (§10) — for ideas, with a default verdict of "do not adopt" |
| I | `tui-pantry` conventions and the three open defects (§11) |

### Held back from delegation — but only after pass 1 confirms it

Four surfaces look bespoke: permanent cross-tab headers, the reactive status bar, the
keymap-derived help modal, and breadcrumb navigation itself. The status bar and help modal share
a root — a **keymap SSOT**, since both derive from the same key table — and should be designed
together rather than adopted separately.

**They are held back, not written off.** The first pass of this document declared animation
bespoke and `tachyonfx` disproved it within one full enumeration. So these four are candidates
for bespoke work *pending* pass 1 finishing: a screen over all 508 rows is exactly the instrument
that would surface a crate whose author describes the same job in different words. Only after
the screen returns empty for an area is "no candidate exists" a measurement rather than a guess.

**The screen has run, and three of the four fell.** Cross-tab headers/containers
(`ratatui-zonekit`, `panes`), the reactive status bar (`hjkl-statusline-tui`, `tui_pane`) and the
keymap-derived help modal (`ratatui-which-key`) all have candidates — every one of them below the
download floor this document's CSV stops at, which is why they read as absent. **Breadcrumb
navigation is the only one that survives**, and it is now a measurement rather than a guess.

Partially covered already, so scope them as gaps rather than greenfield: motion has `tachyonfx`
(§4), markdown rendering has `tui-markdown` and only the *editing* half is missing (§6).
