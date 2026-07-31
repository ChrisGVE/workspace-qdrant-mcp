# Pass 2 — deep evaluation, by area

Started 2026-07-30, against the KEEP set in `crate-screen.csv` and the amended agent table in
`PASS1-SCREEN.md` §5. One section per area; a section is written only when that area has been
judged, so an absent section means *not yet evaluated*, never *nothing found*.

Every verdict here answers the six pass-2 questions from `CRATE-INVENTORY.md`, and question 0
(ratatui 0.30.2) is already settled for anything that reached this document.

**Status:** `RB` and `CAP` are done, and **both are now built and wired**. `B` (tabs/containers)
is done and **adopts nothing**; `ST` (statusline / keymap / which-key) is done and produces
**one recommendation for Chris's review list** (`ratatui-input-manager`) and nothing adopted;
`F` (toasts), `D` (images / graph feed) and `C` (modals, overlays) are done — `C` produces a
second recommendation, `tui-popup`. `G` (scrolling / input / focus / mouse) is done and adopts
nothing; `E` (editing) is done and produces a third recommendation, `tui-input`; `H` (T4
frameworks) and `I` (`tui-pantry` itself) are done and adopt nothing. **`A` (theming, 33) is the
only area left, and it is deliberately last** — it waits on the `OSC 4` decision, which is
Chris's (`handover.md` priority 1).

**§F answers the factual half of area A's first question**, since both reservations carry the
same boilerplate: there is **no public design** behind `ratatui-theme` 0.0.0 or `ratatui-toast`
0.0.0. That does not touch the `OSC 4` decision, which stays Chris's.

**Two findings from §B bind every area that follows, so read them before starting one:** question
6 (licence) is **not** answered by `crate-screen.csv` — a blank field means "not captured" and
`non-standard` has twice meant "not open-source" — and the screen's `ratatui_reqs` column both
over-counts (dev-dependencies) and under-counts (**renderer-agnostic model crates are missing
from the CSV entirely**). Both are written up at the end of §B.

> **Standing instruction (Chris, 20260730):** the crates selected here — and what each is
> *for* — get **reviewed with Chris jointly** before more are adopted. Evaluation continues
> solo and produces recommendations; adoption does not. Two crates are already in
> `wqm-tui/Cargo.toml` and are the first items on that review: `soft_ratatui` (§RB) and
> `termprofile` (§CAP).

---

## RB — render backends. **Answered: YES, with a bounded caveat. Adopted.**

The area was scoped to one question: *can a ratatui frame be rendered to a PNG
deterministically, headless, with no terminal?* Eighteen candidates; the answer needed only
one, and it was answered by building it rather than by reading about it.

**`soft_ratatui` 0.2.0 — ADOPT**, behind the new `png-capture` feature (`wqm-tui/src/capture.rs`,
`examples/frame_png`).

| question | answer |
|---|---|
| maintained | created 2025-05, last release 2026-04-22, 21K downloads |
| gate | `ratatui-core ^0.1.0`; 0.30.2 pins `ratatui-core 0.1.2` — passes |
| imposes a look | no; it is a `Backend`, not a widget set |
| seam | `SoftBackend::get_pixmap_data_as_rgba()` + width/height. PNG encoding is ours (`png` 0.17) |
| pulls in | `rustc-hash`, `embedded-graphics`, a bitmap font atlas. **No GPU, no windowing, no async** |
| licence | MIT OR Apache-2.0 |

### What it buys

It renders a ratatui frame to RGB pixels with **no terminal, no window server, no GPU and no
OS permission prompt**. That sidesteps `config.claude#45`, where the `tui-test-harness` pixel
backend is blocked on an un-granted iTerm2 Automation grant — *for our own frames*. It does
**not** fix that harness, which screenshots any application's real window; the issue stays open
on its own terms. For this crate's storyboard, our own frames are the whole job.

It also finishes what `handover.md` §1 starts. The `.mock` → `freeze` pipeline was abandoned
because it could not render bold, so weight-based verdicts could not be trusted. Measured here:
a capture of `MMMM` regular and a capture of `MMMM` bold differ byte-for-byte, and
`bold_and_regular_are_distinguishable_in_the_pixels` pins it so the defect cannot come back
quietly.

### What it costs — measured, not read

Rendering the sixteen named colours, the indexed range and `Color::Reset` and reading the
resulting pixels back:

| input | rendered | should be |
|---|---|---|
| `Color::Rgb(r,g,b)` | byte-exact | byte-exact |
| `Color::Indexed(i)` | **`Rgb(i*i, 2*i, i)`** — arithmetic on the index, no palette lookup | the xterm 256 table |
| `Color::Cyan` | `#0000ff` (blue) | the terminal's cyan |
| `Color::LightMagenta` | `#8b008b`, *darker* than `Magenta`'s `#ff00ff` | the brighter of the pair |
| `Color::Reset` as foreground | `#ccccff`, a fixed constant | the terminal's foreground |
| `Color::Reset` as background | `#050179` (navy) | the terminal's background |

`Indexed(232)` — the first rung of the greyscale ramp, `#080808` — came out `#40d0e8`, a cyan.
The formula was confirmed by three sample points per cell across `Indexed(0..=15)` and five
points on the ramp, so it is the fill and not an antialiasing artifact.

The named-16 row deserves its own note: they resolve through **X11/CSS colour names**
(`darkred`, `darkgreen`, `gold`, `darkblue`), which is a defensible choice on its own but
disagrees with the indexed path for the *same slot* — `Cyan` and `Indexed(6)` give completely
different answers.

### Consequence, and why it lands well

`Palette::Derived` emits `Color::Rgb` for every neutral, so **it is the one mode this backend
renders truthfully** — and it is the settled default (`handover.md` §7.1) and the mode the
storyboard is authored in (§7.5). `Palette::Indexed` is *entirely* indexed colour and would
render as a cyan ramp. `Palette::Theme` would paint the reserved selector hue blue, and
cyan-means-selected is the one hue rule VISUAL-LANGUAGE §3 has.

So `capture()` **forces `Derived` and restores what it found**, rather than documenting a
caution. A PNG that silently depicted the wrong palette is the exact class of artifact §1 exists
to stop producing — and the old one at least failed *visibly*, by not rendering bold.

Two consequences were fixed rather than tolerated, because both are cases of *a renderer with
no terminal cannot resolve `Reset`, so it invents something*:

- **Layer 0.** VISUAL-LANGUAGE §6 says layer 0 keeps the terminal background and is never
  repainted, which is why `tokens` has no token for it. A capture has no terminal to keep, so
  navy sat under every frame. `capture` now paints the background it was *told* about — the
  faithful reading of layer 0, not an addition to it.
- **`tokens::normal()`.** It returned `Color::Reset` in every mode; under `Derived` it now
  returns `neutral(NORMAL_RUNG)`, which is the foreground endpoint *by construction*. Same
  colour, said explicitly. It also makes this rung consistent with the other ten, which were
  already baked RGB under `Derived`. Recorded as design decision §6.7.

Verified after both: a captured `service-zone` frame against Catppuccin Mocha endpoints is
95.5% `#1e1e2e` (the real background), with `normal` at `#cdd6f4`, `muted` at `#9ea4be` and
`faint` at `#858aa2` — every one byte-exact against the interpolation `tokens::neutral` computes.

**Standing limit, stated so no one reads a capture for the wrong thing:** read a capture for
*layout, weight, glyph and neutral* fidelity. Read a real terminal for **hue** — health green
still comes out `#006400` and the selector still comes out blue, because those are named slots.

### The remaining lead, not taken today

Making hue faithful needs the terminal's *sixteen slots*, not just its background and
foreground. `OSC 4` queries them, and `src/terminal.rs` already owns the `OSC 11`/`OSC 10`
machinery to ask — the same `/dev/tty` descriptor, the same reply parser. That would make
`Palette::Derived` fully theme-faithful **and** make captured hues true, which is a bigger win
than the capture path alone. Not started; the shape is obvious and the cost is one more query
and a 16-entry table.

The other 17 RB candidates were not evaluated individually and do not need to be for this
question. Recorded so the reason is not re-litigated: `ratzilla`, `beamterm-*` and `egui_ratatui`
target browsers or GUI toolkits; `ratatui-wgpu` and `parley_ratatui` need a GPU adapter;
`native-ascii-renderer` opens a native window — all of which reintroduce exactly the host
dependency this question exists to remove.

---

## CAP — capability probing. **Adopt `termprofile`, with one carve-out.**

Two candidates in the screen; `ftui-core` is a T4 framework component, so this is really one.

**`termprofile` 0.2.4 — ADOPT for detection, forcing and the ratatui seam.**

| question | answer |
|---|---|
| maintained | created 2025-10, last release 2026-05-02, CI + codecov, 3.9K downloads |
| gate | `ratatui-core ^0.1` (optional) — passes |
| imposes a look | nothing to impose; it has no widgets |
| seam | every dependency is optional and default features are off |
| pulls in | `ratatui` + `convert` features → `ratatui-core`, `anstyle`, `palette` |
| licence | MIT OR Apache-2.0 |
| MSRV | **1.88.0 — exactly `src/rust`'s `rust-version`**, so it raises no floor at migration |

### Why it is the right shape and not merely adjacent

`handover.md` §12 designed the Encoding axis as *"a capability that is probed and never chosen,
except by the instrument, which must be able to force any of them."* `TermProfile` is that
enum — TrueColor / Ansi256 / Ansi16 / NoColor / **NoTTY** — and it arrives with the forcing
already built:

- `CLICOLOR_FORCE="ansi256"` forces one specific level, which is how the instrument renders a
  degradation frame on a truecolor machine.
- `TermVars::from_source(HashMap)` reads the variables from memory instead of the environment,
  so a *test* can assert a degradation without any terminal at all — the property that makes
  the encoding ladder testable rather than merely implementable.

It also supplies a row the design did not have: **NoTTY**, where output is not a terminal and no
escape sequence should be emitted at all. That is precisely `cargo pantry dump | file`, which
this module already relies on.

The `ratatui` feature converts directly to `ratatui::style::Color`, and the README is explicit
about why that feature exists: `anstyle` has no `Color::Reset` variant. `tokens::normal()` is
`Color::Reset` under two of three palettes, so the intermediate layer would have dropped exactly
the rung we care most about.

Together this closes **workspace-qdrant-mcp#249**, which is open because the palette modes
document a degradation story that no code implements — nothing reads colour depth or `NO_COLOR`.

### The carve-out, and it matters

**Do not use `adapt_color`'s automatic quantisation for the neutral ladder.** `handover.md` §12
already measured that path and rejected it: per-rung nearest-colour quantisation of the tinted
ladder into the 256 palette loses the tint on 4 rungs, overshoots to `+40` on 4 more where the
ideal is `+26`…`+35`, and puts `faint` and `rule_frame` on the *same* index 103 — a new collision,
worse than either palette mode we already have. `adapt_color` is that exact algorithm, so
adopting it wholesale would re-import a defect we have already paid to discover.

The crate anticipates the objection and provides the answer: `ProfileColor::new(rgb, profile)
.ansi_256(240).ansi_16(AnsiColor::White)` lets **us** supply the per-level value. That is §12's
own rule — *the encoder must choose a family once, never per rung* — with `termprofile` deciding
which authored ladder applies rather than deriving one.

So: adopt it for **which encoding are we in**, and keep authorship of **what each rung is in that
encoding**.

### Built 20260730 — `src/encoding.rs`, and the rule turned out to be one comparison

The Encoding axis is now real, which closes the implemented half of **workspace-qdrant-mcp#249**.
The shape it settled into is smaller than the design anticipated:

- `Family` (`None` < `Slots` < `Ramp` < `Rgb`) is *how* a colour is emitted, and it is **ordered**.
- `Palette` (the Source) says which family it *wants*; `Encoding` (the capability) says which it
  *permits*.
- `tokens::family()` is `min` of the two. That single comparison is the whole Source × Encoding
  split, and it makes §12's rule — *the encoder chooses a family once, never per rung* —
  structural rather than a discipline: there is no code path that could quantise per rung,
  because a rung is always recomputed from its own percentage in whichever family won.

Degradation therefore **drops to the lower ladder's own values** rather than approximating the
one above. `tokens::degrading_picks_the_lower_ladder_rather_than_quantising_the_higher_one`
pins it by asserting that `Derived` under `Ansi256` is byte-identical to `Indexed`, and that
`faint` and `rule_frame` do not collide — the specific defect the rejected `adapt_color` path
produced.

**Cost: zero transitive dependencies.** Every one of `termprofile`'s seven dependencies is
optional and no features are enabled, because detection needs none of them.

**Three rows the design did not have, now rendered rather than argued about.** `Palette
Reference` gained `Encoding: ANSI 256`, `Encoding: ANSI 16` and `Encoding: No Color`, all with
`Derived` as the source, so what a user on a weaker terminal receives of the authored frames is
a frame rather than a claim. Measured from those dumps: ANSI 16 collapses **6 of the 11 rungs**,
which is exactly the count §12 predicted from a different direction.

#### Two findings from wiring it

1. **`CLICOLOR_FORCE` outranks `NO_COLOR` in `termprofile`, and that is user-visible.** The
   two standards contradict each other — no-color.org says colour is disabled, bixense's
   CLICOLOR spec says forced colour applies "no matter what" — and the crate resolves it in
   favour of forcing. This was found by a test asserting the opposite and failing. It is now
   pinned (`forcing_outranks_no_color_and_that_is_measured_not_assumed`) rather than worked
   around: the precedence is inherited behaviour, and if upstream changes it, we should notice
   in a test rather than in a frame. **Method note, again:** the assertion was written from the
   *specs* and the specs disagree; only running it produced the answer.
2. **The tab bar carries an alarm in hue alone.** r02 §3 asks for a structural signature first
   with colour reserved, and `Health::glyph` is that signature everywhere else — the tab bar has
   no glyph, so under `No Color` an alarming tab renders identically to a calm one
   (`CLICOLOR_FORCE=no_color cargo pantry dump "Tab Bar"`, *Alarm, Unselected* vs *Default*).
   The gap is not new; the encoding axis only made it observable. Recorded as an open decision
   in `handover.md` §7 — adding a glyph to a tab label is a visual-language change.

One thing that had to be forced rather than documented: `capture()` now forces
`Encoding::TrueColor` alongside `Palette::Derived`. Because a rung is emitted in the *lesser* of
the two, a process that had probed a 16-colour terminal would have degraded a capture's ladder
back onto the slots `soft_ratatui` mis-resolves — and forcing the palette alone looks correct in
every test that never touches the encoding.

---

## B — tabs and containers. **Answered: adopt nothing. Two of the three leads are the wrong kind of thing, and the third is a 60-line function.**

Ten `TB`-tagged KEEPs plus the three crates `PASS1-SCREEN.md` §3 named when it refuted the
"expect bespoke" finding. The refutation was right that candidates *exist*; measuring them says
none of them does the job wqm-tui has.

Everything below was produced by building against the crates, not by reading them. Two throwaway
binaries: one that registers zones and resolves layouts, one that renders each tab candidate into
a `Buffer` and prints every glyph it emitted.

### The container half — `ratatui-zonekit`, `panes`/`panes-ratatui`, `ratatui-hypertile`

**`ratatui-zonekit` 0.1.1 — REJECT. It shares the word "zone" and solves a different problem.**

Pass 1 flagged it as *"named zones, plugin-owned panes — the zone model VISUAL-LANGUAGE §6
describes"*. Measured, the zone model computes no geometry at all:

```
[zonekit] granted name=wqm.search   hint=Tab       area=Rect { x: 0, y: 0, width: 0, height: 0 }
[zonekit] granted name=wqm.status   hint=StatusBar area=Rect { x: 0, y: 0, width: 0, height: 0 }
[zonekit] after host update_area: wqm.search   -> Rect { x: 0, y: 0, width: 80, height: 1 }
[zonekit] after host update_area: wqm.status   -> Rect { x: 0, y: 23, width: 80, height: 1 }
```

A granted zone's area is `Rect::ZERO` and stays there until the **host** calls `update_area`. The
crate is a *plugin-ownership registry* — `ZonePlugin`, `ZoneRegistry`, `ZoneRequest`, a hint the
host may ignore — and the layout it appears to own is the caller's to compute. That is a sound
design for a host application accepting third-party plugins. `wqm-tui` has no plugins and one
author, so the whole apparatus is overhead around a `Layout::vertical` call we already write.

This is `PASS1-SCREEN.md` §6's lesson landing on §6's own remedy: **a keyword search finds crates
that share your vocabulary, not crates that solve your problem** — and re-reading the document
against its own data caught a *missing* crate, while this one was a *present* crate that the
vocabulary flattered. The two failure modes are symmetrical and the second needs a different
instrument: build against it.

Two smaller measurements, recorded because they cost time: the modules docs.rs lists
(`ratatui_zonekit::zone::ZoneSpec`) are **private** and re-exported at the root, so every doc path
is wrong as an import; and there is no `ZoneRequest::status_bar` constructor although
`ZoneHint::StatusBar` exists — the ctor set is `tab`/`sidebar`/`overlay` only.

**`panes` 0.21 + `panes-ratatui` 0.5.2 — REJECT for this crate. Correct, clean, and it returns
the answer we already have.**

The seam is as good as it looks: `panes-ratatui` converts a resolved layout to
`ratatui::layout::Rect` and does nothing else — no widget, no style, no look imposed. On wqm's
own frame it agrees with ratatui exactly:

```
[panes]   tabs -> Rect { x: 0, y: 0, width: 80, height: 1 }
[panes]   body -> Rect { x: 0, y: 1, width: 80, height: 22 }
[panes] status -> Rect { x: 0, y: 23, width: 80, height: 1 }
[panes] ratatui Layout -> [same three rects]
[panes] identical to ratatui Layout for this frame: true
```

The cost is `taffy` 0.12 plus `arrayvec`, `grid`, `slotmap`, `rustc-hash` and `thiserror` — a
CSS flexbox/grid solver, carried for a result ratatui's own solver produced. What would justify
it is a layout wqm-tui cannot express: grids, frame-to-frame diffing, viewport caching, all of
which `panes` has. **Measured against our own source: `wqm-tui` calls `Layout::` exactly once, in
`examples/frame_png`.** A widget crate leaves layout to its host, so there is no layout problem
here to solve yet. Re-open this if the shell in `src/rust/bins/wqm` grows one.

**`ratatui-hypertile` 0.4.1 — REJECT, recorded so it is not re-examined.** A binary-space-partition
tiling engine: user-driven splits, i3-style. wqm's frame is authored, not tiled.

**`tui_pane` 0.5.0, `tuicore` 0.12.0, `turtletap` 0.3.0 — REJECT on dependency weight; each is an
application wearing a library's name.** `tui_pane` pulls 15 dependencies including `tokio`,
`sysinfo`, `tracing-subscriber`, `dark-light` and two `objc2-*` crates — it probes the OS
appearance, which this crate already does for itself via `OSC 11`. `tuicore` pulls `reqwest` and
`rig` (an LLM framework). `turtletap` is a terminal *shell* hosting surfaces. None is admissible
in something that ships inside `wqm`.

### The tab half — measured by what each one draws

VISUAL-LANGUAGE r02 §6 rejects frame borders: zones are divided by rules, boxes are for modals.
So the test is not the README — it is rendering each candidate at wqm's own tab set and printing
the box-drawing characters it emitted.

```
--- tui-tabs TabNav (default) ---
|╭────────────╮╭─────────────────╮╭───────────────╮|
|│   Search   ││   Collections   ││ ▸ Libraries   │|
|┴────────────┴┴─────────────────┴╯               ╰─|
box-drawing glyphs emitted: {'─', '│', '┴', '╭', '╮', '╯', '╰'}

--- ratatui-tabs TabsBar (default) ---
| Search │ Collections │ Libraries │ Rules|
box-drawing glyphs emitted: {'│'}

--- hjkl-tabs-tui build_line() ---
| Search │ Collections │ Libraries │ Rules|
box-drawing glyphs emitted: {'│'}
build_line -> 11 spans: [" ", "Search ", "│", " ", "Collections ", "│", …]
```

**`tui-tabs` 0.1.1 — REJECT, and now measured rather than inferred.** The inventory guessed from
its description that rounded boxes would conflict with §6. They do, across three rows.

**`ratatui-tabs` 0.2.0 — REJECT on licence.** LGPL-3.0-or-later. Its output is otherwise the right
shape, so the licence is the whole verdict; see the licence finding below.

**`ratatui-comfy-tabs` 0.5.12 — REJECT on licence, and this one is not a nuance.** Its
`license-file` is a bespoke *SA-PS:DA* licence whose own text says **"SA-PS:DA is not
open-source"** and **"Commercial use requires a separate license from ComfyHome™"**. Inadmissible
in `wqm` on any terms we could offer.

**`hjkl-tabs` 0.39.1 + `hjkl-tabs-tui` — REJECT, but it is the closest thing here and the reason
is worth stating.** The model crate is the shape this project likes: **zero dependencies**, no
ratatui type in its surface (icon colours are raw `(u8,u8,u8)` triples), and the adapter's
`build_line()` hands back a `Line` of eleven separate spans, so nothing is style-locked. Two
things sink it.

*Its vocabulary is an editor's, not wqm's.* `Tab` carries `dirty`, `icon` and `icon_color`, and
`display_label()` prepends `●` for a dirty buffer. wqm's tabs are a fixed set of collections
whose per-tab state is an **alarm**, not unsaved changes.

*And what wqm draws is different from what it draws.* Our `TabBar` emits **no separator glyph at
all** — two spaces between tabs — plus a jump number per tab held deliberately outside the
inverted block, and selection as an inverted block rather than a highlight style
(`src/widgets/tab_bar.rs:120-149`). Adopting either crate means replacing the separator, the
numbering and the selection model; what is left is the styling of a `Line`, which is the part we
already have.

**The one thing worth taking is not a crate, it is an algorithm.** `hjkl-tabs` computes overflow,
which our `TabBar` does not:

```
[hjkl-tabs] width  80: left=false right=false visible=[Search, Collections, Libraries, Rules, Scratchpad]
[hjkl-tabs] width  40: left=true  right=true  visible=[Collections, Libraries, Rules]
[hjkl-tabs] width  24: left=true  right=true  visible=[Libraries, Rules]
[hjkl-tabs] width  12: left=true  right=true  visible=[Libraries]
```

It grows outward from the active tab, reserves two columns for the `<`/`>` indicators, and always
keeps the active tab present. That is ~60 lines over a `cell_width()` our `Tab` would have to
define differently anyway (its cell is a number span plus `" {label} "`). **Recommendation: write
it, do not depend on it** — and when the narrow-terminal frame is storyboarded, this is the
behaviour to copy.

### A finding that is not about area B — question 6 is not answered

`PASS1-SCREEN.md` §5 says *"question 6 (licence) is already answered in `crate-screen.csv`"*. It
is not, in two ways, and both were found in this area before spreading:

1. **`non-standard` in the licence column can mean "not open-source".** Two KEEPs carry it —
   `ratatui-comfy-tabs` (area B) and **`ratatui-comfy-toaster` (area F)**. Both were opened and
   both are the same SA-PS:DA licence: source-available, non-commercial redistribution only,
   commercial use by separate agreement. `ratatui-comfy-toaster` is one of only **two** toast
   leads pass 1 found, so area F starts one candidate short.
2. **33 KEEP rows have an empty licence field, and empty does not mean absent.** Spot-checked:
   `terminput` is `MIT OR Apache-2.0` and `ftui-core` is `LicenseRef-MIT-OpenAI-Anthropic-Rider`
   — both blank in the CSV. The column records what the fetch captured, not what the crate says.

Restrictive licences among the KEEPs, so no area rediscovers them: 3 × AGPL-3.0-or-later
(`hotl-theme`, `hotl-tui`, `tui-dialog`), 2 × LGPL-3.0-or-later (`ratatui-tabs`,
**`ratatui-which-key`** — an area-ST lead), 4 × GPL (`chromata`, `prismatica`, `mdfrier`,
`the-other-tui-markdown`), 2 × the SA-PS:DA pair above. **Three of pass 1's named leads, in three
different areas, have a licence problem.** Every remaining area must read the licence rather than
the column.

### And a defect in the screen's instrument, from both directions at once

The `ratatui_reqs` column is derived from crates.io's reverse-dependency join. Two measurements
show it mis-classifies exactly the crates whose architecture matches this project's:

- **It over-counts.** `panes`'s row reads `ratatui^0.30`. `panes` has **no ratatui dependency** —
  `cargo tree` gives `rustc-hash`, `taffy`, `thiserror`. Its manifest declares ratatui under
  `[dev-dependencies]`, and the column does not distinguish the two. A renderer-*agnostic* engine
  is recorded as a ratatui widget crate.
- **It under-counts, and this is the worse half.** `hjkl-tabs`, `hjkl-statusline` and
  `hjkl-which-key` — the three **model** crates — **are absent from the CSV entirely**, while
  their `-tui` adapters are present. A model crate with zero ratatui dependency cannot appear in
  a list of ratatui's dependents. The screen is structurally blind to the model/renderer split,
  which is the split `tokens.rs` uses and the one this evaluation most wants to find.

This is the **fifth** instance of the partial-set failure `PASS1-SCREEN.md` §6 records four of,
and the first that is a property of the instrument rather than of a search. Practical
consequence for the remaining areas: **when a candidate is an adapter (`*-tui`, `*-ratatui`),
look up its model crate by hand** — the screen cannot have seen it.

---

## ST — statusline, keymap, which-key. **Answered: the area is two problems, and the leads only answer one. One crate to put on Chris's review list: `ratatui-input-manager`.**

Fourteen `ST`-tagged KEEPs, plus the two model crates the screen could not see (§B). The area was
scoped as one thing — "status bar, help modals, keybindings, judged together because they share a
keymap SSOT". Measuring it splits it in half, and the halves have different answers.

- **The keymap half** — what keys exist, what they do, and the two surfaces that display them. A
  library question, and it has a good answer.
- **The status-value half** — what the status bar *says* about the daemon. `HEALTH-MONITORING.md`
  makes "rendering must never trigger work" disqualifying here, and **no crate in this area is a
  candidate for it**, because the fix is daemon-side maintained state (`CR-035`, owner `N48`). A
  TUI crate cannot supply it and should not be credited for appearing to.

Keeping them apart matters: a crate rejected for the second problem was never competing for it.

### The keymap half — one declaration, both surfaces, measured

**`ratatui-input-manager` 0.4.0 — the one candidate worth Chris's review list. Not adopted (the
standing instruction); recommended.**

`CRATE-INVENTORY.md` §3 reasoned that the status hints and the help modal derive from one key
table, so the keymap should be an SSOT designed deliberately. This crate is that design, working.
Declaring wqm's own bindings once and rendering both surfaces from the result:

```
[input-manager] KEYBINDS entries = 4
[input-manager]   Next tab               ["<Tab>"]
[input-manager]   Search this collection ["</>"]
[input-manager]   Rebuild the index      ["<C-r>"]
[input-manager]   Quit                   ["<q>", "<Esc>"]

--- HelpBar (the status-bar half) ---
|Next tab: <Tab> | Search this collection: </> | Rebuild the index: <C-r> | Qui|
box-drawing glyphs emitted: {}

--- Help (the modal half) ---
|     <Tab> Next tab|
|       </> Search this collection|
|     <C-r> Rebuild the index|
|<q>, <Esc> Quit|
box-drawing glyphs emitted: {}
```

The declaration is a `#[keymap]` attribute on an `impl` block, one `#[keybind(pressed(key=…))]`
per handler — **and the description is the handler's own doc comment**. There is no second table
to drift: the key, the behaviour and the help text are one statement, and `Help` and `HelpBar`
both read `Wqm::KEYBINDS`.

| question | answer |
|---|---|
| maintained | 0.4.0, last release 2026-04-04, MIT OR Apache-2.0 |
| gate | `ratatui-core` and `ratatui-widgets` are **optional** features; builds against 0.30.2 |
| imposes a look | **no — zero box-drawing glyphs by default**, measured above. `Help` accepts an optional `Block` if a modal wants one, which §6 permits |
| seam | `KEYBINDS` is a `&'static [KeyBind<B>]` of `{pressed, description}`. The widgets are conveniences over it; the table is usable without them |
| uplift | key notation is vim-ish (`<C-r>`, `<Esc>`) and comes from a `Display` impl, so our own formatting is a function away, not a fork |
| pulls in | the proc-macro derive, plus `itertools` and `crossterm` under the features we would use — and `crossterm` is already ours |

Two caveats worth stating before anyone treats this as settled. It is generic over an input
**`Backend`** (crossterm / termion / termwiz) rather than over ratatui, so the generic parameter
propagates into any type holding a keymap. And `KEYBINDS` is an associated **const**, so bindings
are compile-time; a user-rebindable keymap would need the `DynKeyMap` path or a different design.
Neither is a problem for the storyboard; both are the kind of thing to notice before it is load-bearing.

**`ratatui-command-palette` 0.1.0 — not needed yet, but it is the same shape and confirms it.**
`PaletteState` is a pure state machine over `&[ActionSpec]`, with `view()` returning a
`PaletteView` and no rendering of its own. A *third* surface derived from one action table. wqm's
storyboard has no command palette, so there is nothing to adopt — the value is the corroboration:
the SSOT that §3 reasoned toward is how this corner of the ecosystem is already built.

### The hjkl family — excellent code, and it carries a theme model that contradicts ours

`hjkl-statusline` and `hjkl-which-key` are the model crates §B found missing from the CSV. Both
are worth reading; neither is adoptable, and the reason is the same one both times.

**`hjkl-which-key` 0.39.1 — REJECT, but one function in it is the best statement of our own
criterion that this evaluation has found.**

```rust
/// Pure function: should the which-key popup be shown right now?
/// Extracted here so tests can drive `now` without mocking `Instant::now()`.
pub fn should_show(pending_at: Option<Instant>, delay: Duration, enabled: bool, now: Instant) -> bool
```

`now` is a **parameter**. The render path reads no clock; the caller supplies time. That is
`HEALTH-MONITORING.md`'s "rendering must never trigger work" honoured at the smallest possible
scale, by an author who reached it from testability rather than from a health brief. **Worth
adopting as a rule for our own widgets whether or not the crate ever is.**

What sinks it: `entries_for` is generic over the action type but bound to `hjkl_keymap::Keymap`
and `hjkl_vim::Mode`. Adopting the popup means adopting their keymap *and* a vim mode model, and
wqm-tui is not modal.

**`hjkl-statusline` 0.39.1 — REJECT, and this is the sharpest finding in the area.**

`Bar::layout(width) -> Vec<Segment>` is a clean width-driven packer: left segments, right
segments, a computed spacer, and `…`-truncation of the last left segment when it does not fit.
Widths are `UnicodeWidthStr`, so CJK and emoji stay aligned. Genuinely reusable arithmetic.

But the model is **renderer-agnostic and not theme-agnostic**. `Segment::Text` carries a `Style`,
and `pub use hjkl_theme::StyleSpec as Style` — so the "renderer-agnostic" model imports their
theme. And that theme is 18 absolute-RGB slots with a Nord default:

```rust
diag_error_fg: Color::rgb(0xff, 0x00, 0x00),   // ANSI Red
diag_warning_fg: Color::rgb(0xff, 0xc0, 0x00), // ANSI Yellow
// "Standard ANSI-named colors: adapt to the terminal palette."
```

The comment is false in the way this project spent a whole session measuring: an absolute RGB
triple cannot adapt to the terminal palette — that is precisely why `Palette::Theme` and
`Palette::Indexed` were both rejected and `Palette::Derived` reads `OSC 11`/`OSC 10` instead
(`handover.md` §7.1). Adopting this model means importing a colour authority that contradicts
`tokens.rs` on the one decision this crate has most carefully settled.

**The general lesson, and it applies to every area still open:** "renderer-agnostic" is a claim
about the *drawing* API. It says nothing about whether the crate also brings a **colour**
authority — and for this project, colour is the settled part. Check for a theme type in the model
crate's public surface, not just for the absence of a ratatui type.

**`hjkl-prompt-tui` — not evaluated.** It is the ex/search prompt bar, which belongs with the
search-input surface rather than with the status bar; it drags in `hjkl-form` and `hjkl-prompt`.
Carry it to area **E** (editing), where the single-line-input question already lives.

### Rejected without a deep read, and why — so nobody re-opens them

| crate | why |
|---|---|
| `glues-core` 0.8.1 | pulls **`gluesql` and `reqwest`** — an application's core with an embedded SQL engine and an HTTP client |
| `gx-tui` 0.1.0 | 10 dependencies including `syntect`, `image`, `icy_sixel` — an application, tagged `ART` for a reason |
| `term-wm-sys-ui-components` 0.8.20-**alpha** | alpha, and pulls `webbrowser`; its overlay builders mutate through `.borrow_mut()` during render |
| `monitrs-tui` 0.2.0 | 34 downloads, first published 2026-07-30, and explicitly "for monitrs" — one application's internals |
| `tui_pane` 0.5.0 | rejected in §B: 15 dependencies including `tokio`, `sysinfo` and two `objc2-*` crates |
| `tui-logger` 0.18.3 | 2.19M downloads and the obvious choice **if** the TUI ever surfaces daemon logs — but it is a global singleton (`lazy_static` log buffer, `.lock()` taken inside `render`). Re-open only when a log pane is actually storyboarded; do not adopt speculatively |

**`scarab-nav-protocol` 0.1.0 — REJECT, and pass 1's reason for it was a vocabulary match.** It
was tagged "protocol for keyboard-driven TUI navigation — keymap-SSOT prior art". Read, it is a
**protobuf schema** (`prost`, one 122-byte `lib.rs` around generated code) by which a TUI reports
its interactive elements to a *host terminal* so the host can paint Vimium-style hints. Its own
README: *"a side-channel communication protocol where the TUI application reports its interactive
layout to the host."* Not a keymap, not an SSOT — an accessibility export. Licence is **MIT**, not
blank as the CSV records (§B).

**`async-callback-manager` 0.1.1 — REJECT, and this one matters because of what it was tagged
as.** Pass 1 recorded it as *"the watch/debounce seam `CR-035` needs"*. Measured, it is an async
**task spawner**: `AsyncCallbackManager::spawn_task`, `AsyncTask::new_future`/`new_stream`, over
`tokio`. It initiates work on demand — the exact opposite of maintained state that readers
observe without triggering a probe. Crediting it here would have inverted the brief's central
point.

### The method note this area earned

Three of pass 1's named leads — `ratatui-zonekit` (§B), `scarab-nav-protocol`, and
`async-callback-manager` — did not survive contact with their own source, and each failed the
same way: **the `reason` column is a hypothesis formed from the crate's description, and a
description names a vocabulary, not a behaviour.** That is not a defect in pass 1, which was a
screen and is allowed to be optimistic. It is what pass 2 is for, and it is the same rule §RB
arrived at from the opposite direction: *a self-description is a claim about intent; a verdict
needs a claim about output.*

One instrument note, recorded because it produced a false positive that nearly reached this page.
A mechanical scan for "does rendering trigger work" — matching clock, I/O, spawn and lock calls
inside functions whose name contains `render`/`draw` — flagged `monitrs-tui` for `Instant::now`.
The match was inside a **test** whose name happens to contain the word "renders". The scan is a
screen, not a verdict, and every hit it produced had to be read before it could be believed.

---

## F — toasts. **Answered: `hjkl-holler` is the bus to copy, and the debounce CR-035 needs cannot come from this area at all.**

Five `TO`-tagged KEEPs, one of which §B removed on licence, plus the `ratatui-toast` placeholder
and `hjkl-holler` — the **fourth** model crate missing from the CSV (§B).

### The placeholder question, answered by reading it

**`ratatui-toast` 0.0.0 — there is no public design behind the reservation.** Its entire `lib.rs`
is `#![doc = include_str!("../README.md")]`, and the README says so in as many words:

> *"This crate is a Ratatui namespace reservation for future work in the dialogs and feedback
> area. It intentionally exposes no public API yet. A future release **may** replace this
> reservation with a real implementation when the design is ready."*

No RFC, no linked issue, no timeline — it points at the main ratatui repository. So "wait for
first-party" is waiting on nothing observable. **The same is true of `ratatui-theme` 0.0.0**,
which carries the identical boilerplate for the styling area. That answers the factual half of
area **A**'s first question — *is there a public design behind the reservation?* **No** — without
touching the `OSC 4` decision that is Chris's.

This also settles `handover.md` §12's asymmetry in the theming case: the argument for waiting was
that a theme is a *format* and formats are what one waits on. That still holds in principle, but
there is no candidate format to wait for — only a reserved name.

### The criterion, and why no toast crate can satisfy it

Chris's *"if it changes, alert"* ties toasts to the status area, and `HEALTH-MONITORING.md`
property 3 requires that **a flapping probe coalesce to one event per settled change** — or the
log volume this whole exercise exists to remove returns as toasts. `hjkl-holler` looked like it
answered that, so it was measured:

```
[holler] 8x identical      active=["8x Warn \"store degraded\""]     history entries: 1
[holler] 4x flapping pair  active=[8 separate entries]               history entries: 8
```

**It collapses repetition, not flapping.** `push` merges into the last entry only when body *and*
severity match the immediately preceding one, so `degraded → healthy → degraded → healthy` yields
four toasts, not one. A settled-change detector has to sit **upstream** of any bus, daemon-side,
exactly where `CR-035` puts it. Recorded because the near-miss is the trap: a bus that dedups
repeats *looks* like it debounces, and would pass a casual review of property 3.

### `hjkl-holler` 0.39.1 — REJECT as a dependency, ADOPT as the shape

| question | answer |
|---|---|
| maintained | 0.39.1, released 2026-07-30, MIT |
| dependencies | **zero** |
| clock | `is_expired(now)`, `is_fading(now)`, `active(now)` — **the caller owns the clock on every read**. `push` reads it once, which is correct: pushing is an event, reading is a render |
| model | severity-keyed default TTLs (Info 2 s / Warn 4 s / Error 6 s), a ring-buffer history with a cap, `dismiss(id)`, and a `count` on collapsed repeats |
| the adapter | `hjkl-holler-tui::render_active` draws *"a floating bordered box"* per toast and takes `&mut Frame`, so it needs a Terminal and cannot be composed into a `Buffer` — the same seam limit as `hjkl-tabs-tui::render` (§B) |

The bus is the second crate in this evaluation (after `hjkl-which-key::should_show`, §ST) whose
render path takes `now` as a parameter rather than reading the clock. Two independent authors
arriving at the same discipline is worth more than either instance: **make it a rule for our own
widgets.** The reason not to depend on it is the same as §ST's — the family's value is 60 lines of
well-judged model, and taking it drags a vocabulary and, one crate over, a colour authority.

One design question this raises for Chris, flagged not answered: **a toast is a floating box, and
r02 §6 reserves boxes for modals.** Whether a transient overlay counts as a modal for that rule is
a visual-language call, and it has to be made before any toast surface is built — it is the same
class of question as §7.6.

> **ANSWERED 20260731 (Chris) — and it did not arrive as a ruling on the rule.** He specified the
> surface: *"a small ephemeral (but long enough to read) rectangle on the lower right corner with an
> audible (configurable) sound"*. So a toast **is** a box and is **not** a modal, and
> VISUAL-LANGUAGE §6 was amended to say what a box *means* — position and lifetime separate the two
> — with §8 added for the surface. This area's verdict held under it: the toast surface was built in
> `wqm-tui/src/widgets/toast.rs` with **no crate adopted**, `hjkl-holler`'s caller-owns-the-clock
> discipline copied rather than depended on, and the flapping limit measured here pinned by a test.
> The configurable half is `UIQ-006` — the knobs are N7's, not this module's to invent.

### The rest of the area

| crate | verdict |
|---|---|
| `ratatui-toaster` 0.1.4 (6.0K downloads, Unlicense OR MIT) | the most-used option, and it is single-toast: `has_toast()`/`hide_toast()`, one `ToastEngine<A>` with an optional `tokio::sync::mpsc::Sender<A>` for actions. A severity **stack** is what the status surface implies, so this is the wrong shape before the tokio question is even reached |
| `ratatui-notifications` 0.1.0 | `tick(delta)` is caller-driven, which is right — but `render(&mut self, frame, area)` takes `&mut self` and a `&mut Frame`, so drawing mutates and needs a Terminal. Pulls `chrono` + `crossterm` + `log` |
| `tui-overlay` 0.1.2 | **carried to area C**, where it belongs: two dependencies (`ratatui-core`, `ratatui-widgets`) and one crate covering drawers, modals, popovers *and* toasts, with `resolve_rect` as pure geometry and easing/slide state the caller ticks. Judge it against the modal surface, which is storyboarded, rather than the toast surface, which is not |
| ~~`ratatui-comfy-toaster`~~ | SA-PS:DA, not open-source (§B) |

---

## D — images and the graph feed. **Answered: premature, and the framing was wrong. Facts recorded so it is not re-derived.**

Four `IM`-tagged KEEPs plus `viuer`, which is not in the CSV because it is not a ratatui widget.

**Nothing here should be adopted, because there is nothing to put it in.** `handover.md` §10's
storyboard backlog is the data-cursor row, the editing cell, the layer-1 modal, and a Views tab
composing them. No graph window is storyboarded. Evaluating image transports now would be
choosing a dependency for a surface whose shape is unknown — and §12 already refused that trade
for theming, on the same grounds.

What *is* worth recording is that the area's framing does not survive contact.
`CRATE-INVENTORY.md` §5 assumed the open question was *"rendering a graph to a raster inside the
TUI needs a layout pass first"* — i.e. graph → image → sixel. Two crates say the raster is
optional:

- **`ratatui-flow` 0.1.1** (MIT, 3 dependencies) — `NodeGraph` with `calculate()`, `positions()`,
  `split(area) -> Vec<Rect>`, `split_named()` and `hit_test(area, x, y)`. It is a **layout engine
  that yields rects**, so the caller renders its own widgets into the node boxes. It draws the
  connections in cells, with `├`/`┬` port glyphs.
- **`gen-tui` 0.2.1** (Apache-2.0) — a full text-cell graph system: `gen-sugiyama` for layered
  layout, `petgraph`, `rstar` for spatial indexing, its own edge router, viewport and cursor. It
  is also 500 KB of source across 22 files, pulls `tachyonfx`, and is explicitly "for Gen" — one
  application's internals, at 221 downloads.

So the real question for the graph surface, when it is storyboarded, is not *which image
protocol* but **cells or pixels** — and the cell answer needs no image dependency, no terminal
capability negotiation, and no interaction with the Encoding axis (§13). That is a design
question for Chris, and it should be asked before any of this area is re-opened.

`ratatui-image` 11.0.6 remains the answer for the pixel branch if it is ever chosen: 620K
downloads, MIT, sixel/kitty/iTerm2/halfblocks. Its cost is the part to remember — **9 required
dependencies** including `image`, `icy_sixel`, `rustix`, `windows` and `rand`, which is a large
addition to something that ships inside `wqm` for one optional view. `viuer` (1.09M downloads) is
not a ratatui widget and would need an adapter, so it is behind `ratatui-image` on every axis.

`tuika-mermaid` 0.1.1 renders Mermaid fenced blocks and belongs with markdown, not here; it is
bound to the `tuika` markdown crate, so it is not usable standalone.

---

## C — modals, overlays, the `tui-widgets` umbrella. **Answered: the bucket is two questions. The modal primitive has a good answer (`tui-popup`); animation is not storyboarded.**

Thirty-three `MO`-tagged KEEPs, plus `tui-overlay` carried over from §F. The tag is over-broad:
roughly a third of it is **animation** (`tachyonfx`, `animate`, `animato`, `cellophane`,
`mixed-signals`, `tui-shader`, `tui-shimmer`, `rattles`, `tui-spinner`, `throbber-widgets-tui`,
`tui-skeleton`, `tui-splitflap`), which is a different question from the one the storyboard asks.

**Only the modal half is live.** `handover.md` §10 item 3 is the **layer-1 modal**, and r02 §6
makes it the one place a box is allowed. Nothing in the backlog animates.

### The measurement that decides it

r02 §6 gives the modal two properties a popup crate must not fight: the border must be **ours** to
style or remove, and a modal **establishes a surface**, so it must paint its own background rather
than let layer 0 show through. Both were measured by pre-filling a buffer with a layer-0 fill and
reading back what survived.

```
--- tui-popup, defaults ---
|······┌Confirm──────────────┐·····|
|······│Delete 3 collections?│·····|
|······└─────────────────────┘·····|
box glyphs: {─ │ ┌ ┐ └ ┘}   bg it painted instead: {"Reset": 69}

--- tui-popup, layer-1 fill + our border ---
box glyphs: {─ │ ┌ ┐ └ ┘}   bg it painted instead: {"Rgb(49, 50, 68)": 69}

--- tui-popup, Borders::NONE ---
|·······Delete 3 collections?······|
box glyphs: {}              bg it painted instead: {"Rgb(49, 50, 68)": 21}

--- tui-overlay, centred 24×3, instant open ---
|·····                        ·····|
box glyphs: {}              bg it painted instead: {"Rgb(49, 50, 68)": 72}
```

**`tui-popup` 0.7.6 — the recommendation for the layer-1 modal, with one default to override.**

| question | answer |
|---|---|
| maintained | ratatui-org, 249K downloads, last release 2026-06-14, MIT OR Apache-2.0 |
| gate | `ratatui-core` + `ratatui-widgets`; builds against 0.30.2 |
| imposes a look | **no.** `borders`, `border_set`, `border_style`, `style` and `title` are all public settable fields; `Borders::NONE` genuinely produces a borderless surface, measured above |
| seam | it is a plain `Widget`/`StatefulWidget` over any body that is `KnownSize + Widget`. `KnownSize` is the only thing it asks of our widgets |
| pulls in | `derive-getters`, `derive_setters`, `document-features` — proc macros, no runtime weight |

**The default to override, and it is exactly the §6 property:** out of the box the popup paints
`Color::Reset` across its 69 cells — it clears to the *terminal* background, which is layer 0. A
modal that reverts to layer 0's colour is not a surface. Setting `.style(bg(layer1))` paints our
RGB across the whole box including the border row, so the fix is one call — but it must be made,
and a wrapper in `wqm-tui` should make it unforgettable rather than leaving it to each call site.
This is the same class of finding as `tokens::normal()` (§6.7): a renderer with no opinion about
the surface beneath it invents one.

**`tui-overlay` 0.1.2 — the smaller, purer primitive, and the reason to keep it in view.** Two
dependencies, and with no `Block` set it paints a bare centred rect of *our* colour and nothing
else — 72 cells, zero glyphs. That is the layer-1 surface with no widget opinion at all, plus
anchor/offset geometry (`resolve_rect`) and an `OverlayState` the caller ticks for open/close
transitions (duration defaults to zero, so it is instant unless asked otherwise). Against it: 1.8K
downloads, 0.1.2, last touched 2026-04-07, one author. Against `tui-popup`: it is not the org's.
**If the modal needs only a surface and a body, `tui-overlay` is the closer fit; if it needs a
titled box, `tui-popup` is the safer dependency.** That is a design call, and it follows from the
modal's shape, which is not yet drawn.

**`tui-widgets` 0.7.10 — the umbrella costs nothing, and this is worth knowing before area G.**
Its only required dependencies are `document-features` and `ratatui-core`; `tui-popup`,
`tui-scrollbar`, `tui-scrollview`, `tui-big-text`, `tui-prompts`, `tui-qrcode` and the rest are
**optional features**. So it is a façade, not a bundle — depending on it and enabling one feature
costs exactly what depending on that crate directly costs. Either form is defensible; the façade
has the advantage that the org version-tracks the set as a whole.

### Rejected, with reasons

| crate | verdict |
|---|---|
| `rat-popup` 3.0.2 / `rat-dialog` 2.0.2 | each drags the `rat-*` family — `rat-focus`, `rat-event`, `rat-reloc`, `rat-cursor`, and for `rat-dialog` the whole `rat-widget` umbrella. That is an application framework's focus and event model, and adopting a popup should not decide those |
| `ratada` 0.5.0 | "driver, modals, forms, pickers, theming in one toolkit" — pulls `clipboard-win`, `pulldown-cmark`, `chrono`, `nucleo-matcher`. A toolkit, and it brings a **theming** authority, which §ST says to check for |
| `tui_confirm_dialog` 0.4.1 | 28K downloads and the TUI does have confirm modals, but it pulls `regex` **and `rand`** for a dialog. Its shape is also fixed (a two-button dialog) where `tui-popup` + our own body is not |
| `hefesto-widgets` 0.7.3 | one dependency and a broad grab-bag (popups, lists, trees, inputs) at 405 downloads — the breadth is the problem: adopting it for a popup takes a position on four other surfaces |
| `hjkl-*-tui` popups (completion, hover, info, menu) | same family verdict as §B/§ST — good models, an editor's vocabulary, and the theme coupling §ST measured |
| ~~`tui-dialog`~~ | **AGPL-3.0-or-later** (§B) |

### The animation half — not evaluated, and that is the verdict

Nothing in the storyboard backlog animates, so choosing an animation library now is the trade §12
refused for theming. Two facts recorded so the area does not start from zero:

- **`tachyonfx` 0.25.1** is the one with mass — 271K downloads, `ratatui-core`, six dependencies,
  MIT. It is also the crate `PASS1-SCREEN.md` §6 uses as its standing example of a search miss, so
  it is already on the record.
- **`animato` 1.7.2** is renderer-agnostic with **one** required dependency (`animato-core`) and
  eighteen optional ones, so a TUI-only slice is cheap. Worth a look *if* animation is ever wanted.

**One sub-question inside the animation bucket is live, and it is not animation.** A search in
flight needs a *loading* state, and r02 has no vocabulary for one. `tui-skeleton` and
`tui-splitflap` both ship as `tui-pantry` components, which makes them relevant to area **I** as
much as here. Carried to **I**; the design question — what a pending result looks like — belongs
to the storyboard, not to a crate screen.

---

## G — scrolling, input, focus, mouse. **Answered: adopt nothing, because three of the four are not this crate's problem and the fourth already works.**

Seventeen `SC`-tagged KEEPs. `CRATE-INVENTORY.md` §8 framed the area around focus — *"focus
management is the harder half, and it is the thing a keyboard-first design needs anyway, with
mouse falling out of it."* That framing is right about the design and wrong about the boundary.

### Input: `wqm-tui` does not read it

Measured on our own source: **`grep -rn "KeyEvent\|MouseEvent\|crossterm" src/` returns nothing.**
The crate renders; the pantry harness and, later, the `wqm` binary read input. So `terminput`,
`ratatui-input-manager`'s event half, `focusable-derive` and `monio` are all decisions for the
binary in `src/rust/bins/wqm`, not for this crate — and taking any of them here would put an input
backend in a widget library's dependency tree.

Recorded for that later decision, because it is the cleanest thing in the area: **`terminput`
0.5.15 has exactly one required dependency (`bitflags`)**, 229K downloads, MIT OR Apache-2.0, with
each backend adapter as a separate crate (`terminput-crossterm`, `-termion`, `-termwiz`). Its
licence is **not** blank, as the CSV records (§B). It is the same model/adapter split this project
uses, done at the input layer.

### Focus: already in the visual language, and already rendered correctly

This is the finding, and it nearly went the other way. A first pass at `grep -rn focus src/`
returns four hits and reads like an absence. Reading what it returned says the opposite —
**r02 encodes focus in the neutral ladder, and `tokens.rs` documents it in place**:

```rust
/// Inactive tabs, unfocused zone bodies, timestamps, paths, key hints.
pub fn muted() -> Color { neutral(62) }
/// Body text of the focused zone; the baseline.
pub fn normal() -> Color { … }
```

and the widget API already carries it: `Collections::new(None)` is the pantry variant *"The
unfocused zone: no row carries the cursor, so nothing competes for the eye"*, against
`Collections::new(Some(0))` for the focused one. **Focus is an `Option<usize>` — the cursor
position, absent when the zone does not have focus** — plus the muted/normal split on body text.
That is a complete rendering answer for a widget crate.

What is left is focus *management*: which zone has it, how Tab moves it, what a modal does to it.
That is host state, exactly as layout was in §B, and it arrives with the shell rather than with a
widget. So:

| crate | verdict |
|---|---|
| `rat-focus` 2.1.1 | 60K downloads and a real focus model, but it pulls `rat-event` **and `ratatui-crossterm`** — adopting focus would decide the event model and the input backend at the same time. Wrong order |
| `ratatui-interact` 0.5.3 | the only crate naming focus *and* mouse together, which is why §8 singled it out. It pulls `crossterm`, `regex`, `thiserror` and `unicode-width` — again an input backend, plus a regex engine, inside a widget crate |
| `focusable-derive` 0.2.9 | a derive macro for focus; MIT OR Apache-2.0, not blank as the CSV records (§B). Nothing to adopt until there is a focus model to derive |

### Scrolling: the answer is known, and there is nothing to scroll yet

**`tui-scrollbar` 0.2.7 is the default answer** — 1.08M downloads, two dependencies
(`document-features`, `ratatui-core`), ratatui-org, MIT OR Apache-2.0, fractional thumb. §C
already established that reaching it through the `tui-widgets` façade costs the same as depending
on it directly, so that is a style choice rather than a trade-off.

It is not needed yet. `Collections` renders four rows; nothing in the storyboard backlog scrolls.
When a result list does, this is the crate — and `tui-scrollview` (410K) is its companion for a
scrollable *viewport* rather than a bar.

`tui-widget-list` 0.15.3 (227K, two ratatui deps) is the other shape: a list that owns its own
scrolling. Judge it against `Collections` when the result list is storyboarded — the question will
be whether variable-height rows are needed, since that is what it buys over a plain list.

### Mouse: the interesting idea, and it is one crate not five

Mouse needs hit-testing, and hit-testing needs to know where a widget *ended up* after render —
which ratatui does not record. Two crates solve it, and both are worth knowing:

- **`rat-reloc` 2.0.2 — standalone, one dependency (`ratatui-core`), 37K downloads.** Unusually for
  the `rat-*` family it drags nothing else: a `RelocatableState` trait plus `relocate_area` /
  `relocate_position` helpers that shift and clip stored rects after the fact. The *idea* — a
  widget's state remembers its own rect so a later mouse event can be resolved against it — is
  reusable whether or not the crate is.
- **`ratatui-sectioned-list` 0.3.0 — zero required dependencies, `ratatui` itself optional.** It
  does layout, focus, scroll *and* hit-test for sectioned variable-height lists, as a pure model.
  It is the model/renderer split done properly, which §B says the screen is structurally blind to
  — and it is here only because someone tagged it by hand. Against it: 214 downloads, Apache-2.0,
  one author. Worth a real look when the result list is designed; too early to adopt.

`tui-panel-select` 0.1.5 (mouse text selection plus clipboard, via `libc` and `base64`) and
`monio` 0.1.1 (OS-level input monitoring through `objc2`/`windows`/`x11`) are both out of scope for
a widget crate by a wide margin.

### Method note

The focus finding is the §B lesson inverted. There, a crate's *description* promised something its
behaviour did not have. Here, our own `grep` promised an absence that our own source did not have —
four hits looked like "focus is unaddressed" until the hits were read, at which point they turned
out to be the design, documented in place. **Both directions have the same remedy: read what the
instrument returned, do not summarise its shape.**

---

## E — editing. **Answered: the fork question dissolves, `tui-input` is the recommendation for the single-line case, and r02's modal caret is only half-expressible by any textarea crate.**

Thirty-two `ED`-tagged KEEPs. This area is live — the **editing cell** is storyboard item 2, and
VISUAL-LANGUAGE §3 already commits to vim modality by specifying two carets: `▏` for insert and a
reverse block for normal.

### Why the org forked — dates answer it, no archaeology needed

| crate | last release | ratatui dependency | repository |
|---|---|---|---|
| `tui-textarea` (rhysd, the 2.2M original) | **2024-10-22** | optional, `^0.29` — hence pass 1's `DROP-VERSION` | `rhysd/tui-textarea` |
| `ratatui-textarea` 0.9.2 (the org) | 2026-06-12 | `ratatui-core ^0.1.1` + `ratatui-widgets ^0.3.1` | `ratatui/ratatui-textarea` |
| `tui-textarea-2` 0.12.1 (srothgan) | 2026-07-10 | none required — backends optional, as the original | `srothgan/tui-textarea` |

The org's README says it plainly: *"This project is a Ratatui fork of tui-textarea and maintained
independently."* The original had gone ~20 months without a release and was pinned behind the 0.30
`ratatui-core`/`ratatui-widgets` split. **The fork is a maintenance rescue, not a design
disagreement** — which is why `PASS1-SCREEN.md` §5's reframing ("why did the org fork" rather than
"which of three") has a boring answer, and that is the useful outcome: there is no divergence to
adjudicate. `tui-textarea-2` is a second, independent rescue of the same crate.

One thing to notice before treating the org fork as the safe default: its built-in keymap is
**Emacs-like** (`C-n`/`C-p`/`C-f`/`C-b`, `M-f`/`M-b`, `C-a`/`C-e`, `C-k`), which is the opposite
modality from the one §3 commits to. Separable — the crate takes `Input`/`Key` values rather than
owning the event loop — but it means adopting it as-shipped would contradict the design.

### The finding that matters more than the fork: the caret

r02 §3 specifies the caret as a **glyph** in insert mode (`▏`) and a **style** in normal mode
(reverse block). Measured across both serious textarea candidates:

- `ratatui-textarea` renders the cursor as `Span::styled(" ", cursor_style)` — a styled **space**
  (`src/highlight.rs:224,259`).
- `edtui` exposes `EditorTheme::cursor_style` and `hide_cursor()`, defaulting to `bg(WHITE)
  fg(BLACK)`.
- **Neither exposes a cursor symbol.** `grep -rn "cursor_symbol\|cursor_char\|set_symbol"` across
  both crates' sources returns nothing relevant.

So a reverse block is expressible by either and **`▏` is expressible by neither**. And `edtui`'s
cursor style is not mode-dependent — nothing in its view layer branches on `EditorMode` for the
caret — so even the modal *switch* between the two carets would be ours to add.

That is a real constraint on this area, and it is the kind that only shows up by looking: any
textarea crate adopted here needs a patch or a wrapper to draw §3's insert caret, and the crates'
own theming stops at styles.

### Recommendations

**`tui-input` 0.15.3 — recommended for the single-line case, which is the one the storyboard
reaches first.** `CRATE-INVENTORY.md` §6 already suspected this and the suspicion holds:

| question | answer |
|---|---|
| maintained | 1.73M downloads, last release 2026-04-18, MIT |
| dependencies | **two, both `unicode-*`** — no ratatui, no input backend. The backends are behind features |
| imposes a look | **nothing at all — it does not render.** `Input` is a model: `value()`, `cursor()`, `visual_cursor()`, `visual_scroll(width)`, and `handle(InputRequest)` |
| seam | the caret is entirely ours, which is exactly what the `▏`-versus-block problem above requires |

It is the same shape as `tokens.rs`: a model with no widget dependency. The one caveat is that it
is *only* a model — grapheme handling, scrolling arithmetic and the request vocabulary come free,
rendering does not.

**For the multi-line cell: nothing yet, and the reason is the caret.** If a block editor is needed,
`ratatui-textarea` is the safer dependency (org-maintained, 324K downloads, four `unicode`/ratatui
deps) and `edtui` is the one that already has vim modality (`EditorMode::Normal`/`Insert`,
229K downloads) at the cost of a required `crossterm` — an input backend inside a widget crate,
which §G says belongs to the binary. Both need caret work. **Decide this when the editing cell is
drawn**, since what the frame needs will settle which compromise is cheaper.

### Rejected

| crate | verdict |
|---|---|
| `tui-textarea` 0.7.0 | unmaintained since 2024-10-22 and behind the 0.30 split. The 2.2M downloads are history, not health |
| `modalkit` 0.0.25 | the vim FSM as a library, renderer-agnostic — and **fourteen** required dependencies including `nom`, `regex`, `ropey`, `intervaltree`, `radix_trie`. An application framework |
| `rat-text` 3.1.0 | **nineteen** required dependencies: the whole `rat-*` family plus `chrono`, `pure-rust-locales`, `regex-cursor`, `ropey`. Same objection as §C/§G — adopting a text widget would decide focus, events and locales |
| `tui-prompts` 0.6.7 | part of `tui-widgets`, but it requires `crossterm` — the input-backend objection again. Revisit from the binary side |
| `hjkl-engine` / `hjkl-buffer` / `hjkl-vim` (+ `-tui` adapters) | the family verdict from §B/§ST/§C. `hjkl-prompt-tui`, carried here from §ST, lands the same way |
| `vix-editor` / `vix-editor-core` | quintuple-licensed including **GPL-2.0-only and GPL-3.0-only** (§B), and an editor *host* rather than a widget |
| `ratatui-code-editor`, `karet-editor`, `lumis`, `tuika-codeformatters`, `vimltui`, `scm-record`, `json-tree-editor`, `tui-canvas`, `ratin`, `tui-line-editor` | all whole editors, viewers or app-specific components. `tui-line-editor` deserves one line because its name suggests otherwise: 71 downloads, published from an application repo (`paperboy-tui`) |

**Markdown-aware block editing still has no candidate**, exactly as the inventory records, and
nothing in the full 32 changes that. `tui-markdown` renders markdown *to* ratatui text; nothing
edits it. The gap is real and narrow: rendering is available, only the editing half is missing.

---

## H — T4 frameworks. **Answered: do not adopt, as expected — but one idea is worth taking and one licence problem is worth knowing.**

Thirty-three `FW`-tagged KEEPs. The default verdict was decided before pass 2 started: adopting a
framework means leaving ratatui, and `project-notes/tui-designer.md` already killed the framework
trade for the design-tooling question. Nothing here disturbs that. The area's value is what it
records.

### The headline lead is retired — by our own code, not by evaluation

`PASS1-SCREEN.md` §4 singled out **FrankenTUI** because `ftui-core` advertises *"terminal
lifecycle and **capabilities**"*, which it called "precisely the capability-probe our #249 is
missing". That is no longer missing: **`src/encoding.rs` is built** (§CAP, §13), on `termprofile`
for detection only, at zero transitive cost. The lead closed while it was still on the list.

Worth stating because it is the pattern: **an evaluation running alongside implementation has to
re-check its own leads against what shipped**, or it recommends a solution to a solved problem.

### The licence problem, and it is worse than §B's

The eleven `ftui-*` crates all declare `license = "LicenseRef-MIT-OpenAI-Anthropic-Rider"` and
their README says *"MIT License (with OpenAI/Anthropic Rider) © 2026 Jeffrey Emanuel. See
`LICENSE`."*

**No `LICENSE` file ships in the package.** Checked in both `ftui-core` 0.5.0 (71 files) and
`ftui-a11y` 0.5.0 — no licence file at any path. A `LicenseRef-` SPDX expression means "the terms
are in a file"; the file is not in the artifact. So the licence terms of the whole family are
**not obtainable from what crates.io distributes**, and a rider that singles out two AI companies
by name is exactly the kind of term one would need to read before depending on it.

This sharpens §B: there the CSV had not *captured* the licence; here the crate does not *ship*
it. Both end at the same rule — **read the licence from the artifact, never from a column** —
but this instance is unresolvable without going to the upstream repository, which a dependency
audit cannot rely on.

### The one idea worth taking: `ftui-a11y`

Pass 1's own note — *"accessibility layer — nothing in our stack has one"* — is correct and the
crate shows what one looks like: an ARIA-shaped semantic tree *beside* the render tree.
`A11yRole::{Window, Dialog, Button, TextInput, List, Table, Tab, TabPanel, ProgressBar,
Separator, Group, Presentation, …}`, plus `LiveRegion::{Polite, Assertive}`, `MotionProfile` and
`ContrastProfile`.

Two of those land directly on open items here, which is why the idea is worth carrying even
though the crate is not adoptable:

- **`LiveRegion` is the structural answer to §7.6.** The tab bar's alarm is hue-only, so a
  `NO_COLOR` user gets nothing (§13, measured). A semantic announcement is a channel that does not
  depend on colour *or* on a glyph — a third option beside the two §7.6 currently offers Chris.
- **`ContrastProfile` is the second independent sighting of the missing instrument.** §14 already
  noted `karet-theme` does WCAG contrast checking, and defect §8.5 — `strong()` resolving *below*
  `normal` — is a contrast measurement nothing in this crate performs. Two crates in two areas
  have it; we have none.

### The rest, in one table

| crate | why not |
|---|---|
| `cursive` 0.21.1 (1.65M) | retained-mode, not ratatui, **last release 2024-08-03** |
| `tuirealm` 4.1.0 (215K), `tui-react` 0.24.0 (397K), `iocraft` (145K), `reratui`, `ratatui-tea`, `eye_declare`, `ratatui-kit`, `tabitha` | Elm/React/Bubble-Tea runtimes. Each would decide the app's whole control flow, which is `src/rust/bins/wqm`'s decision and not a widget crate's |
| `textual-rs` 0.3.16 | a Rust port of Textual with CSS styling. The *idea* — style sheets external to the code — is real but collides head-on with `tokens.rs` being the single colour authority (§ST) |
| `tuika` 0.6.0, `tui-lipan` 0.1.0 (MPL-2.0), `plurimus`, `bobatea`, `santui-core`, `tuicore` | small frameworks; `tuicore` was already rejected in §B for pulling `reqwest` and `rig` |
| `rich_rust` (114K), `rat-salsa` 4.0.3 | a Rich port and the `rat-*` app framework — both the whole-stack commitment this area exists to decline |
| ~~`hotl-tui`~~ | **AGPL-3.0-or-later** (§B), and its author disclaims semver |
| `vtcode-ui` 0.141.8 | tagged `ART` too — one product's design system, published. Worth *reading* for how it drew the seam, which is what the `ART` tag is for; not a candidate |

---

## I — `tui-pantry` itself. **Answered: the convention exists, ours differs from it in two ways, and the four defects have no alternative harness to escape to.**

### There is no competitor, which settles the defect question

`CRATE-INVENTORY.md` §11 recorded that searching "pantry" surfaced no alternative preview
harness. The full 5285-crate screen does not change that: `tui-pantry` 0.4.0 (472 downloads) and
`tui-pantry-macros` 0.4.0, both `taho-inc`, MIT OR Apache-2.0, last released 2026-04-15. So
`handover.md` §8's four third-party defects are **report-or-work-around**, with no migration
available — and all four are currently worked around. That is a stable position rather than a
comfortable one: the harness is a single-author crate at 472 downloads on which every storyboard
frame depends.

### The publishing convention exists, and it is not the one we use

§11 asked: *is there a documented pattern for shipping a widget with its pantry ingredients?*
Measured against `tui-skeleton` 0.3.0 and `tui-splitflap` 0.1.0 (both `jharsono`, MIT OR
Apache-2.0), the answer is yes, and it differs from ours on two points:

| | `tui-skeleton` | `wqm-tui` |
|---|---|---|
| feature name | `pantry` | `tui-pantry` |
| ingredient location | a **sibling file** per widget — `braille_bar.rs` + `braille_bar.ingredient.rs` | an inline `#[cfg(feature = "tui-pantry")] pub mod ingredient` inside the widget file |
| layout helper | `tui_pantry::layout::render_centered` | hand-placed |

Neither difference is a defect. Both are worth a decision rather than a drift:

- The **sibling-file** split keeps the widget file to the widget, which matters here because
  `tab_bar.rs` is 267 lines of which roughly 100 are ingredient boilerplate, and the project has a
  file-size discipline. It also makes the "two touches to add a widget" gotcha (§9) into three,
  so it is a trade rather than a win.
- The **feature name** is cosmetic, but a crate that ends up under `src/rust/crates/` alongside
  others gains from matching the ecosystem's spelling.
- `tui-skeleton` also ships a `use_cases.ingredient.rs` — an 18 KB pantry entry that is *only*
  use cases, separate from the per-widget variants. That is a convention idea worth stealing: the
  storyboard's composed frames are exactly that shape, and they currently have nowhere to live
  except `palette_sheet.rs`.

### The loading-state question carried from §C, and the finding that closes this pass

`tui-skeleton` is genuinely skeleton-loading widgets — pulse, sweep and shimmer placeholders for
data in flight — which is the r02 vocabulary gap §C flagged: a search in flight has no defined
appearance. Not adoptable as-is (it brings its own animation model, and animation is not
storyboarded), but its README states the design property that makes it interesting:

> *"All widgets are stateless — pass `elapsed_ms` from your event loop and the animation state is
> computed purely from the timestamp."*

**That is the third independent sighting of the same discipline in this pass**, from three
unrelated authors:

| crate | area | how it takes time |
|---|---|---|
| `hjkl-which-key` | §ST | `should_show(pending_at, delay, enabled, **now**)` |
| `hjkl-holler` | §F | `active(**now**)`, `is_expired(**now**)`, `is_fading(**now**)` |
| `tui-skeleton` | §I | animation state computed purely from a caller-supplied `elapsed_ms` |

Three crates, three authors, one rule: **a widget never reads the clock — the caller passes time
in.** `HEALTH-MONITORING.md` arrives at the same place from the opposite direction ("rendering
must never trigger work") for reasons about daemon load rather than testability. When a design
constraint and an ecosystem convention converge from unrelated motives, it is worth promoting
from an observation to a rule for our own widgets — and it is the most portable thing this whole
evaluation found.

---

## Review agenda — for the joint session with Chris (20260730)

Written so the review does not have to start by reconstructing what was picked and why. Nothing
here is new research: the adopted rows are §RB and §CAP above, and the pending rows are pass 1's
leads, which are **candidates, not verdicts** — an area with a lead is an area someone still has
to judge.

### Adopted, in `Cargo.toml` today

| crate | purpose | cost | reversibility |
|---|---|---|---|
| `soft_ratatui` 0.2.0 | render a ratatui frame to pixels headlessly, so a session with no eyes can judge a frame | `rustc-hash`, `embedded-graphics`, a bitmap font atlas, plus `png`. Behind `png-capture`, so a shipping build carries none of it | one feature and one module (`capture.rs`); the ANSI surfaces do not depend on it |
| `termprofile` 0.2.4 | probe what the output stream can emit — the Encoding axis | **nothing**: no features enabled, all its dependencies optional | detection sits behind `encoding::detect`; the `Family`/`min` rule is ours and would survive replacing it |

The two questions worth putting to them: is a **design instrument's** dependency judged by the
same standard as a shipping one (both of these end up inside `wqm` under §11), and is
"feature-gated" enough to make the first one a non-commitment.

### Recommended but NOT adopted — the standing instruction holds

| crate | purpose | cost | why it is on the list |
|---|---|---|---|
| `ratatui-input-manager` 0.4.0 | the **keymap SSOT**: one `#[keymap]` declaration whose doc comments are the descriptions, from which the status-bar hints and the help modal both render (§ST) | the proc-macro derive plus `itertools`; `crossterm` is already ours, and `ratatui-core`/`ratatui-widgets` are optional features | it answers the design question `CRATE-INVENTORY.md` §3 posed, and it draws **no box-drawing glyphs by default** — measured, not read. The question for Chris is whether a compile-time `const KEYBINDS` is the right shape given the generic `Backend` parameter it propagates |
| `tui-input` 0.15.3 | **single-line editing** — the search field and any inline edit (§E) | **two dependencies, both `unicode-*`**; no ratatui, no input backend | 1.73M downloads and it does not render at all, so r02's `▏` insert caret stays ours to draw — which matters because no textarea crate can express it |
| `tui-popup` 0.7.6 | the **layer-1 modal** — storyboard item 3, and the one surface r02 §6 allows a box (§C) | three proc-macro dependencies, no runtime weight; ratatui-org, 249K downloads | border and fill are fully ours (`Borders::NONE` works, measured), but its **default clears to `Color::Reset` — i.e. to layer 0**, so a wrapper must set the layer-1 fill rather than each call site remembering. The alternative is `tui-overlay`, purer and much smaller but a single-author 0.1.2; that choice follows from the modal's shape, which is not drawn yet |

Five areas produced **no** recommendation at all — `B` (tabs and containers), the status-*value*
half of `ST`, `D`, `F`, `G`, `H` and `I`. That is a result, not a gap: each section says what was
measured and why nothing earned adoption. Three of them (`D`, the animation third of `C`, and the
graph surface) were declined as **premature** rather than as unsuitable — nothing in
`handover.md` §10's backlog needs them, and choosing a dependency for an undrawn surface is the
trade §12 already refused for theming.

### One rule to take, independent of any adoption

Three unrelated crates in three different areas arrived at the same discipline: **a widget never
reads the clock — the caller passes time in.** `hjkl-which-key::should_show(…, now)` (§ST),
`hjkl-holler::active(now)` (§F), `tui-skeleton`'s animation computed purely from a caller-supplied
`elapsed_ms` (§I). `HEALTH-MONITORING.md` reaches the same place from an unrelated motive —
"rendering must never trigger work", argued from daemon load rather than testability. A constraint
and a convention converging from different directions is worth promoting to a rule for our own
widgets, and it costs nothing to adopt.

### Two things the review should decide that are not crates

- **Is a toast a modal?** r02 §6 reserves boxes for modals, and every toast implementation draws a
  floating box (§F). The answer gates the toast surface before any crate is chosen.
- **Is `LiveRegion` a third option for §7.6?** The tab-bar alarm is hue-only, and §7.6 currently
  offers Chris two choices — add a glyph, or accept that a `NO_COLOR` user does not get it.
  `ftui-a11y` shows a third: a semantic announcement that depends on neither colour nor glyph
  (§H).

### Not yet evaluated — the areas, and what pass 1 left pointing at each

| area | question | pass-1 leads |
|---|---|---|
| **A** theming (33) | ~~is there a public design behind the `ratatui-theme` v0.0.0 reservation?~~ **ANSWERED IN §F: no** — the reservation exposes no API and links only to the main ratatui repo. What is left is the `OSC 4` decision, which is Chris's | `ratatui-theme` (org placeholder), `karet-theme` (**does WCAG contrast checking** — the instrument defect §8.5 needs), `tui-theme-builder`, `ratatui-themekit`, `ratatui-style-presets` |
| ~~**B** tabs, containers~~ | **DONE — adopt nothing (§B).** The leads exist; none does wqm's job. `ratatui-zonekit` computes no geometry, `panes` returns ratatui's own answer plus `taffy`, `hjkl-tabs` is an editor's vocabulary. Overflow is worth writing, not depending on | — |
| ~~**ST** statusline, keymap, which-key~~ | **DONE (§ST) — the area is two problems.** The keymap half has one recommendation, **`ratatui-input-manager`** (below). The status-*value* half has no candidate and cannot have one: it is daemon-side maintained state (`CR-035`/`N48`) | — |
| ~~**C** modals, overlays~~ | **DONE (§C) — `tui-popup` recommended for the layer-1 modal**, with one default that must be overridden: it clears to `Color::Reset`, i.e. to layer 0. The animation third of the bucket is not storyboarded and was deliberately not evaluated | — |
| ~~**D** images, graph feed~~ | **DONE (§D) — premature, and the framing was wrong.** No graph window is storyboarded, and the real question is **cells or pixels**, not which image protocol: `ratatui-flow` lays a graph out in text cells and yields rects. Facts recorded for when it is storyboarded | — |
| ~~**E** editing~~ | **DONE (§E) — `tui-input` recommended for the single-line case.** The fork question dissolves: the original went 20 months without a release, so the org's fork is a maintenance rescue, not a design split. The sharper finding is that **no textarea crate can draw r02's `▏` insert caret** — they all express the cursor as a style on a space |
| ~~**F** toasts~~ | **DONE (§F) — adopt nothing.** The placeholder has no design; `hjkl-holler` is the shape to copy, not a dependency; and the coalescing `CR-035` needs **cannot come from a toast crate** — measured: it collapses repetition, not flapping |
| ~~**G** scrolling, input, focus, mouse~~ | **DONE (§G) — adopt nothing.** `wqm-tui` reads no input at all (measured), focus is already in the ladder as muted-vs-normal plus an `Option<usize>` cursor, and nothing is storyboarded that scrolls. `tui-scrollbar` is the known answer for when something does | — |
| ~~**H** T4 frameworks~~ | **DONE (§H) — do not adopt, as expected.** Its headline lead retired itself: `ftui-core`'s capability probe is what `src/encoding.rs` now is. One idea carried (`ftui-a11y`'s `LiveRegion` is a third option for §7.6) and one warning: the eleven `ftui-*` crates declare a custom licence whose file **does not ship in the package** | — |
| ~~**I** `tui-pantry` itself~~ | **DONE (§I).** No alternative harness exists in 5285 crates, so §8's defects are report-or-work-around. The publishing convention does exist and differs from ours — sibling `*.ingredient.rs` files, a `pantry` feature name, and a use-cases-only entry worth stealing | — |

**A should be read after the `OSC 4` decision** (`handover.md` priority 1): querying the
terminal's sixteen slots changes what a theming crate would have to supply, and possibly whether
one is wanted at all.

**Breadcrumbs remain the one surface with no candidate**, and after a 5285-crate screen that is
a measurement rather than a guess.

## Method note

Both verdicts above were reached by *measuring the candidate*, not by reading its description —
and in the `soft_ratatui` case the description was accurate while the behaviour was broken in a
way no description would have mentioned. `PASS1-SCREEN.md` §6 records that four times in one
document's history a partial set stood in for a complete one. This is the neighbouring failure:
**a self-description is a claim about intent, and a pass-2 verdict needs a claim about output.**
The measurement cost one throwaway binary and twenty minutes.
