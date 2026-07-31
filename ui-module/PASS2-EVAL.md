# Pass 2 — deep evaluation, by area

Started 2026-07-30, against the KEEP set in `crate-screen.csv` and the amended agent table in
`PASS1-SCREEN.md` §5. One section per area; a section is written only when that area has been
judged, so an absent section means *not yet evaluated*, never *nothing found*.

Every verdict here answers the six pass-2 questions from `CRATE-INVENTORY.md`, and question 0
(ratatui 0.30.2) is already settled for anything that reached this document.

**Status:** `RB` and `CAP` are done, and **both are now built and wired**. `B` (tabs/containers)
is done and **adopts nothing**; `ST` (statusline / keymap / which-key) is done and produces
**one recommendation for Chris's review list** (`ratatui-input-manager`) and nothing adopted.
`A` (theming, 33), `C`, `D`, `E`, `F`, `G`, `H`, `I` are **not started** — `PASS1-SCREEN.md` §5
still describes what each of them is.

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

Two areas produced **no** recommendation at all — `B` (tabs and containers) and the status-*value*
half of `ST`. That is a result, not a gap: §B and §ST say what was measured and why nothing
earned adoption.

### Not yet evaluated — the areas, and what pass 1 left pointing at each

| area | question | pass-1 leads |
|---|---|---|
| **A** theming (33) | is there a public design behind the `ratatui-theme` v0.0.0 reservation? | `ratatui-theme` (org placeholder), `karet-theme` (**does WCAG contrast checking** — the instrument defect §8.5 needs), `tui-theme-builder`, `ratatui-themekit`, `ratatui-style-presets` |
| ~~**B** tabs, containers~~ | **DONE — adopt nothing (§B).** The leads exist; none does wqm's job. `ratatui-zonekit` computes no geometry, `panes` returns ratatui's own answer plus `taffy`, `hjkl-tabs` is an editor's vocabulary. Overflow is worth writing, not depending on | — |
| ~~**ST** statusline, keymap, which-key~~ | **DONE (§ST) — the area is two problems.** The keymap half has one recommendation, **`ratatui-input-manager`** (below). The status-*value* half has no candidate and cannot have one: it is daemon-side maintained state (`CR-035`/`N48`) | — |
| **C** modals, overlays | plus the `tui-widgets` umbrella | `tui-widgets` (the org's own), `tachyonfx` (animation, 268K downloads) |
| **D** images, graph feed | how a graph window reaches the terminal | `ratatui-image`, `viuer` |
| **E** editing | untangle the fork situation: `tui-textarea` is DROP-VERSION on `^0.29` while the org fork passes | `ratatui-textarea`, `tui-textarea-2`, `edtui` (vim modality), `tui-input` (single line, 1.72M), **plus `hjkl-prompt-tui` carried over from ST** — it is the search/ex prompt bar, so it belongs with the single-line-input question |
| **F** toasts | plus the `ratatui-toast` v0.0.0 placeholder question — and **Chris's "if it changes, alert" ties this to the status area** | `ratatui-toast`, ~~`ratatui-comfy-toaster`~~ (**SA-PS:DA, not open-source — §B**). The area starts with **one** lead, and it is a v0.0.0 placeholder |
| **G** scrolling, input, focus, mouse | — | `CRATE-INVENTORY.md` §8 |
| **H** T4 frameworks | read for ideas; default verdict is do-not-adopt | `CRATE-INVENTORY.md` §10 |
| **I** `tui-pantry` itself | conventions, and the three open defects in `handover.md` §8 | — |

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
