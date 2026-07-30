# Pass 2 — deep evaluation, by area

Started 2026-07-30, against the KEEP set in `crate-screen.csv` and the amended agent table in
`PASS1-SCREEN.md` §5. One section per area; a section is written only when that area has been
judged, so an absent section means *not yet evaluated*, never *nothing found*.

Every verdict here answers the six pass-2 questions from `CRATE-INVENTORY.md`, and question 0
(ratatui 0.30.2) is already settled for anything that reached this document.

**Status:** `RB` and `CAP` are done. `A` (theming, 33), `B` (tabs/containers), `ST`
(statusline / keymap / which-key), `C`, `D`, `E`, `F`, `G`, `H`, `I` are **not started** —
`PASS1-SCREEN.md` §5 still describes what each of them is.

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

---

## Method note

Both verdicts above were reached by *measuring the candidate*, not by reading its description —
and in the `soft_ratatui` case the description was accurate while the behaviour was broken in a
way no description would have mentioned. `PASS1-SCREEN.md` §6 records that four times in one
document's history a partial set stood in for a complete one. This is the neighbouring failure:
**a self-description is a claim about intent, and a pass-2 verdict needs a claim about output.**
The measurement cost one throwaway binary and twenty minutes.
