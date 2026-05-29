# Handoff Spec: Admin — Behavioral Rules Management

Spec-of-record for the "Behavioral rules" section of the admin UI (served at
`http://localhost:6335/admin/`). Tokens, measurements, and states are extracted
from the source so a future contributor — or a redesign — has the ground truth.

**Stack:** vanilla JS + plain CSS custom properties, no framework, no build step
for the static layer.

**Source:**
- UI markup — `src/typescript/mcp-server/src/admin/static/index.html`
- UI logic — `src/typescript/mcp-server/src/admin/static/app.js`
- Tokens / components — `src/typescript/mcp-server/src/admin/static/style.css`
- REST backend — `src/typescript/mcp-server/src/admin/routes.ts`
- Rule storage / scope logic — `src/typescript/mcp-server/src/tools/rules*.ts`

## Overview

A CRUD panel for the daemon's behavioral rules. Operators list, add, edit, and
delete rules that get injected into every agent session's system prompt. Two
scopes: **Global** (applies to all projects) and **Project** (applies to one
registered tenant). Global + project rules are merged (global first, by priority)
at session start.

## Layout

Single-column flow inside `.container` (`max-width: 1200px`, centered,
`padding: 24px`, `gap: 16px` between cards). The section is one `.card`.

| Region | Composition |
|---|---|
| Card head | `<h2>` "Behavioral rules" + right-aligned `.dim .small` caption, `align-items: baseline`, space-between |
| Intro | `<p class="dim small">` explaining global vs project merge order |
| Controls | `.row` (`align-items: flex-end`, `gap: 8px`, `flex-wrap: wrap`): Scope `<select>` · Project `<select>` (`#rulesProjectField`, `hidden` until scope=project) · Reload button · `#rulesMeta` status text |
| List | `#rulesEmpty` (empty-state text) **or** `.data` table `#rulesTable` |
| Editor | `<form id="ruleForm">`: `<h3>` mode title · `.row` of Label / Title(`.grow`) / Priority · full-width `<textarea rows="4">` · action `.row` (submit · cancel · `#ruleFormMsg`) |

Table columns: **Label** · **Title** · **Priority** (`.num`, right-aligned mono)
· **Content** · **Action**.

## Design Tokens Used

All from `:root` in `style.css` (dark theme, `color-scheme: dark`).

| Token | Value | Usage in this section |
|---|---|---|
| `--bg` | `#0e1116` | App background |
| `--bg-2` | `#161b22` | Card background |
| `--bg-3` | `#1f242c` | Input/select/button background, table-row hover, `<code>` chips |
| `--border` | `#2a3038` | Card border, table cell bottom-border |
| `--border-2` | `#3a414b` | Input/select/button border |
| `--fg` | `#e6edf3` | Primary text, input text |
| `--fg-muted` | `#8b949e` | Labels, `.dim`, table `th`, captions |
| `--fg-dim` | `#6e7681` | Secondary sub-text |
| `--accent` / `--accent-2` | `#3fb950` / `#2da44e` | Primary button (bg `--accent-2`, hover `--accent`) |
| `--link` | `#58a6ff` | `.secondary` button text/border, input focus outline |
| `--err` | `#f85149` | `.danger` button (Delete), error text, toast error border |
| `--mono` | `ui-monospace, …Consolas, monospace` | Inputs, code, priority cell, content cell |
| Font (sans) | `-apple-system, "Segoe UI", Roboto…` | Body, `font-size: 14px`, `line-height: 1.5` |

Spacing scale is **4px-based**; the section's inline `rem` values map to it
(`0.5rem ≈ 8px`, matching the `.row` gap).

## Components

| Component | Variant / id | Props & constraints | Notes |
|---|---|---|---|
| Card | `.card` | radius `8px`, padding `18px 20px` | shared shell |
| Scope select | `#rulesScopeSelect` | options `global` (default) / `project` | inline-styled (selects aren't in the global CSS): `bg-3`, `1px var(--border-2)`, radius `6px`, padding `6px 8px` |
| Project select | `#rulesProjectSelect` | populated from registered projects (`tenantId` + `path`) | container `.field.grow` (`flex: 1 1 320px`); hidden unless scope=project |
| Button | `.secondary` (Reload, Cancel) | transparent bg, `--link` border+text | `.small` = padding `4px 10px`, 12px |
| Button | `.primary` (Add/Update) | `--accent-2` bg, white text | label swaps "Add rule" ⇄ "Update rule" |
| Button | `.danger .small` (Delete) | transparent, `--err` border+text, hover `rgba(248,81,73,.12)` | row action |
| Text input | `#ruleLabelInput` | `maxlength=15`, `required`, mono | becomes `readOnly` in edit mode |
| Text input | `#ruleTitleInput` | `maxlength=50`, optional | in `.field.grow` |
| Number input | `#rulePriorityInput` | `step=1`, optional | empty → omitted from payload |
| Textarea | `#ruleContentInput` | `rows=4`, `required`, `resize: vertical`, mono 12px | |
| Table | `.data` | full-width, collapsed borders, 13px | row hover `--bg-3`; `aria-label` updated per scope |
| Code chip | `<code>` | label cell, `--bg-3`, radius `3px` | |

### Backend contract

All under `/admin/api/rules`, Bearer auth (same token as the MCP transport).

| Method | Query / Body | Success | Failure |
|---|---|---|---|
| `GET` | `?scope=global\|project&projectId&limit` | `200 {success, rules[], message}` | `502` |
| `POST` (add) | `{label, content, scope, projectId?, title?, priority?, tags?}` | `200` | `400` validation · `409 {…, error}` duplicate |
| `PUT` (update) | same as add | `200` | `400 {…, error}` |
| `DELETE` (remove) | `{label, scope, projectId?}` | `200` | `400 {…, error}` |

`projectId` is **required** when `scope=project` (the server 400s otherwise — over
HTTP the daemon can't auto-detect the caller's cwd).

## States and Interactions

| Element | State / trigger | Behavior |
|---|---|---|
| Scope select | `change` | toggle Project field visibility, `resetRuleForm()`, `loadRules()` |
| Project select | `change` | `loadRules()` |
| Reload | `click` | `loadRules()` |
| Form | submit, not editing | `POST`; on success → toast `Added rule X`, reset, reload |
| Form | submit, editing | `PUT`; on success → toast `Updated rule X`, reset, reload |
| Submit button | in-flight | `disabled` (opacity 0.5) for request duration |
| Row · Edit | `click` | fills form, Label set `readOnly`, h3 → "Edit rule: X", submit → "Update rule", Cancel shown, focus textarea |
| Row · Delete | `click` | native `confirm()` → `DELETE` → toast `Deleted rule X`; if deleting the row being edited, reset form |
| Cancel edit | `click` | `resetRuleForm()` (clears, unlocks label, restores "Add rule") |
| Inputs/textarea | `:focus` | `outline: 2px solid --link`, `outline-offset: -1px`, border → `--link` |
| Buttons | `:hover` | `.primary`→`--accent`; `.danger`→`rgba(err,.12)`; default→`--border` |
| Buttons | `:disabled` | `opacity: 0.5`, `cursor: not-allowed` |
| Edit/Delete | rule has no `label` | button `disabled` + `title` "rule has no label — cannot edit/delete" |
| Form message | submit error | `#ruleFormMsg` text + `.error` class (red); reset to `.dim` on success/cancel |
| Toast | any action result | fixed bottom-right, auto-dismiss **3200ms**; error variant = `--err` left border |

## Responsive Behavior

The dashboard is fluid, not breakpoint-heavy. Only one media query exists globally.

| Breakpoint | Changes |
|---|---|
| Desktop / default | controls + form rows lay out horizontally; `.field.grow` flexes from `320px` basis |
| `≤ 720px` | `.grid-3` stat rows collapse to 1 col (affects other sections, not this one directly) |
| Narrow (general) | `.row { flex-wrap: wrap }` makes scope/project/reload and the label/title/priority trio wrap to new lines automatically; textarea is always `width: 100%` |

No dedicated mobile layout for the rules table — it relies on horizontal cell flow
and `word-break` on long content; on very narrow widths it can overflow horizontally.

## Edge Cases

- **Empty (global)**: `#rulesEmpty` → "No rules in this scope yet."
- **Project scope, none selected**: "Select a registered project to view its rules." (table hidden).
- **No registered projects**: project `<select>` shows single option "(no registered projects)".
- **Content overflow**: cell truncates at **120 chars** → `slice(0,118)+'…'`; full text in `title` attribute (hover tooltip only).
- **Duplicate add**: server returns `409` with `error`/`message`; UI shows it inline in `#ruleFormMsg` and as an error toast. No force-add path.
- **Label limits**: hard `maxlength=15` (label), `50` (title) at the input level.
- **Load failure**: "Failed to load: &lt;message&gt;", table hidden, meta cleared.
- **All user content is HTML-escaped** (`escapeHtml`) before injection into table rows and `title` attrs.

## Animation / Motion

Minimal by design (dashboard aesthetic).

| Element | Trigger | Animation | Duration | Easing |
|---|---|---|---|---|
| Buttons | hover | background color swap | instant (no transition declared) | — |
| Inputs | focus | outline appears | instant | — |
| Toast | show/dismiss | visibility toggle (no transition) | dismiss after 3200ms | — |

The progress-bar pulse / width transition is scoped to the registered-projects
indexing cell and does not apply here.

## Accessibility

**Implemented:**
- Native `<select>`, `<input>`, `<textarea>`, `<button>` → keyboard-operable.
- Every field has an associated `<label for>`; visible `2px --link` focus outline.
- Disabled Edit/Delete carry an explanatory `title`.
- Delete uses native `confirm()` (announced by screen readers).
- **Toast** has `role="status"` with `aria-live` toggled `polite`/`assertive`
  (assertive on error) so results are announced. *(item 1)*
- **Rules table** carries an `aria-label` updated per scope — "Global behavioral
  rules" / "Project behavioral rules for &lt;tenant&gt;". *(item 3)*
- **Form error** message uses the `.error` class (red) on failure for sufficient
  contrast, reset to `.dim` on success/cancel. *(item 5)*

**Focus order** (current DOM): Scope → Project → Reload → table Edit/Delete
buttons → Label → Title → Priority → Content → Submit → Cancel. Logical; no
`tabindex` overrides needed.

**Remaining suggestions (not yet done):**
2. **Content tooltip is `title`-only** — not reachable by keyboard. Consider an
   expandable row or `aria-describedby` for the full rule text.
4. **Edit mode change is otherwise silent** — focus moves to the textarea, but a
   visually-hidden live announcement ("Editing rule X") would help SR users.
6. Consider a `transition: background .12s ease` on `button` globally if the
   redesign wants subtle motion on hover.
