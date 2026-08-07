---
name: veridyx
description: An examiner's bench where a job posting is scored and the reading returns as a tagged exhibit, carrying who examined it, when, and on what grounds.
colors:
  # --- Ground: a matte photographic bench, lit from upper-left -------------
  bench: "#1d2024"
  bench-deep: "#141619"
  bench-raised: "#262a2f"
  rule: "#3a4046"
  rule-strong: "#525a62"
  # --- Ink on the bench ---------------------------------------------------
  chalk: "#eef0ee"          # 14.28:1
  chalk-dim: "#a7aeb2"      # 7.27:1
  chalk-faint: "#8b9399"    # 5.24:1
  # --- The tag: buff stock, a component material and never the page -------
  tag: "#c9bfa6"
  tag-edge: "#a89c7e"
  tag-ink: "#22252a"        # 8.41:1 on tag
  tag-ink-dim: "#4a4436"    # 5.29:1 on tag
  tag-ink-faint: "#4f4a3b"  # 4.83:1 on tag
  tag-rule: "#5c5643"       # 4.01:1 — scale rules, centre axis, major ticks
  tag-tick: "#655e49"       # 3.53:1 — minor ticks
  tag-rail: "#a89c7e"       # the unfilled portion of a findings bar
  # --- Reserved ------------------------------------------------------------
  tape: "#c8492f"           # evidence tape
  tape-dim: "#83301f"       # 4.77:1 on tag — the flagged state, nothing else
  clear: "#35502c"          # 4.92:1 on tag — the below-threshold state
  brass: "#b08d4a"          # eyelet, focus ring
  slate-bar: "#365a6b"      # a finding pushing toward legitimate
  # --- Interaction and depth ----------------------------------------------
  bench-hover: "#2f343a"    # secondary button, hover on the bench
  tag-hover: "#d8cfb8"      # primary button, hover on buff
  error-ink: "#e8907c"      # error text on the bench, 6.0:1
  tag-divider: "#b5aa8e"    # the hairline between findings on the tag
shadows:
  # Offset plus soft blur throughout; no zero-offset halos. Alphas are documented
  # rather than tokenised because each is tuned to the material it falls on.
  inset-well: "inset 0 1px 3px rgba(0,0,0,0.42)"     # recessed inputs
  tag-lift: "0 14px 34px rgba(0,0,0,0.42), 0 2px 6px rgba(0,0,0,0.3)"
  button-lift: "0 2px 10px rgba(0,0,0,0.34)"
  eyelet: "0 1px 2px rgba(0,0,0,0.5)"
  record-recess: "rgba(0,0,0,0.16)"                   # the conditions block ground
fonts:
  display: Public Sans 800
  body: Public Sans 300/400/600
  typed: Courier Prime 400/700
seed: 87044a63
---

# Veridyx — the Evidence Tag world

Recorded from the built surface, not from intention. The world was selected by roll
(candidate 4 of 7 grounded directions, seed `87044a63`) and confirmed by the user
against three challengers and the category standard.

## Thesis

**A score is not a verdict — it is an exhibit, tagged with who examined it, when, and
on what grounds.**

This refuses the arrangement every fraud-detection demo ships: near-black ground, one
neon accent, a large score gauge, accent-coloured bars on rounded cards. The product's
argument is that its numbers are lower than the literature's *on purpose*, and a page
that looks like every other ML demo cannot make that argument.

## The world

**Ground.** A matte graphite bench under a lamp — a radial falloff from `30% 0%`, so
light lands where the work happens. Never manila. The evidence-tag world's obvious
rendition is cream card stock, which is also the rendition every model reaches for; buff
appears here only as the tag's own material and as the primary button, sitting *on* the
bench.

**The tag.** Buff stock with a cut corner (`clip-path`) and a brass eyelet, carrying a
real offset-and-blur shadow. It is the same object in both states: the empty tag keeps
the cut corner and eyelet, stepped down, so what arrives filled is recognisably what was
waiting.

**Colour discipline.** Evidence-tape red is reserved for the flagged disposition and the
operating-limit rule. Green marks below-threshold. Brass is structural (eyelet, focus
ring). Neither red nor green is ever a decorative accent, and no third hue exists.

**Type.** Public Sans — the face of official forms, chosen because the world *is*
official forms — at 800 for the wordmark and verdict, 300 for the offer line, 600/700
for labels. Courier Prime carries typed field entries: scores, limits, serials,
timestamps, magnitudes, scale legends. Monospace here is for data and measurement, which
is its legitimate use, not a costume for "technical".

## Components that carry the product

**The verdict tag.** Headline disposition, an examination record (`EXAMINED BY` model,
`AT` timestamp), a measurement scale, a readout row, and numbered grounds.

**The scale** is the piece that did not exist before. A linear 0–1 rule with the
operating limit engraved on it in tape red and the score marked as a diamond. Linear on
purpose: scores are violently skewed — a clear posting lands near 0.0006 and the limit
sits at 0.9758 — so a clear posting shows visible daylight to the limit and a flagged
one sits past it. Compressing the axis to make the mark "look interesting" would flatter
the model by hiding how far from the limit it actually sat.

**The grounds list.** Numbered findings, each with the term, its signed magnitude, its
direction in words, a chevron, and a bar diverging from a drawn centre axis. Bars are
scaled to a **fixed** ±3.5 log-odds axis printed on the tag, so two readings are
comparable; a per-verdict maximum made a −0.95 draw as long as a +3.17.

Direction is never carried by colour alone — words, glyph, and side-of-centre placement
all encode it. This is a product requirement, not a preference: direction is the core
information in an attribution.

**Conditions of examination.** The limitations, set as a quiet bordered record in
chalk-dim prose with bold lead-ins, no red, no warning icon. Rigour is this project's
argument; its caveats must not look like an error the reader has to clear.

## Rules that look like taste and are not

- **Every displayed figure reads from a committed artifact.** The manifest supplies
  threshold, precision, recall, model version and training rows; the drift figure reads
  `experiments/drift.json` and degrades to prose rather than inventing a number. A
  hardcoded `PSI 0.41` nearly shipped because the check for stray literals was watching
  the manifest and this figure came from a different file.
- **The empty tag prints no reading.** `_scale` takes `score: float | None`; passing 0.0
  drew a diamond at the clamp floor, so the empty state advertised a score it did not
  have — on a surface whose whole argument is that no number is typed by hand.
- **Icons are CSS masks, not inline SVG.** Streamlit sanitises `st.html` with DOMPurify
  and strips `<svg>`; every icon was invisible in the deployed app while rendering
  correctly in a plain-HTML harness. Emoji clear the sanitiser and are not an icon
  system. The same sanitiser strips HTML comments, so the direction contract lives in
  `serve/theme.py` as `CONTRACT` and cannot be recovered from the DOM.
- **Only `.st-key-*` and documented `data-testid` selectors.** Streamlit's hashed class
  names change on upgrade and would silently strip the design.
- **Status is a field, not a kicker.** An uppercase tracked label above a large heading
  saying the same thing is the one page scaffold that is banned outright. The
  disposition is real chain-of-custody data, so it became a labelled field in the
  readout row rather than being deleted.
- **Contrast is computed, never eyeballed.** Every ink/ground pair above carries its
  measured ratio. Seven pairs failed on first build, the worst at 2.04:1.

## Known unspent ceiling

Named so a later pass knows what was left rather than missed: the buff is a flat fill
and not a produced paper raster; the eyelet ties to nothing; the tag carries no second
register (`EXHIBIT No.`, filing rule, countersignature); motion is a single
transform-and-opacity entrance where the world owns `clip-path` reveals; and the bench's
declared light direction is not reflected in the tag's straight-down shadow.

Mobile is verified for this stylesheet at a true 390px viewport, but not for Streamlit's
own column stacking — Chrome headless clamps windows to 500px, so the 390 render runs
inside an iframe where Streamlit's websocket does not complete.
