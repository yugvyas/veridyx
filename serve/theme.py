"""The Evidence Tag visual world: tokens, CSS, and the authored components.

Everything visual lives here so the app file stays composition. The two components
that carry the product — the verdict tag and the findings list — are authored HTML
rather than `st.error` + `st.progress` + `st.dataframe`, because they are the product
and should not wear the framework's clothes.

**Selector policy.** Only `.st-key-<key>` (emitted by `st.container(key=...)`) and
documented `data-testid` values are targeted. Streamlit's hashed class names change on
every upgrade; styling against them would silently strip the design mid-submission.
"""

from __future__ import annotations

import html as _html

from veridyx.schema import Verdict

# --------------------------------------------------------------------------------
# Tokens
# --------------------------------------------------------------------------------
#
# The ground is a graphite bench, not manila. The evidence-tag world's obvious
# rendition is cream card stock, which is also the rendition every model reaches for;
# buff appears here as the tag's own material, sitting on the bench, never as the page.

BENCH = "#1d2024"          # ground: matte photographic bench
BENCH_DEEP = "#141619"     # recesses, input wells
BENCH_RAISED = "#262a2f"   # lifted surfaces
RULE = "#3a4046"           # hairline divisions
RULE_STRONG = "#525a62"

TAG = "#c9bfa6"            # buff tag stock
TAG_EDGE = "#a89c7e"       # its cut edge
TAG_INK = "#22252a"        # what is written on the tag

CHALK = "#eef0ee"          # scale rules, measurement marks, primary ink on bench
CHALK_DIM = "#a7aeb2"      # secondary ink
CHALK_FAINT = "#8b9399"    # tertiary. Measured 5.24:1 on BENCH; #7d858b was 4.36.

TAPE = "#c8492f"           # evidence tape. Reserved: the flagged state only.
TAPE_DIM = "#83301f"       # 4.77:1 on buff; #8f3423 measured 4.28, failing small text
BRASS = "#b08d4a"          # eyelet, seals, the "toward fraud" direction
SLATE = "#5b8fa8"          # the "toward legitimate" direction

FONT_SANS = "'Public Sans', 'Helvetica Neue', Helvetica, Arial, sans-serif"
# One shared axis for every findings bar, in log-odds. Contributions on this model
# reach roughly +-3.5; a fixed maximum is what makes two readings comparable.
FINDING_AXIS = 3.5

FONT_TYPED = "'Courier Prime', 'Courier New', Courier, monospace"


def _esc(value: object) -> str:
    return _html.escape(str(value), quote=True)


def _drift_note() -> str:
    """The drift figure, read from the committed artifact rather than typed.

    It came from `experiments/drift.json`, not from `deployment.json`, which is exactly
    why it nearly survived as a hardcoded "PSI 0.41" — the check for stray literals was
    looking at the manifest's numbers. Every figure on this surface traces to a
    committed run; a caveat is not exempt from that because it is prose.
    """
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "experiments" / "drift.json"
    try:
        drift = json.loads(path.read_text())
        return f"PSI {drift['psi']:.2f}, {_esc(drift['psi_verdict'])}"
    except (OSError, KeyError, ValueError):
        # Never invent the number. Say the shape of the claim without a figure.
        return "measured, and significant"


# --------------------------------------------------------------------------------
# Icons — drawn, one stroke weight, currentColor. No emoji.
# --------------------------------------------------------------------------------

# Drawn marks, one stroke weight, tinted by currentColor through a CSS mask.
#
# These were inline <svg> inside st.html and did not survive: Streamlit sanitises that
# body with DOMPurify, which dropped every icon in the deployed app while the same
# markup rendered correctly in a plain-HTML harness. Emoji would clear the sanitiser
# and are banned as an icon system, so the drawings move into the stylesheet, where
# they are CSS and not sanitised HTML.

def _mask(svg: str) -> str:
    """A stroked SVG as a url-encoded mask payload."""
    body = svg.replace('"', "'").replace("#", "%23").replace("<", "%3C").replace(">", "%3E")
    return f'url("data:image/svg+xml,{body}")'


_SVG_TAG = (
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' "
    "stroke='black' stroke-width='1.7' stroke-linecap='round' stroke-linejoin='round'>"
    "<path d='M13.4 3H20a1 1 0 0 1 1 1v6.6a1 1 0 0 1-.3.7l-9 9a1 1 0 0 1-1.4 0"
    "l-6.6-6.6a1 1 0 0 1 0-1.4l9-9a1 1 0 0 1 .7-.3z'/>"
    "<circle cx='17' cy='7' r='1.4'/></svg>"
)
_SVG_CLEAR = (
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' "
    "stroke='black' stroke-width='1.7' stroke-linecap='round' stroke-linejoin='round'>"
    "<circle cx='12' cy='12' r='9'/><path d='M8 12.5l2.6 2.6L16 9.7'/></svg>"
)
_SVG_NOTE = (
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' "
    "stroke='black' stroke-width='1.7' stroke-linecap='round' stroke-linejoin='round'>"
    "<path d='M6 3h8l4 4v14a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1z'/>"
    "<path d='M14 3v4h4M8.5 12h7M8.5 16h4.5'/></svg>"
)
_SVG_UP = (
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16' fill='none' "
    "stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>"
    "<path d='M4 10l4-4 4 4'/></svg>"
)
_SVG_DOWN = (
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16' fill='none' "
    "stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>"
    "<path d='M4 6l4 4 4-4'/></svg>"
)

_ICON_TAG = '<i class="vx-i vx-i--tag" aria-hidden="true"></i>'
_ICON_CLEAR = '<i class="vx-i vx-i--clear" aria-hidden="true"></i>'
_ICON_NOTE = '<i class="vx-i vx-i--note" aria-hidden="true"></i>'
_ICON_TOWARD_FRAUD = '<i class="vx-i vx-i--up" aria-hidden="true"></i>'
_ICON_TOWARD_LEGIT = '<i class="vx-i vx-i--down" aria-hidden="true"></i>'


# --------------------------------------------------------------------------------
# The contract, and the stylesheet
# --------------------------------------------------------------------------------

CONTRACT = """<!--
THESIS: A score is not a verdict — it is an exhibit, tagged with who examined it, when,
and on what grounds. Refuses the category default: near-black ML demo, one neon accent,
a big gauge, accent bars on rounded cards.
OWN-WORLD: A matte graphite bench. Buff tag stock as component material, never the
ground. Evidence-tape red reserved for the flagged state alone; brass and slate carry
direction; chalk-white draws every scale and rule. Public Sans (the face of official
forms) with typewriter entries in the tag's fields.
STORY: A judge understands in one line what this scores, runs an example in one click,
watches the verdict arrive as a tagged exhibit with numbered findings, and reads the
limitations as part of the record rather than as an alarm.
FIRST VIEWPORT: Name and one-line offer top-left; the posting under examination on the
bench beneath, with both worked examples as the first controls; the tag occupies the
right, printed but unfilled until a score is taken, its blank fields teaching what will
arrive; the primary action sits directly under the input.
FORM: candidate 4 of 7 (evidence exhibit tags / chain-of-custody), seed key 87044a63.
FINISH: unreviewed and undocumented is unfinished; this build ends with the finish
review, the verdict, and DESIGN.md
-->"""


def css() -> str:
    """The whole design layer. Injected once, at the top of the app."""
    return f"""{CONTRACT}
<style>
@import url('https://fonts.googleapis.com/css2?family=Public+Sans:ital,wght@0,300;0,400;0,600;0,800;1,400&family=Courier+Prime:wght@400;700&display=swap');

:root {{
  --bench: {BENCH};
  --bench-deep: {BENCH_DEEP};
  --bench-raised: {BENCH_RAISED};
  --rule: {RULE};
  --rule-strong: {RULE_STRONG};
  --tag: {TAG};
  --tag-edge: {TAG_EDGE};
  --tag-ink: {TAG_INK};
  --chalk: {CHALK};
  --chalk-dim: {CHALK_DIM};
  --chalk-faint: {CHALK_FAINT};
  --tape: {TAPE};
  --brass: {BRASS};
  --slate: {SLATE};
}}

/* ---- ground ------------------------------------------------------------- */

.stApp, [data-testid="stAppViewContainer"] {{
  background: var(--bench);
  /* A bench under a lamp: the light falls where the work happens. */
  background-image: radial-gradient(
    120% 80% at 30% 0%, #262b30 0%, {BENCH} 58%, {BENCH_DEEP} 100%);
}}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stDecoration"] {{ display: none; }}
[data-testid="stMainBlockContainer"] {{
  padding: 2.3rem 2rem 4rem;
  max-width: 1180px;
}}
html, body, .stApp {{
  font-family: {FONT_SANS};
  color: var(--chalk);
  -webkit-font-smoothing: antialiased;
}}

/* ---- icons: CSS masks, not sanitised markup ----------------------------- */

.vx-i {{
  display: inline-block; width: 1em; height: 1em; flex: none;
  background-color: currentColor; vertical-align: -0.13em;
  -webkit-mask-repeat: no-repeat; mask-repeat: no-repeat;
  -webkit-mask-position: center; mask-position: center;
  -webkit-mask-size: contain; mask-size: contain;
}}
.vx-i--tag {{ -webkit-mask-image: {_mask(_SVG_TAG)}; mask-image: {_mask(_SVG_TAG)}; }}
.vx-i--clear {{ -webkit-mask-image: {_mask(_SVG_CLEAR)}; mask-image: {_mask(_SVG_CLEAR)}; }}
.vx-i--note {{ -webkit-mask-image: {_mask(_SVG_NOTE)}; mask-image: {_mask(_SVG_NOTE)}; }}
.vx-i--up {{ -webkit-mask-image: {_mask(_SVG_UP)}; mask-image: {_mask(_SVG_UP)}; }}
.vx-i--down {{ -webkit-mask-image: {_mask(_SVG_DOWN)}; mask-image: {_mask(_SVG_DOWN)}; }}

/* ---- masthead ----------------------------------------------------------- */

.vx-masthead {{
  border-bottom: 1px solid var(--rule);
  padding-bottom: 1.05rem;
  /* 2.2rem put the primary action's top edge at ~y780 on a maximised 1440x900
     laptop, whose usable viewport is ~790px. The CTA has to clear the fold on the
     device PRODUCT.md names, not merely inside a 900px synthetic viewport. */
  margin-bottom: 1.5rem;
}}
.vx-wordmark {{
  font-size: clamp(2.6rem, 6vw, 4.1rem);
  font-weight: 800;
  letter-spacing: -0.035em;
  line-height: 0.92;
  color: var(--chalk);
  margin: 0;
}}
.vx-offer {{
  margin: 0.7rem 0 0;
  font-size: clamp(1.02rem, 1.9vw, 1.22rem);
  font-weight: 300;
  line-height: 1.5;
  color: var(--chalk-dim);
  max-width: 62ch;
}}
.vx-offer b {{ color: var(--chalk); font-weight: 600; }}
.vx-provenance {{
  display: flex; flex-wrap: wrap; gap: 0.55rem 1.5rem;
  margin-top: 0.85rem;
  font-family: {FONT_TYPED};
  font-size: 0.735rem;
  letter-spacing: 0.02em;
  color: var(--chalk-faint);
}}
.vx-provenance span {{ white-space: nowrap; }}
/* nowrap sets a floor on the container width; below the breakpoint that floor
   exceeded the viewport and pushed the entire page into horizontal scroll, so
   every line of the masthead was cut off on the right. */
@media (max-width: 720px) {{ .vx-provenance span {{ white-space: normal; }} }}
.vx-provenance b {{ color: var(--chalk-dim); font-weight: 700; }}

/* ---- field labels ------------------------------------------------------- */

.vx-fieldhead {{
  font-size: 0.68rem;
  font-weight: 800;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--chalk-faint);
  margin: 0 0 0.6rem;
  display: flex; align-items: center; gap: 0.6rem;
}}
.vx-fieldhead::after {{
  content: ""; flex: 1; height: 1px; background: var(--rule);
}}

/* ---- inputs: recessed wells in the bench -------------------------------- */

.stTextInput input, .stTextArea textarea {{
  background: var(--bench-deep) !important;
  color: var(--chalk) !important;
  border: 1px solid var(--rule) !important;
  border-radius: 2px !important;
  font-family: {FONT_SANS} !important;
  font-size: 0.94rem !important;
  box-shadow: inset 0 1px 3px rgba(0,0,0,0.42) !important;
  transition: border-color 120ms ease, box-shadow 120ms ease;
}}
.stTextArea textarea {{ line-height: 1.6 !important; }}
.stTextInput input::placeholder, .stTextArea textarea::placeholder {{
  color: {CHALK_FAINT} !important;
}}
.stTextInput input:focus, .stTextArea textarea:focus {{
  border-color: var(--brass) !important;
  box-shadow: inset 0 1px 3px rgba(0,0,0,0.42), 0 0 0 2px rgba(176,141,74,0.32) !important;
  outline: none !important;
}}
.stTextInput label, .stTextArea label, .stCheckbox label {{
  font-size: 0.73rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.1em !important;
  text-transform: uppercase;
  color: var(--chalk-faint) !important;
}}
.stCheckbox label span {{ text-transform: none; letter-spacing: 0; }}

/* ---- buttons ------------------------------------------------------------ */

.stButton button {{
  border-radius: 2px;
  border: 1px solid var(--rule-strong);
  background: var(--bench-raised);
  color: var(--chalk);
  font-family: {FONT_SANS};
  font-weight: 600;
  font-size: 0.83rem;
  letter-spacing: 0.04em;
  padding: 0.6rem 1rem;
  transition: background 120ms ease, border-color 120ms ease, transform 90ms ease;
}}
.stButton button:hover {{
  background: #2f343a; border-color: var(--brass); color: var(--chalk);
}}
.stButton button:active {{ transform: translateY(1px); }}
.stButton button:focus-visible {{
  outline: 2px solid var(--brass); outline-offset: 2px;
}}
.st-key-vx_submit .stButton button,
.st-key-vx_submit button {{
  background: var(--tag);
  border-color: var(--tag-edge);
  color: var(--tag-ink);
  font-weight: 800;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  width: 100%;
  padding: 0.82rem 1rem;
  box-shadow: 0 2px 10px rgba(0,0,0,0.34);
}}
.st-key-vx_submit .stButton button:hover,
.st-key-vx_submit button:hover {{
  background: #d8cfb8; border-color: var(--brass); color: var(--tag-ink);
}}

/* ---- states: error, disabled, busy -------------------------------------- */

.vx-error {{
  color: #e8907c; font-size: 0.85rem; line-height: 1.55; margin: 0.7rem 0 0;
  padding-left: 0.75rem; border-left: 1px solid #e8907c;
}}
.stButton button:disabled, .st-key-vx_submit button:disabled {{
  opacity: 0.5; cursor: not-allowed; border-color: var(--rule);
}}
[data-testid="stStatusWidget"] {{ filter: grayscale(1) brightness(1.4); }}

/* ---- the tag ------------------------------------------------------------ */

.vx-tag {{
  position: relative;
  background: var(--tag);
  color: var(--tag-ink);
  border-radius: 3px;
  padding: 1.5rem 1.5rem 1.25rem;
  box-shadow: 0 14px 34px rgba(0,0,0,0.42), 0 2px 6px rgba(0,0,0,0.3);
  /* the cut corner where the eyelet sits */
  clip-path: polygon(0 0, calc(100% - 26px) 0, 100% 26px, 100% 100%, 0 100%);
}}
.vx-tag::before {{
  content: ""; position: absolute; top: 11px; right: 11px;
  width: 11px; height: 11px; border-radius: 50%;
  background: var(--bench-deep);
  box-shadow: inset 0 0 0 2px {BRASS}, 0 1px 2px rgba(0,0,0,0.5);
}}
.vx-tag--blank {{
  background: transparent;
  color: var(--chalk-faint);
  border: 1px dashed var(--rule-strong);
  box-shadow: none;
  /* keeps the cut corner so the blank card is recognisably the same object that
     arrives filled in; without it the brass eyelet floated on a plain rectangle */
}}
.vx-tag--blank::before {{ box-shadow: inset 0 0 0 2px var(--rule-strong); }}
.vx-tag--blank .vx-tag-rule {{ background: var(--rule); }}
.vx-blank-lede {{
  font-size: 1.28rem; font-weight: 600; letter-spacing: -0.015em;
  line-height: 1.25; color: var(--chalk-dim); margin: 0 0 0.5rem;
}}

.vx-tag-head {{
  display: flex; justify-content: flex-end; margin-bottom: 0.9rem;
}}
/* A field value in the readout row, not a kicker above the headline. The first
   build stacked an uppercase tracked label directly over a 2.9rem heading saying the
   same thing, which is the one page-scaffold shape the floor bans outright. The
   status is real chain-of-custody data, so it became a labelled field instead of
   being deleted. */
/* A value in the readout row, sharing its ink and rhythm. Rendered in colour at
   0.82rem it out-competed a 1.28rem score inside a row whose job is presenting
   measurements — the kicker's emphasis relocated rather than dissolved. State is
   carried by the coloured mark; the words return to the row's own ink. */
.vx-tag-kind {{
  display: flex; align-items: center; gap: 0.4rem;
  font-size: 0.95rem; font-weight: 600; letter-spacing: 0.005em;
  color: var(--tag-ink);
}}
.vx-tag-kind .vx-i {{ font-size: 1.1em; flex: none; }}
.vx-readout-sm {{
  font-family: {FONT_TYPED}; font-size: 0.82rem !important; font-weight: 400 !important;
  color: #4a4436 !important;
}}
.vx-tag--flagged .vx-tag-kind .vx-i {{ color: {TAPE_DIM}; }}
.vx-readout .vx-tag-kind {{ font-family: {FONT_SANS}; }}
.vx-tag--blank .vx-tag-kind .vx-i {{ color: {CHALK_FAINT}; }}
.vx-tag--clear .vx-tag-kind .vx-i {{ color: #35502c; }}  /* 4.92:1 on buff */
.vx-tag-serial {{
  font-family: {FONT_TYPED}; font-size: 0.68rem;
  color: #4f4a3b; text-align: right; line-height: 1.55;
}}
.vx-tag-serial b {{ font-weight: 700; letter-spacing: 0.08em; }}
.vx-axis-note {{
  font-family: {FONT_TYPED}; font-size: 0.6rem; font-weight: 400;
  letter-spacing: 0.02em; text-transform: none; color: #4f4a3b; margin-left: 0.5rem;
}}

.vx-verdict {{
  font-size: clamp(2.1rem, 5vw, 2.9rem);
  font-weight: 800; letter-spacing: -0.03em; line-height: 1.05;
  margin: 0 0 0.15rem;
}}
.vx-tag--flagged .vx-verdict {{ color: {TAPE_DIM}; }}
.vx-tag--clear .vx-verdict {{ color: var(--tag-ink); }}
.vx-verdict-sub {{
  font-size: 0.86rem; line-height: 1.5; color: #4a4436; margin: 0 0 1.15rem;
  max-width: 44ch;
}}

.vx-tag-rule {{ height: 1px; background: #a99d80; margin: 1.1rem 0; }}

/* the scale: score read against an engraved limit */
.vx-scale {{ margin: 0.2rem 0 0.2rem; }}
.vx-scale-track {{
  position: relative; height: 30px; margin: 0 0 0.4rem;
  border-bottom: 1px solid #5c5643;
}}
.vx-scale-tick {{
  position: absolute; bottom: 0; width: 1px; height: 7px; background: #655e49;
}}
.vx-scale-tick--major {{ height: 12px; background: #5c5643; }}
.vx-scale-limit {{
  position: absolute; bottom: 0; width: 2px; height: 30px; background: {TAPE_DIM};
}}
.vx-scale-limit::after {{
  content: attr(data-label);
  position: absolute; bottom: 32px; right: 0; white-space: nowrap;
  font-family: {FONT_TYPED}; font-size: 0.64rem; letter-spacing: 0.04em;
  color: {TAPE_DIM};
}}
.vx-scale-mark {{
  position: absolute; bottom: -5px; width: 13px; height: 13px;
  transform: translateX(-50%) rotate(45deg);
  background: var(--tag-ink); border-radius: 1px;
}}
.vx-tag--flagged .vx-scale-mark {{ background: {TAPE_DIM}; }}
.vx-scale-legend {{
  display: flex; justify-content: space-between;
  font-family: {FONT_TYPED}; font-size: 0.66rem; color: #4f4a3b;
}}

.vx-readout {{
  display: flex; gap: 1.5rem 2rem; flex-wrap: wrap; margin-top: 0.9rem;
}}
.vx-readout div {{ min-width: 0; }}
.vx-readout dt {{
  font-size: 0.63rem; font-weight: 800; letter-spacing: 0.14em;
  text-transform: uppercase; color: #4a4436; margin: 0 0 0.2rem;
}}
.vx-readout dd {{
  font-family: {FONT_TYPED}; font-size: 1.28rem; font-weight: 700;
  color: var(--tag-ink); margin: 0; font-variant-numeric: tabular-nums;
}}

/* ---- findings ----------------------------------------------------------- */

.vx-findings {{ margin: 0; padding: 0; list-style: none; counter-reset: finding; }}
.vx-finding {{
  display: grid;
  grid-template-columns: 1.6rem minmax(0, 1fr) 4.6rem;
  align-items: center; gap: 0.75rem;
  padding: 0.5rem 0;
  border-bottom: 1px solid #b5aa8e;
}}
.vx-finding:last-child {{ border-bottom: 0; }}
.vx-finding-n {{
  font-family: {FONT_TYPED}; font-size: 0.7rem; color: #4f4a3b;
  font-variant-numeric: tabular-nums;
}}
.vx-finding-term {{
  font-size: 0.92rem; font-weight: 600; color: var(--tag-ink);
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}}
.vx-finding-dir {{
  display: block; font-size: 0.67rem; font-weight: 600; letter-spacing: 0.04em;
  color: #4a4436; margin-top: 0.08rem;
}}
.vx-finding-dir svg {{
  width: 11px; height: 11px; vertical-align: -1px; margin-right: 0.25rem;
}}
.vx-finding-mag {{
  font-family: {FONT_TYPED}; font-size: 0.82rem; font-weight: 700;
  text-align: right; font-variant-numeric: tabular-nums; color: var(--tag-ink);
}}
.vx-finding-bar {{
  grid-column: 2 / 4; height: 3px; background: #a89c7e; border-radius: 2px;
  position: relative;
}}
/* The axis the bars diverge from. Without it the rows read as bars beginning at
   arbitrary offsets rather than as signed contributions either side of zero. */
.vx-finding-bar::before {{
  content: ""; position: absolute; left: 50%; top: -4px; bottom: -4px;
  width: 1px; background: #5c5643;
}}
.vx-finding-bar i {{
  position: absolute; top: 0; bottom: 0; display: block; border-radius: 2px;
}}
.vx-finding--fraud .vx-finding-bar i {{ background: {TAPE_DIM}; left: 50%; }}
.vx-finding--legit .vx-finding-bar i {{ background: #365a6b; right: 50%; }}

/* ---- record / conditions ------------------------------------------------ */

.vx-record {{
  border: 1px solid var(--rule);
  border-radius: 2px;
  background: rgba(0,0,0,0.16);
  padding: 1.1rem 1.25rem;
}}
.vx-record-head {{
  display: flex; align-items: center; gap: 0.5rem;
  font-size: 0.68rem; font-weight: 800; letter-spacing: 0.16em;
  text-transform: uppercase; color: var(--chalk-faint); margin-bottom: 0.7rem;
}}
.vx-record-head svg {{ width: 15px; height: 15px; }}
.vx-record p {{
  margin: 0 0 0.6rem; font-size: 0.855rem; line-height: 1.62; color: var(--chalk-dim);
  max-width: 74ch;
}}
.vx-record p:last-child {{ margin-bottom: 0; }}
.vx-record b {{ color: var(--chalk); font-weight: 600; }}

/* ---- colophon ----------------------------------------------------------- */

.vx-colophon {{
  margin-top: 2.6rem; padding-top: 1.1rem; border-top: 1px solid var(--rule);
  display: flex; flex-wrap: wrap; gap: 0.4rem 1.4rem;
  font-family: {FONT_TYPED}; font-size: 0.72rem; color: var(--chalk-faint);
}}
.vx-colophon a {{ color: var(--chalk-dim); text-decoration: none;
  border-bottom: 1px solid var(--rule-strong); }}
.vx-colophon a:hover {{ color: var(--chalk); border-bottom-color: var(--brass); }}

/* ---- the one authored moment: the tag arriving --------------------------- */

@keyframes vx-tag-in {{
  from {{ opacity: 0; transform: translateY(-9px) rotate(-0.5deg); }}
  to   {{ opacity: 1; transform: none; }}
}}
.vx-tag--filled {{
  animation: vx-tag-in 460ms cubic-bezier(0.16, 0.84, 0.34, 1) both;
}}
@media (prefers-reduced-motion: reduce) {{
  .vx-tag--filled {{ animation: none; }}
  .stButton button {{ transition: none; }}
}}

/* ---- narrow ------------------------------------------------------------- */

@media (max-width: 640px) {{
  [data-testid="stMainBlockContainer"] {{ padding: 2rem 1.05rem 3.5rem; }}
  .vx-tag {{ padding: 1.2rem 1.15rem 1rem; }}
  .vx-readout {{ gap: 1.2rem; }}
  .vx-finding {{ grid-template-columns: 1.3rem minmax(0, 1fr) 3.9rem; gap: 0.55rem; }}
  .vx-provenance {{ gap: 0.35rem 1rem; font-size: 0.7rem; }}
  /* Side by side, the state label and the serial each wrapped to two or three lines
     and the tag's head became the most cramped part of its hero component. */
  .vx-tag-head {{ justify-content: flex-start; }}
  .vx-tag-serial {{ text-align: left; }}
  .vx-scale-limit::after {{ font-size: 0.6rem; }}
}}
</style>"""


# --------------------------------------------------------------------------------
# Components
# --------------------------------------------------------------------------------


def _scale(score: float | None, threshold: float, flagged: bool) -> str:
    """The score read against the operating threshold, on a linear 0-1 rule.

    Linear on purpose. Scores on this model are violently skewed — a legitimate posting
    lands near 0.0002 and the limit sits at 0.976 — so a linear rule puts a clear
    posting at the far left with visible daylight to the limit. That distance is the
    thing the old UI never showed, and compressing the axis to make the mark "look
    interesting" would flatter the model by hiding how far from the limit it actually
    sat.
    """
    # `score=None` means nothing has been examined. The first version passed 0.0 and
    # the clamp drew a diamond at 2.4%, so the empty tag printed a reading it did not
    # have — on a surface whose argument is that no number is ever typed by hand.
    mark = ""
    if score is not None:
        pos = max(2.4, min(97.6, score * 100))
        mark = f'<span class="vx-scale-mark" style="left:{pos}%"></span>' 
    limit = max(1.0, min(99.0, threshold * 100))
    ticks = "".join(
        f'<span class="vx-scale-tick{" vx-scale-tick--major" if i % 5 == 0 else ""}" '
        f'style="left:{i * 2}%"></span>'
        for i in range(51)
    )
    return f"""<div class="vx-scale">
  <div class="vx-scale-track">
    {ticks}
    <span class="vx-scale-limit" style="left:{limit}%" data-label="LIMIT {threshold:.4f}"></span>
    {mark}
  </div>
  <div class="vx-scale-legend"><span>0.0000</span><span>1.0000</span></div>
</div>"""


def _findings(verdict: Verdict) -> str:
    if not verdict.top_contributions:
        return '<p class="vx-verdict-sub">No single feature moved this score materially.</p>'

    # A per-verdict peak made bar lengths incomparable between readings: a -0.95 on
    # one tag drew exactly as long as a +3.17 on another, while the story is a visitor
    # running both examples back to back. One fixed axis, printed on the tag, so the
    # lengths mean the same thing everywhere.
    peak = FINDING_AXIS
    rows = []
    for i, c in enumerate(verdict.top_contributions, start=1):
        toward_fraud = c.value > 0
        width = max(2.0, min(50.0, abs(c.value) / peak * 50.0))
        side = "vx-finding--fraud" if toward_fraud else "vx-finding--legit"
        icon = _ICON_TOWARD_FRAUD if toward_fraud else _ICON_TOWARD_LEGIT
        word = "toward fraud" if toward_fraud else "toward legitimate"
        rows.append(
            f'<li class="vx-finding {side}">'
            f'<span class="vx-finding-n">{i:02d}</span>'
            f'<span class="vx-finding-term" title="{_esc(c.feature)}">{_esc(c.feature)}'
            f'<span class="vx-finding-dir">{icon}{word}</span></span>'
            f'<span class="vx-finding-mag">{c.value:+.2f}</span>'
            f'<span class="vx-finding-bar"><i style="width:{width}%"></i></span>'
            f"</li>"
        )
    return f'<ul class="vx-findings">{"".join(rows)}</ul>'


def verdict_tag(verdict: Verdict) -> str:
    """The filled exhibit tag: what was concluded, against what limit, on what grounds."""
    flagged = verdict.flagged
    state = "vx-tag--flagged" if flagged else "vx-tag--clear"
    icon = _ICON_TAG if flagged else _ICON_CLEAR
    kind = "Flagged for review" if flagged else "Below review threshold"
    headline = "Refer to a reviewer" if flagged else "No action"
    if flagged:
        sub = (
            "This posting scored above the operating limit. It is routed to a human "
            "reviewer — the system does not decide."
        )
    else:
        sub = (
            "This posting scored below the operating limit and would not reach a "
            "reviewer's queue."
        )
    stamped = verdict.scored_at.strftime("%Y-%m-%d %H:%M UTC")

    return f"""<div class="vx-tag vx-tag--filled {state}">
  <p class="vx-verdict">{_esc(headline)}</p>
  <p class="vx-verdict-sub">{_esc(sub)}</p>
  {_scale(verdict.score, verdict.threshold_used, flagged)}
  <dl class="vx-readout">
    <div><dt>Score</dt><dd>{verdict.score:.4f}</dd></div>
    <div><dt>Operating limit</dt><dd>{verdict.threshold_used:.4f}</dd></div>
    <div><dt>Disposition</dt>
      <dd class="vx-tag-kind">{icon}{_esc(kind)}</dd></div>
    <div><dt>Examined by</dt><dd class="vx-readout-sm">{_esc(verdict.model_version)}</dd></div>
    <div><dt>At</dt><dd class="vx-readout-sm">{_esc(stamped)}</dd></div>
  </dl>
  <div class="vx-tag-rule"></div>
  <div class="vx-fieldhead" style="color:#4a4436;margin-bottom:0.5rem">
    Grounds <span class="vx-axis-note">bars scaled to &plusmn;{FINDING_AXIS:.1f} log-odds</span></div>
  {_findings(verdict)}
</div>"""


def blank_tag(threshold: float, model_version: str) -> str:
    """The empty state: the same tag, printed but unfilled.

    It teaches by showing the form the answer will take — the limit already engraved on
    the scale, the fields waiting — rather than telling the visitor to enter something.

    The first version set a 2.9rem em-dash where the verdict goes and a row of dashes
    where the score goes. Rendered, both read as a loading skeleton rather than as a
    blank form: a grey bar floating in an empty card looks broken, not expectant. The
    waiting fields now say what they are waiting for.
    """
    return f"""<div class="vx-tag vx-tag--blank">
  <p class="vx-blank-lede">This tag fills in when you take a reading.</p>
  <p class="vx-verdict-sub" style="color:{CHALK_FAINT}">
    You will get the score, its distance from the operating limit already engraved
    below, and every feature that moved it — with the direction each one pushed.
  </p>
  {_scale(None, threshold, False)}
  <dl class="vx-readout">
    <div><dt style="color:{CHALK_FAINT}">Score</dt>
      <dd style="color:{CHALK_FAINT};font-size:1rem;letter-spacing:0.06em">awaiting</dd></div>
    <div><dt style="color:{CHALK_FAINT}">Operating limit</dt>
      <dd style="color:{CHALK_DIM}">{threshold:.4f}</dd></div>
    <div><dt style="color:{CHALK_FAINT}">Disposition</dt>
      <dd class="vx-tag-kind" style="color:{CHALK_FAINT}">
        {_ICON_NOTE}Not yet examined</dd></div>
    <div><dt style="color:{CHALK_FAINT}">Examined by</dt>
      <dd class="vx-readout-sm" style="color:{CHALK_FAINT}">{_esc(model_version)}</dd></div>
  </dl>
</div>"""


def record(manifest: dict) -> str:
    """Conditions of examination. The limitations, as part of the record.

    Same facts the old alert carried, in the same words where it matters. Placed and
    set as provenance rather than as a warning, because rigour is this project's
    argument and it should not look like an error the reader must clear.
    """
    m = manifest["test_metrics_at_operating_threshold"]
    return f"""<div class="vx-record">
  <div class="vx-record-head">{_ICON_NOTE}Conditions of examination</div>
  <p><b>What the limit buys.</b> At {manifest['operating_threshold']:.4f} on the
  held-out fold this model flags {m['true_positives']} of {m['n_positive']} fraudulent
  postings with <b>{m['false_positives']} false alarms</b> — precision
  {m['precision']:.3f}, recall {m['recall']:.3f}. The threshold is set for a desk
  reviewing about {manifest['review_capacity']} postings a day. Low recall is the price
  of that choice, not a defect.</p>
  <p><b>Where it does not apply.</b> Trained on a 2012-2014 US-centric corpus. Against
  a live Indian job feed the score distribution shifts significantly ({_drift_note()}),
  so scores on current non-US postings are not comparable to the figures above.</p>
  <p><b>Known brittleness.</b> The model learned the bigram <i>work from</i> from "work
  from home" and fires on a legitimate "Work From Office"; fintech vocabulary
  (<i>money</i>, <i>income</i>) overlaps with scam vocabulary.</p>
  <p><b>This is a screening aid, not a verdict.</b> Every flag is routed to a human
  reviewer. Nothing here adjudicates.</p>
</div>"""


def masthead(manifest: dict) -> str:
    return f"""<div class="vx-masthead">
  <h1 class="vx-wordmark">Veridyx</h1>
  <p class="vx-offer">Scores a job posting for signs of employment fraud, and returns
  <b>the features that produced the score</b> — because a flag nobody can interrogate
  is one they will rubber-stamp or ignore.</p>
  <div class="vx-provenance">
    <span><b>MODEL</b> {_esc(manifest['model_version'])}</span>
    <span><b>TRAINED ON</b> {manifest['training_rows']:,} postings</span>
    <span><b>SPLIT</b> duplicate clusters held in one fold</span>
    <span><b>LIMIT</b> {manifest['operating_threshold']:.4f}</span>
  </div>
</div>"""


def colophon() -> str:
    return """<div class="vx-colophon">
  <span>Method, data and every figure:
    <a href="https://github.com/yugvyas/veridyx">github.com/yugvyas/veridyx</a></span>
  <span>Live feed evaluated against
    <a href="https://github.com/yugvyas/quantyx">quantyx</a></span>
</div>"""
