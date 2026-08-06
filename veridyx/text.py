"""HTML flattening for job descriptions.

Vendored from `quantyx/pipeline/schema.py` (same author) rather than reimplemented.
It already handles the two cases that break naive stripping:

* Feeds that double-encode (`&amp;lt;p&amp;gt;`), so one unescape pass is not enough.
* Block-level tags whose boundaries carry meaning — collapsing `<li>Python</li>
  <li>SQL</li>` to spaces glues "Python" and "SQL" into "PythonSQL" and destroys the
  word boundaries every downstream tokenizer depends on.

Copied rather than imported because a clean clone of Veridyx has no quantyx beside
it, and an import across sibling repositories would make this package unbuildable
for anyone but its author.
"""

from __future__ import annotations

import html
import re

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t\r\f\v]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")
_BLOCK_RE = re.compile(
    r"</?(?:p|div|br|li|ul|ol|tr|h[1-6]|section|article)\b[^>]*>",
    re.IGNORECASE,
)


def html_to_text(raw: str | None) -> str | None:
    """Flatten an HTML job description to plain text, preserving word boundaries."""
    if not raw:
        return None
    text = html.unescape(raw)
    text = _BLOCK_RE.sub("\n", text)
    text = _TAG_RE.sub(" ", text)
    # Unescape again: some feeds double-encode.
    text = html.unescape(text)
    text = _WS_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _BLANK_LINES_RE.sub("\n\n", text)
    return text.strip() or None


_WORD_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens. Used for near-duplicate shingling and OOV rate.

    Deliberately crude and dependency-free: this feeds the dedup clustering and the
    drift monitor, both of which need a *stable* tokenization that will not shift
    when a library upgrades and silently invalidate committed cluster assignments.
    """
    return _WORD_RE.findall(text.lower())
