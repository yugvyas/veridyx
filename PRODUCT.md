# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

**Primary: a judge at ML Bubble 2026 (TE-BE Advanced Track).** They open the hosted
link once, for a few minutes, in the middle of evaluating a submission alongside many
others. Laptop or phone. They did not read the repository first and will not read it
after. They are technically literate and have seen a great many fraud-detection
projects that report 0.98 F1 and stop there.

Their job on this surface is to decide, quickly, whether this project is serious.

**Secondary: anyone assessing a job posting they are suspicious of.** Real but not who
the surface is designed for. The system is a screening aid that routes a posting to a
human reviewer; it never issues a verdict on its own.

## Product Purpose

Veridyx scores a job posting for signs of employment fraud and returns, with every
score, the features that produced it.

It exists because the standard benchmark for this task is misleading, and the project
is an argument about that as much as it is a classifier. Success on this surface is a
judge understanding within a minute that the numbers here are lower than the
literature's *on purpose*, and why that makes them worth more.

## Positioning

The mechanism a neighbouring project could not truthfully copy: **Veridyx measures what
its own benchmark score is made of, then deploys only the part that survives.**

Two measurements carry it, both reproducible from the repository:

- 46.8% of fraudulent postings in EMSCAD have a near-duplicate twin, so a random split
  scores a model on its memory. Veridyx keeps duplicate clusters inside one fold and
  reports the difference.
- Roughly 0.11 PR-AUC of benchmark performance comes from columns no live job feed
  provides (chiefly an empty `company_profile`). Veridyx deploys the model trained
  without them and states the cost.

The deployed model is therefore the *worse* one on paper and the only honest one in
production. Everything the surface shows is generated from committed artifacts.

## Operating Context

The judge arrives from a link in a slide deck or a repository README, cold, with no
context beyond the project name. They will try at most one or two inputs, most likely
the provided examples, and they will read whatever is in the first viewport.

The model is served from a committed artifact — it never trains on request. A scored
posting returns a `Verdict`: score, decision against the operating threshold, the
contributing features with signed magnitudes, and the model version.

## Capabilities and Constraints

- Scores one posting at a time from: title, description, location, remote flag, and
  whether pay is stated. Nothing else is available at inference, by design.
- Every displayed figure — operating threshold, precision, recall, model version,
  training row count — reads from `artifacts/deployment.json`. None may be hardcoded.
- The operating threshold (0.9758) is chosen for a review desk handling ~50 postings a
  day: on the held-out fold it flags 33 of 133 frauds with **zero** false alarms. Low
  recall is the deliberate price of that choice, not a defect.
- Trained on EMSCAD, a 2012–2014 US-centric corpus. Against a live Indian job feed the
  score distribution shifts significantly (PSI 0.41), so current non-US scores are not
  comparable to the published metrics.
- Known brittle features: the model learned the bigram `work from` from "work from
  home" and fires on a legitimate "Work From Office"; fintech vocabulary (`money`,
  `income`) overlaps with scam vocabulary.
- Hosted on Streamlit Community Cloud, Python 3.14. The runtime dependency list is
  minimal on purpose; anything added that the app does not import can break the deploy.
- Stack is fixed by the existing codebase: Python, Streamlit, a committed LightGBM
  artifact. Design control comes from a custom CSS layer, not from leaving the
  framework.

## Brand Commitments

The name **Veridyx** is the only fixed asset. No logo, mark, committed palette, or
typeface exists — confirmed with the user as a free hand.

Sibling project **quantyx** (same author, cross-linked) has its own committed
"mission-control console" world. Veridyx was explicitly decided **not** to inherit or
imitate it: the suite reads as deliberate through shared engineering standards, not
matching paint.

Voice, as established across the repository: plain, measured, willing to report results
that are unflattering. It states limitations in the same register as findings.

## Evidence on Hand

Real, reproducible, and safe to show:

- `artifacts/deployment.json` — the served model's threshold and held-out metrics.
- `experiments/results.json` — the full matrix, three architectures × two feature
  regimes × two split kinds, including DistilBERT losing to TF-IDF + logistic
  regression at roughly 400× the training cost.
- `experiments/dataset_stats.json` — the duplicate-leakage measurement.
- `experiments/drift.json` and `live_feed.json` — 1,006 real postings from the quantyx
  feed scored, 0 flagged, PSI 0.4068.
- `reports/*.png` with JSON sidecars — every figure, generated.
- Two worked example postings in `serve/scoring.py`, one fraudulent in character and
  one legitimate.

**Absences that must not be fabricated:** no users, no customers, no testimonials, no
production deployment, no uptime or latency claims, no confirmed fraud caught in the
wild. The live-feed run flagged nothing, and that null result is the finding — it must
never be dressed up as a catch.

## Product Principles

1. **Report the unflattering number.** The benchmark-honest score, the transformer
   losing to a three-second baseline, the zero flags on live data. The argument is
   built from results that are inconvenient.
2. **A flag must be interrogable.** A score without its reasons is a score a reviewer
   will either rubber-stamp or ignore. Attribution is not a feature of the product; it
   is the product.
3. **No number is typed by hand.** Every figure traces to a committed artifact, in the
   deck, the README, and this interface alike.
4. **The threshold is a staffing decision.** What gets flagged follows from how many
   postings a human can actually review, not from maximising a metric.
5. **Screening aid, never verdict.** The system routes to a human. Nothing in the
   product may imply it adjudicates.

## Accessibility & Inclusion

No externally imposed standard. Two product-specific requirements: the attribution
display must not carry meaning by colour alone, since direction (toward fraud / toward
legitimate) is its core information; and the surface must be fully legible on a phone,
because judges open links on phones.
