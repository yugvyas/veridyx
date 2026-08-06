---
title: Veridyx
emoji: 🛡️
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.22.0
app_file: app.py
pinned: false
license: mit
---

# Veridyx — fraudulent job posting detection

Scores a job posting for signs of employment fraud and shows the features behind the
score. Source: [github.com/yugvyas/veridyx](https://github.com/yugvyas/veridyx).

## What is served here

The **portable** LightGBM model. That is not a fallback from the transformer — it is
the only model in the comparison that can be both served and explained:

- **DistilBERT** is the ceiling check. It has no attribution path, so a flag it raised
  could not be defended to a reviewer.
- The **FULL** model reaches a higher benchmark score (PR-AUC 0.883 vs 0.773) but does
  so using EMSCAD columns — `company_profile`, `has_company_logo`, `benefits` — that no
  live feed or API caller can supply. It is unservable by construction.

That ~0.11 PR-AUC gap is the honest cost of deployability, and it is measured rather
than glossed over.

## Reading the output

Every response is a `Verdict`: the score, the decision against the operating threshold,
the SHAP contributions that produced it, and the model version. The threshold is chosen
for a review desk of ~50 postings/day, so it is deliberately conservative — on the
held-out test fold it flags 33 postings with **zero false alarms** and misses 100. A
different review capacity implies a different threshold; that trade-off is the design,
not a defect.

## Known limitations, stated plainly

- Trained on EMSCAD (2012-2014, US-centric). Against a live Indian job feed the score
  distribution shifts significantly (PSI 0.41), so scores on current non-US postings
  are not comparable to the published metrics.
- The model has learned some brittle text features. `work from` was learned from "work
  from home" and fires on a legitimate "Work From Office"; fintech vocabulary (`money`,
  `income`) overlaps with scam vocabulary.
- **This is a screening aid, not a verdict.** It is built to route postings to a human
  reviewer, and every deployment assumption in the repository is built around that.

## API

```python
from gradio_client import Client

client = Client("yugvyas/veridyx")
client.predict(
    "Work From Home Data Entry",   # title
    "Earn $5000/week, no experience, wire transfer...",  # description
    "",                            # location
    True,                          # is_remote
    False,                         # has_salary
    api_name="/score",
)
```
