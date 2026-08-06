# Veridyx

[![ci](https://github.com/yugvyas/veridyx/actions/workflows/ci.yml/badge.svg)](https://github.com/yugvyas/veridyx/actions/workflows/ci.yml)

**Fraudulent job posting detection, evaluated honestly and deployed against a live feed.**

Submitted to ML Bubble 2026 (TE-BE Advanced Track). Sibling to
[quantyx](https://github.com/yugvyas/quantyx), whose live job-market feed Veridyx is
evaluated against.

Published results on the EMSCAD benchmark run to ~0.98 F1. This repository reports
lower numbers and argues they are the real ones, because two things inflate the
published figures and neither survives deployment:

1. **Near-duplicate leakage.** 46.8% of fraudulent postings have a near-duplicate twin
   in the dataset. Under a random split, half the fraud test set has a sibling in
   train, and the model is scored on its memory rather than its judgement.
2. **Benchmark-only columns.** EMSCAD's strongest single fraud signal is an empty
   `company_profile` field. No job board, ATS, or API exposes such a field. A model
   leaning on it scores beautifully and cannot be deployed anywhere.

Veridyx measures both, then deploys the model that survives them.

<!-- RESULTS:START -->

### The dataset, after de-duplication

**17,880** postings · **13,327** near-duplicate clusters · **46.8%** of fraudulent postings have a twin

| | share sitting in a near-duplicate cluster |
| --- | --- |
| All postings | 33.9% |
| Fraudulent | **46.8%** (405 of 866) |
| Legitimate | 33.2% |

### Results

Grouped split, held-out test fold. Thresholds chosen on validation.

| model | regime | PR-AUC | F1 | precision | recall |
| --- | --- | ---: | ---: | ---: | ---: |
| lightgbm | full | 0.8829 | 0.7895 | 0.7895 | 0.7895 |
| tfidf-lr | full | 0.8577 | 0.8015 | 0.8140 | 0.7895 |
| distilbert | full | 0.7453 | 0.7265 | 0.7946 | 0.6692 |
| lightgbm | portable | 0.7727 | 0.7000 | 0.7165 | 0.6842 |
| tfidf-lr | portable | 0.7584 | 0.7107 | 0.7890 | 0.6466 |
| distilbert | portable | 0.7183 | 0.6696 | 0.8085 | 0.5714 |

**What the benchmark-only columns are worth.** PR-AUC lost when the fields no live feed provides are removed:

| model | FULL | PORTABLE | difference |
| --- | ---: | ---: | ---: |
| distilbert | 0.7453 | 0.7183 | **+0.0270** |
| lightgbm | 0.8829 | 0.7727 | **+0.1102** |
| tfidf-lr | 0.8577 | 0.7584 | **+0.0992** |

**What a random split is worth.** PR-AUC gained by letting near-duplicates leak across folds:

| model | regime | grouped | naive | inflation |
| --- | --- | ---: | ---: | ---: |
| lightgbm | full | 0.8829 | 0.9079 | **+0.0250** |
| tfidf-lr | full | 0.8577 | 0.9051 | **+0.0474** |
| lightgbm | portable | 0.7727 | 0.8445 | **+0.0718** |
| tfidf-lr | portable | 0.7584 | 0.8195 | **+0.0611** |

### Against a live feed

Scored **1,006** current postings from [quantyx](https://github.com/yugvyas/quantyx) at the deployed threshold of 0.9758.

- **0 flagged** (0.00%); 99th-percentile score 0.0442
- Drift **PSI 0.4068** (significant shift), KS 0.1446, OOV 10.42%
- Mean score 0.0393 → 0.0037

Zero flags is not the same as a clean feed. The score distribution has shifted significantly, so the absence of flags is at least partly the model going quiet on data it was not trained for — which is exactly what the drift monitor exists to distinguish.

<!-- RESULTS:END -->

## The two design decisions

**Two feature regimes, reported side by side.** `FULL` uses every EMSCAD column and
matches the literature. `PORTABLE` uses only what a live ATS or aggregator feed
actually provides — title, description, location, remote flag, salary presence, and
text-derived signals. The gap between them measures how much published performance is
dataset artifact.

This is enforced structurally, not by convention: `features.full_features()` calls
`features.portable_features()` and concatenates a benchmark-only block, and
`ScoreRequest` — the contract the endpoint, the quantyx bridge, and the review sheet
all speak — cannot carry a benchmark-only field. A test asserts it.

**Grouped splits, with the naive delta reported.** Near-duplicate clusters (MinHash +
LSH banding, exact Jaccard confirmation at 0.80) are kept wholly inside one fold. Both
split kinds are computed so the leak can be quantified rather than merely avoided.

## Metrics

Accuracy is not computed anywhere in this repository. At a 4.84% fraud rate, a model
answering "legitimate" to everything is 95.2% accurate and worthless; a test asserts
that such a model scores zero on every metric Veridyx reports. The reported metrics are
PR-AUC, precision, recall, and F1.

## The threshold is a cost decision

Two framings, which disagree — and the disagreement is the point:

- **Minimum expected cost** at an assumed 200:1 false-negative:false-positive ratio
  flags 801 of 2,554 test postings at 16.2% precision. Beyond any review desk.
- **Review capacity** — a desk handling 50 postings/day — flags 34 at 100% precision,
  catching 26% of fraud. This is the deployed operating point.

The cost ratio is a guess, so `veridyx.threshold --sweep` reports how far the answer
moves from 1:1 to 1000:1 rather than resting on it.

## Layout

```
veridyx/
  schema.py      the contract: RawPosting -> ScoreRequest -> Verdict
  data.py        fetch, verify, clean, cluster, split
  dedup.py       MinHash + LSH near-duplicate clustering
  splits.py      grouped and naive partitioning
  features.py    the FULL / PORTABLE regimes
  models/        TF-IDF+LR, LightGBM, DistilBERT behind one interface
  evaluate.py    metrics and the experiment matrix
  threshold.py   cost- and capacity-based threshold selection
  explain.py     SHAP attribution
  drift.py       PSI, KS, out-of-vocabulary rate
  review.py      explained HTML review sheet + verdict log
  train.py       produces the deployable artifact
  adapters/      quantyx live-feed bridge
  stats.py       regenerates the results block above
serve/           framework-agnostic scoring for the endpoint
streamlit_app.py the hosted endpoint (Streamlit Community Cloud)
reports/         generated figures — every one has a JSON sidecar
experiments/     committed results the README and figures read from
```

## On DistilBERT

The transformer loses to both classic models, on both regimes, at roughly 300x the
training cost. That is the reported result rather than a footnote.

Two caveats belong with it, because they bound the claim:

- **It sees text only.** LightGBM consumes the text *and* the engineered feature block;
  DistilBERT consumes only the text. So this compares a text-only transformer against
  text-plus-metadata trees, which is how both are normally deployed, but it is not a
  pure architecture comparison.
- **256 tokens, 3 epochs, no hyperparameter search.** EMSCAD descriptions run to several
  thousand characters, so roughly the first third is seen.

The claim is "DistilBERT out of the box loses to TF-IDF here", not "transformers cannot
work on this task".

## Reproducing

Requires Python 3.11–3.13. **On macOS, LightGBM needs OpenMP:** `brew install libomp`,
without which it fails at import with a `dlopen` error.

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt

.venv/bin/python -m veridyx.data --fetch      # 17,880 rows, 866 fraudulent, checksummed
.venv/bin/python -m veridyx.data --prepare    # cluster and split
.venv/bin/python -m veridyx.evaluate          # the results matrix
.venv/bin/python -m veridyx.threshold --sweep
.venv/bin/python -m veridyx.report            # regenerate every figure
.venv/bin/python -m veridyx.stats             # regenerate the block above
.venv/bin/streamlit run streamlit_app.py      # the endpoint, locally

.venv/bin/pytest                              # bare pytest, not python -m pytest
```

DistilBERT is optional and slow (~25 min per cell on an M4 via MPS):

```bash
.venv/bin/pip install -r requirements-bert.txt
.venv/bin/python -m veridyx.evaluate --models distilbert
```

The live-feed evaluation expects quantyx beside this repository:

```bash
git clone https://github.com/yugvyas/quantyx ../quantyx
.venv/bin/python -m veridyx.adapters.quantyx   # score the live feed
.venv/bin/python -m veridyx.drift              # measure the shift
.venv/bin/python -m veridyx.review             # explained review sheet
```

## Provenance

The dataset is EMSCAD ("Real or Fake? Fake Job Postings", University of the Aegean,
Laboratory of Information & Communication Systems Security), open licence, fetched
from a Hugging Face mirror. Its SHA256 and row/label counts are committed in
`data/dataset.lock.json` and verified on every load; a mismatch is a hard failure. The
payload itself is not committed — it is large and reproducible.

Every figure in `reports/` ships with a JSON sidecar containing exactly the values
drawn, and the results block above is generated from `experiments/`. No number in this
repository or in the accompanying deck is typed by hand.

## Limitations

- EMSCAD is 2012–2014 and US-centric. Against a live Indian job feed the score
  distribution shifts significantly (PSI 0.41).
- The model has learned brittle text features. `work from` was learned from "work from
  home" and fires on a legitimate "Work From Office"; fintech vocabulary (`money`,
  `income`) overlaps with scam vocabulary.
- This is a screening aid that routes postings to a human reviewer, not a verdict.

## Licence

MIT.
