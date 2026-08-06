"""Bridges from real posting sources into the `ScoreRequest` contract.

An adapter's only job is to turn somebody else's schema into ours. It must not score,
threshold, or explain — those live in `veridyx.evaluate`, `veridyx.threshold` and
`veridyx.explain`, and keeping them out of here is what stops a second, subtly
different scoring path from growing beside the first.
"""
