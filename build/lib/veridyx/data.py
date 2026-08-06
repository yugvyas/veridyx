"""Acquire EMSCAD, verify it, clean it, cluster it, split it.

Provenance matters more than usual here. Every headline number in the submission is
computed from this file's output, so the dataset has to be reproducible from a clean
clone by someone who is not the author. That means:

* the payload is fetched, never committed (it is large and redistributable, not ours);
* its SHA256 and row/label counts are committed in `data/dataset.lock.json`;
* a mismatch is a hard failure, not a warning.

Run `python -m veridyx.data --verify` to prove the chain.

The source is the EMSCAD "Real or Fake? Fake Job Postings" set (University of the
Aegean, Laboratory of Information & Communication Systems Security), distributed
under an open licence and mirrored on the Hugging Face Hub. Expected shape:
17,880 rows, 866 fraudulent (4.84%).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from veridyx.dedup import DUPLICATE_THRESHOLD, cluster_near_duplicates
from veridyx.schema import RawPosting
from veridyx.splits import Split, grouped_split, naive_split

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_CSV = RAW_DIR / "emscad.csv"
LOCK_FILE = DATA_DIR / "dataset.lock.json"
CACHE_PARQUET = DATA_DIR / "prepared.parquet"

EXPECTED_ROWS = 17_880
EXPECTED_FRAUD = 866

# Tried in order. Each is a mirror of the same Kaggle/EMSCAD CSV; the first that
# resolves wins and its identity is recorded in the lock file, so a future run can
# tell whether it got the same bytes from the same place.
HF_MIRRORS: tuple[str, ...] = (
    "victor/real-or-fake-fake-jobposting-prediction",
    "shawhin/fake-job-postings",
    "Sathvika-Yerramsetti/fake_job_postings",
)

# EMSCAD column -> RawPosting field. Mirrors differ in casing and in whether the id
# column is present, so mapping is explicit rather than positional.
_COLUMN_ALIASES = {
    "job_id": "job_id",
    "jobid": "job_id",
    "title": "title",
    "location": "location",
    "department": "department",
    "salary_range": "salary_range",
    "company_profile": "company_profile",
    "description": "description",
    "requirements": "requirements",
    "benefits": "benefits",
    "telecommuting": "telecommuting",
    "has_company_logo": "has_company_logo",
    "has_questions": "has_questions",
    "employment_type": "employment_type",
    "required_experience": "required_experience",
    "required_education": "required_education",
    "industry": "industry",
    "function": "function",
    "fraudulent": "fraudulent",
}


class DatasetError(RuntimeError):
    """Raised when the dataset is missing, unreachable, or fails verification."""


# --------------------------------------------------------------------------------
# Acquisition
# --------------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_from_hub() -> tuple[pd.DataFrame, str]:
    """Return (frame, mirror_id) from the first Hugging Face mirror that resolves."""
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - environment problem, not logic
        raise DatasetError(
            "The `datasets` package is required to fetch the dataset. "
            "Install it with: .venv/bin/pip install -r requirements.txt"
        ) from exc

    failures: list[str] = []
    for mirror in HF_MIRRORS:
        try:
            log.info("trying mirror %s", mirror)
            ds = load_dataset(mirror, split="train")
            return ds.to_pandas(), mirror
        except Exception as exc:
            failures.append(f"  {mirror}: {type(exc).__name__}: {exc}")

    raise DatasetError(
        "No Hugging Face mirror resolved. Attempts:\n"
        + "\n".join(failures)
        + f"\n\nFallback: download fake_job_postings.csv from Kaggle by hand and place "
        f"it at {RAW_CSV}, then re-run."
    )


def fetch_raw(force: bool = False) -> Path:
    """Ensure the raw CSV exists on disk. Returns its path.

    Idempotent: an existing file is left alone unless `force`, because re-downloading
    would silently swap the bytes underneath a committed lock file.
    """
    if RAW_CSV.exists() and not force:
        log.info("raw CSV already present at %s", RAW_CSV)
        return RAW_CSV

    frame, mirror = _download_from_hub()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RAW_CSV, index=False)
    log.info("wrote %d rows from %s to %s", len(frame), mirror, RAW_CSV)

    _write_lock(RAW_CSV, frame, mirror)
    return RAW_CSV


def _write_lock(path: Path, frame: pd.DataFrame, mirror: str) -> None:
    lock = {
        "mirror": mirror,
        "sha256": _sha256(path),
        "rows": len(frame),
        "fraudulent": int(pd.to_numeric(frame["fraudulent"], errors="coerce").fillna(0).sum()),
        "columns": sorted(str(c) for c in frame.columns),
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOCK_FILE.write_text(json.dumps(lock, indent=2) + "\n")
    log.info("wrote lock file %s", LOCK_FILE)


def verify(strict: bool = True) -> dict:
    """Check the on-disk CSV against the committed lock file and expected shape.

    `strict` also enforces the published EMSCAD counts. A mirror that has silently
    dropped rows or rebalanced the classes is worse than no data at all, because the
    resulting numbers still look plausible.
    """
    if not RAW_CSV.exists():
        raise DatasetError(f"{RAW_CSV} missing. Run: python -m veridyx.data --fetch")
    if not LOCK_FILE.exists():
        raise DatasetError(f"{LOCK_FILE} missing. Run: python -m veridyx.data --fetch")

    lock = json.loads(LOCK_FILE.read_text())
    actual_sha = _sha256(RAW_CSV)
    if actual_sha != lock["sha256"]:
        raise DatasetError(
            f"checksum mismatch for {RAW_CSV}\n"
            f"  expected {lock['sha256']}\n"
            f"  actual   {actual_sha}\n"
            "The dataset on disk is not the one this repository's results were "
            "computed from. Re-fetch with --fetch --force, or restore the file."
        )

    if strict:
        problems = []
        if lock["rows"] != EXPECTED_ROWS:
            problems.append(f"rows: expected {EXPECTED_ROWS}, lock says {lock['rows']}")
        if lock["fraudulent"] != EXPECTED_FRAUD:
            problems.append(
                f"fraudulent: expected {EXPECTED_FRAUD}, lock says {lock['fraudulent']}"
            )
        if problems:
            raise DatasetError(
                "dataset does not match published EMSCAD shape:\n  "
                + "\n  ".join(problems)
            )

    return lock


# --------------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------------


def _coerce_bool(series: pd.Series) -> pd.Series:
    """EMSCAD mirrors encode flags as 0/1, "0"/"1", or True/False depending on export."""
    if series.dtype == bool:
        return series
    return (
        pd.to_numeric(series, errors="coerce").fillna(0).astype(int).astype(bool)
        if not series.map(type).eq(str).any()
        else series.astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes"})
    )


def load_postings(path: Path | None = None) -> list[RawPosting]:
    """Parse the raw CSV into validated `RawPosting` records.

    Rows that fail validation are dropped with a warning rather than crashing the run:
    a mirror occasionally carries a handful of malformed rows, and losing three of
    17,880 is preferable to being unable to train at all. The count of dropped rows is
    logged so the loss is never silent.
    """
    path = path or RAW_CSV
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])

    renamed = {}
    for col in frame.columns:
        key = str(col).strip().lower()
        if key in _COLUMN_ALIASES:
            renamed[col] = _COLUMN_ALIASES[key]
    frame = frame.rename(columns=renamed)

    missing = {"title", "description", "fraudulent"} - set(frame.columns)
    if missing:
        raise DatasetError(f"{path} is missing required columns: {sorted(missing)}")

    if "job_id" not in frame.columns:
        # Some mirrors drop the id. Positional ids are stable for a fixed file, and
        # the file is checksummed, so this stays reproducible.
        frame["job_id"] = np.arange(1, len(frame) + 1)

    for flag in ("telecommuting", "has_company_logo", "has_questions"):
        frame[flag] = (
            _coerce_bool(frame[flag]) if flag in frame.columns else False
        )
    frame["fraudulent"] = _coerce_bool(frame["fraudulent"])

    postings: list[RawPosting] = []
    dropped = 0
    for record in frame.to_dict(orient="records"):
        clean = {k: (None if pd.isna(v) else v) for k, v in record.items()}
        try:
            postings.append(RawPosting(**{k: v for k, v in clean.items() if k in RawPosting.model_fields}))
        except Exception as exc:
            dropped += 1
            log.warning("dropped row job_id=%s: %s", clean.get("job_id"), exc)

    if dropped:
        log.warning("dropped %d/%d rows during validation", dropped, len(frame))
    return postings


# --------------------------------------------------------------------------------
# The prepared dataset
# --------------------------------------------------------------------------------


@dataclass(frozen=True)
class Dataset:
    """Everything downstream needs, assembled once.

    `frame` carries one row per posting with the cleaned columns, the near-duplicate
    `cluster` id, and both fold assignments (`fold_grouped`, `fold_naive`) side by
    side — so any evaluation can switch between them without re-splitting and without
    the risk of comparing two differently-seeded partitions.
    """

    frame: pd.DataFrame
    postings: list[RawPosting]
    grouped: Split
    naive: Split
    cluster_stats: dict

    @property
    def labels(self) -> np.ndarray:
        return self.frame["fraudulent"].to_numpy(dtype=bool)


def _duplicate_stats(clusters, labels: np.ndarray, n: int) -> dict:
    """Quantify the leakage the grouped split exists to prevent.

    The per-class breakdown is the number that matters. An overall duplicate share
    says the dataset is repetitive; a *fraudulent* duplicate share says the positive
    class specifically has near-twins, which is what makes a random split flatter a
    model. Reporting only the overall figure would understate the problem.
    """
    sizes = np.bincount(clusters.labels)
    in_multi = sizes[clusters.labels] > 1

    def share(mask: np.ndarray) -> float:
        return round(float(in_multi[mask].mean()), 4) if mask.any() else 0.0

    return {
        "n_documents": n,
        "n_clusters": clusters.n_clusters,
        "n_in_multi_document_clusters": clusters.n_documents_in_multi_clusters,
        "confirmed_pairs": clusters.confirmed_pairs,
        "duplicate_share_overall": share(np.ones(n, dtype=bool)),
        "duplicate_share_fraudulent": share(labels),
        "duplicate_share_legitimate": share(~labels),
        "n_fraudulent_with_duplicate": int(in_multi[labels].sum()),
        "n_fraudulent": int(labels.sum()),
        "largest_cluster_size": int(sizes.max()) if n else 0,
        "threshold": DUPLICATE_THRESHOLD,
    }


def prepare(seed: int = 0, path: Path | None = None) -> Dataset:
    """Load, cluster, and split. This is the single entry point for every experiment."""
    postings = load_postings(path)
    log.info("loaded %d postings", len(postings))

    requests = [p.to_score_request() for p in postings]
    texts = [r.text() for r in requests]

    clusters = cluster_near_duplicates(texts, seed=seed)
    log.info(
        "near-duplicate clustering: %d clusters over %d documents "
        "(%d documents sit in a multi-document cluster)",
        clusters.n_clusters,
        len(texts),
        clusters.n_documents_in_multi_clusters,
    )

    labels = np.array([p.fraudulent for p in postings], dtype=bool)
    grouped = grouped_split(clusters.labels, labels, seed=seed)
    naive = naive_split(labels, seed=seed)

    frame = pd.DataFrame(
        {
            "job_id": [p.job_id for p in postings],
            "title": [p.title for p in postings],
            "text": texts,
            "fraudulent": labels,
            "cluster": clusters.labels,
            "fold_grouped": grouped.fold,
            "fold_naive": naive.fold,
        }
    )

    cluster_stats = _duplicate_stats(clusters, labels, len(texts))
    return Dataset(
        frame=frame,
        postings=postings,
        grouped=grouped,
        naive=naive,
        cluster_stats=cluster_stats,
    )


EXPERIMENTS_DIR = ROOT / "experiments"
DATASET_STATS_FILE = EXPERIMENTS_DIR / "dataset_stats.json"


def _write_dataset_stats(ds: Dataset) -> Path:
    """Commit the leakage measurement as data, not as a number in a slide.

    Everything the deck claims about duplication reads from this file, so the claim
    and the computation cannot drift apart.
    """
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "near_duplicates": ds.cluster_stats,
        "splits": {
            split.kind: {
                "seed": split.seed,
                "folds": split.summary(ds.labels),
            }
            for split in (ds.grouped, ds.naive)
        },
    }
    DATASET_STATS_FILE.write_text(json.dumps(payload, indent=2, default=float) + "\n")
    log.info("wrote %s", DATASET_STATS_FILE)
    return DATASET_STATS_FILE


# --------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--fetch", action="store_true", help="download the dataset")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    parser.add_argument("--verify", action="store_true", help="check checksum and shape")
    parser.add_argument("--prepare", action="store_true", help="cluster and split; print summary")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not (args.fetch or args.verify or args.prepare):
        parser.error("pass at least one of --fetch, --verify, --prepare")

    if args.fetch:
        fetch_raw(force=args.force)

    if args.verify or args.fetch:
        lock = verify()
        rate = lock["fraudulent"] / lock["rows"]
        print(f"dataset OK  mirror={lock['mirror']}")
        print(f"  rows        {lock['rows']:,}")
        print(f"  fraudulent  {lock['fraudulent']:,}  ({rate:.2%})")
        print(f"  sha256      {lock['sha256']}")

    if args.prepare:
        ds = prepare(seed=args.seed)
        _write_dataset_stats(ds)
        print("\nnear-duplicate clustering")
        for k, v in ds.cluster_stats.items():
            print(f"  {k:32s} {v}")
        for split in (ds.grouped, ds.naive):
            print(f"\n{split.kind} split (seed={split.seed})")
            for fold, stats in split.summary(ds.labels).items():
                print(
                    f"  {fold:5s} n={stats['n']:6,d}  fraud={stats['n_fraud']:4,d}  "
                    f"rate={stats['fraud_rate']:.2%}"
                )

    return 0


if __name__ == "__main__":
    sys.exit(_main())
