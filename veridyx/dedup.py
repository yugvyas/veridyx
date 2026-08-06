"""Near-duplicate detection over posting text, via MinHash + LSH banding.

Why this module exists at all:

EMSCAD contains scam campaigns — the same fraudulent template posted many times with
a company name or a city swapped. Under a random train/test split those siblings land
on both sides, and a model that has memorised one recognises the other. The reported
F1 then measures memorisation while looking exactly like generalisation. Published
results on this dataset are inflated by precisely this.

The fix is to cluster near-duplicates and keep every cluster wholly inside one fold.
This file finds the clusters; `veridyx.data` uses them.

Implemented directly rather than via `datasketch` for two reasons: it is ~80 lines,
and cluster assignments are committed and must not shift when a dependency changes
its hash function underneath us.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from veridyx.text import tokenize

# A 61-bit Mersenne prime. Permutations are (a * h + b) mod MERSENNE, the standard
# universal-hash family for MinHash.
_MERSENNE = (1 << 61) - 1

SHINGLE_SIZE = 5
NUM_PERM = 128
# 16 bands x 8 rows. Candidate pairs surface at Jaccard ~= (1/16)^(1/8) ~= 0.71,
# comfortably below the 0.80 we actually require, so banding over-generates and the
# exact check below does the deciding. Missing a duplicate is the costly error here;
# an extra candidate pair costs microseconds.
NUM_BANDS = 16
ROWS_PER_BAND = NUM_PERM // NUM_BANDS

DUPLICATE_THRESHOLD = 0.80

# Long descriptions are truncated for shingling only. Boilerplate benefits/EEO text
# at the tail of a posting is near-identical across unrelated legitimate listings and
# would manufacture false clusters if allowed to dominate the signature.
MAX_TOKENS = 1200


def _shingle_hashes(text: str) -> np.ndarray:
    """64-bit hashes of overlapping k-token shingles, deduplicated."""
    tokens = tokenize(text)[:MAX_TOKENS]
    if len(tokens) < SHINGLE_SIZE:
        # Too short to shingle: fall back to the whole token string so that short
        # postings still collide with their own exact duplicates.
        if not tokens:
            return np.empty(0, dtype=np.uint64)
        shingles = {" ".join(tokens)}
    else:
        shingles = {
            " ".join(tokens[i : i + SHINGLE_SIZE])
            for i in range(len(tokens) - SHINGLE_SIZE + 1)
        }
    return np.fromiter(
        (hash_shingle(s) for s in shingles), dtype=np.uint64, count=len(shingles)
    )


def hash_shingle(shingle: str) -> int:
    """Stable 64-bit hash. `hash()` is salted per-process and must not be used."""
    import hashlib

    return int.from_bytes(hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest(), "big")


def _permutations(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.integers(1, _MERSENNE, size=NUM_PERM, dtype=np.int64)
    b = rng.integers(0, _MERSENNE, size=NUM_PERM, dtype=np.int64)
    return a, b


def signatures(texts: list[str], seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """MinHash signatures and a validity mask.

    Returns `(sigs, has_shingles)`. Documents with no usable tokens get a placeholder
    all-max signature and `has_shingles[i] == False`.

    The mask is not decoration. An all-max placeholder is *identical* for every empty
    document, so without it two empty postings collide in every band and get merged at
    Jaccard 1.0 — the clustering would conclude that two postings with no text are the
    same scam campaign. Callers must skip masked-out rows rather than rely on the
    signature being unmatchable.
    """
    a, b = _permutations(seed)
    out = np.full((len(texts), NUM_PERM), np.iinfo(np.uint64).max, dtype=np.uint64)
    has_shingles = np.zeros(len(texts), dtype=bool)
    for i, text in enumerate(texts):
        h = _shingle_hashes(text)
        if h.size == 0:
            continue
        has_shingles[i] = True
        # Work in Python-int-free numpy: (a * h + b) % MERSENNE for every permutation.
        # int64 overflow is avoided by masking h into 61 bits first.
        hm = (h & _MERSENNE).astype(np.int64)
        perm = (a[:, None] * hm[None, :] + b[:, None]) % _MERSENNE
        out[i] = perm.min(axis=1).astype(np.uint64)
    return out, has_shingles


def _jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """Estimated Jaccard similarity: the fraction of agreeing MinHash slots."""
    return float(np.count_nonzero(sig_a == sig_b) / sig_a.size)


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


@dataclass(frozen=True)
class ClusterResult:
    """Cluster id per document, plus the pair count that produced them."""

    labels: np.ndarray  # shape (n,), int; documents sharing an id are near-duplicates
    confirmed_pairs: int

    @property
    def n_clusters(self) -> int:
        return int(np.unique(self.labels).size)

    @property
    def n_documents_in_multi_clusters(self) -> int:
        _, counts = np.unique(self.labels, return_counts=True)
        return int(counts[counts > 1].sum())


def cluster_near_duplicates(
    texts: list[str],
    seed: int = 0,
    threshold: float = DUPLICATE_THRESHOLD,
) -> ClusterResult:
    """Group texts into near-duplicate clusters. Singletons get their own cluster.

    Two-stage: LSH banding proposes candidate pairs, then every candidate is checked
    against the exact signature Jaccard. Banding alone would merge documents at ~0.71
    similarity, which over-clusters and would throw away legitimate training rows.
    """
    n = len(texts)
    if n == 0:
        return ClusterResult(labels=np.empty(0, dtype=np.int64), confirmed_pairs=0)

    sigs, has_shingles = signatures(texts, seed=seed)
    uf = _UnionFind(n)

    # Empty documents are excluded from banding entirely. Their placeholder signatures
    # are identical to each other, so including them would cluster every text-less
    # posting into one giant false campaign.
    eligible = np.flatnonzero(has_shingles)

    candidates: set[tuple[int, int]] = set()
    for band in range(NUM_BANDS):
        lo = band * ROWS_PER_BAND
        band_rows = sigs[:, lo : lo + ROWS_PER_BAND]
        buckets: dict[bytes, list[int]] = {}
        for i in eligible:
            buckets.setdefault(band_rows[i].tobytes(), []).append(int(i))
        for members in buckets.values():
            if len(members) < 2:
                continue
            # A bucket of size m implies m*(m-1)/2 pairs. Campaign clusters can be
            # large, so chain them (i0-i1, i0-i2, ...) instead: union-find makes the
            # transitive closure equivalent at linear cost.
            head = members[0]
            for other in members[1:]:
                candidates.add((head, other))

    confirmed = 0
    for i, j in candidates:
        if uf.find(i) == uf.find(j):
            continue
        if _jaccard(sigs[i], sigs[j]) >= threshold:
            uf.union(i, j)
            confirmed += 1

    roots = np.fromiter((uf.find(i) for i in range(n)), dtype=np.int64, count=n)
    # Renumber roots to a dense 0..k-1 range so cluster ids are stable and readable.
    _, labels = np.unique(roots, return_inverse=True)
    return ClusterResult(labels=labels.astype(np.int64), confirmed_pairs=confirmed)
