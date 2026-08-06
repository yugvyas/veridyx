"""Near-duplicate clustering behaviour.

The fixtures here are deliberately at realistic length. An early version of this file
used four-sentence postings, and a campaign variant that swapped only the company and
city still fell to Jaccard 0.78 — below threshold — purely because six changed tokens
are a large fraction of a fifty-shingle document. Real EMSCAD postings run to several
thousand characters, where the same swap is noise. Short fixtures would have argued
for lowering the threshold to fix a problem that only existed in the test.
"""

from __future__ import annotations

import numpy as np

from veridyx.dedup import cluster_near_duplicates, hash_shingle, signatures

_SCAM = (
    "Work from home data entry position. No experience required. Earn up to $5000 "
    "per week processing simple forms from your own computer. Send your resume and "
    "a copy of your identification to begin immediately. Payment is made weekly by "
    "wire transfer. Limited positions available so apply today. This is a genuine "
    "opportunity to work flexible hours around your existing commitments and family "
    "responsibilities. Successful applicants will be contacted within twenty four "
    "hours of submitting the application form below. You will be responsible for "
    "entering customer details into our secure online portal, verifying that the "
    "information matches the supporting documentation, and forwarding completed "
    "batches to your assigned supervisor at the end of each working day. No prior "
    "office experience is necessary as full training will be provided remotely at "
    "no cost to you. All you need is a reliable internet connection, a personal "
    "computer running any modern operating system, and the ability to commit to a "
    "minimum of ten hours per week. Payment is processed every Friday without fail "
    "and there are no deductions of any kind from your weekly earnings. Positions "
    "are strictly limited and are allocated on a first come first served basis so "
    "we strongly encourage interested candidates to respond at their earliest "
    "convenience to avoid disappointment in this recruitment round."
)

_LEGIT = (
    "We are looking for a data engineer to join our platform team. You will build "
    "and maintain batch and streaming pipelines, work closely with analysts, and "
    "help evolve our warehouse model. Requires strong SQL, Python, and experience "
    "with cloud data infrastructure at scale. Our team owns the ingestion layer "
    "that serves every downstream analytics consumer in the business, and you will "
    "have significant input into its architecture as we migrate from a batch first "
    "design to an incremental model. The ideal candidate has shipped production "
    "data systems, is comfortable reasoning about correctness under late arriving "
    "and out of order events, and cares about the operational experience of the "
    "people who depend on those systems. We work in small autonomous teams with a "
    "strong emphasis on code review, written design documents, and shared on call "
    "responsibility. Familiarity with dbt, Airflow, or an equivalent orchestration "
    "framework is valuable but not required if you have equivalent experience "
    "elsewhere. We offer a competitive salary, meaningful equity, a genuine hybrid "
    "working policy, and a substantial annual budget for conferences and training."
)


def _variant(text: str, company: str, city: str) -> str:
    """The realistic campaign mutation: swap the identifying nouns, keep the body."""
    return f"{company} is hiring in {city}. {text}"


class TestSignatures:
    def test_shape_and_determinism(self):
        sigs_a, mask_a = signatures([_SCAM, _LEGIT], seed=0)
        sigs_b, mask_b = signatures([_SCAM, _LEGIT], seed=0)
        assert sigs_a.shape == (2, 128)
        assert (sigs_a == sigs_b).all()
        assert (mask_a == mask_b).all()
        assert mask_a.all()

    def test_empty_text_is_masked_out(self):
        _, mask = signatures(["", "   ", _LEGIT], seed=0)
        assert not mask[0]
        assert not mask[1]
        assert mask[2]

    def test_shingle_hash_is_stable_across_processes(self):
        """`hash()` is salted per process; a committed cluster id must not move."""
        assert hash_shingle("a b c d e") == hash_shingle("a b c d e")
        assert hash_shingle("a b c d e") != hash_shingle("a b c d f")


class TestClustering:
    def test_campaign_variants_cluster_together(self):
        texts = [
            _variant(_SCAM, "Acme Corp", "Denver"),
            _variant(_SCAM, "Globex Ltd", "Austin"),
            _variant(_SCAM, "Initech", "Portland"),
            _LEGIT,
        ]
        labels = cluster_near_duplicates(texts, seed=0).labels
        assert labels[0] == labels[1] == labels[2]
        assert labels[3] != labels[0]

    def test_unrelated_documents_stay_separate(self):
        labels = cluster_near_duplicates([_SCAM, _LEGIT], seed=0).labels
        assert labels[0] != labels[1]

    def test_exact_duplicates_always_cluster(self):
        labels = cluster_near_duplicates([_LEGIT, _LEGIT, _SCAM], seed=0).labels
        assert labels[0] == labels[1]
        assert labels[2] != labels[0]

    def test_empty_documents_do_not_collapse_together(self):
        """Two postings with no text are not evidence of a shared campaign.

        Empty documents share an identical placeholder signature, so they collide in
        every LSH band. Without an explicit mask they merge into one giant false
        cluster that would then be assigned wholesale to a single fold.
        """
        labels = cluster_near_duplicates(["", "", _LEGIT], seed=0).labels
        assert labels[0] != labels[1]
        assert labels[0] != labels[2]

    def test_singletons_each_get_an_id(self):
        result = cluster_near_duplicates([_SCAM, _LEGIT], seed=0)
        assert result.n_clusters == 2
        assert result.n_documents_in_multi_clusters == 0

    def test_empty_input(self):
        result = cluster_near_duplicates([], seed=0)
        assert result.n_clusters == 0
        assert result.labels.shape == (0,)

    def test_labels_are_dense_and_zero_based(self):
        texts = [_SCAM, _SCAM, _LEGIT, "Completely different posting about baking bread."]
        labels = cluster_near_duplicates(texts, seed=0).labels
        assert set(labels.tolist()) == set(range(len(np.unique(labels))))

    def test_transitive_closure_links_a_chain(self):
        """A drifting campaign: each posting resembles its neighbour, ends differ.

        Union-find is what makes the whole chain one cluster. Pairwise thresholding
        alone would split it, and the tail of a campaign would leak across folds.
        """
        variants = [
            _variant(_SCAM, f"Company {i}", f"City {i}") for i in range(6)
        ]
        labels = cluster_near_duplicates(variants, seed=0).labels
        assert len(set(labels.tolist())) == 1
