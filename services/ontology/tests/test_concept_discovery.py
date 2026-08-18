"""Bottom-up concept discovery: the document proposes, the model only names.

The claim being tested is about direction. Asked which entity types a domain has, a model answers
from what it knows about domains, and an omission leaves no trace — nothing downstream can miss a
concept that was never mentioned. Here the candidates are the document's own noun phrases, so a
frequent one cannot be skipped, and the model's role is narrowed to naming a group.

The deterministic half (extraction, counting, clustering) is tested for real. The naming and
merging calls are scripted, since what they do is the model's judgement and not this code's.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("spacy", reason="the [discovery] extra is not installed")
pytest.importorskip("sklearn", reason="the [discovery] extra is not installed")
pytest.importorskip("sentence_transformers", reason="the [discovery] extra is not installed")

from loka_ontology.concept_discovery import (  # noqa: E402 - after the availability guard
    ClusterNamer,
    Concept,
    cluster_terms,
    discover_concepts,
    extract_terms,
)

_TEXT = """
Independent sellers list items on our marketplace. A shopper pays once. The same item is often
offered by more than one seller, so the seller is a fact about the line, not about the item.
Some items cannot go on the standard service: above a certain weight the parcel needs freight.
Those are bulky items. A bulky item is still an ordinary item in every other respect.
"""


class _Namer:
    """Names every group and merges nothing, so a test sees the clustering rather than a model."""

    def __init__(self, merge: list[list[str]] | None = None) -> None:
        self.merge_groups = merge or []
        self.messages = SimpleNamespace(create=self._create)
        self.asked: list[str] = []

    def _create(self, *, system: str, messages: list[dict[str, Any]], **kw: Any) -> Any:
        self.asked.append("naming" if "naming concepts" in system else "merging")
        if "naming concepts" in system:
            groups = [g for g in messages[0]["content"].strip().split("\n") if g]
            reply: dict[str, Any] = {
                "names": [
                    {"group": int(g.split(":")[0]), "name": f"Concept{g.split(':')[0]}"}
                    for g in groups
                ]
            }
        else:
            reply = {"same": self.merge_groups}
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=json.dumps(reply))]
        )


# ---- extraction ----

def test_both_the_phrase_and_its_head_noun_are_counted() -> None:
    """A chunker returns "the same item", "a bulky item", "some items" for one word. Counting
    phrases alone splits a document's central vocabulary into fragments."""
    counts = extract_terms(_TEXT)
    assert counts["item"] >= 4          # the head noun, consolidated across phrasings
    assert counts["bulky item"] >= 2    # and the phrase itself, which means something else


def test_plurals_and_determiners_do_not_create_separate_terms() -> None:
    counts = extract_terms(_TEXT)
    assert "items" not in counts
    assert "sellers" not in counts
    assert "the seller" not in counts


def test_grammatical_scaffolding_is_dropped() -> None:
    """spaCy returns "it", "that", "the same thing". The paper notes these have to go before
    abstraction; left in, they cluster together and get named as though they were a concept."""
    counts = extract_terms("It is that thing. This one. Something happened to them.")
    assert not {"it", "that", "this", "one", "something", "them"} & set(counts)


def test_a_word_is_not_mangled_into_a_non_word() -> None:
    """Stripping a trailing "s" by hand turns "this" into "thi", which then slips past a stop
    list that was watching for "this". Lemmatisation is used instead."""
    counts = extract_terms(_TEXT)
    assert not [t for t in counts if t.startswith("thi") and t != "thing"]


# ---- clustering ----

def test_clustering_needs_no_cluster_count() -> None:
    """The number of concepts in a domain is what is being discovered, which rules out any
    algorithm that has to be told it."""
    terms = ["seller", "vendor", "merchant", "parcel", "package", "shipment", "monday", "friday"]
    clusters = cluster_terms(terms)
    assert 1 < len(clusters) < len(terms)
    joined = {frozenset(c) for c in clusters}
    assert any({"parcel", "package", "shipment"} <= c for c in joined)
    assert any({"seller", "vendor", "merchant"} <= c for c in joined)


def test_the_preference_adapts_to_the_document() -> None:
    """preference is a self-similarity and has to sit on the scale of the similarities, which
    depends on the text. A constant below all of them collapses the vocabulary into one cluster —
    which is exactly what a hard-coded -0.35 did."""
    terms = [t for t, _ in extract_terms(_TEXT).most_common()]
    assert len(cluster_terms(terms, quantile=0.95)) > len(cluster_terms(terms, quantile=0.3))


# ---- the whole pass ----

def test_concepts_come_from_the_document_and_are_only_named_by_the_model() -> None:
    client = _Namer()
    namer = ClusterNamer(client=client, model="test", domain="marketplace")
    result = discover_concepts(_TEXT, namer)
    assert result.terms_extracted > 10
    assert result.concepts
    for concept in result.concepts:
        assert concept.evidence in extract_terms(_TEXT)  # a phrase the document contains
        assert concept.occurrences > 0
    assert client.asked == ["naming", "merging"]


def test_what_was_set_aside_is_returned_not_discarded() -> None:
    """A reviewer disagreeing with the result needs to see the cluster that fell below the size
    floor, or they can only accept the output as given."""
    namer = ClusterNamer(client=_Namer(), model="test")
    result = discover_concepts(_TEXT, namer, min_cluster_size=3)
    assert result.small_clusters
    assert result.rare_terms
    assert result.as_dict()["dropped_small_clusters"]


def test_duplicate_concepts_are_merged_keeping_the_better_covered_name() -> None:
    """Clustering is deliberately set to over-split, so merging is the repair. The survivor is
    the one covering more of the document; keeping the first would be arbitrary."""
    namer = ClusterNamer(client=_Namer(merge=[["Buyer", "Shopper"]]), model="test")
    a = Concept(name="Buyer", terms=("buyer",), occurrences=2, evidence="buyer")
    b = Concept(name="Shopper", terms=("shopper",), occurrences=9, evidence="shopper")
    # Buyer is a fifth of Shopper: the shape of a split-off fragment, which is what merging is for
    kept, merged = namer.merge([a, b])
    assert [c.name for c in kept] == ["Shopper"]
    assert kept[0].occurrences == 11
    assert set(kept[0].terms) == {"shopper", "buyer"}
    assert merged == [("Buyer", "Shopper")]


def test_a_group_the_model_declines_to_name_is_left_out() -> None:
    """Returning null for an unnameable group is how the paper's naming step filters clusters
    that are noise rather than concepts."""
    class _Refuses(_Namer):
        def _create(self, *, system: str, messages: list[dict[str, Any]], **kw: Any) -> Any:
            if "naming concepts" in system:
                groups = [g for g in messages[0]["content"].strip().split("\n") if g]
                reply = {"names": [{"group": i, "name": None} for i, _ in enumerate(groups)]}
                return SimpleNamespace(
                    content=[SimpleNamespace(type="text", text=json.dumps(reply))]
                )
            return super()._create(system=system, messages=messages, **kw)

    result = discover_concepts(_TEXT, ClusterNamer(client=_Refuses(), model="test"))
    assert result.concepts == []
    assert result.terms_extracted > 0  # the document was read; nothing was nameable


def test_two_concepts_of_comparable_weight_are_not_merged() -> None:
    """The failure this rule exists for. On a real document the model proposed folding
    Marketplace into Seller and Parcel into Purchase — relationships, not identities. Both pairs
    were of comparable coverage, which is the shape of two concepts rather than one split in
    two, and a marketplace's sellers and its shoppers appear together in every sentence about a
    sale precisely because they are opposites."""
    namer = ClusterNamer(client=_Namer(merge=[["Marketplace", "Seller"]]), model="test")
    marketplace = Concept(
        name="Marketplace", terms=("marketplace",), occurrences=17, evidence="marketplace"
    )
    seller = Concept(name="Seller", terms=("seller",), occurrences=30, evidence="seller")
    kept, merged = namer.merge([marketplace, seller])
    assert {c.name for c in kept} == {"Marketplace", "Seller"}
    assert merged == []
    assert namer.refused_merges == [("Marketplace", "Seller")]  # declined, and said so


def test_a_concept_paired_with_itself_is_not_a_merge() -> None:
    """Asked which concepts are the same, the model returned every concept paired with itself —
    thirty-four groups of the form [X, X]. The size rule refused them all, since a thing is
    exactly as large as itself, but being right by accident is not being right."""
    namer = ClusterNamer(client=_Namer(merge=[["Seller", "Seller"]]), model="test")
    seller = Concept(name="Seller", terms=("seller",), occurrences=30, evidence="seller")
    other = Concept(name="Parcel", terms=("parcel",), occurrences=8, evidence="parcel")
    kept, merged = namer.merge([seller, other])
    assert {c.name for c in kept} == {"Seller", "Parcel"}
    assert merged == []
    assert namer.refused_merges == []  # nothing was refused: there was nothing to refuse
