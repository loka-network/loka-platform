"""Discover entity types bottom-up, from the nouns the document actually contains.

This is the concept-discovery half of LLM4Onto (Ouyang et al., Semantic Web Journal),
implemented as the paper describes it:

    noun phrases (spaCy) → drop the semantically empty ones → sentence embeddings
    → affinity propagation → drop clusters below a size floor
    → the model NAMES each cluster → merge clusters that turn out to be one concept

The direction is what matters, and it is the opposite of asking a model for an ontology. There,
the concepts come from the model and the text is evidence; a concept the model does not think of
is invisible, and nothing downstream can notice the omission. Here the candidates come from the
document — every repeated noun phrase in it is one — and the model's only job is to say what a
group of them is a kind of. It cannot introduce a concept, and it cannot skip a frequent one.

The paper's stated purpose for this is completeness: the clustering "ensures that the framework
does not omit any entity-type corresponding ontologies present in the text". That is a different
guarantee from the one a citation check gives. A citation check catches invention; only reading
the text exhaustively catches omission.

Affinity propagation is used rather than k-means because the number of concepts in a domain is
not known before the domain is analysed, and it does not need one. Its ``preference`` is set to
favour many small clusters, following the paper: over-splitting is repaired by the naming and
merging steps, whereas two distinct concepts merged into one cluster are lost.

Requires the ``[discovery]`` extra: spaCy (with a model), sentence-transformers, scikit-learn.
Everything else in this package runs without them.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

#: Clusters smaller than this are dropped. The paper's rationale: a cluster has to hold enough
#: members for the naming step to have something to abstract from, and sparse clusters are mostly
#: noise. It is a floor on evidence, not on importance — see ``ConceptDiscovery.rare_terms``.
MIN_CLUSTER_SIZE = 2

#: Minimum occurrences for a term to enter clustering. One, i.e. no floor, following the paper:
#: its only threshold is on cluster size, "only clusters with a number of entities not less than
#: n are retained". Filtering terms first is worse than it sounds — a concept mentioned once
#: under each of six wordings has six terms and no frequent one, and dropping them all removes
#: the concept while the counts still look reasonable. On our document a floor of 2 left 31
#: terms and 6 usable clusters; no floor leaves 158 and 39, and the extra clusters are real
#: (carrier, delivery, route, threshold).
MIN_TERM_OCCURRENCES = 1

#: How much smaller a concept must be than the one it is folded into.
#:
#: A threshold fitted to observed failures rather than derived, and the refusals are reported so
#: a reviewer can overrule it. The reasoning behind the shape: a merge repairs a split, the
#: split-off piece is the smaller one, and two concepts of similar weight are not that. The
#: value separates the merges that were right (Buyer at a fifth of Shopper) from the ones that
#: were not (Marketplace at over half of Seller, Parcel at over half of Purchase).
_MERGE_FRAGMENT_RATIO = 0.4

#: Quantile of the observed similarities used as affinity propagation's ``preference``.
#:
#: ``preference`` is each point's self-similarity — how readily it becomes its own exemplar — so
#: it has to sit on the same scale as the similarity matrix, and that scale depends on the
#: document. A constant cannot: at -0.35, below every similarity these terms produce, nothing
#: becomes an exemplar and the whole vocabulary collapses into one cluster. Taking a quantile of
#: the actual similarities makes it adapt.
#:
#: The value is a tuning choice, not a derivation. The paper favours finer clusters — splitting
#: is repaired by naming and merging, fusion is not — but pushed far enough the split stops
#: following meaning: at 0.9 on an eight-term example "package" clusters with "monday". 0.6 was
#: picked by reading the output on two inputs of very different size, where it grouped seller
#: variants, item variants, and delivery vocabulary correctly in both. Expect to revisit it on a
#: corpus rather than a document.
CLUSTER_PREFERENCE_QUANTILE = 0.6

#: Noun phrases that are grammatical scaffolding rather than domain vocabulary. spaCy's chunker
#: returns "it", "that", "the same thing"; the paper notes these have to go before abstraction,
#: and that such a lexicon is easy to build.
_EMPTY = frozenset({
    "it", "that", "this", "these", "those", "which", "what", "who", "whom", "whose",
    "they", "them", "we", "us", "you", "he", "she", "him", "her", "i", "me",
    "one", "ones", "someone", "something", "anything", "nothing", "everything",
    "the same thing", "the same", "the other", "the others", "each other",
    "the first", "the second", "the third", "the rest", "the way", "the case",
    "the point", "the reason", "the thing", "the fact", "the part", "the number",
    "the following", "the above", "the below", "the latter", "the former",
    "the whole", "all", "both", "any", "some", "many", "most", "few", "several",
    "example", "examples", "none", "no one", "anyone", "everyone", "nobody",
    "kind", "kinds", "sort", "sorts", "lot", "lots", "bit", "piece",
})

_DETERMINERS = re.compile(r"^(the|a|an|this|that|these|those|its|their|our|your|his|her)\s+")


class DiscoveryUnavailable(RuntimeError):
    """The optional dependencies for bottom-up discovery are not installed."""


@dataclass
class Concept:
    """One named cluster: what the model called it, and the terms it was named from."""

    name: str
    terms: tuple[str, ...]
    occurrences: int
    evidence: str = ""  # the most frequent term, which is the phrase in the text

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "terms": list(self.terms),
            "occurrences": self.occurrences,
            "evidence": self.evidence,
        }


@dataclass
class ConceptDiscovery:
    """What was found in the document, and what was set aside on the way."""

    concepts: list[Concept] = field(default_factory=list)
    #: terms that occurred too rarely to be candidates
    rare_terms: list[tuple[str, int]] = field(default_factory=list)
    #: clusters dropped for being below the size floor, with their terms
    small_clusters: list[tuple[str, ...]] = field(default_factory=list)
    #: clusters the merge step judged to be one concept
    merged: list[tuple[str, ...]] = field(default_factory=list)
    #: merges the model proposed and the size rule declined
    refused_merges: list[tuple[str, ...]] = field(default_factory=list)
    terms_extracted: int = 0
    clusters_formed: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "terms_extracted": self.terms_extracted,
            "clusters_formed": self.clusters_formed,
            "concepts": [c.as_dict() for c in self.concepts],
            "dropped_small_clusters": [list(c) for c in self.small_clusters],
            "rare_terms": [{"term": t, "occurrences": n} for t, n in self.rare_terms[:40]],
            "merged": [list(m) for m in self.merged],
            "refused_merges": [list(m) for m in self.refused_merges],
        }


def _concepts_in(
    item: dict[str, Any], cluster: tuple[str, ...]
) -> list[tuple[str, tuple[str, ...]]]:
    """(name, phrases) per concept named from one cluster, in three reply shapes.

    Which phrases belong to which concept is the part that matters, and it was missing at first:
    every concept split out of a cluster was given the whole cluster's phrases, so three
    concepts arrived at the merge step carrying identical evidence and it judged them the same
    thing — correctly, from what it could see. Splitting a cluster without dividing its phrases
    replaces one failure with another.

    Phrases are kept only if the cluster actually holds them. A phrase invented here would be
    evidence that cites nothing, which the grounding check would report as a concept absent from
    the source — an accusation against the model for something this parser did.
    """
    raw = item.get("concepts")
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        out = []
        for entry in raw:
            name = re.sub(r"[^A-Za-z0-9]", "", str(entry.get("name") or ""))
            phrases = tuple(
                p for p in (entry.get("phrases") or []) if isinstance(p, str) and p in cluster
            )
            if name:
                out.append((name, phrases))
        return out

    # Older shapes, kept because a reply is not a contract: a list of bare names, or one name.
    names = item.get("names") if isinstance(item.get("names"), list) else None
    if names is None:
        names = [item["name"]] if item.get("name") else []
    return [
        (cleaned, ())
        for n in names
        if (cleaned := re.sub(r"[^A-Za-z0-9]", "", str(n or "")))
    ]


def _require(module: str, hint: str) -> Any:
    try:
        return __import__(module)
    except ImportError as exc:
        raise DiscoveryUnavailable(
            f"bottom-up concept discovery needs {module!r}: {hint}"
        ) from exc


def extract_terms(text: str, *, min_occurrences: int = MIN_TERM_OCCURRENCES) -> Counter[str]:
    """Nouns and noun phrases in the text, minus the semantically empty ones, counted.

    Both, not either. A chunker returns "the same item", "a bulky item", "some items" — three
    strings for one word, so counting phrases alone splits a document's central vocabulary into
    fragments that each fall under the frequency floor. "item" occurs six times in our document
    and phrase-counting found it once. So each chunk contributes its head noun (lemmatised, which
    also handles the plural) as well as the phrase itself.

    Leading determiners are stripped so "the shopper" and "a shopper" are one term. Counting is
    what makes a frequency floor possible, and that floor is what separates a domain's vocabulary
    from its incidentals.
    """
    spacy = _require("spacy", "pip install 'loka-ontology[discovery]' and a spaCy model")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError as exc:
        raise DiscoveryUnavailable(
            "spaCy model 'en_core_web_sm' is not installed: "
            "python -m spacy download en_core_web_sm"
        ) from exc

    counts: Counter[str] = Counter()
    for chunk in nlp(text).noun_chunks:
        head = chunk.root.lemma_.strip().lower()
        # Lemmatise the head inside the phrase too, rather than stripping a trailing "s" by
        # hand: "this" is not a plural, and a rule that says otherwise turns it into "thi" and
        # slips it past a stop list that was watching for "this".
        # Compared by index, not identity: spaCy builds Token objects on demand, so
        # ``t is chunk.root`` is False even for the same token and the head silently keeps its
        # plural — "sellers" and "seller" then count as two terms for one word.
        words = [
            (t.lemma_ if t.i == chunk.root.i else t.text).strip().lower()
            for t in chunk
            if not t.is_punct and t.text.strip()
        ]
        phrase = _DETERMINERS.sub("", " ".join(words))
        phrase = re.sub(r"[^a-z0-9 \-]", "", phrase).strip()
        raw = chunk.text.strip().lower()
        if raw in _EMPTY or _DETERMINERS.sub("", raw) in _EMPTY:
            continue
        # A set, because a one-word chunk has the same phrase and head, and counting it twice
        # would inflate single-word terms against multi-word ones for no reason.
        for term in {phrase, head}:
            if not term or len(term) < 3 or term in _EMPTY:
                continue
            if all(token in _EMPTY for token in term.split()):
                continue
            counts[term] += 1
    return Counter({t: n for t, n in counts.items() if n >= min_occurrences})


def cluster_terms(
    terms: Sequence[str], *, quantile: float = CLUSTER_PREFERENCE_QUANTILE
) -> list[list[str]]:
    """Group semantically similar terms. Affinity propagation, so the count is not supplied.

    A domain's concept count is exactly what is being discovered, which rules out any algorithm
    that has to be told it. The preference is set toward finer clusters because the naming and
    merging steps can repair a split concept, and nothing can repair two concepts fused into one.
    """
    if len(terms) < 2:
        return [[t] for t in terms]
    st = _require(
        "sentence_transformers", "pip install 'loka-ontology[discovery]'"
    )
    sk = _require("sklearn", "pip install 'loka-ontology[discovery]'")
    import numpy as np  # noqa: PLC0415 - arrives with the optional extra
    from sklearn.cluster import AffinityPropagation  # noqa: PLC0415 - optional extra

    model = st.SentenceTransformer("all-MiniLM-L6-v2")
    vectors = model.encode(list(terms), normalize_embeddings=True)
    similarity = vectors @ vectors.T  # cosine, since the vectors are unit length

    off_diagonal = similarity[~np.eye(len(terms), dtype=bool)]
    preference = float(np.quantile(off_diagonal, quantile))
    ap = AffinityPropagation(
        affinity="precomputed", preference=preference, random_state=0, max_iter=400
    )
    labels = ap.fit_predict(similarity)
    del sk  # imported to fail early with a clear message; the estimator is what is used

    grouped: dict[int, list[str]] = {}
    for term, label in zip(terms, labels, strict=True):
        grouped.setdefault(int(label), []).append(term)
    return [grouped[k] for k in sorted(grouped)]


class ClusterNamer:
    """The model's role here: name a group of terms, and say when two groups are one concept.

    Deliberately the whole of its role. It is never asked which concepts the domain has — that
    was decided by what the document contains — so it has no opening to supply one from
    elsewhere. Naming is also the task it is best at, and the cheapest to check: the name has to
    be a plausible generalisation of terms a reader can see.
    """

    NAME = (
        "You are naming concepts for a domain ontology in the {domain} domain. Each group below "
        "is a set of phrases taken from one document. Give the entity types the phrases are "
        "instances or aspects of. Use the domain's own naming style, CamelCase, singular.\n"
        "A group may hold more than one concept — clustering put the phrases together by "
        "similarity, which is not the same as their being one kind of thing. When it does, name "
        "every concept in it, and say which phrases belong to which. Do not choose one and drop "
        "the rest. Every phrase you assign must come from that group; a phrase that belongs to "
        "none of the concepts you name is left out.\n"
        "Return an empty list for a group that names nothing: phrases too mixed to be one kind "
        "of thing, or about the document rather than the domain. That this is a briefing or a "
        "manual, who wrote it and who it is for are not part of the {domain} domain.\n"
        "Give a short reason for each group, in `why`. A group you cannot write a reason for is "
        "one to return empty.\n"
        'Reply with ONLY JSON: {{"names": [{{"group": <index>, "concepts": '
        '[{{"name": <CamelCase>, "phrases": [<phrase from this group>, ...]}}, ...], '
        '"why": <one short line>}}]}}. No prose, no code fences.\n\n'
        "Examples — ['cat', 'dog', 'horse'] gives one concept, Animal, holding all three. "
        "['shopper', 'business', 'marketplace', 'online marketplace'] gives two: Shopper from "
        "['shopper'], and Marketplace from ['marketplace', 'online marketplace'] — a buyer and "
        "the platform they buy through are two kinds of thing that happen to be discussed "
        "together, and 'business' belongs to neither. "
        "['last part', 'part', 'fact', 'case'] gives none: these are ways of referring to parts "
        "of the text, not kinds of thing the domain has."
    )
    MERGE = (
        "These are named concepts from one document, each with the phrases it was named from. "
        "Clustering deliberately over-splits, so ONE concept may appear twice under different "
        "names — for example Buyer and Purchaser. Your job is to find only that: two names for "
        "the same thing.\n"
        "Do NOT group concepts that are merely related, that interact, or that belong to the "
        "same part of the domain. A seller and a shopper both appear in every sale and are not "
        "the same concept. A parcel and the purchase it belongs to are not the same concept. "
        "If in doubt, do not group.\n"
        "Every name below is distinct: never pair a name with itself.\n"
        'Reply with ONLY JSON: {"same": [[<name>, <name>], ...]}. An empty list is the expected '
        "answer for most documents, and is the right answer when you are unsure. "
        "No prose, no code fences."
    )

    def __init__(
        self,
        *,
        client: Any,
        model: str = "claude-opus-4-8",
        domain: str = "business",
        max_tokens: int = 16000,
    ) -> None:
        self._client = client
        self._model = model
        self._domain = domain
        self._max_tokens = max_tokens
        self.calls: list[dict[str, Any]] = []
        #: merges declined for joining two of the document's principal concepts
        self.refused_merges: list[tuple[str, ...]] = []

    def _ask(self, stage: str, system: str, user: str) -> dict[str, Any]:
        import time  # noqa: PLC0415 - kept next to its single use

        from .builder import EXTRACTION_TEMPERATURE  # noqa: PLC0415 - avoids a cycle
        from .staged_builder import _json_object  # noqa: PLC0415 - avoids a cycle at import

        started = time.monotonic()
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=EXTRACTION_TEMPERATURE,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        elapsed = time.monotonic() - started
        text = "".join(
            getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text"
        )
        self.calls.append({
            "stage": stage,
            "reply_chars": len(text),
            "seconds": round(elapsed, 1),
            "requests": getattr(resp, "requests", 1),
            "budget": getattr(resp, "budget", None),
        })
        return _json_object(text, stage)

    def name(self, clusters: Sequence[Sequence[str]], counts: Counter[str]) -> list[Concept]:
        listing = "\n".join(
            f"{i}: {list(c)}" for i, c in enumerate(clusters)
        )
        obj = self._ask("naming", self.NAME.format(domain=self._domain), listing)
        named: list[Concept] = []
        for item in obj.get("names", []):
            if not isinstance(item, dict):
                continue
            index = item.get("group")
            if not isinstance(index, int) or not (0 <= index < len(clusters)):
                continue

            # A group may name more than one concept. Clustering groups phrases by similarity,
            # which is not the same as their being one kind of thing: on this document it put
            # 'shopper' with 'marketplace', and a reply shaped to hold one name per group could
            # only answer Marketplace. The buyer then vanished from the ontology, and nothing
            # reported it — not the checklist, not the grounding list. An answer that cannot
            # express the case does not fail, it silently picks.
            cluster = tuple(clusters[index])
            for name, phrases in _concepts_in(item, cluster):
                terms = phrases or cluster
                named.append(
                    Concept(
                        name=name,
                        terms=terms,
                        occurrences=sum(counts.get(t, 0) for t in terms),
                        evidence=max(terms, key=lambda t: counts.get(t, 0)),
                    )
                )
        return [c for c in named if c.name]

    def merge(self, concepts: Sequence[Concept]) -> tuple[list[Concept], list[tuple[str, ...]]]:
        """Fold concepts the model judges to be one. Repairs the over-splitting the clustering
        preference deliberately allows."""
        if len(concepts) < 2:
            return list(concepts), []
        listing = "\n".join(f"{c.name}: {list(c.terms)}" for c in concepts)
        obj = self._ask("merging", self.MERGE, listing)
        by_name = {c.name: c for c in concepts}
        merged: list[tuple[str, ...]] = []
        refused: list[tuple[str, ...]] = []
        dropped: set[str] = set()
        # Merging exists to repair over-splitting: to rejoin a fragment to the concept it was
        # split from. A fragment is by nature the smaller piece, so two concepts of comparable
        # coverage are two concepts, not one said twice. Asked for duplicates a model reaches
        # instead for whatever co-occurs — on a real document it merged Marketplace into Seller
        # and Parcel into Purchase, which are relationships, not identities.
        share = {c.name: c.occurrences for c in concepts}
        for group in obj.get("same", []) or []:
            names = [str(n) for n in group if str(n) in by_name] if isinstance(group, list) else []
            # Asked which concepts are the same, the model returned every concept paired with
            # itself — thirty-four groups of the form [X, X]. The size rule happened to refuse
            # them all, since a thing is exactly as large as itself, but being right by accident
            # is not being right. Duplicates within a group are dropped before anything else.
            names = list(dict.fromkeys(names))
            if len(names) < 2:
                continue
            weights = [share[n] for n in names]
            if min(weights) >= _MERGE_FRAGMENT_RATIO * max(weights):
                refused.append(tuple(names))
                continue
            # Keep the one that covers the most of the document; it is the reading the text
            # supports best, and the alternative — keeping the first — is arbitrary.
            keeper = max(names, key=lambda n: by_name[n].occurrences)
            terms = tuple(dict.fromkeys(t for n in names for t in by_name[n].terms))
            by_name[keeper] = Concept(
                name=keeper,
                terms=terms,
                occurrences=sum(by_name[n].occurrences for n in names),
                evidence=by_name[keeper].evidence,
            )
            dropped.update(n for n in names if n != keeper)
            merged.append(tuple(names))
        self.refused_merges = refused
        return [c for n, c in by_name.items() if n not in dropped], merged


def discover_concepts(
    text: str,
    namer: ClusterNamer,
    *,
    min_occurrences: int = MIN_TERM_OCCURRENCES,
    min_cluster_size: int = MIN_CLUSTER_SIZE,
    quantile: float = CLUSTER_PREFERENCE_QUANTILE,
) -> ConceptDiscovery:
    """The full bottom-up pass. Everything set aside on the way is returned, not discarded.

    What was dropped is as much a finding as what was kept: a reviewer disagreeing with the
    result needs to see the frequent term that formed no cluster and the cluster that fell under
    the size floor, or they can only accept the output as given.
    """
    counts = extract_terms(text, min_occurrences=min_occurrences)
    result = ConceptDiscovery(terms_extracted=len(counts))
    if not counts:
        return result

    all_terms = [t for t, _ in counts.most_common()]
    clusters = cluster_terms(all_terms, quantile=quantile)
    result.clusters_formed = len(clusters)

    keep = [c for c in clusters if len(c) >= min_cluster_size]
    result.small_clusters = [tuple(c) for c in clusters if len(c) < min_cluster_size]
    # A term that formed no usable cluster is still in the document. Reported so an omission can
    # be seen rather than inferred from its absence.
    result.rare_terms = [
        (t, counts[t]) for c in result.small_clusters for t in c
    ]
    result.rare_terms.sort(key=lambda pair: -pair[1])
    if not keep:
        return result

    named = namer.name(keep, counts)
    result.concepts, result.merged = namer.merge(named)
    result.refused_merges = list(namer.refused_merges)
    result.concepts.sort(key=lambda c: -c.occurrences)
    return result


class ClusterFirstBuilder:
    """LLM4Onto's direction: the document proposes the concepts, the model only names them.

    Every other builder here asks a model which entity types the domain has. That question can be
    answered from what a model knows about domains in general, so an omission is undetectable —
    nothing downstream can miss what was never mentioned. This one takes the candidates from the
    text: each noun phrase in the document is one, they are clustered by meaning, and the model's
    role is reduced to naming a group and saying when two groups are the same concept.

    Relations, attributes and act classes then run exactly as in the staged builder, so the
    difference between this and that one is the concept step and nothing else.
    """

    paradigm = "cluster_first"

    def __init__(
        self,
        *,
        client: Any,
        model: str = "claude-opus-4-8",
        domain: str = "business",
        max_tokens: int = 16000,
    ) -> None:
        from .staged_builder import StagedLLMBuilder  # noqa: PLC0415 - avoids an import cycle

        self._namer = ClusterNamer(client=client, model=model, domain=domain)
        self._stages = StagedLLMBuilder(client=client, model=model, max_tokens=max_tokens)
        self.discovery: ConceptDiscovery | None = None

    @property
    def model(self) -> str:
        return self._stages.model

    @property
    def grounding(self) -> Any:
        return self._stages.grounding

    @property
    def stage_calls(self) -> list[dict[str, Any]]:
        return self._namer.calls + self._stages.stage_calls

    @property
    def notes(self) -> dict[str, Any]:
        out: dict[str, Any] = dict(self._stages.notes)
        if self.discovery is not None:
            out["discovery"] = self.discovery.as_dict()
        return out

    @property
    def system_prompt(self) -> str:
        return (
            f"[cluster naming] {self._namer.NAME}\n\n---\n\n"
            f"[cluster merging] {self._namer.MERGE}\n\n---\n\n"
            + self._stages.system_prompt
        )

    def propose(self, texts: Sequence[str]) -> Any:
        from .builder import EntityDraft, OntologyBuildError  # noqa: PLC0415 - import cycle
        from .staged_builder import OntologyDraft  # noqa: PLC0415 - import cycle

        source = "\n".join(texts)
        self.discovery = discover_concepts(source, self._namer)
        if not self.discovery.concepts:
            raise OntologyBuildError(
                "clustering the document's noun phrases produced no nameable concept; "
                f"{self.discovery.terms_extracted} terms formed "
                f"{self.discovery.clusters_formed} clusters"
            )

        entities = [EntityDraft(name=c.name) for c in self.discovery.concepts]
        # Every concept here came from the text by construction, so the citation check has
        # nothing to catch — it is recorded anyway, so the grounding rate stays comparable
        # across paradigms rather than being absent for this one.
        for c in self.discovery.concepts:
            self._stages.grounding.note(c.name, c.evidence, source)

        attrs = self._stages._attributes(source, entities)
        entities = [
            EntityDraft(name=e.name, subtype_of=None, attributes=attrs.get(e.name, ()))
            for e in entities
        ]
        relations = self._stages._relations(source, entities)
        verbs = self._stages._verbs(relations)
        names = tuple(e.name for e in entities)
        return OntologyDraft(
            entities=tuple(entities),
            relations=relations,
            verbs=verbs,
            data_needs=names,
            method_needs=(),
            facets={"factual": names, "cognitive": (), "communication": ()},
        )
