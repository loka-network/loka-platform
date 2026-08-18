"""Four ways to get an ontology out of a document, and one check applied to all of them.

Ouyang et al. (LLM4Onto, Semantic Web Journal) separate three paradigms for building an ontology
from raw text. The distinction is entirely about how relations are obtained; the concepts can be
asked for the same way in each.

    T-O          the whole ontology in one reply. Simple, and the reply is what runs out of
                 budget on a domain of any size — splitting the *document* does not help, since
                 it is the *answer* that is too long.
    R-O          two passes: first ask which relation types the document uses at all, then
                 generate triples restricted to that vocabulary.
    HT-R-O       work at the level of instances. Map each mention to a type, extract triples
                 between mentions from short fragments, then lift them to type level through the
                 mapping. A type-level relation exists because instances of it were found, not
                 because it was asked for.

This module implements the last three (the first is ``LLMBuilder``, one call), sharing the stages
that do not differ so a comparison between them is a comparison of the paradigm rather than of
four separately-written programs:

    entities → attributes (batched) → RELATIONS (differs) → verb act-classes

Applied to all of them: every proposal must return the span of source text it was read from, and
the span is then looked for in the source. The paper prevents invention by construction — it
clusters noun phrases actually present in the text and asks the model only to name the clusters,
at the cost of a POS tagger, an embedding model and a clustering pass. This takes the same
principle from the other end, which suits a system whose posture is that model output is checked
rather than trusted. Detection instead of prevention, and the disagreement is kept either way:
an unfound name may be a correct generalisation, and deleting it would replace a visible
disagreement with a silent one.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from .builder import _BASE_TYPE, _VERB_CLASS, EntityDraft, OntologyBuildError, OntologyDraft

#: Entities per call in the attribute stage.
#:
#: Sized from a measurement rather than caution. On a 4.8k-character document the longest reply
#: any stage produced was 6,212 characters — roughly 1,550 tokens against a 32,000 ceiling, so
#: twenty times the headroom. At six per batch a twenty-type ontology cost four attribute calls,
#: and at ~200s per call with a reasoning model those four are most of the wall clock: the
#: paradigms that add stages on top of them timed out at thirty minutes. Batching wider trades
#: headroom nobody was using for calls that were the actual cost.
_ATTR_BATCH = 12

#: Independent calls run at once, up to this many. Attribute batches do not depend on each
#: other and neither do fragments, yet they were issued one at a time — so a paradigm's wall
#: clock was the sum of calls that could all have been in flight together. These are HTTP
#: requests waiting on a model, so threads are the right tool and the bound exists to stay
#: polite to the endpoint rather than to protect anything here.
_MAX_CONCURRENCY = 4

#: Target size of a fragment in the instance stage. The paper notes the extraction "performs
#: this task more effectively in short text fragments"; paragraph boundaries are used so a
#: fragment is a unit of meaning rather than a fixed number of characters.
#:
#: Raised from 900 for the same reason as the attribute batch: at 900 our document became seven
#: fragments, seven calls, and the paradigm did not finish inside half an hour. "Short" here is
#: relative to a whole document, and 1,800 characters is still one or two sections.
_FRAGMENT_CHARS = 1800

_WORD = re.compile(r"[a-z0-9]+")


@dataclass
class Grounding:
    """Which proposals were found in the source text and which were not.

    A record, not a filter. A name absent from the text is not thereby wrong: "Shopper" for a
    document that says "customer", "PurchaseLine" for one that says "each line of a purchase".
    What is unacceptable is the two being indistinguishable in the result.
    """

    grounded: dict[str, str] = field(default_factory=dict)  # proposal -> evidence found
    ungrounded: dict[str, str] = field(default_factory=dict)  # proposal -> evidence claimed

    def note(self, proposal: str, evidence: str, source: str) -> bool:
        ok = _appears_in(evidence, source) or _appears_in(proposal, source)
        (self.grounded if ok else self.ungrounded)[proposal] = evidence
        return ok

    def as_dict(self) -> dict[str, Any]:
        total = len(self.grounded) + len(self.ungrounded)
        return {
            "checked": total,
            "grounded": len(self.grounded),
            "ungrounded": sorted(self.ungrounded),
            "rate": round(len(self.grounded) / total, 4) if total else 0.0,
            "evidence": dict(self.grounded),
        }


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _appears_in(phrase: str, source: str) -> bool:
    """Whether a phrase occurs in the source, allowing for the ways a name gets written.

    Compared on word tokens rather than characters, so CamelCase, snake_case and spacing do not
    decide it: "PurchaseLine" is looked for as "purchase line", and a trailing plural is
    tolerated. Deliberately no fuzzier — a check that matches anything reports everything as
    grounded and certifies nothing.
    """
    if not phrase:
        return False
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", phrase)
    want = _tokens(spaced)
    if not want:
        return False
    have = _tokens(source)

    def same(a: str, b: str) -> bool:
        return a == b or a.rstrip("s") == b.rstrip("s")

    n = len(want)
    return any(
        all(same(want[j], have[i + j]) for j in range(n)) for i in range(len(have) - n + 1)
    )


def _fragments(source: str, size: int = _FRAGMENT_CHARS) -> list[str]:
    """Split on paragraph boundaries, packing up to ``size``. A fragment that is a paragraph is
    a unit of meaning; one that is a fixed slice cuts sentences in half."""
    out: list[str] = []
    current = ""
    for para in re.split(r"\n\s*\n", source.strip()):
        para = para.strip()
        if not para:
            continue
        if current and len(current) + len(para) + 2 > size:
            out.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        out.append(current)
    return out or [source]


def _json_object(text: str, stage: str) -> dict[str, Any]:
    start, end = text.find("{"), text.rfind("}")
    if start == -1:
        raise OntologyBuildError(f"stage {stage!r} returned no JSON object: {text[:200]!r}")
    candidate = text[start : end + 1] if end > start else text[start:]
    try:
        obj: Any = json.loads(candidate)
    except json.JSONDecodeError as exc:
        cut = candidate.count("{") > candidate.count("}")
        raise OntologyBuildError(
            (f"stage {stage!r} was cut off mid-JSON" if cut
             else f"stage {stage!r} returned malformed JSON")
            + f" at char {exc.pos} of {len(candidate)} ({exc.msg}). Ends: ...{candidate[-160:]!r}"
        ) from exc
    if not isinstance(obj, dict):
        raise OntologyBuildError(f"stage {stage!r} returned {type(obj).__name__}, not an object")
    return obj


class _StagedBuilder:
    """Shared stages. Subclasses differ only in how relations are obtained."""

    #: name of the paradigm, recorded on the draft
    paradigm = "staged"

    ENTITIES = (
        "List the entity types a domain ontology for this text needs. An entity type is a kind "
        "of thing the domain has many of, not a one-off. Reply with ONLY JSON: "
        '{"entities": [{"name": <CamelCase>, "subtype_of": <name|null>, '
        '"evidence": <the exact phrase from the text that denotes it>}]}. '
        "The evidence must be copied from the text, not paraphrased. If a type is implied rather "
        "than named, give the phrase that implies it. No prose, no code fences."
    )
    ATTRIBUTES = (
        "For each entity type listed, give the attributes the text states it has. Reply with "
        'ONLY JSON: {"attributes": {"<EntityType>": [{"name": <snake_case>, '
        '"type": "string|integer|double|boolean|timestamp|date", '
        '"evidence": <the exact phrase from the text>}]}}. '
        "Only attributes the text actually mentions. An empty list is a valid answer. "
        "No prose, no code fences."
    )
    VERBS = (
        "Classify each relation verb by act class. factual = a change or fact in the objective "
        "world; communicative = a message between parties; institutional = an act that changes "
        'standing or authority by declaration. Reply with ONLY JSON: {"verbs": [[<VERB>, '
        '"factual|communicative|institutional"]]}. No prose, no code fences.'
    )

    def __init__(
        self, *, client: Any, model: str = "claude-opus-4-8", max_tokens: int = 4000
    ) -> None:
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self.grounding = Grounding()
        self.stage_calls: list[dict[str, Any]] = []
        #: paradigm-specific findings worth reporting alongside the draft
        self.notes: dict[str, Any] = {}

    @property
    def model(self) -> str:
        return self._model

    def _prompts(self) -> list[tuple[str, str]]:
        return [("entities", self.ENTITIES), ("attributes", self.ATTRIBUTES)]

    @property
    def system_prompt(self) -> str:
        """Every instruction this builder sends, in order. What ran, for the record."""
        return "\n\n---\n\n".join(f"[{name}] {p}" for name, p in self._prompts())

    def _ask_many(
        self, jobs: Sequence[tuple[str, str, str]]
    ) -> list[dict[str, Any] | Exception]:
        """Issue independent calls together, keeping the order of ``jobs``.

        An exception is returned rather than raised so one bad batch does not discard the others:
        a stage that failed on batch three still has one and two, and dropping them would cost a
        round trip each to learn nothing new.
        """
        if len(jobs) == 1:
            try:
                return [self._ask(*jobs[0])]
            except Exception as exc:  # noqa: BLE001 - handed back to the caller to decide
                return [exc]
        out: list[dict[str, Any] | Exception] = [ValueError("not run")] * len(jobs)
        with ThreadPoolExecutor(max_workers=min(_MAX_CONCURRENCY, len(jobs))) as pool:
            futures = {pool.submit(self._ask, *job): i for i, job in enumerate(jobs)}
            for future in as_completed(futures):
                index = futures[future]
                try:
                    out[index] = future.result()
                except Exception as exc:  # noqa: BLE001 - see above
                    out[index] = exc
        return out

    def _ask(self, stage: str, system: str, user: str) -> dict[str, Any]:
        # Timed per stage. Wall clock is what decides whether a paradigm is usable at all — two
        # of them could not finish inside half an hour — and a total tells you that without
        # telling you which stage to shorten.
        started = time.monotonic()
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        elapsed = time.monotonic() - started
        text = "".join(
            getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text"
        )
        self.stage_calls.append(
            {"stage": stage, "reply_chars": len(text), "seconds": round(elapsed, 1)}
        )
        return _json_object(text, stage)

    # ---- shared stages ----

    def _entities(self, source: str) -> list[EntityDraft]:
        obj = self._ask("entities", self.ENTITIES, source)
        out: list[EntityDraft] = []
        seen: set[str] = set()
        for e in obj.get("entities", []):
            if not isinstance(e, dict):
                continue
            name = str(e.get("name") or "").strip()
            if not name or name in seen:
                continue
            seen.add(name)
            self.grounding.note(name, str(e.get("evidence") or ""), source)
            out.append(EntityDraft(name=name, subtype_of=e.get("subtype_of") or None))
        # A subtype pointing at a type that was not proposed cannot load, and dropping the link
        # keeps the type: losing "BulkyItem is a kind of Item" beats losing BulkyItem.
        names = {e.name for e in out}
        return [
            EntityDraft(name=e.name, subtype_of=e.subtype_of if e.subtype_of in names else None)
            for e in out
        ]

    def _attributes(
        self, source: str, entities: list[EntityDraft]
    ) -> dict[str, tuple[tuple[str, str], ...]]:
        found: dict[str, tuple[tuple[str, str], ...]] = {}
        batches = [entities[i : i + _ATTR_BATCH] for i in range(0, len(entities), _ATTR_BATCH)]
        replies = self._ask_many([
            (
                f"attributes[{i}]",
                self.ATTRIBUTES,
                f"Entity types: {', '.join(e.name for e in batch)}\n\nText:\n{source}",
            )
            for i, batch in enumerate(batches)
        ])
        for batch, obj in zip(batches, replies, strict=True):
            if isinstance(obj, Exception):
                # Attributes are optional in a draft; a batch that failed leaves its types
                # without any, which the review checklist already knows how to report.
                self.notes.setdefault("failed_stages", []).append(f"attributes: {obj}")
                continue
            in_batch = {e.name for e in batch}
            for entity, attrs in (obj.get("attributes") or {}).items():
                if entity not in in_batch or not isinstance(attrs, list):
                    continue
                collected: list[tuple[str, str]] = []
                seen: set[str] = set()
                for a in attrs:
                    if not isinstance(a, dict) or not a.get("name"):
                        continue
                    aname = re.sub(r"[^A-Za-z0-9_]", "_", str(a["name"])).strip("_")
                    if not aname or aname in seen:
                        continue
                    seen.add(aname)
                    self.grounding.note(f"{entity}.{aname}", str(a.get("evidence") or ""), source)
                    collected.append(
                        (aname, _BASE_TYPE.get(str(a.get("type", "string")).lower(), "string"))
                    )
                found[entity] = tuple(collected)
        return found

    def _verbs(self, relations: Sequence[tuple[str, str, str]]) -> tuple[tuple[str, str], ...]:
        verbs = sorted({v for _, v, _ in relations})
        if not verbs:
            return ()
        obj = self._ask("verbs", self.VERBS, "Verbs: " + ", ".join(verbs))
        classified = {
            str(v[0]).upper(): (str(v[1]) if str(v[1]) in _VERB_CLASS else "factual")
            for v in obj.get("verbs", [])
            if isinstance(v, list) and len(v) == 2
        }
        # A verb the classifier skipped still has to exist or the relation using it cannot load.
        # Defaulting to factual is a decision, and the review checklist is where it surfaces.
        return tuple((v.upper(), classified.get(v.upper(), "factual")) for v in verbs)

    def _relations(
        self, source: str, entities: list[EntityDraft]
    ) -> tuple[tuple[str, str, str], ...]:
        raise NotImplementedError

    def _accept(self, src: str, verb: str, tgt: str, evidence: str, names: set[str],
                source: str, seen: set[str]) -> tuple[str, str, str] | None:
        """Shared admission for one proposed relation: endpoints must be types this run
        proposed, and the evidence must be findable."""
        if src not in names or tgt not in names or not verb:
            # Stage disagreement is information: this stage saw a type the entity stage did not.
            self.grounding.ungrounded[f"{src} -{verb}-> {tgt}"] = "endpoint not an entity type"
            return None
        key = f"{src} -{verb}-> {tgt}"
        if key in seen:
            return None
        seen.add(key)
        self.grounding.note(key, evidence, source)
        return (src, verb, tgt)

    def propose(self, texts: Sequence[str]) -> OntologyDraft:
        source = "\n".join(texts)
        entities = self._entities(source)
        if not entities:
            raise OntologyBuildError("stage 'entities' proposed no entity types")
        attrs = self._attributes(source, entities)
        entities = [
            EntityDraft(name=e.name, subtype_of=e.subtype_of, attributes=attrs.get(e.name, ()))
            for e in entities
        ]
        relations = self._relations(source, entities)
        verbs = self._verbs(relations)

        names = tuple(e.name for e in entities)
        return OntologyDraft(
            entities=tuple(entities),
            relations=relations,
            verbs=verbs,
            data_needs=names,
            method_needs=(),
            facets={"factual": names, "cognitive": (), "communication": ()},
        )


class StagedLLMBuilder(_StagedBuilder):
    """T-O, decomposed. The relations are asked for directly, at type level, with a citation.

    The paradigm is unchanged from the single-shot form — the model is still asked what relates
    to what — but the answer no longer has to carry the whole ontology, and each proposal names
    the sentence it came from.
    """

    paradigm = "staged"

    RELATIONS = (
        "Given these entity types, extract the relations the text states between them. Do not "
        "propose relations that would be reasonable in this domain — only ones this text asserts. "
        'Reply with ONLY JSON: {"relations": [{"from": <EntityType>, "verb": <lower_snake>, '
        '"to": <EntityType>, "evidence": <the sentence from the text that states it>}]}. '
        "The evidence must be a sentence copied from the text. No prose, no code fences."
    )

    def _prompts(self) -> list[tuple[str, str]]:
        return [
            ("entities", self.ENTITIES), ("attributes", self.ATTRIBUTES),
            ("relations", self.RELATIONS), ("verbs", self.VERBS),
        ]

    def _relations(
        self, source: str, entities: list[EntityDraft]
    ) -> tuple[tuple[str, str, str], ...]:
        names = {e.name for e in entities}
        obj = self._ask(
            "relations", self.RELATIONS,
            f"Entity types: {', '.join(sorted(names))}\n\nText:\n{source}",
        )
        out: list[tuple[str, str, str]] = []
        seen: set[str] = set()
        for r in obj.get("relations", []):
            if not isinstance(r, dict):
                continue
            triple = self._accept(
                str(r.get("from") or ""), str(r.get("verb") or ""), str(r.get("to") or ""),
                str(r.get("evidence") or ""), names, source, seen,
            )
            if triple:
                out.append(triple)
        return tuple(out)


class RelationFirstBuilder(_StagedBuilder):
    """R-O. Fix the relation vocabulary first, then extract only within it.

    Asked for relations freely, a model reaches for whichever verb the sentence in front of it
    suggests, so one relation arrives as ``lists`` here, ``offers`` there and ``sells`` in a
    third place. They are the same edge, and an ontology carrying all three has three ways to
    walk it and no way to know they agree.

    Deciding the vocabulary in its own pass and then constraining generation to it makes the
    normalisation a step with an output that can be inspected, rather than something a reviewer
    has to notice afterwards. What it costs: a distinction the vocabulary pass did not
    anticipate has nowhere to go, so the constrained pass will file it under the nearest
    available verb. Whatever it could not place is reported.
    """

    paradigm = "relation_first"

    VOCABULARY = (
        "List the distinct kinds of relation this text asserts between things. Merge wordings "
        "that mean the same thing into one. Reply with ONLY JSON: "
        '{"relations": [{"verb": <lower_snake>, "means": <one short line>, '
        '"evidence": <a phrase from the text using it>}]}. No prose, no code fences."'
    )
    RELATIONS = (
        "Extract the relations the text states between the given entity types, using ONLY the "
        "verbs from the supplied vocabulary. If the text asserts a relation that no supplied "
        'verb fits, put it in "unplaced" instead of forcing it. Reply with ONLY JSON: '
        '{"relations": [{"from": <EntityType>, "verb": <one of the vocabulary>, '
        '"to": <EntityType>, "evidence": <the sentence from the text>}], '
        '"unplaced": [{"description": <what the text asserts>, "evidence": <the sentence>}]}. '
        "No prose, no code fences."
    )

    def _prompts(self) -> list[tuple[str, str]]:
        return [
            ("entities", self.ENTITIES), ("attributes", self.ATTRIBUTES),
            ("vocabulary", self.VOCABULARY), ("relations", self.RELATIONS), ("verbs", self.VERBS),
        ]

    def _relations(
        self, source: str, entities: list[EntityDraft]
    ) -> tuple[tuple[str, str, str], ...]:
        vocab_obj = self._ask("vocabulary", self.VOCABULARY, source)
        vocabulary: list[str] = []
        for r in vocab_obj.get("relations", []):
            if isinstance(r, dict) and r.get("verb"):
                verb = re.sub(r"[^a-z0-9_]", "_", str(r["verb"]).lower()).strip("_")
                if verb and verb not in vocabulary:
                    vocabulary.append(verb)
                    self.grounding.note(
                        f"verb:{verb}", str(r.get("evidence") or ""), source
                    )
        if not vocabulary:
            raise OntologyBuildError("stage 'vocabulary' proposed no relation types")
        self.notes["vocabulary"] = vocabulary

        names = {e.name for e in entities}
        obj = self._ask(
            "relations", self.RELATIONS,
            f"Entity types: {', '.join(sorted(names))}\n"
            f"Relation vocabulary: {', '.join(vocabulary)}\n\nText:\n{source}",
        )
        allowed = set(vocabulary)
        out: list[tuple[str, str, str]] = []
        seen: set[str] = set()
        off_vocabulary: list[str] = []
        for r in obj.get("relations", []):
            if not isinstance(r, dict):
                continue
            verb = str(r.get("verb") or "")
            if verb not in allowed:
                # The constraint was not honoured. Kept as a finding rather than accepted: the
                # point of fixing a vocabulary is lost if generation may leave it silently.
                off_vocabulary.append(verb)
                continue
            triple = self._accept(
                str(r.get("from") or ""), verb, str(r.get("to") or ""),
                str(r.get("evidence") or ""), names, source, seen,
            )
            if triple:
                out.append(triple)
        self.notes["off_vocabulary"] = off_vocabulary
        # What the fixed vocabulary could not express. The cost of this paradigm, made visible.
        self.notes["unplaced"] = [
            str(u.get("description") or "")
            for u in obj.get("unplaced", []) or []
            if isinstance(u, dict)
        ]
        return tuple(out)


class HeadTailBuilder(_StagedBuilder):
    """HT-R-O. Extract between instances, then lift to types through the mapping.

    The other two paradigms ask what relation holds between two *types* — a question about the
    domain, which a model can answer from what it knows about domains in general. This asks what
    a specific sentence says about two named things, which it can only answer from the sentence.

    A type-level relation then exists because instances of it were found, not because it was
    proposed: ``(Seller, lists, Item)`` is admitted when the fragments yielded, say, three
    mentions of a particular seller listing a particular item. That count is the support, and it
    is reported, because on a short document most relations will rest on a single occurrence and
    a reader should be able to see that rather than infer it.

    Extraction runs per fragment, which is where the paper reports it works best and which also
    keeps each reply small.
    """

    paradigm = "head_tail"

    #: minimum instance triples behind a type-level relation. One, because a short document
    #: states most things once; raising it on a corpus is the point of counting at all.
    MIN_SUPPORT = 1

    MENTIONS = (
        "Find the concrete things this text mentions, and say which of the given entity types "
        'each is an instance of. Reply with ONLY JSON: {"mentions": [{"mention": <the exact '
        'words from the text>, "type": <one of the entity types>}]}. Only things the text names '
        "or describes specifically. No prose, no code fences."
    )
    INSTANCE_TRIPLES = (
        "For this fragment, extract what it states between the things it mentions. Report what "
        "the sentence says, not what is plausible. Reply with ONLY JSON: "
        '{"triples": [{"head": <the exact words>, "relation": <lower_snake>, '
        '"tail": <the exact words>, "evidence": <the sentence>}]}. '
        "An empty list is a valid answer. No prose, no code fences."
    )

    def _prompts(self) -> list[tuple[str, str]]:
        return [
            ("entities", self.ENTITIES), ("attributes", self.ATTRIBUTES),
            ("mentions", self.MENTIONS), ("instance triples", self.INSTANCE_TRIPLES),
            ("verbs", self.VERBS),
        ]

    def _mentions(self, source: str, names: set[str]) -> dict[str, str]:
        obj = self._ask(
            "mentions", self.MENTIONS,
            f"Entity types: {', '.join(sorted(names))}\n\nText:\n{source}",
        )
        mapping: dict[str, str] = {}
        for m in obj.get("mentions", []):
            if not isinstance(m, dict):
                continue
            mention, mtype = str(m.get("mention") or "").strip(), str(m.get("type") or "")
            if not mention or mtype not in names:
                continue
            if not _appears_in(mention, source):
                # A mention is by definition a piece of the text. One that is not there is the
                # clearest possible sign the model is working from memory instead.
                self.grounding.ungrounded[f"mention:{mention}"] = "not present in the text"
                continue
            mapping[mention.lower()] = mtype
        return mapping

    def _lookup(self, phrase: str, mapping: dict[str, str]) -> str | None:
        """The type of a mention, allowing the fragment stage to word it slightly differently."""
        key = phrase.strip().lower()
        if key in mapping:
            return mapping[key]
        for mention, mtype in mapping.items():
            if _appears_in(mention, phrase) or _appears_in(phrase, mention):
                return mtype
        return None

    def _relations(
        self, source: str, entities: list[EntityDraft]
    ) -> tuple[tuple[str, str, str], ...]:
        names = {e.name for e in entities}
        mapping = self._mentions(source, names)
        if not mapping:
            raise OntologyBuildError(
                "stage 'mentions' found nothing in the text to map onto the entity types; "
                "without instances there is nothing to lift to type level"
            )
        self.notes["mentions"] = len(mapping)

        support: dict[tuple[str, str, str], list[str]] = {}
        unmapped: list[str] = []
        fragments = _fragments(source)
        self.notes["fragments"] = len(fragments)
        replies = self._ask_many([
            (f"instance triples[{i}]", self.INSTANCE_TRIPLES, fragment)
            for i, fragment in enumerate(fragments)
        ])
        for obj in replies:
            if isinstance(obj, Exception):
                self.notes.setdefault("failed_stages", []).append(f"instance triples: {obj}")
                continue
            for t in obj.get("triples", []):
                if not isinstance(t, dict):
                    continue
                head, rel, tail = (
                    str(t.get("head") or ""), str(t.get("relation") or ""),
                    str(t.get("tail") or ""),
                )
                evidence = str(t.get("evidence") or "")
                htype, ttype = self._lookup(head, mapping), self._lookup(tail, mapping)
                if not (htype and ttype and rel):
                    # A triple whose ends cannot be typed cannot be lifted. Recorded, because it
                    # is usually a real statement about something the entity stage missed.
                    unmapped.append(f"{head} -{rel}-> {tail}")
                    continue
                verb = re.sub(r"[^a-z0-9_]", "_", rel.lower()).strip("_") or "relates"
                support.setdefault((htype, verb, ttype), []).append(evidence or head)

        self.notes["unmapped_triples"] = unmapped
        self.notes["support"] = {
            f"{h} -{v}-> {t}": len(ev) for (h, v, t), ev in sorted(support.items())
        }

        out: list[tuple[str, str, str]] = []
        seen: set[str] = set()
        for (htype, verb, ttype), evidence in support.items():
            if len(evidence) < self.MIN_SUPPORT:
                continue
            triple = self._accept(htype, verb, ttype, evidence[0], names, source, seen)
            if triple:
                out.append(triple)
        return tuple(out)


def _paradigms() -> dict[str, Any]:
    """Paradigm name -> builder. ``cluster_first`` needs the optional discovery extra, so it is
    offered only when that is installed rather than failing at request time."""
    out: dict[str, Any] = {
        "staged": StagedLLMBuilder,
        "relation_first": RelationFirstBuilder,
        "head_tail": HeadTailBuilder,
    }
    try:
        from .concept_discovery import ClusterFirstBuilder

        out["cluster_first"] = ClusterFirstBuilder
    except ImportError:  # pragma: no cover - only when the module itself cannot be imported
        pass
    return out


#: paradigm name -> builder, for the API and for comparing them on one document
PARADIGMS: dict[str, Any] = _paradigms()
