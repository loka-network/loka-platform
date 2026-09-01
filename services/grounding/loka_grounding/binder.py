"""Stage ② — the deterministic binder: validate a proposal against the ontology into q*.

This is the verifiable core of grounding. It takes a (possibly wrong) :class:`QueryProposal`
and either returns a signed :class:`~loka_schemas.TypedQuery` or rejects it with a structured
error. No language model runs here — binding is pure, so a well-typed q* is guaranteed before
anything reaches the compiler.
"""

from __future__ import annotations

from loka_schemas import OntologyView, TypedQuery

from .models import (
    TASK_TYPES,
    EmptyProposal,
    QueryProposal,
    UnknownAttribute,
    UnknownTarget,
    UnknownTaskType,
)
from .proposer import QueryProposer


def bind(
    proposal: QueryProposal,
    ontology: OntologyView,
    *,
    query_id: str,
    signature: str | None = None,
) -> TypedQuery:
    """Validate ``proposal`` against ``ontology`` and produce a typed query q*.

    Raises :class:`UnknownTaskType`, :class:`EmptyProposal`, or :class:`UnknownTarget` — the
    same fail-fast discipline the compiler uses, one stage earlier.
    """
    if proposal.task_type not in TASK_TYPES:
        raise UnknownTaskType(
            f"task type {proposal.task_type!r} not in {sorted(TASK_TYPES)}"
        )
    if not proposal.targets:
        raise EmptyProposal("proposal has no candidate targets to ground")
    unknown = [t for t in proposal.targets if not ontology.has_entity(t)]
    if unknown:
        raise UnknownTarget(f"targets not in ontology {ontology.version}: {unknown}")

    # An attribute is checked against the types the question is about, not against the ontology
    # at large. Asking a Seller for a Product's weight is a question this ontology cannot answer
    # even though it declares `weight`, and a check that looked everywhere would let it through.
    declared: dict[str, list[str]] = {}
    for target in proposal.targets:
        for name in ontology.properties_of(target):
            declared.setdefault(name, []).append(target)
    undeclared = [a for a in proposal.attributes if a not in declared]
    if undeclared:
        raise UnknownAttribute(
            f"{sorted(undeclared)} not declared by "
            f"{list(proposal.targets)} in ontology {ontology.version}; "
            f"declared: {sorted(declared)}"
        )

    return TypedQuery(
        query_id=query_id,
        task_type=proposal.task_type,
        targets=proposal.targets,
        attributes=proposal.attributes,
        signature=signature,
    )


def ground(
    question: str,
    proposer: QueryProposer,
    ontology: OntologyView,
    *,
    query_id: str,
    signature: str | None = None,
) -> TypedQuery:
    """End-to-end grounding: propose (Stage ①) then bind (Stage ②).

    The proposer may be the keyword reference or the LLM-backed one — the binder validates
    either identically, so an ill-typed proposal is rejected regardless of its source.
    """
    proposal = proposer.propose(question)
    return bind(proposal, ontology, query_id=query_id, signature=signature)
