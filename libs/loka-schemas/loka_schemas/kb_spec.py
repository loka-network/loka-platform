"""KBSpec — the output of Workflow A (ontology generation).

Workflow A: domain texts + prompt -> LLM -> ontology + acquired knowledge, split
into DATA (needed) and METHODS (needed), stored in the KB. A ``KBSpec`` is that output as one
validated object: a loadable ontology definition plus the data/method needs and the three
Factual / Cognitive / Communication facets.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class KBSpec:
    """A built knowledge-base specification (ontology + DATA needs + METHODS needs)."""

    ontology_yaml: str  # a definition that loka_ontology.load_ontology_str accepts
    data_needs: tuple[str, ...] = ()  # KB.DATA: what data the ontology requires
    method_needs: tuple[str, ...] = ()  # KB.METHODS: what computations queries will need
    # the three facets: factual / cognitive / communication
    facets: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    #: Which proposed terms became entity types and which did not, with the reason for each.
    #: Carried on the result rather than on the builder that produced it: one of the builders
    #: exposes ``notes`` as a property that rebuilds a dict per call, so writing there succeeds
    #: and is then discarded — the report would be silently absent rather than wrong.
    types: Mapping[str, object] = field(default_factory=dict)
