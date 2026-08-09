"""KBSpec — the output of Workflow A (ontology generation).

The professor's slide 7: domain texts + prompt -> LLM -> ontology + acquired knowledge, split
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
    facets: Mapping[str, tuple[str, ...]] = field(default_factory=dict)  # factual/cognitive/communication
