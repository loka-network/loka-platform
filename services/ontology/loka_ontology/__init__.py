"""loka_ontology — the ontology engine Ω.

Public API: load an ontology definition, query subtypes, check type bindings.
Downstream services (grounding, compiler) access the ontology only through this API.
"""

from .engine import BindingCheck, OntologyEngine
from .loader import OntologyLoadError, load_ontology, load_ontology_str
from .model import (
    EntityType,
    Ontology,
    Relation,
    TypingConstraint,
    Verb,
    VerbClass,
)

__all__ = [
    "OntologyEngine",
    "BindingCheck",
    "load_ontology",
    "load_ontology_str",
    "OntologyLoadError",
    "Ontology",
    "EntityType",
    "Verb",
    "VerbClass",
    "Relation",
    "TypingConstraint",
]
