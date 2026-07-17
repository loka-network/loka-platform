"""loka_ontology — the ontology engine Ω.

Public API: load an ontology definition, query subtypes, check type bindings.
Downstream services (grounding, compiler) access the ontology only through this API.
"""

from .engine import BindingCheck, OntologyEngine
from .infer import (
    guess_primary_key,
    infer_base_type,
    infer_entity_type,
    infer_from_adapter,
    infer_ontology_from_rows,
    to_yaml,
)
from .loader import OntologyLoadError, load_ontology, load_ontology_str
from .model import (
    BaseType,
    Cardinality,
    EntityType,
    Ontology,
    Property,
    Relation,
    TypingConstraint,
    Verb,
    VerbClass,
)
from .souffle_checker import SouffleTypeChecker, SouffleUnavailable, souffle_available

__all__ = [
    "OntologyEngine",
    "BindingCheck",
    "SouffleTypeChecker",
    "SouffleUnavailable",
    "souffle_available",
    "load_ontology",
    "load_ontology_str",
    "OntologyLoadError",
    "infer_ontology_from_rows",
    "infer_from_adapter",
    "infer_entity_type",
    "infer_base_type",
    "guess_primary_key",
    "to_yaml",
    "Ontology",
    "EntityType",
    "Property",
    "BaseType",
    "Verb",
    "VerbClass",
    "Relation",
    "Cardinality",
    "TypingConstraint",
]
