"""loka_ontology —— 本体论引擎 Ω。

对外公开 API:加载本体论定义、查询子类型、校验类型绑定。
下游(语义接地、Compiler)只通过这些访问本体论。
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
