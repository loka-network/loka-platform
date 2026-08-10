"""Health scenario config — sourced from and validated against the ontology.

The projection method's outcome / dial / control fields are *references into* the health ontology.
``method_spec(engine)`` makes the ontology load-bearing: every attribute the method uses must be
declared on the ``Country`` entity in Ω, otherwise the method is rejected. This is why the ontology
is not decorative — it is the contract the method is bound to, and queries are type-checked against
it (a target entity/attribute not in Ω is refused, not guessed).
"""

from __future__ import annotations

import os
from typing import Any

ENTITY = "Country"
OUTCOME = "under5_mortality"
DIAL = "health_exp_per_capita"
CONTROLS = [
    "gdp_per_capita", "immunization_dpt", "sanitation_access",
    "water_access", "fertility_rate", "urban_pct",
]
LOG_COLS = ["health_exp_per_capita", "gdp_per_capita"]


def load_health_ontology() -> Any | None:
    """Load the health ontology Ω into an engine (env override, else repo/cwd examples/)."""
    from loka_ontology import OntologyEngine, load_ontology_str

    here = os.path.dirname(__file__)
    for p in (
        os.getenv("LOKA_HEALTH_ONTOLOGY"),
        os.path.join(here, "..", "..", "..", "examples", "health_ontology.yaml"),
        os.path.join(os.getcwd(), "examples", "health_ontology.yaml"),
    ):
        if p and os.path.exists(p):
            return OntologyEngine(load_ontology_str(open(p).read()))
    return None


def method_spec(engine: Any | None = None) -> dict[str, Any]:
    """The projection method spec. If an ontology engine is given, VALIDATE against it (raises)."""
    spec: dict[str, Any] = {
        "entity": ENTITY, "outcome": OUTCOME, "dial": DIAL,
        "controls": list(CONTROLS), "log_cols": list(LOG_COLS),
        "ontology_validated": False,
    }
    if engine is not None:
        if not engine.has_entity(ENTITY):
            raise ValueError(f"ontology has no entity '{ENTITY}'")
        props = set(engine.properties_of(ENTITY))
        missing = [a for a in [OUTCOME, DIAL, *CONTROLS] if a not in props]
        if missing:
            raise ValueError(f"method references attributes not in ontology: {missing}")
        spec["ontology_validated"] = True
    return spec
