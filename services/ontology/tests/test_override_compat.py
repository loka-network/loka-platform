"""CΩ rule R7 — a subtype may not break substitutability when it overrides a property.

Ω declares that properties are inherited along ⪯ and that a subtype may redeclare one. Without a
compatibility rule that redeclaration can silently destroy substitutability: ``SmallCountry`` could
declare ``gdp: string`` over ``Country.gdp: double`` and still load, so a ``SmallCountry`` would no
longer be usable wherever a ``Country`` is expected. R7 closes that.
"""

from __future__ import annotations

import pytest
from loka_ontology.loader import OntologyLoadError, load_ontology_str


def _onto(parent_type: str, child_type: str, parent_extra: str = "", child_extra: str = "") -> str:
    return (
        "version: t\n"
        "entities:\n"
        "  - type: Country\n"
        "    properties:\n"
        f"      - {{name: gdp, type: {parent_type}{parent_extra}}}\n"
        "  - type: SmallCountry\n"
        "    subtype_of: Country\n"
        "    properties:\n"
        f"      - {{name: gdp, type: {child_type}{child_extra}}}\n"
    )


def test_override_with_an_unrelated_type_is_rejected() -> None:
    with pytest.raises(OntologyLoadError, match="substitutability"):
        load_ontology_str(_onto("double", "string"))


def test_override_widening_the_type_is_rejected() -> None:
    # integer -> double admits values the supertype forbids, so the subtype is not substitutable.
    with pytest.raises(OntologyLoadError, match="substitutability"):
        load_ontology_str(_onto("integer", "double"))


def test_override_with_the_same_type_is_accepted() -> None:
    load_ontology_str(_onto("double", "double"))


def test_override_narrowing_the_type_is_accepted() -> None:
    load_ontology_str(_onto("double", "integer"))    # every integer is a valid double
    load_ontology_str(_onto("timestamp", "date"))    # every date is a valid timestamp


def test_override_relaxing_required_to_optional_is_rejected() -> None:
    with pytest.raises(OntologyLoadError, match="may not relax a requirement"):
        load_ontology_str(_onto("double", "double", parent_extra=", required: true"))


def test_override_is_checked_against_every_ancestor_not_just_the_parent() -> None:
    # A grandchild must stay compatible with the type declared two levels up.
    yaml = (
        "version: t\n"
        "entities:\n"
        "  - type: A\n"
        "    properties:\n"
        "      - {name: v, type: double}\n"
        "  - type: B\n"
        "    subtype_of: A\n"
        "  - type: C\n"
        "    subtype_of: B\n"
        "    properties:\n"
        "      - {name: v, type: string}\n"
    )
    with pytest.raises(OntologyLoadError, match="substitutability"):
        load_ontology_str(yaml)


def test_the_shipped_health_ontology_still_loads() -> None:
    import os

    path = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "examples", "health_ontology.yaml"
    )
    if not os.path.exists(path):
        pytest.skip("examples not present in this env")
    with open(path) as f:
        load_ontology_str(f.read())
