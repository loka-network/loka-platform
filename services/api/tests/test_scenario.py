"""The ontology is load-bearing: the projection method is validated against Ω, not hardcoded."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from loka_api.app import create_app
from loka_api.scenario import method_spec


def test_scenario_endpoint_reports_ontology_binding() -> None:
    body = TestClient(create_app()).get("/scenario").json()
    assert body["entity"] == "Country"
    if body["attributes"]:  # ontology file present in this env
        assert body["method"]["ontology_validated"] is True
        assert "under5_mortality" in body["attributes"]
        assert "health_exp_per_capita" in body["attributes"]


def test_method_spec_rejects_attribute_not_in_ontology() -> None:
    # Stub engine whose Country entity is missing the method's control attributes.
    class _Engine:
        def has_entity(self, name: str) -> bool:
            return name == "Country"

        def properties_of(self, name: str) -> dict[str, object]:
            return {"under5_mortality": None, "health_exp_per_capita": None}  # controls missing

    with pytest.raises(ValueError, match="not in ontology"):
        method_spec(_Engine())


def test_method_spec_rejects_missing_entity() -> None:
    class _Engine:
        def has_entity(self, name: str) -> bool:
            return False

        def properties_of(self, name: str) -> dict[str, object]:
            return {}

    with pytest.raises(ValueError, match="no entity"):
        method_spec(_Engine())
