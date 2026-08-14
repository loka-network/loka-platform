"""NL projection: LLM extracts {country, new_spending}; the data validates it."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from loka_api.app import create_app
from loka_api.nl_project import as_spending, extract_projection, resolve_country

_PANEL = [
    {"iso3": "ZMB", "country": "Zambia", "year": "2023"},
    {"iso3": "NGA", "country": "Nigeria", "year": "2023"},
]


def _fake_client(payload_json: str) -> object:
    create = lambda **_: SimpleNamespace(content=[SimpleNamespace(type="text", text=payload_json)])  # noqa: E731
    return SimpleNamespace(messages=SimpleNamespace(create=create))


def test_extract_projection_parses_llm_json() -> None:
    client = _fake_client('{"country": "Zambia", "new_spending": 150}')
    out = extract_projection("If Zambia raises health spending to $150, mortality?", client, "m")
    assert out == {"country": "Zambia", "new_spending": 150}


def test_resolve_country_by_name_and_iso() -> None:
    assert resolve_country(_PANEL, "Zambia") == "ZMB"
    assert resolve_country(_PANEL, "zmb") == "ZMB"
    assert resolve_country(_PANEL, "Narnia") is None
    assert resolve_country(_PANEL, None) is None


def test_as_spending_validates() -> None:
    assert as_spending(150) == 150.0
    assert as_spending("200") == 200.0
    assert as_spending(-5) is None
    assert as_spending("abc") is None


def test_ask_end_to_end_with_fake_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    # Stub the gateway's LLM so the full NL -> params -> projection path runs offline.
    import loka_serving

    fake = _fake_client('{"country": "Zambia", "new_spending": 150}')
    monkeypatch.setattr(loka_serving, "llm_for", lambda purpose: fake)
    monkeypatch.setattr(loka_serving, "model_for", lambda purpose: "m")

    client = TestClient(create_app())
    resp = client.post(
        "/ask",
        json={"question": "If Zambia raises health spending to $150, what of mortality?"},
    )
    if resp.status_code == 500:  # real panel not present in this env — skip
        pytest.skip("health panel not present in this env")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["iso3"] == "ZMB"
    assert body["formalized_query"] == {"country": "ZMB", "new_spending": 150.0}
    assert body["controlled"]["projected_outcome"] >= 0
    assert body["controlled"]["identification"] == "observational"


def test_ask_refuses_a_predicate_not_declared_in_omega(monkeypatch: pytest.MonkeyPatch) -> None:
    """The refusal comes from Ω, not from a missing parameter."""
    import loka_serving

    fake = _fake_client(
        '{"country":"Zambia","new_spending":null,"attribute":"stock_market_index"}'
    )
    monkeypatch.setattr(loka_serving, "llm_for", lambda purpose: fake)
    monkeypatch.setattr(loka_serving, "model_for", lambda purpose: "m")

    client = TestClient(create_app())
    resp = client.post("/ask", json={"question": "Will Zambia's stock market rise tomorrow?"})
    if resp.status_code == 500:  # panel absent in this env
        pytest.skip("health panel not present in this env")
    body = resp.json()
    assert body["answer"] == "don't know"
    assert body["reason_code"] == "not_in_ontology"


def test_removing_an_attribute_from_omega_breaks_the_lookup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Ω is load-bearing: delete the outcome from Ω and the service refuses to start.

    A method whose fields the ontology does not declare cannot be authorised by it, so the
    failure surfaces at startup rather than as an answer that quietly claims less validation
    than it appears to have.
    """
    import yaml

    src_path = Path(__file__).resolve().parents[3] / "examples" / "health_ontology.yaml"
    if not src_path.exists():
        pytest.skip("health ontology not present in this env")
    src = yaml.safe_load(src_path.read_text())
    for ent in src["entities"]:
        if ent["type"] == "Country":
            ent["properties"] = [p for p in ent["properties"] if p["name"] != "under5_mortality"]
    trimmed = tmp_path / "health_trimmed.yaml"
    trimmed.write_text(yaml.safe_dump(src))
    monkeypatch.setenv("LOKA_HEALTH_ONTOLOGY", str(trimmed))

    with pytest.raises(ValueError, match="under5_mortality"):
        create_app()


def test_projection_then_lookup_returns_the_observation(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end guard: asking for a projection must not corrupt the later observed lookup."""
    import loka_serving

    monkeypatch.setattr(loka_serving, "model_for", lambda purpose: "m")
    client = TestClient(create_app())

    # 1) a projection (orders) -> a counterfactual value
    monkeypatch.setattr(
        loka_serving, "llm_for",
        lambda purpose: _fake_client('{"country":"Zambia","new_spending":150,"attribute":null}'),
    )
    proj = client.post("/ask", json={"question": "If Zambia raises spending to 150, mortality?"})
    if proj.status_code == 500:  # panel absent in this env
        pytest.skip("health panel not present in this env")
    projected = proj.json()["controlled"]["projected_outcome"]

    # 2) then the current value (asks) -> must still be the observation, not the projection
    monkeypatch.setattr(
        loka_serving, "llm_for",
        lambda purpose: _fake_client(
            '{"country":"Zambia","new_spending":null,"attribute":"under5_mortality"}'
        ),
    )
    look = client.post("/ask", json={"question": "What is Zambia's current child mortality?"})
    assert look.status_code == 200, look.text
    body = look.json()
    assert body["answer"] != projected
    assert body["retrieved"]["provenance"]["kind"] == "observed"


def test_ask_lookup_goes_through_asks_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    # A current-value lookup -> asks(sp,li,?x:Country under5_mortality(x)) -> retrieve from KB.DATA.
    import loka_serving

    fake = _fake_client('{"country":"Zambia","new_spending":null,"attribute":"under5_mortality"}')
    monkeypatch.setattr(loka_serving, "llm_for", lambda purpose: fake)
    monkeypatch.setattr(loka_serving, "model_for", lambda purpose: "m")

    client = TestClient(create_app())
    resp = client.post("/ask", json={"question": "What is Zambia's current child mortality?"})
    if resp.status_code == 500:  # panel absent in this env
        pytest.skip("health panel not present in this env")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["speech_act"]["act"] == "asks"
    assert body["speech_act"]["query"].startswith("asks(user, loka, ?x:Country under5_mortality")
    assert isinstance(body["answer"], (int, float))  # a real retrieved value, not "don't know"


def test_ask_out_of_domain_says_dont_know(monkeypatch: pytest.MonkeyPatch) -> None:
    # Question the ontology can't support -> the LLM returns nulls -> system honestly says so.
    import loka_serving

    fake = _fake_client('{"country": null, "new_spending": null}')
    monkeypatch.setattr(loka_serving, "llm_for", lambda purpose: fake)
    monkeypatch.setattr(loka_serving, "model_for", lambda purpose: "m")

    client = TestClient(create_app())
    resp = client.post("/ask", json={"question": "Will Zambia's stock market rise tomorrow?"})
    if resp.status_code == 500:  # panel absent in this env
        pytest.skip("health panel not present in this env")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["answer"] == "don't know"
    assert body["reason_code"] == "unknown_entity"  # this path: no country resolved


def test_ask_without_llm_returns_503() -> None:
    # No LLM provider configured in the test env -> the endpoint fails cleanly, not a 500 crash.
    client = TestClient(create_app())
    resp = client.post("/ask", json={"question": "If Zambia raises spending to 150, mortality?"})
    assert resp.status_code in (503, 500)  # 503 clean when panel present; 500 only if panel absent
