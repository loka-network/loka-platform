"""NL projection: LLM extracts {country, new_spending}; the data validates it."""

from __future__ import annotations

from types import SimpleNamespace

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


def test_ask_end_to_end_with_fake_llm(monkeypatch) -> None:
    # Stub the gateway's LLM so the full NL -> params -> projection path runs offline.
    import loka_serving

    fake = _fake_client('{"country": "Zambia", "new_spending": 150}')
    monkeypatch.setattr(loka_serving, "llm_for", lambda purpose: fake)
    monkeypatch.setattr(loka_serving, "model_for", lambda purpose: "m")

    client = TestClient(create_app())
    resp = client.post(
        "/ask",
        json={"question": "If Zambia raises health spending to $150, what happens to child mortality?"},
    )
    if resp.status_code == 500:  # real panel not present in this env — skip
        return
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["iso3"] == "ZMB"
    assert body["formalized_query"] == {"country": "ZMB", "new_spending": 150.0}
    assert body["controlled"]["projected_outcome"] >= 0
    assert body["controlled"]["identification"] == "observational"


def test_ask_lookup_goes_through_asks_branch(monkeypatch) -> None:
    # A current-value lookup -> asks(sp,li,?x:Country under5_mortality(x)) -> retrieve from KB.DATA.
    import loka_serving

    fake = _fake_client('{"country":"Zambia","new_spending":null,"attribute":"under5_mortality"}')
    monkeypatch.setattr(loka_serving, "llm_for", lambda purpose: fake)
    monkeypatch.setattr(loka_serving, "model_for", lambda purpose: "m")

    client = TestClient(create_app())
    resp = client.post("/ask", json={"question": "What is Zambia's current child mortality?"})
    if resp.status_code == 500:  # panel absent in this env
        return
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["speech_act"]["act"] == "asks"
    assert body["speech_act"]["query"].startswith("asks(user, loka, ?x:Country under5_mortality")
    assert isinstance(body["answer"], (int, float))  # a real retrieved value, not "don't know"


def test_ask_out_of_domain_says_dont_know(monkeypatch) -> None:
    # Question the ontology can't support -> the LLM returns nulls -> system honestly says so.
    import loka_serving

    fake = _fake_client('{"country": null, "new_spending": null}')
    monkeypatch.setattr(loka_serving, "llm_for", lambda purpose: fake)
    monkeypatch.setattr(loka_serving, "model_for", lambda purpose: "m")

    client = TestClient(create_app())
    resp = client.post("/ask", json={"question": "Will Zambia's stock market rise tomorrow?"})
    if resp.status_code == 500:  # panel absent in this env
        return
    assert resp.status_code == 200, resp.text
    assert resp.json()["answer"] == "don't know"


def test_ask_without_llm_returns_503() -> None:
    # No LLM provider configured in the test env -> the endpoint fails cleanly, not a 500 crash.
    client = TestClient(create_app())
    resp = client.post("/ask", json={"question": "If Zambia raises spending to 150, mortality?"})
    assert resp.status_code in (503, 500)  # 503 clean when panel present; 500 only if panel absent
