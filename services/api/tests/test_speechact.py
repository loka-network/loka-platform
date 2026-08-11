"""The speech-act layer realizes Sifakis slide 7: asks/orders/informs + add-P-to-KB."""

from __future__ import annotations

import pytest
from loka_api.speechact import KB, Asks, Informs, Method, Orders, Provenance, dispatch


def test_asks_renders_like_slide7() -> None:
    q = Asks("user", "loka", var_type="Country", entity_id="ZMB", predicate="under5_mortality")
    assert q.render() == "asks(user, loka, ?x:Country under5_mortality(x=ZMB))"


def test_orders_renders_like_slide7() -> None:
    q = Orders(
        "user", "loka", method="project", in_types=("Country", "health_exp_per_capita"),
        out_type="under5_mortality", entity_id="ZMB", predicate="under5_mortality",
        args={"new_spending": 150.0},
    )
    r = q.render()
    assert r.startswith("orders(user, loka, project[Country,health_exp_per_capita->under5_mortality]")
    assert "under5_mortality(ZMB, project(new_spending=150.0))" in r


def test_informs_dont_know_renders_quoted() -> None:
    assert Informs("loka", "user", "don't know").render() == 'informs(loka, user, "don\'t know")'


def test_asks_retrieves_data_with_its_provenance() -> None:
    kb = KB()
    kb.add_fact("ZMB", "under5_mortality", 58.4,
                provenance=Provenance(kind="observed", source="worldbank:WDI", vintage="2023"))
    q = Asks("user", "loka", "Country", "ZMB", "under5_mortality")
    reply = dispatch(q, kb)
    assert reply.content["value"] == 58.4
    assert reply.content["provenance"] == {
        "kind": "observed", "source": "worldbank:WDI", "vintage": "2023",
    }
    assert reply.render() == "informs(loka, user, under5_mortality(ZMB)=58.4)"


def test_asks_unknown_data_says_dont_know() -> None:
    q = Asks("user", "loka", "Country", "ZMB", "gdp_per_capita")
    reply = dispatch(q, KB())  # empty KB.DATA
    assert reply.content == "don't know"


def test_orders_writes_into_a_counterfactual_not_the_actual_world() -> None:
    kb = KB()
    kb.register_method(Method(
        name="project", in_types=("Country",), out_type="under5_mortality",
        fn=lambda iso, new_spending: {"value": 41.2, "detail": {"projected": 41.2}},
    ))
    q = Orders(
        "user", "loka", method="project", in_types=("Country",), out_type="under5_mortality",
        entity_id="ZMB", predicate="under5_mortality", args={"iso": "ZMB", "new_spending": 150.0},
    )
    reply = dispatch(q, kb)
    sid = reply.content["scenario_id"]
    assert reply.content["value"] == 41.2
    assert kb.retrieve("ZMB", "under5_mortality", sid) == 41.2  # counterfactual world
    assert not kb.has_data("ZMB", "under5_mortality")           # actual world untouched


def test_projection_never_overwrites_an_observation() -> None:
    """The bug this guards: an orders result used to land on the observed value's key."""
    kb = KB()
    kb.add_fact("ZMB", "under5_mortality", 49.1,
                provenance=Provenance(kind="observed", source="worldbank:WDI", vintage="2023"))
    kb.register_method(Method(
        name="project", in_types=("Country",), out_type="under5_mortality",
        fn=lambda iso, new_spending: {"value": 48.551, "detail": {}},
    ))
    dispatch(Orders("user", "loka", method="project", in_types=("Country",),
                    out_type="under5_mortality", entity_id="ZMB", predicate="under5_mortality",
                    args={"iso": "ZMB", "new_spending": 150.0}), kb)

    reply = dispatch(Asks("user", "loka", "Country", "ZMB", "under5_mortality"), kb)
    assert reply.content["value"] == 49.1                       # still the observation
    assert reply.content["provenance"]["kind"] == "observed"


def test_derived_fact_cannot_be_written_into_the_actual_world() -> None:
    kb = KB()
    with pytest.raises(ValueError):
        kb.add_fact("ZMB", "under5_mortality", 48.551,
                    provenance=Provenance(kind="derived", method="project"))


def test_orders_unknown_method_says_dont_know() -> None:
    q = Orders(
        "user", "loka", method="nope", in_types=(), out_type="x",
        entity_id="ZMB", predicate="x", args={},
    )
    reply = dispatch(q, KB())  # empty KB.METHODS
    assert reply.content == "don't know"
