"""The speech-act layer realizes Sifakis slide 7: asks/orders/informs + add-P-to-KB."""

from __future__ import annotations

from loka_api.speechact import KB, Asks, Informs, Method, Orders, dispatch


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


def test_asks_retrieves_data_and_writes_back() -> None:
    kb = KB()
    kb.add_fact("ZMB", "under5_mortality", 58.4)
    q = Asks("user", "loka", "Country", "ZMB", "under5_mortality")
    reply = dispatch(q, kb)
    assert reply.content == {"entity": "ZMB", "predicate": "under5_mortality", "value": 58.4}
    assert reply.render() == "informs(loka, user, under5_mortality(ZMB)=58.4)"


def test_asks_unknown_data_says_dont_know() -> None:
    q = Asks("user", "loka", "Country", "ZMB", "gdp_per_capita")
    reply = dispatch(q, KB())  # empty KB.DATA
    assert reply.content == "don't know"


def test_orders_applies_method_and_adds_P_to_kb() -> None:
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
    assert reply.content["value"] == 41.2
    # runtime: the informed predicate P was written back into KB.DATA
    assert kb.has_data("ZMB", "under5_mortality")
    assert kb.retrieve("ZMB", "under5_mortality") == 41.2


def test_orders_unknown_method_says_dont_know() -> None:
    q = Orders(
        "user", "loka", method="nope", in_types=(), out_type="x",
        entity_id="ZMB", predicate="x", args={},
    )
    reply = dispatch(q, KB())  # empty KB.METHODS
    assert reply.content == "don't know"
