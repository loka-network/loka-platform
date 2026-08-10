"""NL -> formalized projection query (the professor's Workflow B: question -> LLM -> q).

The LLM *proposes* the method inputs (which country, what new health-spending level); the caller
*disposes* by validating them against the panel (an unknown country / non-numeric level is
rejected, not guessed). Uses an injected client exposing ``messages.create`` — so Claude, an
OpenAI-compatible proxy, or a fake in tests all work.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

_SYSTEM = (
    "You extract a health-spending projection request. Reply with ONLY a JSON object: "
    '{"country": <country name or ISO3 code, or null>, '
    '"new_spending": <number: new health spending per capita in USD, or null>}. '
    "If the question is not about changing a country's health spending to project child "
    "mortality, use nulls. No prose, no code fences."
)


def extract_projection(question: str, client: Any, model: str) -> dict[str, Any]:
    """Ask the LLM to propose {country, new_spending} from a natural-language question."""
    resp = client.messages.create(
        model=model,
        max_tokens=200,
        system=_SYSTEM,
        messages=[{"role": "user", "content": question}],
    )
    text = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text")
    start, end = text.find("{"), text.rfind("}")
    obj = json.loads(text[start : end + 1]) if start != -1 else {}
    return {"country": obj.get("country"), "new_spending": obj.get("new_spending")}


_FORMALIZE_SYSTEM = (
    "Classify a question about a country's child health into a formal query. Reply with ONLY a "
    'JSON object: {"country": <name or ISO3, or null>, '
    '"new_spending": <new health spending per capita in USD if the question CHANGES spending to '
    'project mortality, else null>, '
    '"attribute": <the single attribute being looked up if the question just ASKS for a current '
    'value (one of: under5_mortality, health_exp_per_capita, gdp_per_capita, immunization_dpt, '
    'sanitation_access, water_access, fertility_rate, urban_pct), else null>}. '
    "A projection (change spending -> mortality) sets new_spending. A lookup (current value) sets "
    "attribute. If neither applies, use nulls. No prose, no code fences."
)


def formalize_query(question: str, client: Any, model: str) -> dict[str, Any]:
    """NL -> a classified query: intent = 'order' (apply method) | 'ask' (lookup DATA) | 'none'.

    ``order`` (Sifakis orders(...)) changes a dial and applies the projection method; ``ask``
    (asks(...)) retrieves a current attribute from KB.DATA. Intent is inferred from the fields so
    it is robust to models that only fill some of them.
    """
    resp = client.messages.create(
        model=model,
        max_tokens=200,
        system=_FORMALIZE_SYSTEM,
        messages=[{"role": "user", "content": question}],
    )
    text = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text")
    start, end = text.find("{"), text.rfind("}")
    obj = json.loads(text[start : end + 1]) if start != -1 else {}
    country, new_spending, attribute = obj.get("country"), obj.get("new_spending"), obj.get("attribute")
    if as_spending(new_spending) is not None:
        intent = "order"
    elif attribute:
        intent = "ask"
    else:
        intent = "none"
    return {"intent": intent, "country": country, "new_spending": new_spending, "attribute": attribute}


def resolve_country(panel: Sequence[Mapping[str, Any]], name_or_iso: Any) -> str | None:
    """Resolve an LLM-proposed country (ISO3 or name) to an ISO3 present in the panel, or None."""
    if not name_or_iso:
        return None
    q = str(name_or_iso).strip().lower()
    iso = {r["iso3"].lower(): r["iso3"] for r in panel}
    if q in iso:
        return iso[q]
    names = {r["country"].lower(): r["iso3"] for r in panel}
    if q in names:
        return names[q]
    for nm, code in names.items():  # loose contains-match ("thailand" vs "thailand, kingdom of")
        if q in nm or nm in q:
            return code
    return None


def as_spending(value: Any) -> float | None:
    """Validate the proposed new spending is a positive number."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None
