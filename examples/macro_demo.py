"""End-to-end macro demo: the two Sifakis workflows, in process, no server needed.

Run:  PYTHONPATH=services/api python examples/macro_demo.py

It (A) builds a KB from a short macro text, (B) ingests state data + an identified causal claim,
then answers two questions — an ``asks``/DATA one and an ``orders``/METHOD one — printing the
formalized query, the retrieval, and the decision memorandum for each.
"""

from __future__ import annotations

import json

from loka_api.ingest import ingest_causal, ingest_data
from loka_api.orchestrator import answer
from loka_api.world import world_from_kbspec
from loka_ontology import build

TEXT = (
    "The Central Bank sets the Policy Rate. A cut in the Policy Rate weakens the Exchange "
    "Rate, which affects GDP. Analysts forecast GDP and estimate the effect of a rate change."
)

DATA = [
    {"entity": "GDP", "instance": "US", "property": "value", "value": 2.1},
    {"entity": "PolicyRate", "instance": "Fed", "property": "value", "value": 5.25},
]

CAUSAL = [
    {
        "cause": "PolicyRate",
        "effect": "GDP",
        "mean": -0.8,
        "se": 0.25,
        "identification_status": "quasi_experimental",
        "evidence_refs": ["nber:w12345"],
    }
]

QUESTIONS = [
    ("d1", "Give the GDP reading."),  # asks / DATA
    ("d2", "What happens to GDP if the PolicyRate is cut?"),  # orders / METHOD (causal)
]


def _show(title: str, obj: object) -> None:
    print(f"\n--- {title} ---")
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def main() -> None:
    # A · Workflow A — build a KB from domain text.
    spec = build([TEXT])
    print("=== Workflow A: built KB ===")
    print(spec.ontology_yaml)
    print("DATA needs:   ", list(spec.data_needs))
    print("METHOD needs: ", list(spec.method_needs))

    world = world_from_kbspec(spec)
    ingest_data(world, DATA)
    ingest_causal(world, CAUSAL)
    print("\n(ingested %d data rows, %d causal claims)" % (len(DATA), len(CAUSAL)))

    # B · Workflow B — answer questions against the built + ingested KB.
    for qid, question in QUESTIONS:
        print("\n================================================================")
        print(f"Q: {question}")
        res = answer(world, question, query_id=qid)
        _show("formalized query q*", res["formalized_query"])
        _show("retrieval (asks/DATA or orders/METHOD)", res["retrieval"])
        _show("decision memorandum", res["decision"])
        print("stages:", res["stages"])


if __name__ == "__main__":
    main()
