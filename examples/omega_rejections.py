"""Every load-time rule, fired on the mistake it exists to catch.

CΩ is the set of rules that decide whether an ontology is admissible at all. They run when an
ontology is loaded, before any question is asked of it, and they do not depend on a model being
accurate — an ontology that fails one does not load, whoever or whatever produced it.

A rule nobody can see fire is a rule nobody can check. So each one is triggered here by a
mistake a language model actually makes when asked to extract an ontology from prose: a
misspelt supertype, an invented type name, a norm conditioned on an attribute that does not
exist. Every case below is a real rejection with its real message.

    python examples/omega_rejections.py
"""

from __future__ import annotations

from loka_ontology import OntologyLoadError, load_ontology_str

#: A small, valid ontology. Each case below breaks exactly one thing about it, so what a rule
#: rejects is legible from the diff rather than from the prose around it.
BASE = """version: v1
entities:
  - type: Item
    properties: [{name: item_id, type: string}, {name: weight, type: double}]
  - type: BulkyItem
    subtype_of: Item
  - type: Seller
    properties: [{name: seller_id, type: string}, {name: on_time_rate, type: double}]
verbs:
  - {name: SUSPEND, class: institutional}
constraints:
  - {verb: SUSPEND, agent_must_be: Seller, target_must_be: [Seller]}
actions:
  - {name: SuspendSeller, verb: SUSPEND, target: Seller, guard: "on_time_rate < 1"}
"""

CASES: list[tuple[str, str, str]] = [
    (
        "R1",
        "the same property declared twice on one entity",
        BASE.replace(
            "{name: weight, type: double}",
            "{name: weight, type: double}, {name: weight, type: string}",
        ),
    ),
    (
        "R2",
        "a base type outside the permitted set",
        BASE.replace("type: double", "type: currency"),
    ),
    (
        "R3",
        "a verb act class outside the three-way partition",
        BASE.replace("class: institutional", "class: legal"),
    ),
    (
        "R4",
        "a supertype that is not defined",
        BASE.replace("subtype_of: Item", "subtype_of: Items"),
    ),
    (
        "R5",
        "a relation endpoint that is not an entity type",
        BASE + "relations:\n  - {name: sold_by, from: Item, to: Vendor, via: seller_id}\n",
    ),
    (
        "R6",
        "a typing constraint naming an entity type that does not exist",
        BASE.replace("agent_must_be: Seller", "agent_must_be: Regulator"),
    ),
    (
        "R7",
        "an action using a verb that was never declared",
        BASE.replace("verb: SUSPEND, target: Seller", "verb: BAN, target: Seller"),
    ),
    (
        "R8",
        "a cycle in the subtype order",
        BASE.replace("  - type: Seller\n", "  - type: Seller\n    subtype_of: BulkyItem\n")
        .replace("subtype_of: Item", "subtype_of: Seller"),
    ),
    (
        "R9",
        "a subtype override that breaks substitutability",
        BASE.replace(
            "  - type: BulkyItem\n    subtype_of: Item\n",
            "  - type: BulkyItem\n    subtype_of: Item\n"
            "    properties: [{name: weight, type: string}]\n",
        ),
    ),
    (
        "R10",
        "a relation whose traversal field is not declared on both ends",
        BASE + "relations:\n  - {name: sold_by, from: Item, to: Seller, via: vendor_ref}\n",
    ),
    (
        "R11a",
        "a norm governing an action that does not exist",
        BASE + 'norms:\n  - {name: N1, action: BanSeller, status: forbidden, '
               'when: "weight >= 1"}\n',
    ),
    (
        "R11b",
        "a deontic status outside permitted / mandatory / forbidden",
        BASE + 'norms:\n  - {name: N1, action: SuspendSeller, status: recommended, '
               'when: "weight >= 1"}\n',
    ),
    (
        "R11c",
        "a norm governing an uncontrollable action",
        BASE.replace('guard: "on_time_rate < 1"', 'guard: "on_time_rate < 1", controllable: false')
        + 'norms:\n  - {name: N1, action: SuspendSeller, status: forbidden, '
          'when: "on_time_rate < 1"}\n',
    ),
    (
        "R11d",
        "a norm conditioned on an attribute nothing declares",
        BASE + 'norms:\n  - {name: N1, action: SuspendSeller, status: forbidden, '
               'when: "open_disputes >= 1"}\n',
    ),
    (
        "R11e",
        "a norm conditioned on a real attribute of the wrong entity",
        # The case R11 used to let through, and the reason the check is scoped to the act's own
        # participants: every name here resolves, so an ontology-wide search finds nothing wrong.
        # Only asking *where the condition will be read* catches it.
        BASE + 'norms:\n  - {name: N1, action: SuspendSeller, status: forbidden, '
               'when: "weight <= 1"}\n',
    ),
    (
        "R12a",
        "a guard naming an attribute nothing declares",
        BASE.replace('guard: "on_time_rate < 1"', 'guard: "open_disputes >= 1"'),
    ),
    (
        "R12b",
        "a guard naming a real attribute of the wrong entity",
        # weight is declared — on Item. The act applies to a Seller, so the guard can never be
        # read, and the act would simply never be proposed. Nothing else in the ontology is
        # wrong, which is why an ontology-wide name search does not find this.
        BASE.replace('guard: "on_time_rate < 1"', 'guard: "weight <= 100"'),
    ),
]


def main() -> None:
    try:
        load_ontology_str(BASE)
    except OntologyLoadError as exc:  # pragma: no cover - the baseline must load
        raise SystemExit(
            f"the baseline does not load, so nothing below means anything: {exc}"
        ) from exc
    print(f"baseline ontology loads.\n{len(CASES)} cases, one per rule:\n")

    missed = []
    for rule, what, yaml_text in CASES:
        try:
            load_ontology_str(yaml_text)
        except OntologyLoadError as exc:
            print(f"  {rule:<5} {what}\n        -> {exc}\n")
        else:
            missed.append(rule)
            print(f"  {rule:<5} {what}\n        -> NOT REJECTED\n")

    if missed:
        raise SystemExit(f"rules that did not fire: {', '.join(missed)}")
    print("every case was rejected at load time.")


if __name__ == "__main__":
    main()
