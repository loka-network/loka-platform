"""The verb-signature notation, translated into the ontology structures already in use.

A review of the previous notation made three objections, and this module is the answer to them:

  * *Verbs define relations — you do not need relations.*  A relation here is a verb that
    carries ``via``: ``contains(subject: Order, object: OrderItem) via order_id``. There is no
    separate relation set to keep in step with the verb set.
  * *Actions are verbs.*  An action here is a verb that carries a guard or an effect. There is
    no separate action set either.
  * *Constraints are state predicates on attributes.*  ``predicates:`` holds those. What the
    old notation called a constraint — which entity types a verb may be used with — is now the
    verb's own signature, where a reader expects to find it.

What the objection does not mean is that the data those primitives carried can be dropped. The
traversal key decides how a multi-hop query is walked; cardinality decides what the data is
allowed to look like; the controllable/uncontrollable split decides what may be proposed. All of
it survives as part of a verb. Nothing here is a rename: the notation changed, the content did
not, and :mod:`scripts.check_syntax_equivalence` asserts the two files load to the same object.

The translation targets the existing loader rather than replacing it, so every load-time rule
still runs and no downstream module sees a new shape.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from .loader import OntologyLoadError, _parse
from .model import Ontology

#: ``name(role: Type, role: Type, ...)`` — the signature as it is written in the document.
_SIGNATURE = re.compile(r"^\s*(?P<name>\w+)\s*\((?P<params>.*)\)\s*$")
_PARAM = re.compile(r"^\s*(?P<role>\w+)\s*:\s*(?P<type>\w+)\s*$")

#: ``predicate_name(var) := expression``
_PREDICATE = re.compile(r"^\s*(?P<name>\w+)\s*\((?P<var>\w*)\)\s*:=\s*(?P<expr>.+?)\s*$")

#: A predicate used in a norm's ``when``, applied to an argument: ``thin_evidence(x)``.
_APPLIED = re.compile(r"^\s*(?P<name>\w+)\s*\([^)]*\)\s*$")

#: ``Modal(verb(args)) when condition``
_NORM = re.compile(
    r"^\s*(?P<modal>May|Must|Forbidden)\s*\(\s*(?P<verb>\w+)\s*\([^)]*\)\s*\)"
    r"(?:\s+when\s+(?P<when>.+?))?\s*$"
)

_MODALITY = {"May": "permitted", "Must": "mandatory", "Forbidden": "forbidden"}


def load_verb_syntax(path: str | Path) -> Ontology:
    """Load an ontology written in the verb-signature notation."""
    return load_verb_syntax_str(Path(path).read_text(encoding="utf-8"))


def load_verb_syntax_str(text: str) -> Ontology:
    """Load from a YAML string; convenient for tests and for the equivalence check."""
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise OntologyLoadError("top level must be a mapping (dict)")
    return _parse(to_classic(raw))


def to_classic(raw: dict[str, Any]) -> dict[str, Any]:
    """Translate the verb-signature document into the shape the existing loader parses.

    Kept separate from loading so the translation can be inspected and diffed on its own — a
    reader who wants to know what the new notation means can read the dict it produces rather
    than take this module's word for it.
    """
    out: dict[str, Any] = {"version": str(raw.get("version", "v0"))}
    out["entities"] = _entities(raw.get("entities") or {})

    predicates = _predicates(raw.get("predicates") or [])
    verbs, relations, constraints, actions = _verbs(raw.get("verbs") or [])

    out["verbs"] = verbs
    out["relations"] = relations
    out["constraints"] = constraints
    out["actions"] = actions
    out["norms"] = _norms(raw.get("norms") or [], actions, predicates)
    return out


def _entities(block: dict[str, Any]) -> list[dict[str, Any]]:
    """``has`` becomes the property list; ``is`` becomes the supertype."""
    entities: list[dict[str, Any]] = []
    for name, body in block.items():
        body = body or {}
        item: dict[str, Any] = {"type": name}
        if body.get("is"):
            item["subtype_of"] = body["is"]
        if body.get("backing"):
            item["backing"] = body["backing"]

        props: list[dict[str, Any]] = []
        for attr, spec in (body.get("has") or {}).items():
            spec = spec or {}
            prop: dict[str, Any] = {"name": attr, "type": spec.get("type", "string")}
            if spec.get("required"):
                prop["required"] = True
            # ``doc`` rather than ``description``: the attribute table in the document is read
            # left to right, and a long key pushes the type out of alignment.
            if spec.get("doc"):
                prop["description"] = spec["doc"]
            props.append(prop)
        item["properties"] = props
        entities.append(item)
    return entities


def _signature(sig: str) -> tuple[str, dict[str, str]]:
    """``ship_standard(subject: Seller, object: Product)`` → name and role-to-type mapping."""
    m = _SIGNATURE.match(sig)
    if not m:
        raise OntologyLoadError(
            f"malformed verb signature {sig!r}; expected name(role: Type, role: Type, ...)"
        )
    roles: dict[str, str] = {}
    params = m.group("params").strip()
    if params:
        for part in params.split(","):
            pm = _PARAM.match(part)
            if not pm:
                raise OntologyLoadError(
                    f"malformed parameter {part.strip()!r} in {sig!r}; expected role: Type"
                )
            roles[pm.group("role")] = pm.group("type")
    if "subject" not in roles:
        raise OntologyLoadError(f"verb {sig!r} declares no subject")
    return m.group("name"), roles


def _verbs(
    block: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split the verb list into the four structures the existing loader expects.

    A verb carrying ``via`` is a relation and nothing else — registering it in the verb set as
    well would put a name there that no act class describes. A verb carrying a guard, an effect
    or ``controllable`` is an action, and its signature is what supplies the typing constraint.
    """
    verbs: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    constraints: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []

    seen_verbs: set[str] = set()
    seen_constraints: set[tuple[str, str, str]] = set()

    for item in block:
        if "sig" not in item:
            raise OntologyLoadError(f"verb entry has no 'sig': {item!r}")
        name, roles = _signature(item["sig"])

        if item.get("via"):
            if "object" not in roles:
                raise OntologyLoadError(f"relational verb {name!r} declares no object")
            rel: dict[str, Any] = {
                "name": name,
                "from": roles["subject"],
                "to": roles["object"],
                "via": item["via"],
            }
            # ``each`` is optional on purpose: a missing cardinality is recorded as missing, not
            # silently widened to the unconstrained one. Review needs to see the difference.
            if item.get("each"):
                rel["cardinality"] = item["each"]
            relations.append(rel)
            continue

        verb_name = name.upper()
        if verb_name not in seen_verbs:
            if not item.get("act"):
                raise OntologyLoadError(f"verb {name!r} declares no act class")
            verbs.append({"name": verb_name, "class": item["act"]})
            seen_verbs.add(verb_name)

        target = roles.get("object", roles["subject"])
        key = (verb_name, roles["subject"], target)
        if key not in seen_constraints:
            constraints.append(
                {
                    "verb": verb_name,
                    "agent_must_be": roles["subject"],
                    "target_must_be": [target],
                }
            )
            seen_constraints.add(key)

        action: dict[str, Any] = {
            "name": item.get("as") or _camel(name),
            "verb": verb_name,
            "target": target,
        }
        # The third role the review's notation allows. It is not part of the typing constraint —
        # who may act on whom is subject and object — but its attributes are in scope for the
        # guard, which is the whole reason the role exists.
        if "complement" in roles:
            action["complement"] = roles["complement"]
        if item.get("guard"):
            action["guard"] = item["guard"]
        if item.get("effect"):
            action["effect"] = item["effect"]
        if "controllable" in item:
            action["controllable"] = item["controllable"]
        actions.append(action)

    return verbs, relations, constraints, actions


def _predicates(block: list[str]) -> dict[str, str]:
    """``thin_evidence(s) := delivered_lines < 20`` → {"thin_evidence": "delivered_lines < 20"}."""
    out: dict[str, str] = {}
    for line in block:
        m = _PREDICATE.match(line)
        if not m:
            raise OntologyLoadError(
                f"malformed predicate {line!r}; expected name(var) := expression"
            )
        out[m.group("name")] = m.group("expr")
    return out


def _norms(
    block: list[dict[str, Any]], actions: list[dict[str, Any]], predicates: dict[str, str]
) -> list[dict[str, Any]]:
    """``Forbidden(suspend_seller(s, x)) when thin_evidence(x)`` → a norm on that action."""
    by_verb: dict[str, list[str]] = {}
    for a in actions:
        by_verb.setdefault(a["verb"], []).append(a["name"])

    norms: list[dict[str, Any]] = []
    for i, item in enumerate(block):
        rule = item.get("rule") if isinstance(item, dict) else item
        if not isinstance(rule, str):
            raise OntologyLoadError(f"norm entry has no 'rule': {item!r}")
        m = _NORM.match(rule)
        if not m:
            raise OntologyLoadError(
                f"malformed norm {rule!r}; expected May|Must|Forbidden(verb(args)) [when cond]"
            )

        verb_name = m.group("verb").upper()
        candidates = by_verb.get(verb_name, [])
        if not candidates:
            # Left to the load-time rules to report against a name they already know how to
            # explain, rather than raising a second, differently worded error here.
            action_name = m.group("verb")
        elif len(candidates) > 1:
            raise OntologyLoadError(
                f"norm {rule!r} names verb {m.group('verb')!r}, which {len(candidates)} acts "
                f"share ({', '.join(candidates)}); name the act with 'on:' to say which"
            )
        else:
            action_name = candidates[0]

        when = (m.group("when") or "").strip()
        condition = _resolve(when, predicates)

        norm: dict[str, Any] = {
            "name": item.get("name") if isinstance(item, dict) else None,
            "action": item.get("on") if isinstance(item, dict) else None,
            "status": _MODALITY[m.group("modal")],
        }
        norm["name"] = norm["name"] or f"N{i + 1}"
        norm["action"] = norm["action"] or action_name
        if condition:
            norm["when"] = condition
        if isinstance(item, dict) and item.get("why"):
            norm["rationale"] = item["why"]
        norms.append(norm)
    return norms


def _resolve(when: str, predicates: dict[str, str]) -> str:
    """Substitute a named state predicate for the expression it stands for.

    ``thin_evidence(x)`` becomes ``delivered_lines < 20``. A name with no declaration is left as
    written rather than dropped: the load-time check for norms conditioned on attributes nothing
    declares is the right place for that to be reported, and it explains the consequence — the
    norm can never fire, so the act it guards is permitted unconditionally. Silently discarding
    the condition here would turn that into a norm that reads as unconditional.
    """
    if not when:
        return ""
    m = _APPLIED.match(when)
    if m and m.group("name") in predicates:
        return predicates[m.group("name")]
    return predicates.get(when, when)


def _camel(name: str) -> str:
    return "".join(part.capitalize() for part in name.split("_"))
