"""Assert the verb-signature notation loads to the same ontology as the notation it replaces.

The claim being tested is narrow and worth stating exactly: the new notation is a change of
*notation*, not of content. If that holds, nothing downstream of the loader has to change —
traversal, the SQL planner, the action layer and the review checklist all keep receiving the
object they already receive. If it does not hold, the difference is printed field by field, so
what was lost in translation is visible rather than argued about.

    python scripts/check_syntax_equivalence.py
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "services" / "ontology"))

from loka_ontology.loader import load_ontology  # noqa: E402
from loka_ontology.model import Ontology  # noqa: E402
from loka_ontology.verb_syntax import load_verb_syntax  # noqa: E402

#: Every ontology that exists in both notations. A notation shown to work on the one domain it
#: was designed against has not been shown to work; the thin domain is here because a missing
#: role is easiest to overlook where there is almost nothing to declare.
PAIRS = [
    (ROOT / "examples" / "supply_ontology.yaml", ROOT / "examples" / "supply_ontology_v3.yaml"),
    (ROOT / "examples" / "health_ontology.yaml", ROOT / "examples" / "health_ontology_v3.yaml"),
]


#: Fields that document the ontology rather than define it. They are compared, but a difference
#: in them is not a difference in what the ontology *is* — the new file says "walked by verb X"
#: where the old one said "traversed by relation X", which is the notation change itself showing
#: up in prose. Folding them into the structural comparison would report the change as a defect.
PROSE = {"description", "rationale"}


def _strip(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if k not in PROSE}


def _normalise(onto: Ontology, *, prose: bool) -> dict[str, Any]:
    """Order-independent view. Two files may list the same relations in a different order and
    still be the same ontology; a diff that reports that as a difference is noise.

    ``prose=False`` drops the documentation fields, leaving what the ontology declares.
    """
    keep = (lambda d: d) if prose else _strip
    return {
        "version": onto.version,
        "entities": sorted(
            (
                {
                    "name": e.name,
                    "subtype_of": e.subtype_of,
                    "backing": e.backing,
                    "properties": sorted(
                        (keep(asdict(p)) for p in e.properties), key=lambda p: str(p["name"])
                    ),
                }
                for e in onto.entities.values()
            ),
            key=lambda e: str(e["name"]),
        ),
        "verbs": sorted((asdict(v) for v in onto.verbs.values()), key=lambda v: str(v["name"])),
        "relations": sorted((asdict(r) for r in onto.relations), key=lambda r: str(r["name"])),
        "constraints": sorted(
            (asdict(c) for c in onto.constraints),
            key=lambda c: (str(c["verb"]), str(c["agent_must_be"])),
        ),
        "actions": sorted((keep(asdict(a)) for a in onto.actions), key=lambda a: str(a["name"])),
        "norms": sorted((keep(asdict(n)) for n in onto.norms), key=lambda n: str(n["name"])),
    }


def _diff(section: str, old: Any, new: Any) -> list[str]:
    if old == new:
        return []
    out = [f"  {section}:"]
    if isinstance(old, list) and isinstance(new, list):
        for item in old:
            if item not in new:
                out.append(f"    - only in the classic file: {item}")
        for item in new:
            if item not in old:
                out.append(f"    + only in the verb-syntax file: {item}")
    else:
        out.append(f"    classic     : {old}")
        out.append(f"    verb syntax : {new}")
    return out


def _check(classic_path: Path, verb_path: Path) -> list[str]:
    old, new = load_ontology(classic_path), load_verb_syntax(verb_path)
    classic = _normalise(old, prose=False)
    verb = _normalise(new, prose=False)

    print(f"  {classic_path.name}  ==  {verb_path.name}")
    counts = {k: (len(v) if isinstance(v, list) else 1) for k, v in verb.items()}
    print("    " + ", ".join(f"{k} {n}" for k, n in counts.items() if k != "version"))

    problems: list[str] = []
    for section in classic:
        problems += _diff(section, classic[section], verb[section])

    # Reported, not enforced. A reader should know the documentation strings moved even though
    # the declarations did not.
    prose = [
        line
        for section in classic
        for line in _diff(
            section,
            _normalise(old, prose=True)[section],
            _normalise(new, prose=True)[section],
        )
    ]
    if prose:
        n = sum(1 for line in prose if line.strip().startswith(("-", "+"))) // 2
        print(f"    {n} documentation string(s) differ (wording only)")
    return problems


def main() -> None:
    print(f"{len(PAIRS)} ontologies, each written both ways:\n")
    problems: list[str] = []
    for classic_path, verb_path in PAIRS:
        problems += _check(classic_path, verb_path)
        print()

    if problems:
        print("the notations do NOT declare the same ontology:\n")
        print("\n".join(problems))
        raise SystemExit(1)

    print("every pair declares the same ontology, field for field.")
    print("nothing downstream of the loader sees a different object.")


if __name__ == "__main__":
    main()
