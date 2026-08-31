"""Run one extraction over one document against several models, and tabulate the spread.

Review of the previous draft put it plainly: *if your prompts are well designed, the results
should be almost independent from the models.* The draft had reported the opposite — the same
document and the same paradigm returning 41, 40 and 204 concepts — and attributed it to the
endpoint. That attribution was too comfortable. Batching does move a token boundary; it does not
turn forty concepts into two hundred. An instruction that never says how many is one the model
has to guess at, and models guess differently.

So the claim is now testable rather than asserted, in both directions. Either the constrained
prompt brings the models together, in which case the review was right and the fix is recorded
here with the numbers that show it; or it does not, and this prints the evidence for saying so.

Each model runs the same document the same number of times. What is measured:

  entities / relations   what the extraction produced
  grounded               share of concepts whose cited evidence is findable in the source
  loads                  whether the draft passes the load-time rules at all
  review                 length of the checklist a reviewer would be handed
  spread                 max - min across repeats, per model — the number under dispute

Models are described in a JSON file so that adding one is not a code change:

    [
      {"label": "deepseek-chat", "provider": "openai", "model": "deepseek-chat",
       "base_url": "https://api.deepseek.com/v1", "api_key_env": "DEEPSEEK_API_KEY"},
      {"label": "claude", "provider": "anthropic", "model": "claude-opus-4-8",
       "api_key_env": "ANTHROPIC_API_KEY"}
    ]

    python scripts/cross_model_extraction.py --models models.json \
        --text examples/supply_domain.md --repeat 3 --json out.json

A model whose key is absent is reported as skipped, with the variable that was missing. It is
not quietly dropped: a comparison that silently ran on one model would read as a comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parent.parent
for pkg in ("ontology", "serving", "api", "schemas"):
    candidate = ROOT / "services" / pkg
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))

#: The env vars a model spec may set. Restored afterwards so one model's configuration cannot
#: leak into the next — a leak would show up as agreement, which is the answer being looked for.
_VARS = (
    "LOKA_LLM_PROVIDER",
    "LOKA_LLM_BASE_URL",
    "LOKA_LLM_API_KEY",
    "LOKA_LLM_MODEL",
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
)


@contextmanager
def _configured(spec: dict[str, Any]) -> Iterator[None]:
    saved = {k: os.environ.get(k) for k in _VARS}
    try:
        for k in _VARS:
            os.environ.pop(k, None)
        os.environ["LOKA_LLM_PROVIDER"] = spec.get("provider", "openai")
        os.environ["LOKA_LLM_MODEL"] = spec["model"]
        if spec.get("base_url"):
            os.environ["LOKA_LLM_BASE_URL"] = spec["base_url"]
        key = os.environ.get(spec.get("api_key_env", "")) if spec.get("api_key_env") else None
        if key:
            os.environ["LOKA_LLM_API_KEY"] = key
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _run_once(spec: dict[str, Any], text: str, method: str) -> dict[str, Any]:
    """One extraction. Failures are recorded, not raised: a model that cannot complete the task
    is a result about that model, and losing the runs that did complete to it would be worse."""
    from loka_ontology import PARADIGMS
    from loka_ontology.builder import build
    from loka_ontology.loader import OntologyLoadError, load_ontology_str
    from loka_serving import llm_for

    started = time.time()
    with _configured(spec):
        try:
            builder = PARADIGMS[method](client=llm_for("ontology"), model=spec["model"])
            kb = build([text], builder)
        except Exception as exc:  # noqa: BLE001 - the point is to record it
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                    "seconds": round(time.time() - started, 1)}

    out: dict[str, Any] = {"ok": True, "seconds": round(time.time() - started, 1)}
    try:
        onto = load_ontology_str(kb.ontology_yaml)
        out["loads"] = True
        out["entities"] = len(onto.entities)
        out["relations"] = len(onto.relations)
    except OntologyLoadError as exc:
        out["loads"] = False
        out["load_error"] = str(exc)

    notes = getattr(builder, "notes", {}) or {}
    grounding = notes.get("grounding") or {}
    if isinstance(grounding, dict) and "grounded" in grounding:
        total = grounding.get("grounded", 0) + len(grounding.get("ungrounded", []) or [])
        out["grounded"] = round(grounding["grounded"] / total, 3) if total else None
    if notes.get("degenerate_entity_replies"):
        # Kept because it is the failure the constrained prompt was written against: the reply
        # that had to be thrown away and asked for again.
        out["degenerate_replies"] = notes["degenerate_entity_replies"]

    try:
        from loka_api.ontology_store import review_checklist

        out["review"] = len(review_checklist(kb.ontology_yaml))
    except Exception:  # noqa: BLE001 - the checklist is a nicety here, not the measurement
        pass
    return out


def _spread(runs: list[dict[str, Any]], field: str) -> str:
    values = [r[field] for r in runs if r.get(field) is not None]
    if not values:
        return "-"
    if len(values) == 1:
        return str(values[0])
    return f"{min(values)}-{max(values)}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", required=True, help="JSON file describing the models")
    ap.add_argument("--text", default="examples/supply_domain.md")
    ap.add_argument("--method", default="staged", help="extraction paradigm")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--json", dest="json_out", help="write the full results here")
    args = ap.parse_args()

    text = Path(args.text).read_text(encoding="utf-8")
    specs = json.loads(Path(args.models).read_text(encoding="utf-8"))

    print(f"document : {args.text}  ({len(text)} characters)")
    print(f"paradigm : {args.method}")
    print(f"repeats  : {args.repeat} per model\n")

    results: dict[str, Any] = {}
    for spec in specs:
        label = spec.get("label", spec["model"])
        needed = spec.get("api_key_env")
        if needed and not os.environ.get(needed):
            print(f"{label:22s} SKIPPED — {needed} is not set")
            results[label] = {"skipped": needed}
            continue

        runs = [_run_once(spec, text, args.method) for _ in range(args.repeat)]
        results[label] = runs
        failed = [r for r in runs if not r.get("ok")]
        ok = [r for r in runs if r.get("ok")]
        line = (
            f"{label:22s} entities {_spread(ok, 'entities'):>9s}   "
            f"relations {_spread(ok, 'relations'):>7s}   "
            f"grounded {_spread(ok, 'grounded'):>11s}   "
            f"review {_spread(ok, 'review'):>7s}"
        )
        if failed:
            line += f"   ({len(failed)}/{len(runs)} failed)"
        print(line)
        for r in failed:
            print(f"{'':22s}   -> {r['error']}")

    ran = {k: v for k, v in results.items() if isinstance(v, list) and any(r.get("ok") for r in v)}
    print()
    if len(ran) < 2:
        print(
            f"{len(ran)} model(s) produced a result, so nothing here is a cross-model comparison."
        )
    else:
        counts = [
            statistics.median([r["entities"] for r in runs if r.get("entities") is not None])
            for runs in ran.values()
            if any(r.get("entities") is not None for r in runs)
        ]
        if counts:
            print(
                f"median entity count across {len(counts)} models: "
                f"{min(counts):.0f} to {max(counts):.0f}"
            )
            print(
                "the claim under test is that a well-designed prompt makes this range small; "
                "the number above is the answer, whichever way it comes out."
            )

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nfull results written to {args.json_out}")


if __name__ == "__main__":
    main()
