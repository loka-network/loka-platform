"""Run every extraction paradigm over one document and tabulate what each produced.

Ouyang et al. (LLM4Onto, Semantic Web Journal) separate three paradigms for building an ontology
from text and argue that decomposing the task reduces hallucination. The paper contains no
ablation between them, so the argument is made structurally rather than measured. This measures
it on one document, which is the only way to find out whether it holds for documents like ours —
a few hundred words of closed-domain business prose, not the open-domain corpora the paper
evaluates on.

The numbers are reported whichever way they come out. If a paradigm the paper favours produces
less on this document, that is the finding.

    python examples/compare_extraction.py                       # against a running API
    python examples/compare_extraction.py --url http://host:8100
    python examples/compare_extraction.py --text examples/supply_domain.md
    python examples/compare_extraction.py --json out.json       # keep the full responses

Requires the API to have a model configured (LOKA_LLM_BUILD=1). Each paradigm is a separate
extraction, so this costs several model calls per run.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
import urllib.error
import urllib.request
from typing import Any

METHODS = ("single_shot", "staged", "relation_first", "head_tail", "cluster_first")


def _post(url: str, payload: dict[str, Any], timeout: float) -> tuple[int, Any]:
    """POST and always return a (status, body) pair.

    A paradigm that hangs must not take the comparison with it. These runs are minutes long —
    five paradigms, several model calls each — so an exception escaping here throws away every
    result already collected and the whole thing starts over. A timeout is a finding about that
    paradigm, which is exactly what this script exists to record.
    """
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - operator's URL
            return resp.status, json.load(resp)
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.load(exc)
        except Exception:  # noqa: BLE001 - a non-JSON error body is still worth reporting
            return exc.code, {"detail": exc.reason}
    except (TimeoutError, socket.timeout) as exc:  # noqa: UP041 - socket.timeout on older 3.x
        return 0, {"detail": (
            f"no reply within {timeout:.0f}s. The server may still be working: this is the "
            f"client giving up, not the extraction failing. {exc}"
        )}
    except urllib.error.URLError as exc:
        return 0, {"detail": f"could not reach {url}: {exc.reason}"}


def _summarise(method: str, status: int, body: Any, seconds: float) -> dict[str, Any]:
    if status != 200:
        detail = body.get("detail") if isinstance(body, dict) else body
        cause = detail.get("cause") if isinstance(detail, dict) else detail
        return {"method": method, "ok": False, "seconds": round(seconds, 1), "error": str(cause)}

    yaml_text = body.get("ontology_yaml", "")
    prov = body.get("provenance", {})
    # Without a model configured every method falls to the rule-based builder, which ignores the
    # paradigm entirely. Four identical rows would then read as "the paradigms make no
    # difference", which is the opposite of what happened.
    builder = body.get("builder")
    grounding = prov.get("grounding") or {}
    calls = prov.get("stage_calls") or []
    return {
        "method": method,
        "ok": True,
        "builder": builder,
        "ran_paradigm": prov.get("extraction", "n/a" if builder != "llm" else method),
        "seconds": round(seconds, 1),
        "entities": yaml_text.count("\n  - type: "),
        "relations": yaml_text.count("\n  - {name: ") - yaml_text.count("class:"),
        "verbs": yaml_text.count("class:"),
        "calls": len(calls) or 1,
        # The longest single reply is the number that decides whether a bigger domain would have
        # been truncated. It is the reason to decompose at all.
        "longest_reply": max((c.get("reply_chars", 0) for c in calls), default=None),
        # Where the wall clock went. A total says a paradigm is too slow; this says which stage.
        "slowest_stage": (
            max(calls, key=lambda c: c.get("seconds", 0)).get("stage") if calls else None
        ),
        "grounded": grounding.get("grounded"),
        "checked": grounding.get("checked"),
        "grounding_rate": grounding.get("rate"),
        "ungrounded": grounding.get("ungrounded") or [],
        "review_items": len(body.get("review", [])),
        "notes": prov.get("paradigm_notes") or {},
        "ontology_id": body.get("ontology_id"),
    }


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://localhost:8100")
    ap.add_argument("--text", default=os.path.join(here, "supply_domain.md"))
    # A staged paradigm is several model calls, and a reasoning model spends minutes per call,
    # so the whole extraction can take far longer than a single-reply one.
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--json", dest="out", help="write the full responses here")
    ap.add_argument("--methods", nargs="*", default=list(METHODS))
    args = ap.parse_args()

    with open(args.text, encoding="utf-8") as f:
        text = f.read()
    print(f"document: {args.text}  ({len(text)} chars)")
    print(f"api:      {args.url}\n")

    rows: list[dict[str, Any]] = []
    full: dict[str, Any] = {}
    for method in args.methods:
        print(f"  {method} ...", end="", flush=True)
        started = time.monotonic()
        status, body = _post(
            f"{args.url}/build-kb", {"texts": [text], "method": method}, args.timeout
        )
        row = _summarise(method, status, body, time.monotonic() - started)
        full[method] = {"status": status, "body": body}
        rows.append(row)
        print(
            f" ok ({row['seconds']}s)" if row["ok"]
            else f" FAILED after {row['seconds']}s: {row['error'][:90]}"
        )
        # Written after every method, not at the end: a run that dies on the last paradigm
        # should not cost the four that already completed.
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(full, f, indent=1, ensure_ascii=False)

    if any(r.get("ok") and r.get("builder") != "llm" for r in rows):
        print(
            "\n!! no model was used: every method fell to the rule-based builder, which ignores\n"
            "   the paradigm. The rows below are the same extraction four times, not a\n"
            "   comparison. Set LOKA_LLM_BUILD=1 and a model on the API and re-run."
        )

    def cell(row: dict[str, Any], key: str) -> str:
        value = row.get(key)
        return "-" if value is None else str(value)

    print()
    header = ("method", "ok", "secs", "ents", "rels", "verbs", "calls", "longest",
              "grounded", "review")
    print(f"{header[0]:<15}{header[1]:<5}{header[2]:>7}{header[3]:>6}{header[4]:>6}"
          f"{header[5]:>7}{header[6]:>7}{header[7]:>9}{header[8]:>10}{header[9]:>8}")
    print("-" * 80)
    for row in rows:
        grounded = (
            f"{row['grounded']}/{row['checked']}"
            if row.get("checked") else "-"
        )
        print(
            f"{row['method']:<15}{'y' if row['ok'] else 'n':<5}{cell(row, 'seconds'):>7}"
            f"{cell(row, 'entities'):>6}{cell(row, 'relations'):>6}{cell(row, 'verbs'):>7}"
            f"{cell(row, 'calls'):>7}{cell(row, 'longest_reply'):>9}{grounded:>10}"
            f"{cell(row, 'review_items'):>8}"
        )
    for row in rows:
        if not row["ok"]:
            print(f"\n{row['method']}: {row['error']}")

    for row in rows:
        if row.get("ungrounded"):
            print(f"\n{row['method']}: proposed but not found in the document — "
                  f"{', '.join(row['ungrounded'])}")
        if row.get("slowest_stage"):
            print(f"{row['method']}: slowest stage = {row['slowest_stage']}")
        for key, value in (row.get("notes") or {}).items():
            print(f"{row['method']}: {key} = {value}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(full, f, indent=1, ensure_ascii=False)
        print(f"\nfull responses -> {args.out}")


if __name__ == "__main__":
    main()
