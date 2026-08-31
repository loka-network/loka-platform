"""Record one extraction in full: every instruction sent, every reply as it came back.

Review of the previous draft asked five times, in five places, for a real example. What it got
was two sentences of invented prose — *Independent sellers list items on our marketplace* — and
a claim that a real document works the same way, only larger. That claim is the thing under
doubt, so it cannot be the thing that stands in for the evidence.

This runs the extraction on the real document and writes down what happened. Nothing is
summarised: the system prompt is the string that was sent, the reply is the text that came back
before anything parsed it, and the ontology is what the loader accepted. A reader who thinks the
result is wrong can see which stage produced it.

The recording happens at the client boundary — a proxy around ``messages.create`` — so no
production path is altered to make it possible. What is recorded here is what runs when nobody
is recording.

    python scripts/extraction_transcript.py                      # staged, the real document
    python scripts/extraction_transcript.py --method cluster_first
    python scripts/extraction_transcript.py --out transcript.md

Needs a model configured (LOKA_LLM_* or OPENAI_*) and costs one extraction — four calls for
`staged`, more if a stage is retried.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
for pkg in ("ontology", "serving", "api", "adapters"):
    candidate = ROOT / "services" / pkg
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))


@dataclass
class Call:
    system: str
    user: str
    reply: str
    seconds: float
    error: str | None = None


@dataclass
class Recorder:
    """Wraps an LLM client and keeps what crossed it, in order."""

    inner: Any
    calls: list[Call] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.messages = _Messages(self)


class _Messages:
    def __init__(self, rec: Recorder) -> None:
        self._rec = rec

    def create(self, **kwargs: Any) -> Any:
        system = str(kwargs.get("system", ""))
        msgs = kwargs.get("messages") or []
        user = str(msgs[0].get("content", "")) if msgs else ""
        started = time.time()
        try:
            resp = self._rec.inner.messages.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - a failed call is part of the record
            self._rec.calls.append(
                Call(system, user, "", round(time.time() - started, 1), f"{type(exc).__name__}: {exc}")
            )
            raise
        text = "".join(
            getattr(b, "text", "") for b in getattr(resp, "content", []) or []
        )
        self._rec.calls.append(Call(system, user, text, round(time.time() - started, 1)))
        return resp


def _fence(text: str, lang: str = "") -> str:
    # A reply may itself contain a fence. Widening this one keeps the block from ending early
    # and silently truncating the evidence.
    ticks = "`" * max(3, max((len(m) for m in _runs(text)), default=0) + 1)
    return f"{ticks}{lang}\n{text.strip()}\n{ticks}"


def _runs(text: str) -> list[str]:
    out, cur = [], ""
    for ch in text:
        if ch == "`":
            cur += ch
        else:
            if cur:
                out.append(cur)
            cur = ""
    if cur:
        out.append(cur)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--text", default="examples/supply_domain.md")
    ap.add_argument("--method", default="staged")
    ap.add_argument("--out", default="extraction_transcript.md")
    args = ap.parse_args()

    from loka_ontology import PARADIGMS
    from loka_ontology.builder import build
    from loka_ontology.loader import load_ontology_str
    from loka_serving import llm_for
    from loka_serving.client import default_model

    source = Path(args.text).read_text(encoding="utf-8")
    recorder = Recorder(llm_for("ontology"))
    # Resolved here rather than left to the builder's default, so the name in the transcript is
    # the name that was sent. A record whose model field is a class default is not a record.
    builder = PARADIGMS[args.method](client=recorder, model=default_model())

    started = time.time()
    failure: str | None = None
    kb = None
    try:
        kb = build([source], builder)
    except Exception as exc:  # noqa: BLE001 - a failed extraction is still a transcript
        failure = f"{type(exc).__name__}: {exc}"
    elapsed = round(time.time() - started, 1)

    out: list[str] = []
    w = out.append

    w(f"# Extraction transcript — `{args.method}`\n")
    w(f"- document: `{args.text}` — {len(source)} characters, {len(source.split())} words")
    w(f"- model: `{getattr(builder, 'model', '?')}`")
    w(f"- calls: {len(recorder.calls)}")
    w(f"- wall clock: {elapsed}s")
    if failure:
        w(f"- **the extraction failed**: {failure}")
    w("")

    w("## 0. The document\n")
    w("Every phrase quoted as evidence below is checked against this text.\n")
    w(_fence(source, "text"))
    w("")

    for i, call in enumerate(recorder.calls, 1):
        w(f"## Call {i} — {call.seconds}s\n")
        w("**Sent (system):**\n")
        w(_fence(call.system))
        w("")
        if call.user and call.user != source:
            w("**Sent (content):**\n")
            w(_fence(call.user))
            w("")
        elif call.user == source:
            w("**Sent (content):** the whole document, above.\n")
        if call.error:
            w(f"**Failed:** {call.error}\n")
        else:
            w("**Returned, before anything parsed it:**\n")
            w(_fence(call.reply, "json"))
        w("")

    if kb is not None:
        onto = load_ontology_str(kb.ontology_yaml)
        w("## The ontology that loaded\n")
        w(
            f"{len(onto.entities)} entity types, {len(onto.relations)} relations, "
            f"{len(onto.verbs)} verbs. It passed every load-time rule; that is what "
            "*loaded* means here.\n"
        )
        w(_fence(kb.ontology_yaml, "yaml"))
        w("")

        notes = getattr(builder, "notes", {}) or {}
        # On the builder itself, not in its notes: `notes` collects what a stage remarked on,
        # while grounding is a running record the stages write into as they go.
        recorder_g = getattr(builder, "grounding", None)
        grounding = recorder_g.as_dict() if recorder_g is not None else {}
        if grounding:
            w("## Grounding\n")
            w(
                "Every proposal had to cite a phrase from the document, and the citation is "
                "checked back against the text on word tokens — so CamelCase and plurals do not "
                "decide it. What could not be found is listed, not removed: an unfound name may "
                "be a correct generalisation the text words differently, and the two being "
                "indistinguishable in the result is the thing that is unacceptable.\n"
            )
            w(f"- checked: {grounding.get('checked')}")
            w(f"- found in the source: {grounding.get('grounded')} "
              f"({grounding.get('rate', 0) * 100:.1f}%)")
            ungrounded = grounding.get("ungrounded") or []
            w(f"- not found: {len(ungrounded)}\n")
            for name in ungrounded:
                w(f"  - `{name}`")
            w("")

        for key in ("entities_below_mention_floor", "entities_without_mentions",
                    "degenerate_entity_replies"):
            if notes.get(key):
                w(f"**{key}**: {notes[key]}\n")

        try:
            from loka_api.ontology_store import review_checklist

            items = review_checklist(kb.ontology_yaml)
            w("## What a reviewer is handed\n")
            w(
                f"{len(items)} items — what a machine reading prose could not decide. It is not "
                "a list of mistakes; it is the list of questions extraction cannot answer.\n"
            )
            for item in items:
                w(f"- **{item['kind']}** — `{item.get('target')}`: {item.get('detail')}")
            w("")
        except Exception as exc:  # noqa: BLE001
            w(f"(the checklist could not be produced: {exc})\n")

    Path(args.out).write_text("\n".join(out), encoding="utf-8")
    print(f"{len(recorder.calls)} calls, {elapsed}s -> {args.out}")
    if failure:
        print(f"the extraction failed: {failure}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
