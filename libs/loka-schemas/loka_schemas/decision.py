"""DecisionMemo — the policy stage's output (S6 PolicyFormer).

Three blocks (recommended / mandated / contingency) plus an audit manifest hash so a run can
be replayed. Every figure should eventually trace to evidence; the reference/stub policy fills
the shell and the real PolicyFormer fills the substance.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DecisionMemo:
    """Decision memorandum returned to the speaker, with an audit manifest for replay."""

    query_id: str
    recommendation: str
    rationale: str
    block_A_recommended: Mapping[str, object] = field(default_factory=dict)
    block_B_mandated: Mapping[str, object] = field(default_factory=dict)
    block_C_contingency: Mapping[str, object] = field(default_factory=dict)
    evidence_refs: tuple[str, ...] = ()
    audit_manifest: str = ""
