"""Admissibility matrix — which identification status may be used for which purpose.

A simplified, explicit form of doc Table 3. The key rules preserved:
  - Block A justification requires structural / experimental / quasi-experimental.
  - Hard constraints come from definitional / institutional / structural.
  - Authority restrictions come from institutional rules.
  - Simulator-derived claims are admissible for nothing here (they live in the hypothesis
    layer and only feed G4 review).
"""

from __future__ import annotations

from loka_schemas import IdentificationStatus, UseCase

S = IdentificationStatus

_ADMISSIBLE: dict[UseCase, frozenset[IdentificationStatus]] = {
    UseCase.HARD_CONSTRAINT: frozenset({S.DEFINITIONAL, S.INSTITUTIONAL, S.STRUCTURAL}),
    UseCase.FORECAST_CONDITIONING: frozenset(
        {
            S.DEFINITIONAL,
            S.STRUCTURAL,
            S.EXPERIMENTAL,
            S.QUASI_EXPERIMENTAL,
            S.OBSERVATIONAL,
            S.PREDICTIVE_ONLY,
            S.EXPERT_PRIOR,
        }
    ),
    UseCase.BLOCK_A_JUSTIFICATION: frozenset(
        {S.STRUCTURAL, S.EXPERIMENTAL, S.QUASI_EXPERIMENTAL}
    ),
    UseCase.AUTHORITY_RESTRICTION: frozenset({S.INSTITUTIONAL}),
}


def is_admissible(status: IdentificationStatus, use_case: UseCase) -> bool:
    """Whether a claim with ``status`` may be used for ``use_case``."""
    return status in _ADMISSIBLE.get(use_case, frozenset())
