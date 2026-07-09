"""Tests for the admissibility matrix (doc Table 3, simplified)."""

from __future__ import annotations

from loka_causal import is_admissible
from loka_schemas import IdentificationStatus as S
from loka_schemas import UseCase


def test_block_a_requires_structural_or_experimental() -> None:
    assert is_admissible(S.STRUCTURAL, UseCase.BLOCK_A_JUSTIFICATION)
    assert is_admissible(S.EXPERIMENTAL, UseCase.BLOCK_A_JUSTIFICATION)
    assert is_admissible(S.QUASI_EXPERIMENTAL, UseCase.BLOCK_A_JUSTIFICATION)
    assert not is_admissible(S.OBSERVATIONAL, UseCase.BLOCK_A_JUSTIFICATION)
    assert not is_admissible(S.EXPERT_PRIOR, UseCase.BLOCK_A_JUSTIFICATION)


def test_simulator_derived_is_never_admissible() -> None:
    for uc in UseCase:
        assert not is_admissible(S.SIMULATOR_DERIVED, uc)


def test_hard_constraint_sources() -> None:
    assert is_admissible(S.DEFINITIONAL, UseCase.HARD_CONSTRAINT)
    assert is_admissible(S.INSTITUTIONAL, UseCase.HARD_CONSTRAINT)
    assert not is_admissible(S.OBSERVATIONAL, UseCase.HARD_CONSTRAINT)


def test_authority_restriction_is_institutional() -> None:
    assert is_admissible(S.INSTITUTIONAL, UseCase.AUTHORITY_RESTRICTION)
    assert not is_admissible(S.STRUCTURAL, UseCase.AUTHORITY_RESTRICTION)


def test_forecast_allows_predictive_residual_but_block_a_does_not() -> None:
    assert is_admissible(S.PREDICTIVE_ONLY, UseCase.FORECAST_CONDITIONING)
    assert not is_admissible(S.PREDICTIVE_ONLY, UseCase.BLOCK_A_JUSTIFICATION)
