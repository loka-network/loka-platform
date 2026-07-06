"""Mission Profile — the customer-signed configuration (doc §4.1.3).

Declares what the system may optimise for and what it may recommend: mandate, welfare
functional, hard constraints, authority. Customer-supplied and read-only to the model; the
model never derives or edits it. Loaded once per deployment; verified (signed) before use.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WelfareTerm:
    """One weighted objective in the welfare functional."""

    name: str
    weight: float


@dataclass(frozen=True)
class WelfareFunctional:
    """The scalar the decider optimises: a weighted combination of named objectives."""

    terms: tuple[WelfareTerm, ...]


@dataclass(frozen=True)
class HardConstraint:
    """A rule a recommendation may not violate (legal / statutory / prudential)."""

    name: str
    description: str


@dataclass(frozen=True)
class AuthorityRule:
    """Who may approve which class of decision."""

    actor: str
    decision_class: str


@dataclass(frozen=True)
class MissionProfile:
    """A complete, customer-signed mission profile."""

    version: str
    mandate: str
    welfare: WelfareFunctional
    hard_constraints: tuple[HardConstraint, ...] = ()
    authority: tuple[AuthorityRule, ...] = ()
    signature: str | None = None  # customer signature; must be present before use

    @property
    def is_signed(self) -> bool:
        return bool(self.signature)
