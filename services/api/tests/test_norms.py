"""N(s, ac) — the deontic layer: permitted | mandatory | forbidden.

A two-valued model (act / do not act) can say an action was taken when it should not have been.
It cannot say an action was *not* taken when it should have been, because "not acting" is its
default and defaults are not violations. Obligation needs a third value, and these tests are
written around the cases the two-valued version could not state at all.
"""

from __future__ import annotations

from typing import Any

import pytest
from loka_api.actions import propose_actions
from loka_ontology import OntologyEngine, OntologyLoadError, load_ontology_str
from loka_schemas import (
    ManifestPins,
    ScenarioStatePackage,
    ScenarioWorldModel,
    WelfareFunctional,
)

_ONTOLOGY = """
version: v1
entities:
  - type: Seller
    properties:
      - {name: seller_id, type: string, required: true}
  - type: Customer
    properties:
      - {name: customer_id, type: string, required: true}
      - {name: delay_days, type: integer}
      - {name: open_disputes, type: integer}
      - {name: channel_open, type: integer}
      - {name: blackout, type: integer}
verbs:
  - {name: NOTIFY_DELAY, class: communicative}
  - {name: SUSPEND_SELLER, class: institutional}
constraints:
  - {verb: NOTIFY_DELAY, agent_must_be: Seller, target_must_be: [Customer]}
  - {verb: SUSPEND_SELLER, agent_must_be: Seller, target_must_be: [Seller]}
actions:
  - {name: NotifyDelay, verb: NOTIFY_DELAY, target: Customer, guard: "delay_days >= 1"}
  - {name: SuspendSeller, verb: SUSPEND_SELLER, target: Seller, guard: "on_time_rate < 100"}
norms:
  - {name: LateOrdersMustBeDisclosed, action: NotifyDelay, status: mandatory,
     when: "delay_days >= 3"}
  - {name: NoSuspensionWhileDisputed, action: SuspendSeller, status: forbidden,
     when: "open_disputes >= 1"}
"""


class _World:
    def __init__(self, engine: Any) -> None:
        self.engine = engine


def _propose(ontology: str, state: dict[str, object]) -> dict[str, Any]:
    world = _World(OntologyEngine(load_ontology_str(ontology)))
    wqt = ScenarioWorldModel(
        scenario_id="s",
        query_id="q",
        state_package=ScenarioStatePackage(entities=("Seller",), state_slice=state),
        welfare=WelfareFunctional(terms=()),
        hard_constraints=(),
        manifest=ManifestPins(omega_version="v1", et_snapshot="x", mission_version="m1"),
    )
    return {p.action_name: p for p in propose_actions(world, wqt)}


# ---- the case a two-valued model cannot state ----

def test_an_obligation_makes_not_acting_the_violation() -> None:
    by_name = _propose(_ONTOLOGY, {"Order.1.delay_days": 5})
    notify = by_name["NotifyDelay"]
    assert notify.deontic_status == "mandatory"
    assert notify.status == "required"          # not merely "proposed"
    assert notify.omission_violates is True     # staying silent is itself the violation
    assert notify.norm == "LateOrdersMustBeDisclosed"


def test_below_the_threshold_the_same_action_is_only_permitted() -> None:
    """The norm is conditional on the state, so the status has to move with it — otherwise it is
    a label on the action, not a function of (s, ac)."""
    notify = _propose(_ONTOLOGY, {"Order.1.delay_days": 1})["NotifyDelay"]
    assert notify.deontic_status == "permitted"
    assert notify.status == "proposed"
    assert notify.omission_violates is False


def test_a_forbidden_action_is_blocked_and_names_the_norm() -> None:
    suspend = _propose(_ONTOLOGY, {"Seller.s1.open_disputes": 2})["SuspendSeller"]
    assert suspend.deontic_status == "forbidden"
    assert suspend.status == "blocked"
    assert suspend.blocked_by == "norm: NoSuspensionWhileDisputed"


def test_the_same_action_is_available_once_the_condition_lifts() -> None:
    suspend = _propose(_ONTOLOGY, {"Seller.s1.open_disputes": 0})["SuspendSeller"]
    assert suspend.deontic_status == "permitted"
    assert suspend.status == "proposed"


# ---- a norm is not a guard ----

def test_an_obligation_whose_guard_fails_is_reported_not_hidden() -> None:
    """Obliged to act, unable to act. The system cannot act its way out of this state, and a
    reader has to be able to see that — so the obligation stays visible on a blocked proposal
    rather than being dropped because acting is impossible."""
    onto = _ONTOLOGY.replace('guard: "delay_days >= 1"', 'guard: "channel_open >= 1"')
    notify = _propose(onto, {"Order.1.delay_days": 5, "Order.1.channel_open": 0})["NotifyDelay"]
    assert notify.deontic_status == "mandatory"   # the obligation still holds
    assert notify.guard_status == "not_satisfied"  # and it cannot be discharged
    assert notify.status == "blocked"


def test_an_unreadable_condition_leaves_the_norm_silent() -> None:
    """A norm whose condition cannot be evaluated must not be treated as holding: an obligation
    asserted on a condition nobody can check is not one anyone can be held to."""
    onto = _ONTOLOGY.replace('when: "delay_days >= 3"', 'when: "the delay is material"')
    notify = _propose(onto, {"Order.1.delay_days": 5})["NotifyDelay"]
    assert notify.deontic_status == "permitted"
    assert notify.norm is None


# ---- contradictory norms are a defect, not a tie to break ----

def test_a_state_that_is_both_obliged_and_forbidden_is_reported() -> None:
    onto = _ONTOLOGY + (
        "  - {name: NeverNotifyDuringBlackout, action: NotifyDelay, status: forbidden,\n"
        '     when: "blackout >= 1"}\n'
    )
    notify = _propose(onto, {"Order.1.delay_days": 5, "Order.1.blackout": 1})["NotifyDelay"]
    assert notify.normative_conflict == (
        "LateOrdersMustBeDisclosed",
        "NeverNotifyDuringBlackout",
    )
    assert notify.deontic_status == "forbidden"  # forbidden wins, but the pair is still named


# ---- an obligation does not grant autonomy ----

def test_a_mandatory_action_still_stops_at_the_human_boundary() -> None:
    notify = _propose(_ONTOLOGY, {"Order.1.delay_days": 5})["NotifyDelay"]
    assert notify.requires_confirmation is True


# ---- A = Au ⊎ Ac ----

def test_an_uncontrollable_action_is_never_proposed() -> None:
    onto = _ONTOLOGY.replace(
        "norms:",
        "  - {name: CarrierStrike, verb: NOTIFY_DELAY, target: Customer, controllable: false}\n"
        "norms:",
    )
    names = _propose(onto, {"Order.1.delay_days": 5}).keys()
    assert "CarrierStrike" not in names
    assert "NotifyDelay" in names  # the controllable half is unaffected


# ---- CΩ R9 / R10: a norm that cannot fire is caught at load time ----

def test_a_norm_on_an_undefined_action_is_refused() -> None:
    bad = _ONTOLOGY.replace("action: NotifyDelay", "action: NotifyDelayy", 1)
    with pytest.raises(OntologyLoadError, match="NotifyDelayy, which is not defined"):
        load_ontology_str(bad)


def test_a_norm_on_an_uncontrollable_action_is_refused() -> None:
    """N(s, ac) is defined on the controllable half of A. Forbidding something nobody chose has
    no addressee."""
    bad = _ONTOLOGY.replace(
        'actions:\n  - {name: NotifyDelay, verb: NOTIFY_DELAY, target: Customer, '
        'guard: "delay_days >= 1"}',
        'actions:\n  - {name: NotifyDelay, verb: NOTIFY_DELAY, target: Customer, '
        'guard: "delay_days >= 1", controllable: false}',
    )
    with pytest.raises(OntologyLoadError, match="uncontrollable"):
        load_ontology_str(bad)


def test_a_norm_conditioned_on_an_undeclared_attribute_is_refused() -> None:
    """R11. The rule that catches a typo in a condition — and this one was written because the
    supply ontology shipped with two: norms conditioned on ``delay_days`` and ``open_disputes``
    when Ω declares ``days_late`` and nothing about disputes at all. Both could never fire, so
    the forbidding one permitted, unconditionally, exactly what it was written to forbid."""
    bad = _ONTOLOGY.replace('when: "delay_days >= 3"', 'when: "delya_days >= 3"')
    with pytest.raises(OntologyLoadError, match="could never fire"):
        load_ontology_str(bad)


def test_r11_does_not_reject_a_condition_it_cannot_read() -> None:
    """A prose condition is unevaluable, not wrong. It is already handled at runtime by leaving
    the norm silent, which is visible; refusing to load would block a legitimate draft."""
    ok = _ONTOLOGY.replace('when: "delay_days >= 3"', 'when: "the delay is material"')
    load_ontology_str(ok)  # loads


def test_an_unknown_deontic_status_names_what_is_allowed() -> None:
    bad = _ONTOLOGY.replace("status: mandatory", "status: recommended")
    with pytest.raises(OntologyLoadError, match="expected one of forbidden, mandatory, permitted"):
        load_ontology_str(bad)
