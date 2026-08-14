"""The decision half for the health scenario: Simulation -> Policy -> Decision memorandum.

The chain does not stop at a query answer — a formalized query flows through:

    ... -> Simulation: scenario evaluation and selection
        -> Policy: control policy -> decision memorandum + replayable audit -> response

This module connects Workflow B's ``orders`` result (a controlled projection with a 95% interval)
to that chain, so the demo produces a governed artefact — scenarios, a chosen policy, and a
replayable audit hash — instead of a bare number.

It is a basic, honest stand-in for the full simulator and policy model (no agent society, no
CVaR welfare functional) — labelled ``basic`` in the output. The constraint it enforces is sourced
from the ontology action's ``guard`` (load-bearing: the governance gate is Ω's, not hardcoded).
"""

from __future__ import annotations

import hashlib
from typing import Any


def evaluate_scenarios(projection: dict[str, Any]) -> list[dict[str, Any]]:
    """Scenario evaluation (simulator stand-in): bound the projection by the *effect's* CI.

    Under-5 mortality is a *lower-is-better* outcome, so the interval's upper bound is the adverse
    scenario and the lower bound the favourable one. The bounds come from ``interval_95``, which is
    the CI implied by the effect's standard error — not the far wider level prediction interval,
    which answers "where can one country sit relative to the fitted surface" and would produce
    absurd scenarios.

    The ``prob`` weights are placeholders (a nominal/bounds split), not calibrated probabilities;
    a real simulator would produce a distribution. They are labelled as such in the output.
    """
    point = projection["projected_outcome"]
    lo, hi = projection["interval_95"]
    return [
        {"scenario_id": "nominal", "kind": "nominal", "under5_mortality": point,
         "prob": 0.6, "prob_basis": "placeholder"},
        {"scenario_id": "adverse", "kind": "adverse", "under5_mortality": hi,
         "prob": 0.2, "prob_basis": "placeholder"},
        {"scenario_id": "favorable", "kind": "favorable", "under5_mortality": lo,
         "prob": 0.2, "prob_basis": "placeholder"},
    ]


# Below this share of explained variation, a determined direction is reported together with the
# fact that the model explains little. The value is a judgement, so it is published in the memo
# rather than applied silently — a reader who disagrees can see what was applied.
WEAK_FIT_R2 = 0.10


def _explanatory_power(projection: dict[str, Any]) -> dict[str, Any]:
    """How much of the outcome the fitted model accounts for.

    An interval says whether an effect can be told apart from zero; it says nothing about whether
    the relationship matters for any individual case. Those are different failures — one is "we
    cannot tell", the other is "we can tell, and it accounts for almost nothing" — and a report
    carrying only a point estimate and an interval makes them look alike.
    """
    fit = projection.get("fit", {})
    r2 = fit.get("r2")
    return {
        "r2": r2,
        "n": fit.get("n"),
        "weak_fit_threshold": WEAK_FIT_R2,
        "explains_little": isinstance(r2, (int, float)) and r2 < WEAK_FIT_R2,
    }


def _admissibility(projection: dict[str, Any]) -> dict[str, Any]:
    """Whether the evidence behind this projection may justify a recommendation.

    An identification status records *how* an effect was established; the admissibility matrix
    records what each kind of establishment is good enough for. They are separate questions, and
    a system that reports the first without applying the second leaves the reader to know the
    rule. A projection fitted across observed data is ``observational`` — correlational, with no
    identification strategy — and the matrix does not admit that as the justification for a
    recommendation, only for conditioning a forecast.

    The check is reported rather than enforced by refusal: the number is still useful, and the
    caller may be asking for conditioning. What must not happen is a recommendation resting on
    evidence the matrix rejects, without saying so.
    """
    from loka_causal import is_admissible
    from loka_schemas import IdentificationStatus, UseCase

    label = projection.get("identification", "observational")
    try:
        status = IdentificationStatus(label)
    except ValueError:
        # An unrecognised label is admitted for nothing. Treating it as safe would make a typo
        # the way to bypass the matrix.
        return {
            "identification": label,
            "recognised": False,
            "admissible_for": [],
            "may_justify_a_recommendation": False,
        }

    admitted = [u.value for u in UseCase if is_admissible(status, u)]
    return {
        "identification": status.value,
        "recognised": True,
        "admissible_for": admitted,
        "may_justify_a_recommendation": UseCase.BLOCK_A_JUSTIFICATION.value in admitted,
    }


def _audit_inputs(
    projection: dict[str, Any], *, iso: str, ontology_version: str, method_name: str
) -> dict[str, Any]:
    """Everything a replay needs: the authority, the method, the inputs, and the data it saw.

    The earlier hash bound only Ω's version, the method, and the query inputs. That is not enough
    to replay a decision: the projection is fitted across the whole panel, so a revision to *other*
    countries' rows changes this country's answer while every hashed field stays identical — two
    different answers under one hash. Including the fitted sample's digest (and the control values
    held fixed, and the sample size) closes that: same inputs and same data reproduce the hash;
    a data revision produces a different one.
    """
    fit = projection.get("fit", {})
    controls = projection.get("controls_held_fixed", {})
    return {
        "ontology_version": ontology_version,
        "method": method_name,
        "entity": iso,
        "outcome": projection["outcome"],
        "dial": projection["dial"],
        "dial_change": f"{projection['current_dial']}->{projection['new_dial']}",
        "outcome_current": projection["current_outcome"],
        "controls_held_fixed": {k: controls[k] for k in sorted(controls)},
        "sample_digest": fit.get("sample_digest"),
        "sample_n": fit.get("n"),
        "sample_params": fit.get("params"),
        # Without a sample digest the hash cannot distinguish two answers computed from different
        # data — say so rather than let the audit trail look sound when it is not.
        "replayable": fit.get("sample_digest") is not None,
    }


def _audit_preimage(inputs: dict[str, Any]) -> str:
    """The exact string hashed — published alongside the hash so anyone can recompute it."""
    controls = "|".join(f"{k}={v}" for k, v in inputs["controls_held_fixed"].items())
    return (
        f"{inputs['ontology_version']}|{inputs['method']}|{inputs['entity']}|"
        f"{inputs['outcome']}|{inputs['dial']}|{inputs['dial_change']}|"
        f"{inputs['outcome_current']}|{controls}|"
        f"{inputs['sample_digest']}|{inputs['sample_n']}|{inputs['sample_params']}"
    )


def select_and_decide(
    projection: dict[str, Any],
    scenarios: list[dict[str, Any]],
    *,
    iso: str,
    ontology_version: str,
    guard: str,
    method_name: str,
) -> dict[str, Any]:
    """Scenario selection, welfare judgement, guard, and audit hash.

    Welfare objective is to minimise under-5 mortality. The recommendation states whether the
    proposed spending improves it, always carrying the projection's identification label so the
    read stays honest (observational, not a causal guarantee). The audit hash binds Ω's version,
    the method, and the exact inputs so the decision is replayable (replayable audit).
    """
    current = projection["current_outcome"]
    chosen = next(s for s in scenarios if s["kind"] == "nominal")
    projected = chosen["under5_mortality"]
    delta = round(projected - current, 3)

    # Is the effect distinguishable from zero? If its 95% CI straddles 0, the data does not
    # support claiming a direction, and the memo must not claim one.
    eff_lo, eff_hi = projection.get("effect_interval_95", [None, None])
    significant = eff_lo is not None and (eff_lo > 0 or eff_hi < 0)

    new_dial = projection["new_dial"]
    audit_inputs = _audit_inputs(
        projection, iso=iso, ontology_version=ontology_version, method_name=method_name
    )
    audit = hashlib.sha256(_audit_preimage(audit_inputs).encode()).hexdigest()[:16]

    power = _explanatory_power(projection)

    if not significant:
        rec = (
            f"No effect distinguishable from zero: raising {projection['dial']} to {new_dial} "
            f"for {iso} shifts {projection['outcome']} by {delta} "
            f"(95% CI {projection.get('effect_interval_95')}), an interval that includes 0. "
            f"The data does not support claiming this change reduces {projection['outcome']}."
        )
    else:
        improves = delta < 0
        verb = "reduces" if improves else "increases"
        rec = (
            f"Raising {projection['dial']} to {new_dial} {verb} {projection['outcome']} for {iso}: "
            f"{current} -> {projected} ({'−' if improves else '+'}{abs(delta)}, "
            f"95% CI {projection.get('effect_interval_95')})."
        )
        if power["explains_little"]:
            # A precise estimate of a relationship that accounts for little of what is observed
            # is a different situation from an imprecise one, and must not read the same.
            rec += (
                f" The direction is determined, but the fitted model accounts for "
                f"{power['r2'] * 100:.1f}% of the variation in {projection['outcome']} "
                f"(r²={power['r2']}, n={power['n']}): the estimate is precise about a "
                f"relationship that explains little of any individual case."
            )

    admissibility = _admissibility(projection)
    if not admissibility.get("may_justify_a_recommendation", False):
        # Said in the recommendation, not left in a field: a reader who takes the sentence at
        # face value would otherwise have to know the admissibility rule to apply it themselves.
        allowed = ", ".join(admissibility["admissible_for"]) or "no declared use"
        rec += (
            f" This rests on {admissibility['identification']} evidence, which the admissibility "
            f"matrix does not accept as the justification for a recommendation — it is admitted "
            f"for {allowed}. Read the figure as conditioning, not as grounds for acting."
        )

    return {
        "recommendation": rec,
        "effect": delta,
        "effect_interval_95": projection.get("effect_interval_95"),
        "effect_distinguishable_from_zero": significant,
        "explanatory_power": power,
        "admissibility": admissibility,
        "welfare_objective": f"minimize {projection['outcome']}",
        "chosen_scenario": chosen["scenario_id"],
        "contingency_scenario": "adverse",
        "constraints_enforced": [guard] if guard else [],
        "constraint_satisfied": (new_dial or 0) > 0,  # the guard: health_exp_per_capita > 0
        "identification": projection.get("identification", "observational"),
        "audit_manifest": audit,
        "audit_inputs": audit_inputs,  # what the hash binds — recompute it to verify a replay
        "policy_engine": "basic (the full policy model is not implemented)",
    }


def evaluate_and_decide(
    projection: dict[str, Any], *, iso: str, ontology_version: str, guard: str, method_name: str
) -> dict[str, Any]:
    """Run the projection through Simulation -> Policy and return {scenarios, decision}."""
    scenarios = evaluate_scenarios(projection)
    decision = select_and_decide(
        projection, scenarios, iso=iso, ontology_version=ontology_version,
        guard=guard, method_name=method_name,
    )
    return {"scenarios": scenarios, "decision": decision}
