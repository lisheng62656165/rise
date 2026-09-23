"""Schema-agnostic event credit for the EDS-ECA v2 prototype.

This module is deliberately outcome-free.  It turns an existing public event
ledger into a small state-transition view, scores *changes* relative to the
current incumbent, and chooses a bounded repair window.  It does not decide
whether a task passed and it never replays literal arguments or identifiers.
"""

from __future__ import annotations

import os
import re
from typing import Any, Mapping, Sequence


HIGH_RISK = {
    "failure_boundary",
    "repeated_failed_mutation",
    "missing_readback_candidate",
    "pending_consent_or_commit",
    "unknown_evidence",
    "target_drift_candidate",
    "parameter_change_candidate",
    "order_violation_candidate",
}

SEVERE_RISK = {
    "failure_boundary",
    "repeated_failed_mutation",
    "target_drift_candidate",
    "parameter_change_candidate",
    "order_violation_candidate",
}

RISK_WEIGHTS = {
    "failure_boundary": 3.0,
    "repeated_failed_mutation": 3.0,
    "target_drift_candidate": 2.0,
    "parameter_change_candidate": 2.0,
    "order_violation_candidate": 2.0,
    "unknown_evidence": 1.5,
    "missing_readback_candidate": 1.0,
    "pending_consent_or_commit": 0.5,
}

READ_PREFIXES = ("get_", "search_", "list_", "lookup_", "query_", "find_", "read_")
WRITE_PREFIXES = (
    "add_", "create_", "remove_", "delete_", "send_", "set_", "update_",
    "change_", "modify_", "cancel_", "book_", "purchase_", "commit_",
)


def _normalized_symbol(value: Any) -> str:
    """Normalize common tool-name spellings without retaining literal values."""
    text = str(value or "").strip()
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()


def _event_key(event: Mapping[str, Any]) -> tuple[Any, ...]:
    """Match public event roles without requiring stable object identifiers.

    Candidate rollouts run in fresh environments.  A dataset may therefore
    expose a different booking/cart/order identifier even when the operation
    is semantically the same.  Matching on literal refs turns that harmless
    renaming into a fake loss and gain.  The fingerprint keeps public
    structure that distinguishes operations while using only the *shape* of
    the referenced entities, never their values.
    """
    arguments = event.get("arguments") or {}
    shapes = event.get("argument_shapes")
    if shapes is None:
        shapes = sorted(argument_shape(str(key), value) for key, value in arguments.items())
    entity_count = int(event.get("entity_reference_count", len(event.get("entity_refs") or ())))
    if _role_aware_enabled():
        # External benchmarks expose different numbers of public references
        # for the same semantic mutation.  Preserve presence versus absence
        # and cap multiplicity at two; exact counts remain the default for
        # the historical StateBench configuration.
        entity_count = min(entity_count, 2)
    return (
        str(event.get("event_class") or "other"),
        operation_shape(event.get("operation"), event.get("event_class")),
        # Read-only tools are often renamed across benchmarks and should keep
        # the broad structural alignment.  Role-aware matching is reserved for
        # high-risk mutations, where conflating cancellation, refund, and
        # booking can corrupt incremental recovery credit.
        tool_role_shape(event.get("tool"))
        if _role_aware_enabled()
        and str(event.get("event_class") or "") == "mutation"
        and (
            # The explicit general mode must align ordinary successful
            # mutations too: external ledgers often attach risk tags only to
            # failures, not to the successful action that follows recovery.
            tool_role_shape(event.get("tool")) != "generic"
            or _risk_tags(event) & HIGH_RISK
        )
        else "generic",
        tuple(sorted(str(value) for value in shapes)),
        interaction_shape(event.get("interaction")),
        entity_count,
    )


def interaction_shape(value: Any) -> str:
    """Normalize benchmark-specific transaction-phase labels."""
    lowered = _normalized_symbol(value)
    if not lowered or lowered in {"none", "read", "observe", "observation"}:
        return "none"
    if any(token in lowered for token in ("preview", "draft", "quote", "simulate", "dry_run")):
        return "preview"
    if any(token in lowered for token in (
        "commit", "committed", "confirm", "confirmed", "final", "execute", "execution",
    )):
        return "commit"
    if any(token in lowered for token in ("retry", "recover", "repair", "rollback")):
        return "recovery"
    return "other"


def operation_shape(value: Any, event_class: Any = None) -> str:
    """Map dataset-specific operation names to a stable public action class."""
    lowered = _normalized_symbol(value)
    read_words = ("read", "get", "fetch", "retrieve", "search", "list", "lookup", "query", "find", "observe", "inspect")
    write_words = (
        "write", "create", "add", "update", "edit", "delete", "remove", "send", "set",
        "change", "modify", "cancel", "book", "purchase", "commit", "execute", "apply",
        # Recovery and fulfillment verbs are common across tool benchmarks,
        # but are often absent from dataset-specific operation enums.
        "refund", "return", "reimburse", "restore", "rollback", "retry",
        "confirm", "approve", "ship", "deliver", "fulfill", "close",
    )
    if lowered in set(read_words) or any(
        lowered.startswith(f"{word}_") or f"_{word}_" in f"_{lowered}_"
        for word in read_words
    ):
        return "read"
    if lowered in set(write_words) or any(
        lowered.startswith(f"{word}_") or f"_{word}_" in f"_{lowered}_"
        for word in write_words
    ):
        return "write"
    if str(event_class or "").strip().lower() in {"mutation", "verification", "observation"}:
        return "write" if str(event_class).strip().lower() == "mutation" else "read"
    return "unknown"


def _role_aware_enabled() -> bool:
    """Enable semantic tool-role matching only for an explicit ablation."""
    return os.environ.get("EDS_ECA_ROLE_AWARE", "0").lower() in {
        "1", "true", "yes", "on",
    }


def tool_role_shape(value: Any) -> str:
    """Map tool names to portable transaction roles for event alignment.

    Literal tool names are often benchmark-specific, while operation-only
    alignment merges materially different writes (for example cancellation and
    refund).  Role tokens retain the decision-relevant distinction without
    exposing identifiers or requiring a dataset-specific tool registry.
    """
    lowered = _normalized_symbol(value)
    role_patterns = (
        ("cancellation", ("cancel", "void", "revoke")),
        ("return", ("return", "rma")),
        ("refund", ("refund", "reimburse", "credit")),
        ("booking", ("book", "reservation", "reserve")),
        ("cart", ("cart", "basket")),
        ("payment", ("payment", "pay", "charge")),
        ("shipping", ("ship", "delivery", "shipment")),
        ("promotion", ("promo", "promotion", "discount", "coupon")),
        ("account", ("account", "customer", "user", "profile")),
        ("flight", ("flight", "itinerary", "fare")),
        ("product", ("product", "item", "variant", "inventory")),
        ("policy", ("policy", "eligib", "rule")),
    )
    for role, tokens in role_patterns:
        if any(token in lowered for token in tokens):
            return role
    return "generic"


def public_receipt_text(value: Any) -> str:
    """Extract a public receipt status from string, boolean, or JSON-like data."""
    if isinstance(value, bool):
        return "SUCCESS" if value else "FAILED"
    if isinstance(value, Mapping):
        # Keep only public status fields; never serialize arbitrary payloads
        # that could contain identifiers or evaluator output.
        fields = ("status", "state", "result", "outcome", "success", "ok", "error")
        parts = []
        for field in fields:
            if field in value:
                item = value[field]
                if isinstance(item, bool):
                    parts.append("SUCCESS" if item else "FAILED")
                elif isinstance(item, (str, int, float)):
                    parts.append(str(item))
        return " ".join(parts) or "UNKNOWN"
    return str(value or "UNKNOWN")


def argument_shape(key: str, value: Any) -> str:
    """Map dataset-specific argument names to a small public type shape."""
    lowered = key.lower()
    if isinstance(value, bool):
        return "boolean"
    if any(token in lowered for token in ("date", "time", "day", "month")):
        return "temporal"
    if any(token in lowered for token in (
        "id", "ref", "code", "number", "sku", "entity", "object", "item",
        "order", "booking", "reservation", "product", "user", "account",
    )):
        return "identifier"
    if any(token in lowered for token in ("price", "fee", "cost", "total", "amount", "count", "qty")):
        return "numeric"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "numeric"
    return "text"


def normalize_external_public_event(event: Mapping[str, Any], position: int = -1) -> dict[str, Any]:
    """Map a public event from another tool benchmark into the v2 schema.

    Adapters may expose only a tool name, argument keys, and a receipt class.
    This function deliberately converts argument keys to typed placeholders;
    literal values, hidden state, and evaluator fields never enter the ledger.
    """
    tool = str(event.get("tool") or event.get("name") or "unknown")
    lowered = _normalized_symbol(tool)
    operation = str(event.get("operation") or "").upper()
    if operation not in {"READ", "WRITE", "CREATE", "UPDATE", "DELETE", "MUTATE"}:
        shape = operation_shape(lowered)
        operation = "READ" if shape == "read" else "WRITE" if shape == "write" else "UNKNOWN"
    receipt = public_receipt_text(
        event.get("result") or event.get("receipt_status")
        or event.get("result_category") or "UNKNOWN"
    ).upper()
    if any(token in receipt for token in ("FAIL", "ERROR", "REJECT", "DENIED", "INVALID")):
        result = "FAILED"
    elif any(token in receipt for token in ("SUCCESS", "RECEIPT", "ACKNOWLEDGED", "OK")):
        result = "SUCCESS"
    else:
        result = "UNKNOWN"
    if result == "FAILED":
        event_class = "failure"
    elif operation in {"WRITE", "CREATE", "UPDATE", "DELETE", "MUTATE"}:
        event_class = "mutation"
    elif operation == "READ":
        event_class = "verification" if event.get("verifies_event_id") else "observation"
    else:
        event_class = "other"
    argument_keys = event.get("argument_keys")
    if argument_keys is None:
        argument_keys = list((event.get("arguments") or {}).keys())
    arguments = {str(key): "<public-shape>" for key in argument_keys}
    tags = {str(value) for value in event.get("risk_tags") or ()}
    if result == "FAILED":
        tags.add("failure_boundary")
    if result == "UNKNOWN":
        tags.add("unknown_evidence")
    return {
        "event_id": str(event.get("event_id") or f"external-{position:04d}"),
        "position": int(event.get("position", position)),
        "event_class": event_class,
        "operation": operation,
        "tool": tool,
        "entity_refs": ["public-entity"] * int(event.get("entity_reference_count") or 0),
        "arguments": arguments,
        "result": result,
        "interaction": str(event.get("interaction") or ""),
        "risk_tags": sorted(tags),
        "dependencies": list(event.get("dependencies") or ()),
    }


def _risk_tags(event: Mapping[str, Any]) -> set[str]:
    return set(str(x) for x in event.get("risk_tags") or ())


def _is_verified(event: Mapping[str, Any]) -> bool:
    tags = _risk_tags(event)
    if "missing_readback_candidate" in tags or "unknown_evidence" in tags:
        return False
    result = str(event.get("result") or event.get("result_category") or "").upper()
    if result in {"FAILED", "REJECTED", "ERROR", "DENIED", "INVALID"}:
        return False
    if (
        _role_aware_enabled()
        and str(event.get("event_class") or "") == "mutation"
    ):
        # In the explicit cross-dataset mode, a successful receipt is only an
        # action acknowledgement.  State evidence requires a public read-back
        # link; this avoids rewarding unverified recovery mutations.
        return bool(event.get("readback_event_ids"))
    return (
        str(event.get("event_class") or "") == "verification"
        or bool(event.get("readback_event_ids"))
        or result in {"SUCCESS", "ACKNOWLEDGED"}
        and str(event.get("event_class") or "") in {"mutation", "verification"}
    )


def canonical_event(event: Mapping[str, Any]) -> dict[str, Any]:
    """Return a dataset-independent event record without literal replay data."""
    tags = sorted(_risk_tags(event))
    arguments = event.get("arguments") or {}
    entity_count = int(
        event.get("entity_reference_count", len(event.get("entity_refs") or ()))
    )
    return {
        "event_id": str(event.get("event_id") or ""),
        "position": int(event.get("position", -1)),
        "event_class": str(event.get("event_class") or "other"),
        "operation": str(event.get("operation") or "UNKNOWN"),
        "interaction": interaction_shape(event.get("interaction")),
        "argument_shapes": sorted(
            argument_shape(str(key), value) for key, value in arguments.items()
        ),
        "entity_reference_count": entity_count,
        "result_category": str(
            event.get("result") or event.get("result_category") or "UNKNOWN"
        ),
        "verified": _is_verified(event),
        "risk_tags": tags,
        "dependencies": sorted(str(x) for x in event.get("dependencies") or ()),
    }


def public_risk_events(ledger: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Propagate hierarchical public-risk evidence onto atomic events.

    The event analyzer stores FRD/RDS evidence at recovery-window and
    recovery-decision levels, while ECA compares atomic events.  Reading only
    ``raw_events`` silently drops those risk tags.  This adapter preserves the
    original public event payload and adds only tags derived from the same
    public ledger; it never introduces evaluator outcomes or hidden state.
    """
    events = [dict(event) for event in ledger.get("raw_events") or ()]
    by_id = {
        str(event.get("event_id") or ""): event
        for event in events if event.get("event_id")
    }
    def add_tag(event_id: Any, tag: str) -> None:
        event = by_id.get(str(event_id or ""))
        if event is None:
            return
        tags = set(str(value) for value in event.get("risk_tags") or ())
        tags.add(tag)
        event["risk_tags"] = sorted(tags)
    for window in ledger.get("recovery_windows") or ():
        tags = [str(value) for value in window.get("risk_tags") or ()]
        if window.get("status") == "FAILURE" and "failure_boundary" not in tags:
            tags.append("failure_boundary")
        for event in window.get("events") or ():
            event_id = event.get("event_id")
            for tag in tags:
                add_tag(event_id, tag)
        for event_id in window.get("member_event_ids") or ():
            for tag in tags:
                add_tag(event_id, tag)
    decision_tags = {
        "parameter_change_candidate": "parameter_change_candidate",
        "target_drift_candidate": "target_drift_candidate",
        "order_violation_candidate": "order_violation_candidate",
        "repeated_failed_mutation": "repeated_failed_mutation",
        "missing_readback_candidate": "missing_readback_candidate",
    }
    for decision in ledger.get("recovery_decision_events") or ():
        kind = str(decision.get("kind") or "")
        tag = decision_tags.get(kind)
        if tag is None:
            continue
        for event_id in (
            list(decision.get("source_event_ids") or ())
            + list(decision.get("member_event_ids") or ())
        ):
            add_tag(event_id, tag)
    return events


def event_credit(event: Mapping[str, Any]) -> float:
    """Credit an event using public structure only; values are clipped to [-1, 1]."""
    tags = _risk_tags(event)
    event_class = str(event.get("event_class") or "")
    if event_class == "verification" and not _is_verified(event):
        # A failed/unknown read-back is evidence against completion, never a
        # positive verification bonus.
        return -0.25
    value = 0.0
    if event_class in {"mutation", "verification"}:
        value += 0.20
    if _is_verified(event):
        value += 0.45
    if event_class == "verification":
        value += 0.15
    value += min(0.20, 0.05 * len(event.get("dependencies") or ()))
    value -= 0.30 * len(tags & HIGH_RISK)
    if "repeated_failed_mutation" in tags:
        value -= 0.25
    return max(-1.0, min(1.0, value))


def _best_by_key(events: Sequence[Mapping[str, Any]]) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    result: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for event in events:
        key = _event_key(event)
        if key not in result or event_credit(event) > event_credit(result[key]):
            result[key] = event
    return result


def _credit_buckets(
    events: Sequence[Mapping[str, Any]],
) -> dict[tuple[Any, ...], list[float]]:
    """Retain repeated public effects while remaining identifier invariant.

    A single best event per fingerprint loses required multiplicity in tasks
    such as updating two cart items or changing two reservations.  Sorted
    credit buckets provide an order-independent multiset representation.
    """
    result: dict[tuple[Any, ...], list[float]] = {}
    for event in events:
        result.setdefault(_event_key(event), []).append(event_credit(event))
    for values in result.values():
        values.sort(reverse=True)
    return result


def _risk_units(events: Sequence[Mapping[str, Any]]) -> set[tuple[tuple[Any, ...], str]]:
    """Return unique semantic risk units rather than counting trace length."""
    units: set[tuple[tuple[Any, ...], str]] = set()
    for event in events:
        key = _event_key(event)
        for tag in _risk_tags(event) & HIGH_RISK:
            units.add((key, tag))
    return units


def _risk_mass(events: Sequence[Mapping[str, Any]]) -> float:
    """Measure unique public risk with severity instead of event-count bias."""
    return sum(RISK_WEIGHTS.get(tag, 1.0) for _, tag in _risk_units(events))


def _severe_risk_units(events: Sequence[Mapping[str, Any]]) -> set[tuple[tuple[Any, ...], str]]:
    return {
        (key, tag)
        for key, tag in _risk_units(events)
        if tag in SEVERE_RISK
    }


def _coverage_profile(events: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Count distinct public effects without collapsing task coverage.

    ``_best_by_key`` is intentionally used for credit, but its one-event-per-
    fingerprint view can hide two required writes with the same public shape.
    This small profile retains capped multiplicity for outcome-blind progress
    checks.  It uses only event class, operation, verification, and risk tags.
    """
    semantic = [(_event_key(event), event) for event in events]
    verified_mutations = sum(
        1 for _, event in semantic
        if str(event.get("event_class") or "") == "mutation" and _is_verified(event)
    )
    verifications = sum(
        1 for _, event in semantic
        if str(event.get("event_class") or "") == "verification" and _is_verified(event)
    )
    commit_events = sum(
        1 for _, event in semantic
        if interaction_shape(event.get("interaction")) == "commit"
        and str(event.get("event_class") or "") in {"mutation", "verification"}
    )
    verified_commit_events = sum(
        1 for _, event in semantic
        if interaction_shape(event.get("interaction")) == "commit"
        and str(event.get("event_class") or "") in {"mutation", "verification"}
        and _is_verified(event)
    )
    action_shapes = len({
        (key[0], key[1], key[2]) for key, _ in semantic
        if key[0] in {"mutation", "verification"}
    })
    unresolved = sum(
        1 for _, event in semantic
        if _risk_tags(event) & {"failure_boundary", "unknown_evidence"}
    )
    return {
        "verified_mutations": verified_mutations,
        "verifications": verifications,
        "commit_events": commit_events,
        "verified_commit_events": verified_commit_events,
        "action_shapes": action_shapes,
        "unresolved_public_events": unresolved,
    }


def _coverage_delta(
    incumbent_events: Sequence[Mapping[str, Any]],
    candidate_events: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    before = _coverage_profile(incumbent_events)
    after = _coverage_profile(candidate_events)
    return {key: after[key] - before[key] for key in before}


def public_coverage_contract(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build a compact, outcome-blind coverage contract for the next rollout."""
    profile = _coverage_profile(events)
    return {
        "schema_version": "public_coverage_contract_v1",
        "verified_mutations_observed": profile["verified_mutations"],
        "verifications_observed": profile["verifications"],
        "action_shapes_observed": profile["action_shapes"],
        "unresolved_public_events_observed": profile["unresolved_public_events"],
        "generation_directive": (
            "Preserve the visible task coverage represented by verified actions; "
            "reacquire every current entity and argument, do not omit a second "
            "same-shaped action, and read back each consequential mutation."
        ),
        "public_only": True,
        "outcome_used": False,
    }


def incremental_credit(
    incumbent_events: Sequence[Mapping[str, Any]],
    candidate_events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute candidate-minus-incumbent credit, avoiding event-count bias."""
    old = _credit_buckets(incumbent_events)
    new = _credit_buckets(candidate_events)
    keys = set(old) | set(new)
    delta = 0.0
    gains: list[dict[str, Any]] = []
    losses: list[dict[str, Any]] = []
    for key in sorted(keys, key=str):
        before_values = old.get(key, [])
        after_values = new.get(key, [])
        width = max(len(before_values), len(after_values))
        for occurrence in range(width):
            before = before_values[occurrence] if occurrence < len(before_values) else 0.0
            after = after_values[occurrence] if occurrence < len(after_values) else 0.0
            change = round(after - before, 4)
            delta += change
            item = {
                "event_key": list(key),
                "occurrence": occurrence,
                "delta": change,
            }
            if change > 0.05:
                gains.append(item)
            elif change < -0.05:
                losses.append(item)
    incumbent_risks = len(_risk_units(incumbent_events))
    candidate_risks = len(_risk_units(candidate_events))
    incumbent_mass = _risk_mass(incumbent_events)
    candidate_mass = _risk_mass(candidate_events)
    new_severe = _severe_risk_units(candidate_events) - _severe_risk_units(incumbent_events)
    delta -= 0.25 * max(0.0, candidate_mass - incumbent_mass)
    coverage = _coverage_delta(incumbent_events, candidate_events)
    # Verified effects and recovery resolution are positive; unresolved
    # public failures remain a cost.  The value is deliberately small because
    # the MIMO selector remains the primary quality signal.
    coverage_score = (
        0.35 * coverage["verified_mutations"]
        + 0.15 * coverage["verifications"]
        + 0.10 * coverage["action_shapes"]
        - 0.25 * coverage["unresolved_public_events"]
    )
    return {
        "delta_credit": round(delta, 4),
        "event_gains": gains[:8],
        "event_losses": losses[:8],
        "incumbent_risk_count": incumbent_risks,
        "candidate_risk_count": candidate_risks,
        "incumbent_risk_mass": round(incumbent_mass, 3),
        "candidate_risk_mass": round(candidate_mass, 3),
        "new_severe_risk_count": len(new_severe),
        "new_severe_risk_tags": sorted({tag for _, tag in new_severe}),
        "coverage_delta": coverage,
        "coverage_score": round(coverage_score, 3),
        "public_only": True,
        "outcome_used": False,
    }


def select_refinement_window(
    events: Sequence[Mapping[str, Any]], max_events: int = 8,
) -> dict[str, Any]:
    """Select only the highest-confidence public risk window for local refinement."""
    risky = [e for e in events if _risk_tags(e) & HIGH_RISK]
    if not risky:
        return {
            "should_continue": False,
            "reason": "no_public_repairable_risk",
            "event_ids": [],
            "public_only": True,
            "outcome_used": False,
        }
    ranked = sorted(
        risky,
        key=lambda e: (
            max((RISK_WEIGHTS.get(tag, 1.0) for tag in _risk_tags(e) & HIGH_RISK), default=0.0),
            sum(RISK_WEIGHTS.get(tag, 1.0) for tag in _risk_tags(e) & HIGH_RISK),
            -int(e.get("position", -1)),
        ),
        reverse=True,
    )
    chosen = ranked[:max_events]
    positions = [int(e.get("position", -1)) for e in chosen]
    return {
        "should_continue": True,
        "reason": "public_repairable_risk",
        "event_ids": [str(e.get("event_id") or "") for e in chosen],
        "position_span": [min(positions), max(positions)] if positions else [],
        "risk_tags": sorted(set().union(*(_risk_tags(e) for e in chosen))),
        "public_only": True,
        "outcome_used": False,
    }


def accept_candidate(
    incumbent_events: Sequence[Mapping[str, Any]],
    candidate_events: Sequence[Mapping[str, Any]],
    *,
    epsilon: float = 0.05,
) -> tuple[bool, dict[str, Any]]:
    """Accept only a net public improvement without newly introduced risk."""
    comparison = incremental_credit(incumbent_events, candidate_events)
    accepted = (
        comparison["delta_credit"] > epsilon
        and comparison["candidate_risk_mass"] <= comparison["incumbent_risk_mass"]
    )
    comparison["accepted"] = accepted
    comparison["epsilon"] = epsilon
    return accepted, comparison


def accept_candidate_v2_progressive(
    incumbent_events: Sequence[Mapping[str, Any]],
    candidate_events: Sequence[Mapping[str, Any]],
    *,
    risk_tolerance_mass: float = 1.0,
    credit_floor: float = 0.0,
) -> tuple[bool, dict[str, Any]]:
    """Accept only a strictly useful public update.

    The relaxed v2 gate admits a neutral candidate when its risk mass is tied
    with the incumbent.  Across repeated rounds that turns harmless sampling
    variation into incumbent drift.  This gate keeps the useful flexibility
    for equivalent public traces, but requires either positive incremental
    credit, lower risk, or additional verified coverage.
    """
    comparison = incremental_credit(incumbent_events, candidate_events)
    coverage = comparison["coverage_delta"]
    mass_increase = (
        comparison["candidate_risk_mass"] - comparison["incumbent_risk_mass"]
    )
    risk_reduction = mass_increase < 0
    preserves_public_effects = (
        coverage["verified_mutations"] >= 0
        and coverage["action_shapes"] >= 0
        and coverage["verified_commit_events"] >= 0
    )
    positive_credit = comparison["delta_credit"] > credit_floor
    useful_credit = positive_credit and preserves_public_effects
    accepted = (
        comparison["new_severe_risk_count"] == 0
        and mass_increase <= risk_tolerance_mass
        and (
            useful_credit
            or (risk_reduction and coverage["action_shapes"] >= 0)
        )
    )
    comparison.update({
        "accepted": accepted,
        "credit_floor": credit_floor,
        "risk_tolerance_mass": risk_tolerance_mass,
        "risk_mass_increase": round(mass_increase, 3),
        "preserves_public_effects": preserves_public_effects,
        "useful_public_credit": useful_credit,
        "positive_public_progress": bool(
            useful_credit
            or (risk_reduction and coverage["action_shapes"] >= 0)
        ),
        "policy": "strict-positive-or-risk-reducing public-progress acceptance",
    })
    return accepted, comparison


def accept_candidate_v2_terminal(
    incumbent_events: Sequence[Mapping[str, Any]],
    candidate_events: Sequence[Mapping[str, Any]],
    *,
    risk_tolerance_mass: float = 0.0,
    credit_floor: float = 0.0,
) -> tuple[bool, dict[str, Any]]:
    """Apply a terminal-round evidence requirement.

    A late proposal is the last chance to change the incumbent, so risk
    reduction alone is insufficient: the proposal must also add verified
    commit evidence and preserve the public action coverage.  This remains
    dataset-independent and uses only the two public event ledgers.
    """
    accepted, comparison = accept_candidate_v2_progressive(
        incumbent_events,
        candidate_events,
        risk_tolerance_mass=risk_tolerance_mass,
        credit_floor=credit_floor,
    )
    coverage = comparison["coverage_delta"]
    terminal_evidence = (
        comparison["delta_credit"] > credit_floor
        and coverage["verified_commit_events"] > 0
        and coverage["action_shapes"] >= 0
        and coverage["verified_mutations"] >= 0
        and comparison["candidate_risk_mass"] <= comparison["incumbent_risk_mass"]
    )
    accepted = bool(accepted and terminal_evidence)
    comparison.update({
        "accepted": accepted,
        "terminal_evidence": terminal_evidence,
        "policy": "terminal verified-commit positive-progress acceptance",
    })
    return accepted, comparison


def accept_candidate_v2_selective(
    incumbent_events: Sequence[Mapping[str, Any]],
    candidate_events: Sequence[Mapping[str, Any]],
    *,
    risk_tolerance_mass: float = 0.0,
    credit_floor: float = 0.0,
) -> tuple[bool, dict[str, Any]]:
    """Require a strict, observable public-evidence gain before replacement."""
    comparison = incremental_credit(incumbent_events, candidate_events)
    coverage = comparison["coverage_delta"]
    mass_increase = (
        comparison["candidate_risk_mass"] - comparison["incumbent_risk_mass"]
    )
    evidence_gain = (
        coverage["verified_mutations"] > 0
        or coverage["verifications"] > 0
        or coverage["unresolved_public_events"] < 0
        or coverage["verified_commit_events"] > 0
    )
    accepted = bool(
        comparison["new_severe_risk_count"] == 0
        and mass_increase <= risk_tolerance_mass
        and comparison["delta_credit"] > credit_floor
        and evidence_gain
        and coverage["verified_mutations"] >= 0
        and coverage["verified_commit_events"] >= 0
    )
    comparison.update({
        "accepted": accepted,
        "credit_floor": credit_floor,
        "risk_tolerance_mass": risk_tolerance_mass,
        "risk_mass_increase": round(mass_increase, 3),
        "evidence_gain": evidence_gain,
        "policy": "selective verified-evidence public-progress acceptance",
    })
    return accepted, comparison


def accept_candidate_v2_recovery_first(
    incumbent_events: Sequence[Mapping[str, Any]],
    candidate_events: Sequence[Mapping[str, Any]],
    *,
    credit_floor: float = 0.0,
) -> tuple[bool, dict[str, Any]]:
    """Replace only when a public recovery problem is actually addressed.

    An acknowledged mutation is not sufficient evidence to replace a clean
    incumbent: an incorrect mutation can receive the same receipt.  The
    recovery window is therefore the trigger, while the comparison remains
    outcome-free and dataset-independent.
    """
    comparison = incremental_credit(incumbent_events, candidate_events)
    coverage = comparison["coverage_delta"]
    mass_change = (
        comparison["candidate_risk_mass"] - comparison["incumbent_risk_mass"]
    )
    incumbent_has_risk = comparison["incumbent_risk_count"] > 0
    risk_reduced = mass_change < 0
    unresolved_reduced = coverage["unresolved_public_events"] < 0
    strong_verified_commit = (
        coverage["verified_commit_events"] > 0
        and coverage["verified_mutations"] >= 0
        and coverage["verifications"] >= 0
        and coverage["action_shapes"] >= 0
        and comparison["delta_credit"] > credit_floor
    )
    accepted = bool(
        incumbent_has_risk
        and comparison["new_severe_risk_count"] == 0
        and mass_change <= 0
        and (
            risk_reduced
            or (unresolved_reduced and comparison["delta_credit"] > credit_floor)
            or strong_verified_commit
        )
        and coverage["verified_mutations"] >= 0
        and coverage["verified_commit_events"] >= 0
    )
    comparison.update({
        "accepted": accepted,
        "credit_floor": credit_floor,
        "risk_mass_increase": round(mass_change, 3),
        "incumbent_has_risk": incumbent_has_risk,
        "risk_reduced": risk_reduced,
        "unresolved_reduced": unresolved_reduced,
        "strong_verified_commit": strong_verified_commit,
        "policy": "recovery-first public-risk replacement",
    })
    return accepted, comparison


def accept_candidate_v2_relaxed(
    incumbent_events: Sequence[Mapping[str, Any]],
    candidate_events: Sequence[Mapping[str, Any]],
    *,
    risk_tolerance_mass: float = 1.0,
    credit_floor: float = -0.25,
) -> tuple[bool, dict[str, Any]]:
    """Apply a bounded-regret public-risk guard after the selector.

    The original v2 gate required positive event credit *and* zero additional
    risk.  That is too strict when a fresh rollout uses a different but
    semantically equivalent event decomposition: it rejects selector choices
    even when the only change is a small evidence gap.  Severe recovery risks
    still cannot pass, while the selector is allowed to resolve a disagreement
    with a modest structural cost.
    """
    comparison = incremental_credit(incumbent_events, candidate_events)
    mass_increase = (
        comparison["candidate_risk_mass"] - comparison["incumbent_risk_mass"]
    )
    coverage = comparison["coverage_delta"]
    public_progress = (
        comparison["delta_credit"] > 0.0
        or (mass_increase < 0.0 and coverage["action_shapes"] >= 0)
        or coverage["unresolved_public_events"] < 0
        or coverage["verifications"] > 0
        or coverage["verified_commit_events"] > 0
    )
    accepted = (
        comparison["new_severe_risk_count"] == 0
        and mass_increase <= risk_tolerance_mass
        and comparison["delta_credit"] >= credit_floor
        and comparison["candidate_risk_mass"]
        <= comparison["incumbent_risk_mass"] + risk_tolerance_mass
        and comparison["coverage_delta"]["verified_commit_events"] >= 0
        and coverage["action_shapes"] >= 0
        and public_progress
    )
    comparison.update({
        "accepted": accepted,
        "credit_floor": credit_floor,
        "risk_tolerance_mass": risk_tolerance_mass,
        "risk_mass_increase": round(mass_increase, 3),
        "preserves_verified_commit": comparison["coverage_delta"]["verified_commit_events"] >= 0,
        "positive_public_progress": public_progress,
        "policy": "selector-qualified bounded-regret semantic-risk acceptance",
    })
    return accepted, comparison


def accept_candidate_v2_balanced(
    incumbent_events: Sequence[Mapping[str, Any]],
    candidate_events: Sequence[Mapping[str, Any]],
    *,
    risk_tolerance_mass: float = 1.5,
    credit_floor: float = -0.35,
) -> tuple[bool, dict[str, Any]]:
    """Use selector-compatible bounded regret with public coverage evidence.

    This is the cross-dataset variant of v2.  It allows a selected rollout to
    express a different number of semantically equivalent actions, provided it
    adds public verification/coverage or removes an unresolved public event.
    Newly introduced severe recovery risks remain an unconditional veto.
    """
    comparison = incremental_credit(incumbent_events, candidate_events)
    coverage = comparison["coverage_delta"]
    progress = (
        coverage["verified_mutations"]
        + coverage["verifications"]
        + coverage["action_shapes"]
        - coverage["unresolved_public_events"]
    )
    mass_increase = (
        comparison["candidate_risk_mass"] - comparison["incumbent_risk_mass"]
    )
    accepted = (
        comparison["new_severe_risk_count"] == 0
        and mass_increase <= risk_tolerance_mass
        and comparison["delta_credit"] >= credit_floor
        and (progress >= 0 or comparison["delta_credit"] >= 0.0)
    )
    comparison.update({
        "accepted": accepted,
        "credit_floor": credit_floor,
        "risk_tolerance_mass": risk_tolerance_mass,
        "risk_mass_increase": round(mass_increase, 3),
        "coverage_progress": progress,
        "policy": "selector-qualified coverage-aware bounded-regret acceptance",
    })
    return accepted, comparison


def accept_candidate_v2_adaptive(
    incumbent_events: Sequence[Mapping[str, Any]],
    candidate_events: Sequence[Mapping[str, Any]],
    *,
    risk_tolerance_mass: float = 3.0,
    credit_floor: float = -0.65,
) -> tuple[bool, dict[str, Any]]:
    """Relax v2 for heterogeneous schemas while retaining catastrophic guards.

    Public event boundaries vary across domains.  Treating every newly tagged
    target/parameter/order risk as a hard veto therefore suppresses useful
    selector choices.  Only newly introduced failure boundaries and repeated
    failed mutations are unconditional vetoes; coverage collapse and a large
    risk increase remain bounded guards.  No evaluator outcome is consulted.
    """
    comparison = incremental_credit(incumbent_events, candidate_events)
    coverage = comparison["coverage_delta"]
    catastrophic = {"failure_boundary", "repeated_failed_mutation"}
    old_catastrophic = {
        (key, tag) for key, tag in _risk_units(incumbent_events)
        if tag in catastrophic
    }
    new_catastrophic = {
        (key, tag) for key, tag in _risk_units(candidate_events)
        if tag in catastrophic
    } - old_catastrophic
    verified_delta = coverage["verified_mutations"] + coverage["verifications"]
    coverage_collapse = (
        coverage["verified_mutations"] < 0
        and coverage["unresolved_public_events"] > 0
    )
    mass_increase = (
        comparison["candidate_risk_mass"] - comparison["incumbent_risk_mass"]
    )
    accepted = (
        not new_catastrophic
        and not coverage_collapse
        and mass_increase <= risk_tolerance_mass
        and comparison["delta_credit"] >= credit_floor
        and (verified_delta >= 0 or coverage["unresolved_public_events"] <= 0)
    )
    comparison.update({
        "accepted": accepted,
        "credit_floor": credit_floor,
        "risk_tolerance_mass": risk_tolerance_mass,
        "risk_mass_increase": round(mass_increase, 3),
        "new_catastrophic_risk_count": len(new_catastrophic),
        "coverage_collapse": coverage_collapse,
        "verified_coverage_delta": verified_delta,
        "policy": "selector-qualified adaptive catastrophic-risk acceptance",
    })
    return accepted, comparison


def accept_candidate_v3(
    incumbent_events: Sequence[Mapping[str, Any]],
    candidate_events: Sequence[Mapping[str, Any]],
    *,
    risk_tolerance: int = 0,
    credit_floor: float = -0.10,
) -> tuple[bool, dict[str, Any]]:
    """Use the selector as the quality signal while retaining a public-risk guard.

    The v2 positive-credit gate is useful for a single proposal, but it is too
    conservative for a disagreement-triggered challenger: event extraction can
    change the number of equivalent events without changing the public effect.
    v3 therefore rejects only newly introduced high-risk structure (up to an
    explicit tolerance) or a large credit regression.  It never reads outcomes.
    """
    comparison = incremental_credit(incumbent_events, candidate_events)
    accepted = (
        comparison["candidate_risk_count"]
        <= comparison["incumbent_risk_count"] + risk_tolerance
        and comparison["delta_credit"] >= credit_floor
    )
    comparison.update({
        "accepted": accepted,
        "credit_floor": credit_floor,
        "risk_tolerance": risk_tolerance,
        "policy": "selector-qualified bounded-regret public-risk acceptance",
    })
    return accepted, comparison


def accept_candidate_v2_selector_guard(
    incumbent_events: Sequence[Mapping[str, Any]],
    candidate_events: Sequence[Mapping[str, Any]],
    *,
    risk_tolerance_mass: float = 2.0,
    credit_floor: float = -1.00,
) -> tuple[bool, dict[str, Any]]:
    """Keep selector quality primary while vetoing observable public damage.

    Event granularity differs across datasets, so ordinary risk-tag changes
    should not override a selector decision.  This guard only blocks newly
    introduced catastrophic failures, loss of verified commit/mutation
    coverage, and a clear public coverage collapse.
    """
    comparison = incremental_credit(incumbent_events, candidate_events)
    coverage = comparison["coverage_delta"]
    catastrophic = {"failure_boundary", "repeated_failed_mutation"}
    old_catastrophic = {
        (key, tag) for key, tag in _risk_units(incumbent_events)
        if tag in catastrophic
    }
    new_catastrophic = {
        (key, tag) for key, tag in _risk_units(candidate_events)
        if tag in catastrophic
    } - old_catastrophic
    mass_increase = (
        comparison["candidate_risk_mass"] - comparison["incumbent_risk_mass"]
    )
    coverage_collapse = (
        coverage["verified_mutations"] < 0
        or coverage["verified_commit_events"] < 0
        or (
            coverage["action_shapes"] < 0
            and coverage["unresolved_public_events"] > 0
        )
    )
    accepted = bool(
        not new_catastrophic
        and not coverage_collapse
        and mass_increase <= risk_tolerance_mass
        and comparison["delta_credit"] >= credit_floor
    )
    comparison.update({
        "accepted": accepted,
        "credit_floor": credit_floor,
        "risk_tolerance_mass": risk_tolerance_mass,
        "risk_mass_increase": round(mass_increase, 3),
        "new_catastrophic_risk_count": len(new_catastrophic),
        "coverage_collapse": coverage_collapse,
        "policy": "selector-primary catastrophic-and-coverage guard",
    })
    return accepted, comparison
