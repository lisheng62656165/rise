"""Public-only StateTrace-CORT representation and lightweight energy model."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from state_trace_prot import compile_public_graph


FEATURE_NAMES = (
    "confirmed_effect",
    "accepted_unconfirmed_effect",
    "contradicted_effect",
    "open_recovery",
    "discharged_recovery",
    "contradicted_recovery",
    "open_verification",
    "discharged_verification",
    "exact_failed_repeat",
)

GOOD_STATES = {"publicly_confirmed", "discharged"}
BAD_STATES = {"contradicted"}
STATE_QUALITY = {
    "publicly_confirmed": 1.0,
    "discharged": 1.0,
    "accepted": -0.25,
    "unknown": 0.0,
    "open": -1.0,
    "contradicted": -1.5,
}


def _same_entity(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    a = set(left.get("entity_refs") or ())
    b = set(right.get("entity_refs") or ())
    return bool(a and b and a & b)


def _entity_types(obligation: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(sorted({str(value).split("=", 1)[0] for value in obligation.get("entity_refs") or ()}))


def _applicable_read_exists(event: Mapping[str, Any], events: Sequence[Mapping[str, Any]]) -> bool:
    refs = set(event.get("entity_refs") or ())
    types = {str(value).split("=", 1)[0] for value in refs}
    if not refs:
        return False
    for candidate in events:
        if candidate.get("operation") != "READ":
            continue
        candidate_refs = set(candidate.get("entity_refs") or ())
        candidate_types = {str(value).split("=", 1)[0] for value in candidate_refs}
        if refs & candidate_refs or types & candidate_types:
            return True
    return False


def _obligation_key(obligation: Mapping[str, Any]) -> tuple[Any, ...]:
    return (obligation.get("type"), obligation.get("tool"), _entity_types(obligation))


def compile_effect_obligation_graph(
    trajectory: Mapping[str, Any], tool_schemas: Any = None
) -> dict[str, Any]:
    graph = compile_public_graph(trajectory, tool_schemas)
    events = list(graph.get("events") or ())
    obligations: list[dict[str, Any]] = []

    for index, event in enumerate(events):
        if event.get("operation") != "WRITE":
            continue
        status = str(event.get("result_status") or "UNKNOWN")
        if status == "DETERMINISTIC_REJECTION":
            effect_state = "contradicted"
        elif event.get("readback_event_ids"):
            effect_state = "publicly_confirmed"
        elif status == "ACKNOWLEDGED":
            effect_state = "accepted"
        else:
            effect_state = "unknown"
        obligations.append({
            "obligation_id": f"effect-{event['event_id']}",
            "type": "EFFECT",
            "tool": event.get("tool"),
            "entity_refs": list(event.get("entity_refs") or ()),
            "state": effect_state,
            "supporting_event_ids": [event["event_id"], *list(event.get("readback_event_ids") or ())],
        })

        if event.get("readback_event_ids") or (status == "ACKNOWLEDGED" and _applicable_read_exists(event, events)):
            obligations.append({
                "obligation_id": f"verify-{event['event_id']}",
                "type": "VERIFY",
                "tool": event.get("tool"),
                "entity_refs": list(event.get("entity_refs") or ()),
                "state": "discharged" if event.get("readback_event_ids") else "open",
                "supporting_event_ids": list(event.get("readback_event_ids") or ()),
            })

        if status != "DETERMINISTIC_REJECTION":
            continue
        later_writes = [
            item for item in events[index + 1:]
            if item.get("operation") == "WRITE"
            and item.get("tool") == event.get("tool")
            and (_same_entity(event, item) or not event.get("entity_refs") or not item.get("entity_refs"))
        ]
        correction = next((item for item in later_writes if item.get("arguments") != event.get("arguments")), None)
        exact_repeat = next((item for item in later_writes if item.get("arguments") == event.get("arguments")), None)
        if correction and correction.get("result_status") in {"ACKNOWLEDGED", "READBACK_CONFIRMED"}:
            recovery_state, evidence = "discharged", [event["event_id"], correction["event_id"]]
        elif exact_repeat:
            recovery_state, evidence = "contradicted", [event["event_id"], exact_repeat["event_id"]]
        else:
            recovery_state, evidence = "open", [event["event_id"]]
        obligations.append({
            "obligation_id": f"recovery-{event['event_id']}",
            "type": "RECOVERY",
            "tool": event.get("tool"),
            "entity_refs": list(event.get("entity_refs") or ()),
            "state": recovery_state,
            "supporting_event_ids": evidence,
        })

    user_text = "\n".join(
        str(message.get("content") or "")
        for message in trajectory.get("conversation") or ()
        if message.get("role") == "user"
    )
    return {
        **graph,
        "goal_obligation": {"type": "GOAL", "public_text": user_text},
        "obligations": obligations,
        "public_only": True,
    }


def partial_obligation_transport(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    right_by_key: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for obligation in right.get("obligations") or ():
        right_by_key.setdefault(_obligation_key(obligation), []).append(obligation)
    matches, unmatched_left = [], []
    used: set[str] = set()
    for obligation in left.get("obligations") or ():
        candidates = [
            value for value in right_by_key.get(_obligation_key(obligation), ())
            if str(value.get("obligation_id")) not in used
        ]
        if not candidates:
            unmatched_left.append(obligation.get("obligation_id"))
            continue
        match = min(
            candidates,
            key=lambda value: (
                value.get("state") != obligation.get("state"),
                abs(STATE_QUALITY.get(str(value.get("state")), 0.0) - STATE_QUALITY.get(str(obligation.get("state")), 0.0)),
                str(value.get("obligation_id")),
            ),
        )
        used.add(str(match.get("obligation_id")))
        matches.append({
            "left": obligation.get("obligation_id"),
            "right": match.get("obligation_id"),
            "left_state": obligation.get("state"),
            "right_state": match.get("state"),
        })
    unmatched_right = [
        obligation.get("obligation_id")
        for obligation in right.get("obligations") or ()
        if str(obligation.get("obligation_id")) not in used
    ]
    return {"matches": matches, "unmatched_left": unmatched_left, "unmatched_right": unmatched_right}


def transport_advantage(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    """Return right-vs-left public obligation quality under partial transport."""
    transport = partial_obligation_transport(left, right)
    left_by_id = {item.get("obligation_id"): item for item in left.get("obligations") or ()}
    right_by_id = {item.get("obligation_id"): item for item in right.get("obligations") or ()}
    advantage = sum(
        STATE_QUALITY.get(str(match.get("right_state")), 0.0)
        - STATE_QUALITY.get(str(match.get("left_state")), 0.0)
        for match in transport["matches"]
    )
    advantage += sum(-STATE_QUALITY.get(str(left_by_id[item].get("state")), 0.0) for item in transport["unmatched_left"])
    advantage += sum(STATE_QUALITY.get(str(right_by_id[item].get("state")), 0.0) for item in transport["unmatched_right"])
    return advantage


def graph_features(graph: Mapping[str, Any]) -> dict[str, float]:
    values = {name: 0.0 for name in FEATURE_NAMES}
    for obligation in graph.get("obligations") or ():
        kind, state = obligation.get("type"), obligation.get("state")
        if kind == "EFFECT":
            if state == "publicly_confirmed":
                values["confirmed_effect"] += 1
            elif state == "accepted":
                values["accepted_unconfirmed_effect"] += 1
            elif state == "contradicted":
                values["contradicted_effect"] += 1
        elif kind == "RECOVERY":
            values[f"{state}_recovery"] += 1
        elif kind == "VERIFY":
            values[f"{state}_verification"] += 1
    values["exact_failed_repeat"] = float(len((graph.get("risks") or {}).get("exact_failed_repeat") or ()))
    return values


def admissible_interventions(graph: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Create graph corruptions whose added public violation has a local proof."""
    results = []
    for obligation in graph.get("obligations") or ():
        kind, state = obligation.get("type"), obligation.get("state")
        if kind == "EFFECT" and state == "publicly_confirmed":
            changed = copy.deepcopy(graph)
            target = next(item for item in changed["obligations"] if item["obligation_id"] == obligation["obligation_id"])
            target["state"] = "accepted"
            target["supporting_event_ids"] = target["supporting_event_ids"][:1]
            verification = next((
                item for item in changed["obligations"]
                if item.get("type") == "VERIFY"
                and item.get("tool") == obligation.get("tool")
                and item.get("entity_refs") == obligation.get("entity_refs")
                and item.get("state") == "discharged"
            ), None)
            if verification is not None:
                verification["state"] = "open"
                verification["supporting_event_ids"] = []
            else:
                changed["obligations"].append({
                    "obligation_id": f"synthetic-verify-{obligation['obligation_id']}",
                    "type": "VERIFY", "tool": obligation.get("tool"),
                    "entity_refs": list(obligation.get("entity_refs") or ()),
                    "state": "open", "supporting_event_ids": [],
                })
            results.append({"type": "remove_readback", "graph": changed, "proof_event_ids": obligation.get("supporting_event_ids", [])[1:]})
        if kind == "RECOVERY" and state == "discharged":
            changed = copy.deepcopy(graph)
            target = next(item for item in changed["obligations"] if item["obligation_id"] == obligation["obligation_id"])
            target["state"] = "open"
            target["supporting_event_ids"] = target["supporting_event_ids"][:1]
            results.append({"type": "remove_recovery", "graph": changed, "proof_event_ids": obligation.get("supporting_event_ids", [])[1:]})
        if kind == "RECOVERY" and state == "open":
            changed = copy.deepcopy(graph)
            target = next(item for item in changed["obligations"] if item["obligation_id"] == obligation["obligation_id"])
            target["state"] = "contradicted"
            changed.setdefault("risks", {}).setdefault("exact_failed_repeat", []).append(f"synthetic-{obligation['obligation_id']}")
            results.append({"type": "repeat_failed_mutation", "graph": changed, "proof_event_ids": obligation.get("supporting_event_ids", [])})
    return results


def semantics_preserving_control(graph: Mapping[str, Any]) -> dict[str, Any]:
    """Reorder independent serialized lists without changing public semantics."""
    changed = copy.deepcopy(graph)
    changed["events"] = list(reversed(changed.get("events") or ()))
    changed["obligations"] = list(reversed(changed.get("obligations") or ()))
    changed["edges"] = list(reversed(changed.get("edges") or ()))
    return changed


@dataclass
class LinearEnergyModel:
    weights: dict[str, float]
    threshold: float = 0.0
    transport_weight: float = 0.0

    def score(self, graph: Mapping[str, Any]) -> float:
        features = graph_features(graph)
        return sum(self.weights.get(name, 0.0) * features[name] for name in FEATURE_NAMES)

    def comparison_improvement(self, base: Mapping[str, Any], proposal: Mapping[str, Any]) -> float:
        return self.score(base) - self.score(proposal) + self.transport_weight * transport_advantage(base, proposal)

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(FEATURE_NAMES), "weights": self.weights,
            "threshold": self.threshold, "transport_weight": self.transport_weight,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LinearEnergyModel":
        return cls(
            weights={str(k): float(v) for k, v in value["weights"].items()},
            threshold=float(value.get("threshold", 0.0)),
            transport_weight=float(value.get("transport_weight", 0.0)),
        )


def fit_pairwise_energy(
    pairs: Iterable[tuple[Mapping[str, Any], Mapping[str, Any]]],
    *, epochs: int = 300, learning_rate: float = 0.03, l2: float = 1e-3,
    use_transport: bool = True,
) -> LinearEnergyModel:
    examples = [
        (graph_features(clean), graph_features(harmful), transport_advantage(harmful, clean))
        for clean, harmful in pairs
    ]
    if not examples:
        return LinearEnergyModel({name: 0.0 for name in FEATURE_NAMES})
    weights = {name: 0.0 for name in FEATURE_NAMES}
    transport_weight = 0.0
    for _ in range(epochs):
        gradient = {name: l2 * weights[name] for name in FEATURE_NAMES}
        transport_gradient = l2 * transport_weight
        for clean, harmful, transport_signal in examples:
            delta = {name: harmful[name] - clean[name] for name in FEATURE_NAMES}
            margin = sum(weights[name] * delta[name] for name in FEATURE_NAMES)
            if use_transport:
                margin += transport_weight * transport_signal
            factor = -1.0 / (1.0 + math.exp(min(40.0, max(-40.0, margin))))
            for name in FEATURE_NAMES:
                gradient[name] += factor * delta[name] / len(examples)
            if use_transport:
                transport_gradient += factor * transport_signal / len(examples)
        for name in FEATURE_NAMES:
            weights[name] -= learning_rate * gradient[name]
        if use_transport:
            transport_weight -= learning_rate * transport_gradient
    return LinearEnergyModel(weights, transport_weight=transport_weight)


def qualify_proposal(
    graphs: Sequence[Mapping[str, Any]], base_index: int, proposed_index: int,
    model: LinearEnergyModel,
) -> tuple[int, dict[str, Any]]:
    if len(graphs) != 2 or base_index not in (0, 1) or proposed_index not in (0, 1):
        raise ValueError("CORT requires two candidates and valid indices")
    transport = partial_obligation_transport(graphs[0], graphs[1])
    energies = [model.score(graph) for graph in graphs]
    independent_improvement = energies[base_index] - energies[proposed_index]
    aligned_advantage = transport_advantage(graphs[base_index], graphs[proposed_index])
    improvement = independent_improvement + model.transport_weight * aligned_advantage
    distinguishable = graph_features(graphs[0]) != graph_features(graphs[1])
    qualified = proposed_index != base_index and distinguishable and improvement > model.threshold
    selected = proposed_index if qualified else base_index
    return selected, {
        "qualified": qualified,
        "energies": energies,
        "independent_improvement": independent_improvement,
        "transport_advantage": aligned_advantage,
        "transport_weight": model.transport_weight,
        "improvement": improvement,
        "threshold": model.threshold,
        "transport": transport,
        "feature_vectors": [graph_features(graph) for graph in graphs],
    }
