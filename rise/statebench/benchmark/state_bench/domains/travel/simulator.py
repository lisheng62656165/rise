"""Travel domain user simulator prompt builder.

Assembles the simulator prompt from task definition and user profile.
This is travel-specific: it references airline preferences, cabin classes,
loyalty tiers, hotel/car rental neutrality, etc.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from state_bench.domains.travel.schemas import EnvironmentData
from state_bench.domains.travel.user_attributes import AIRLINE_NAMES
from state_bench.schemas import TaskDefinition

_BASE_RULES_PATH = Path(__file__).resolve().parent / "prompts" / "user_sim_base.md"


def _is_redundant_known_info(item: str, *, name: str, user_id: str) -> bool:
    normalized = item.strip().lower()
    return normalized in {
        f"your name is {name}".lower(),
        f"your user id: {user_id}".lower(),
        f"your user id is {user_id}".lower(),
        f"user id: {user_id}".lower(),
        f"user id is {user_id}".lower(),
    }


def build_simulator_prompt(
    task: TaskDefinition,
    env_data: EnvironmentData,
    user_id: str,
) -> str:
    """Assemble the simulator prompt from structured UserSimulatorConfig.

    Order:
    1. Preamble (role + precedence note)
    2. Identity (name, tier, points, constraints, preferences)
    3. Task Context (user_sim_context + known_info + unknown_info)
    4. Base Rules (shared canonical rules from user_sim_base.md)
    5. Task-Specific Rules
    """
    user = next((candidate for candidate in env_data.users if candidate.user_id == user_id), None)
    if user is None:
        raise ValueError(f"Task env does not contain user {user_id!r}")

    sim = task.user_simulator

    # --- 0. Preamble ---
    sections: list[str] = [
        "You are a simulated customer in a conversation with a travel agent. "
        "Your opening message has already been sent. Respond naturally based on the identity, context, and rules below.\n\n"
        "**Important:** Task-specific rules take precedence over base rules if there is a conflict."
    ]

    # --- 1. Identity ---
    name = user.name
    loyalty_tier = user.loyalty_tier
    loyalty_points = user.loyalty_points
    budget = user.budget
    preferences = user.preferences

    placeholders: dict[str, Any] = {
        "user_name": name,
        "loyalty_tier": loyalty_tier,
        "loyalty_points": str(loyalty_points),
        "budget": f"${budget}",
    }
    for key, value in preferences.items():
        if key == "airline" and value:
            placeholders[key] = AIRLINE_NAMES.get(value, value)
        elif isinstance(value, bool):
            placeholders[key] = "yes" if value else "no"
        elif key == "max_stops":
            placeholders[key] = "nonstop" if value == 0 else f"up to {value} stop(s)"
        else:
            placeholders[key] = str(value) if value else "no preference"

    identity_lines = [
        "## User Identity\n",
        f"- Name: {name}",
        f"- User ID: {user_id}",
        f"- Loyalty tier: {loyalty_tier}",
        f"- Loyalty points: {loyalty_points}",
        f"- Budget: {placeholders.get('budget', 'none')} — you will NOT accept anything above this amount",
        "- Preferences:",
        f"  - Meal: {placeholders.get('meal_preference', 'no preference')}",
        f"  - Seat: {placeholders.get('seat_type', 'no preference')}",
        f"  - WiFi: {placeholders.get('add_wifi', 'no')}",
        f"  - Extra legroom: {placeholders.get('add_extra_legroom', 'no')}",
        f"  - Insurance: {placeholders.get('add_insurance', 'no')}",
    ]

    # What you know contains task-specific facts only.
    know_items = [item for item in sim.known_info if not _is_redundant_known_info(item, name=name, user_id=user_id)]
    identity_lines.append("\n### What you know")
    for item in know_items:
        identity_lines.append(f"- {item}")

    # What you don't know
    if sim.unknown_info:
        identity_lines.append("\n### What you don't know")
        for item in sim.unknown_info:
            identity_lines.append(f"- {item}")

    sections.append("\n".join(identity_lines))

    # --- 2. Task Context ---
    sections.append(f"## Task Context\n\n{sim.user_sim_context}")

    # --- 3. Base Rules (shared canonical rules) ---
    sections.append(_BASE_RULES_PATH.read_text())

    # --- 4. Task-Specific Rules ---
    if sim.task_rules:
        rule_lines = ["## Task-Specific Rules\n"]
        for i, rule in enumerate(sim.task_rules, 1):
            rule_lines.append(f"{i}. {rule}")
        sections.append("\n".join(rule_lines))

    return "\n\n---\n\n".join(sections)
