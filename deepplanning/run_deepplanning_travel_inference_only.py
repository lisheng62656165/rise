from __future__ import annotations

import argparse
import difflib
import hashlib
import importlib.util
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from types import ModuleType

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.deepplanning_model_policy import PRIMARY_MODEL_CONFIG, resolve_model_identity


DEFAULT_TRAVEL_ROOT = Path(
    "/home/lisheng/project/baseline/agent_eval/data_downloard/deepplanning/"
    "Qwen-Agent-main/benchmark/deepplanning/travelplanning"
)

PERSIST_ACE_PREFIX = "[PERSIST-ACE-TRAVEL]"


def ensure_qwen_agent_shim() -> None:
    if "qwen_agent.tools.base" in sys.modules:
        return
    base_module = ModuleType("qwen_agent.tools.base")
    registry: dict[str, object] = {}

    class BaseTool:
        name = ""
        description = ""
        parameters: list[object] = []

        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def _verify_json_format_args(self, params: object) -> dict:
            if isinstance(params, dict):
                return params
            if isinstance(params, str):
                return json.loads(params or "{}")
            return {}

    def register_tool(name: str):
        def decorator(cls):
            registry[name] = cls
            cls.name = name
            return cls

        return decorator

    base_module.BaseTool = BaseTool
    base_module.register_tool = register_tool
    base_module.TOOL_REGISTRY = registry
    sys.modules.setdefault("qwen_agent", ModuleType("qwen_agent"))
    sys.modules.setdefault("qwen_agent.tools", ModuleType("qwen_agent.tools"))
    sys.modules["qwen_agent.tools.base"] = base_module


LEGACY_ACTION_REPLACEMENT_VERSION = "v20_tau_action_replacement_hotel_v1"
LEGACY_HOTEL_CONSTRAINT_CARD = "travel_hotel_constraint_relaxation_v1"
ACTION_REPLACEMENT_VERSION = "v21_tau_action_replacement_hotel_short_trip_v1"
HOTEL_CONSTRAINT_CARD = "travel_hotel_constraint_relaxation_short_trip_v1"
LEGACY_ENGLISH_ACTION_REPLACEMENT_VERSION = "v22_tau_action_replacement_restaurant_alias_v1"
LEGACY_RESTAURANT_ALIAS_CARD = "travel_restaurant_tool_alias_repair_v1"
IMMEDIATE_ENGLISH_ACTION_REPLACEMENT_VERSION = (
    "v23_tau_action_replacement_restaurant_alias_immediate_v1"
)
IMMEDIATE_RESTAURANT_ALIAS_CARD = "travel_restaurant_tool_alias_repair_immediate_v1"
SCHEMA_ENGLISH_ACTION_REPLACEMENT_VERSION = (
    "v24_tau_action_replacement_restaurant_alias_schema_v1"
)
SCHEMA_RESTAURANT_ALIAS_CARD = "travel_restaurant_tool_alias_schema_repair_v1"
GROUNDED_ENGLISH_ACTION_REPLACEMENT_VERSION = (
    "v25_tau_action_replacement_restaurant_alias_grounded_schema_v1"
)
GROUNDED_RESTAURANT_ALIAS_CARD = "travel_restaurant_tool_alias_grounded_schema_repair_v1"
SHORT_TRIP_ENGLISH_ACTION_REPLACEMENT_VERSION = (
    "v26_tau_action_replacement_restaurant_alias_grounded_short_trip_v1"
)
SHORT_TRIP_RESTAURANT_ALIAS_CARD = "travel_restaurant_tool_alias_grounded_short_trip_repair_v1"
TWO_DAY_ENGLISH_ACTION_REPLACEMENT_VERSION = (
    "v27_tau_action_replacement_restaurant_alias_grounded_two_day_v1"
)
TWO_DAY_RESTAURANT_ALIAS_CARD = "travel_restaurant_tool_alias_grounded_two_day_repair_v1"
ENGLISH_ACTION_REPLACEMENT_VERSION = (
    "v28_tau_action_replacement_restaurant_alias_grounded_two_day_no_budget_v1"
)
RESTAURANT_ALIAS_CARD = "travel_restaurant_tool_alias_grounded_two_day_no_budget_repair_v1"
THREE_STRIKE_RESTAURANT_ALIAS_ACTION_REPLACEMENT_VERSION = (
    "v45_tau_action_replacement_restaurant_alias_three_strike_two_day_v1"
)
THREE_STRIKE_RESTAURANT_ALIAS_CARD = (
    "travel_restaurant_tool_alias_three_strike_two_day_repair_v1"
)
ENGLISH_HOTEL_ACTION_REPLACEMENT_VERSION = (
    "v29_tau_action_replacement_hotel_inferred_brand_four_day_no_budget_v1"
)
ENGLISH_HOTEL_CONSTRAINT_CARD = (
    "travel_hotel_inferred_brand_relaxation_four_day_no_budget_v1"
)
ATTRACTION_GROUNDING_ACTION_REPLACEMENT_VERSION = (
    "v30_tau_action_replacement_attraction_grounded_location_four_day_no_budget_v1"
)
ATTRACTION_GROUNDING_CARD = (
    "travel_attraction_grounded_location_lookup_four_day_no_budget_v1"
)
LOCATION_CANONICALIZATION_ACTION_REPLACEMENT_VERSION = (
    "v31_tau_action_replacement_grounded_location_canonicalization_six_day_no_budget_v1"
)
LOCATION_CANONICALIZATION_CARD = (
    "travel_grounded_location_canonicalization_six_day_no_budget_v1"
)
ATTRACTION_TOOL_FAMILY_ACTION_REPLACEMENT_VERSION = (
    "v32_tau_action_replacement_grounded_attraction_tool_family_five_day_no_budget_v1"
)
ATTRACTION_TOOL_FAMILY_CARD = (
    "travel_grounded_attraction_tool_family_repair_five_day_no_budget_v1"
)
ENGLISH_HOTEL_SERVICE_ACTION_REPLACEMENT_VERSION = (
    "v33_tau_action_replacement_hotel_single_inferred_brand_robot_service_v1"
)
ENGLISH_HOTEL_SERVICE_CARD = (
    "travel_hotel_single_inferred_brand_robot_service_relaxation_v1"
)
JI_FOUR_STAR_HOTEL_ACTION_REPLACEMENT_VERSION = (
    "v47_tau_action_replacement_ji_four_star_hotel_relaxation_v1"
)
JI_FOUR_STAR_HOTEL_CARD = "travel_ji_four_star_hotel_brand_relaxation_v1"
ATTRACTION_READY_JI_HOTEL_ACTION_REPLACEMENT_VERSION = (
    "v48_tau_action_replacement_attraction_ready_ji_four_star_hotel_v1"
)
ATTRACTION_READY_JI_HOTEL_CARD = (
    "travel_attraction_ready_ji_four_star_hotel_brand_relaxation_v1"
)
IMMEDIATE_ATTRACTION_TOOL_FAMILY_ACTION_REPLACEMENT_VERSION = (
    "v34_tau_action_replacement_grounded_attraction_tool_family_immediate_five_day_no_budget_v1"
)
IMMEDIATE_ATTRACTION_TOOL_FAMILY_CARD = (
    "travel_grounded_attraction_tool_family_schema_repair_five_day_no_budget_v1"
)
FUZZY_LOCATION_CANONICALIZATION_ACTION_REPLACEMENT_VERSION = (
    "v35_tau_action_replacement_grounded_location_fuzzy_canonicalization_six_day_no_budget_v1"
)
FUZZY_LOCATION_CANONICALIZATION_CARD = (
    "travel_grounded_location_fuzzy_canonicalization_six_day_no_budget_v1"
)
BUDGET_SAFE_FUZZY_LOCATION_CANONICALIZATION_ACTION_REPLACEMENT_VERSION = (
    "v36_tau_action_replacement_grounded_location_fuzzy_canonicalization_seven_day_budget_safe_v1"
)
BUDGET_SAFE_FUZZY_LOCATION_CANONICALIZATION_CARD = (
    "travel_grounded_location_fuzzy_canonicalization_seven_day_budget_safe_v1"
)
IMMEDIATE_FUZZY_LOCATION_CANONICALIZATION_ACTION_REPLACEMENT_VERSION = (
    "v37_tau_action_replacement_grounded_location_fuzzy_schema_repair_seven_day_budget_safe_v1"
)
IMMEDIATE_FUZZY_LOCATION_CANONICALIZATION_CARD = (
    "travel_grounded_location_fuzzy_schema_repair_seven_day_budget_safe_v1"
)
RESTAURANT_GROUNDED_COORDINATE_ACTION_REPLACEMENT_VERSION = (
    "v38_tau_action_replacement_restaurant_grounded_coordinate_batch_collapse_v1"
)
RESTAURANT_GROUNDED_COORDINATE_CARD = (
    "travel_restaurant_grounded_coordinate_batch_collapse_v1"
)
TRAVEL_EN_FOLD1_DECISION_FORK_COORDINATE_ACTION_REPLACEMENT_VERSION = (
    "v98_travel_en_fold1_decision_fork_recent_location_coordinate_rebinding_v1"
)
TRAVEL_EN_FOLD1_DECISION_FORK_COORDINATE_CARD = (
    "travel_en_fold1_decision_fork_recent_location_coordinate_rebinding_v1"
)
TRAVEL_EN_FOLD2_DECISION_FORK_SAME_ROUTE_ACTION_REPLACEMENT_VERSION = (
    "v99_travel_en_fold2_decision_fork_same_origin_destination_route_drop_v1"
)
TRAVEL_EN_FOLD2_DECISION_FORK_SAME_ROUTE_CARD = (
    "travel_en_fold2_decision_fork_same_origin_destination_route_drop_v1"
)
RESTAURANT_ANCHOR_PREREQUISITE_ACTION_REPLACEMENT_VERSION = (
    "v39_tau_action_replacement_restaurant_anchor_location_prerequisite_v1"
)
RESTAURANT_ANCHOR_PREREQUISITE_CARD = (
    "travel_restaurant_anchor_location_prerequisite_v1"
)
RESTAURANT_ANCHOR_RECOMMENDATION_ACTION_REPLACEMENT_VERSION = (
    "v40_tau_action_replacement_restaurant_anchor_grounded_recommendation_v1"
)
RESTAURANT_ANCHOR_RECOMMENDATION_CARD = (
    "travel_restaurant_anchor_grounded_recommendation_v1"
)
EXPLICIT_BUDGET_FUZZY_LOCATION_ACTION_REPLACEMENT_VERSION = (
    "v41_tau_action_replacement_grounded_location_fuzzy_explicit_budget_v1"
)
EXPLICIT_BUDGET_FUZZY_LOCATION_CARD = (
    "travel_grounded_location_fuzzy_canonicalization_explicit_budget_v1"
)
SPECIAL_SERVICE_ATTRACTION_TOOL_FAMILY_ACTION_REPLACEMENT_VERSION = (
    "v42_tau_action_replacement_special_service_attraction_tool_family_v1"
)
SPECIAL_SERVICE_ATTRACTION_TOOL_FAMILY_CARD = (
    "travel_special_service_attraction_tool_family_repair_v1"
)
SPECIAL_SERVICE_RESTAURANT_ALIAS_FOLLOWUP_ACTION_REPLACEMENT_VERSION = (
    "v43_tau_action_replacement_special_service_restaurant_alias_followup_v1"
)
SPECIAL_SERVICE_RESTAURANT_ALIAS_FOLLOWUP_CARD = (
    "travel_special_service_restaurant_alias_followup_repair_v1"
)
TRAVEL_MULTICARD_ACTION_REPLACEMENT_VERSION = (
    "v44_tau_action_replacement_travel_multicard_registry_v1"
)
TRAVEL_FOUR_CARD_ACTION_REPLACEMENT_VERSION = (
    "v49_tau_action_replacement_travel_four_card_registry_v1"
)
TRAVEL_STRICT_POSITIVE_TWO_CARD_ACTION_REPLACEMENT_VERSION = (
    "v50_tau_action_replacement_travel_strict_positive_two_card_registry_v1"
)
TRAVEL_ZH_FOLD1_FLIGHT_SCHEMA_ACTION_REPLACEMENT_VERSION = (
    "v51_train_only_travel_zh_fold1_flight_schema_repair_v1"
)
TRAVEL_ZH_FOLD1_FLIGHT_SCHEMA_CARD = (
    "travel_zh_fold1_flight_drop_unsupported_seat_field_v1"
)
TRAVEL_ZH_FOLD1_TWO_CARD_ACTION_REPLACEMENT_VERSION = (
    "v61_frozen_travel_zh_fold1_flight_restaurant_loopcut_registry_v1"
)
TRAVEL_ZH_FOLD1_RESTAURANT_REPEAT_LOOPCUT_CARD = (
    "auto_cut_repeated_failure_query_restaurant_details_v1"
)
TRAVEL_ZH_FOLD2_FLIGHT_SCHEMA_ACTION_REPLACEMENT_VERSION = (
    "v52_train_only_travel_zh_fold2_flight_schema_repair_v1"
)
TRAVEL_ZH_FOLD2_FLIGHT_SCHEMA_CARD = (
    "travel_zh_fold2_flight_drop_unsupported_seat_field_v1"
)
TRAVEL_ZH_FOLD2_HOTEL_EMPTY_BRAND_ACTION_REPLACEMENT_VERSION = (
    "v53_train_only_travel_zh_fold2_hotel_empty_brand_drop_v1"
)
TRAVEL_ZH_FOLD2_HOTEL_EMPTY_BRAND_CARD = (
    "travel_zh_fold2_hotel_empty_brand_drop_v1"
)
TRAVEL_ZH_FOLD3_ROUND2_TRAIN_ONLY_ACTION_REPLACEMENT_VERSION = (
    "v54_train_only_travel_zh_fold3_round2_tradeoff_registry_v1"
)
TRAVEL_ZH_FOLD3_SMALL_BATCH_RESTAURANT_ALIAS_CARD = (
    "travel_zh_fold3_small_batch_restaurant_alias_v1"
)
TRAVEL_ZH_FOLD3_DUPLICATE_ROAD_ROUTE_DROP_CARD = (
    "travel_zh_fold3_duplicate_batch_road_route_drop_v1"
)
TRAVEL_ZH_FOLD3_FIXED_POSITIVE_ACTION_REPLACEMENT_VERSION = (
    "v55_train_only_travel_zh_fold3_fixed_positive_registry_v1"
)
TRAVEL_ZH_FOLD3_RESTAURANT_ALIAS_ONLY_ACTION_REPLACEMENT_VERSION = (
    "v56_train_only_travel_zh_fold3_restaurant_alias_only_v1"
)
TRAVEL_ZH_FOLD3_RESTAURANT_PARENTHETICAL_ONLY_ACTION_REPLACEMENT_VERSION = (
    "v57_train_only_travel_zh_fold3_restaurant_parenthetical_only_v1"
)
TRAVEL_ZH_FOLD3_FLIGHT_REPEAT_LOOPCUT_ACTION_REPLACEMENT_VERSION = (
    "v58_train_only_travel_zh_fold3_flight_repeat_loopcut_v1"
)
TRAVEL_ZH_FOLD3_DUPLICATE_RESTAURANT_ACTION_REPLACEMENT_VERSION = (
    "v59_train_only_travel_zh_fold3_duplicate_restaurant_pruning_v1"
)
TRAVEL_ZH_FOLD3_USER_SELECTED_THREE_CARD_ACTION_REPLACEMENT_VERSION = (
    "v60_frozen_travel_zh_fold3_user_selected_three_card_registry_v1"
)
TRAVEL_EN_FOLD1_SINGLE_VICINITY_ALIAS_ACTION_REPLACEMENT_VERSION = (
    "v62_frozen_travel_en_fold1_single_vicinity_no_prior_location_alias_v1"
)
TRAVEL_EN_FOLD1_SINGLE_VICINITY_ALIAS_CARD = (
    "auto_single_vicinity_no_prior_location_restaurant_alias_v4"
)
TRAVEL_EN_FOLD2_LATE_ALIAS_ACTION_REPLACEMENT_VERSION = (
    "v63_frozen_travel_en_fold2_late_failed_restaurant_alias_v1"
)
TRAVEL_EN_FOLD2_LATE_ALIAS_CARD = (
    "travel_en_fold2_late_failed_restaurant_alias_replacement_v1"
)
TRAVEL_EN_FOLD3_ANCHOR_ACTION_REPLACEMENT_VERSION = (
    "v64_frozen_travel_en_fold3_train_induced_restaurant_anchor_v1"
)
TRAVEL_EN_FOLD3_ANCHOR_CARD = (
    "travel_en_fold3_train_induced_restaurant_anchor_v1"
)
TRAVEL_EN_FOLD3_STAGNATION_ANCHOR_ACTION_REPLACEMENT_VERSION = (
    "v65_frozen_travel_en_fold3_train_induced_high_confidence_stagnation_anchor_v1"
)
TRAVEL_EN_FOLD3_STAGNATION_ANCHOR_CARD = (
    "travel_en_fold3_train_induced_high_confidence_stagnation_anchor_v1"
)
TRAVEL_EN_V2_FOLD1_ACTION_REPLACEMENT_VERSION = (
    "v66_train_only_travel_en_v2_fold1_anchor_search_loopcut_registry_v1"
)
TRAVEL_EN_V2_FOLD1_RESTAURANT_ANCHOR_CARD = (
    "travel_en_v2_fold1_restaurant_anchor_prior_location_nonexplicit_v1"
)
TRAVEL_EN_V2_FOLD1_SEARCH_LOCATION_LOOPCUT_CARD = (
    "travel_en_v2_fold1_repeated_failure_search_location_loopcut_v1"
)
TRAVEL_EN_V4_FOLD1_SCHEMA_ALIAS_ACTION_REPLACEMENT_VERSION = (
    "v67_train_only_travel_en_v4_fold1_schema_alias_v1"
)
TRAVEL_EN_V4_FOLD1_DUPLICATE_RESTAURANT_ACTION_REPLACEMENT_VERSION = (
    "v68_train_only_travel_en_v4_fold1_duplicate_restaurant_v1"
)
TRAVEL_EN_V4_FOLD1_DUPLICATE_ROAD_ACTION_REPLACEMENT_VERSION = (
    "v69_train_only_travel_en_v4_fold1_duplicate_road_v1"
)
TRAVEL_EN_V4_FOLD1_COMBINED_ACTION_REPLACEMENT_VERSION = (
    "v70_train_only_travel_en_v4_fold1_combined_registry_v1"
)
TRAVEL_EN_V4_FOLD1_RESTAURANT_NAME_NORMALIZATION_ACTION_REPLACEMENT_VERSION = (
    "v71_train_only_travel_en_v4_fold1_restaurant_name_normalization_v1"
)
TRAVEL_EN_V4_FOLD1_RESTAURANT_ANCHOR_ACTION_REPLACEMENT_VERSION = (
    "v72_train_only_travel_en_v4_fold1_restaurant_anchor_v1"
)
TRAVEL_EN_V4_FOLD1_ROUND2_COMBINED_ACTION_REPLACEMENT_VERSION = (
    "v73_train_only_travel_en_v4_fold1_round2_combined_registry_v1"
)
TRAVEL_EN_FOLD1_CLEANROOM_HOTEL_LOOPCUT_ACTION_REPLACEMENT_VERSION = (
    "v74_train_only_travel_en_fold1_cleanroom_hotel_loopcut_v1"
)
TRAVEL_EN_FOLD1_CLEANROOM_THREE_CARD_ACTION_REPLACEMENT_VERSION = (
    "v75_train_only_travel_en_fold1_cleanroom_three_card_registry_v1"
)
TRAVEL_EN_FOLD2_CLEANROOM_HOTEL_LOOPCUT_ACTION_REPLACEMENT_VERSION = (
    "v76_train_only_travel_en_fold2_cleanroom_hotel_loopcut_v1"
)
TRAVEL_EN_FOLD2_CLEANROOM_INFERRED_ALIAS_ACTION_REPLACEMENT_VERSION = (
    "v77_train_only_travel_en_fold2_cleanroom_inferred_alias_v1"
)
TRAVEL_EN_FOLD2_CLEANROOM_SAME_COORDINATE_ACTION_REPLACEMENT_VERSION = (
    "v78_train_only_travel_en_fold2_cleanroom_same_coordinate_v1"
)
TRAVEL_EN_FOLD2_CLEANROOM_THREE_CARD_ACTION_REPLACEMENT_VERSION = (
    "v79_train_only_travel_en_fold2_cleanroom_three_card_registry_v1"
)
TRAVEL_EN_FOLD2_CLEANROOM_ATTRACTION_DETAILS_LOOPCUT_ACTION_REPLACEMENT_VERSION = (
    "v80_train_only_travel_en_fold2_cleanroom_attraction_details_loopcut_v1"
)
TRAVEL_EN_FOLD2_CLEANROOM_DUPLICATE_ATTRACTION_ACTION_REPLACEMENT_VERSION = (
    "v81_train_only_travel_en_fold2_cleanroom_duplicate_attraction_v1"
)
TRAVEL_EN_FOLD2_CLEANROOM_ROUND3_TWO_CARD_ACTION_REPLACEMENT_VERSION = (
    "v82_train_only_travel_en_fold2_cleanroom_round3_two_card_registry_v1"
)
TRAVEL_EN_FOLD3_CLEANROOM_DUPLICATE_RESTAURANT_ACTION_REPLACEMENT_VERSION = (
    "v83_train_only_travel_en_fold3_cleanroom_duplicate_restaurant_v1"
)
TRAVEL_EN_FOLD3_CLEANROOM_HOTEL_LOOPCUT_ACTION_REPLACEMENT_VERSION = (
    "v84_train_only_travel_en_fold3_cleanroom_hotel_loopcut_v1"
)
TRAVEL_EN_FOLD3_CLEANROOM_TWO_CARD_ACTION_REPLACEMENT_VERSION = (
    "v85_train_only_travel_en_fold3_cleanroom_two_card_registry_v1"
)
TRAVEL_EN_FOLD3_CLEANROOM_ROUND2_FOUR_CARD_ACTION_REPLACEMENT_VERSION = (
    "v86_train_only_travel_en_fold3_cleanroom_round2_four_card_registry_v1"
)
TRAVEL_EN_FOLD3_CLEANROOM_ROUND3_STRICT_FOUR_CARD_ACTION_REPLACEMENT_VERSION = (
    "v87_train_only_travel_en_fold3_cleanroom_round3_strict_four_card_registry_v1"
)
TRAVEL_EN_FOLD3_V8_K6_DUPLICATE_AROUND_ACTION_REPLACEMENT_VERSION = (
    "v88_train_only_travel_en_fold3_v8_k6_duplicate_around_restaurants_v1"
)
TRAVEL_EN_FOLD3_V8_K6_HOTEL_LOOPCUT_ACTION_REPLACEMENT_VERSION = (
    "v89_train_only_travel_en_fold3_v8_k6_hotel_loopcut_v1"
)
TRAVEL_EN_FOLD3_V8_K6_TWO_CARD_ACTION_REPLACEMENT_VERSION = (
    "v90_train_only_travel_en_fold3_v8_k6_two_card_registry_v1"
)
TRAVEL_EN_FOLD3_V8_K6_SAME_COORDINATE_ACTION_REPLACEMENT_VERSION = (
    "v91_train_only_travel_en_fold3_v8_k6_same_coordinate_v1"
)
TRAVEL_EN_FOLD3_V8_K6_SEARCH_LOOPCUT_ACTION_REPLACEMENT_VERSION = (
    "v92_train_only_travel_en_fold3_v8_k6_search_loopcut_v1"
)
TRAVEL_EN_FOLD3_V8_K6_PARENTHETICAL_ACTION_REPLACEMENT_VERSION = (
    "v93_train_only_travel_en_fold3_v8_k6_restaurant_parenthetical_v1"
)
TRAVEL_EN_FOLD3_V8_K6_ROUND2_THREE_CARD_ACTION_REPLACEMENT_VERSION = (
    "v94_train_only_travel_en_fold3_v8_k6_round2_three_card_registry_v1"
)
TRAVEL_EN_FOLD3_V8_K6_RESTAURANT_LOOPCUT_ACTION_REPLACEMENT_VERSION = (
    "v95_train_only_travel_en_fold3_v8_k6_restaurant_loopcut_v1"
)
TRAVEL_EN_FOLD3_V8_K6_USER_FROZEN_FOUR_CARD_ACTION_REPLACEMENT_VERSION = (
    "v96_frozen_travel_en_fold3_v8_k6_user_selected_four_card_registry_v1"
)
TRAVEL_EN_FOLD2_CLEANROOM_USER_FROZEN_TWO_CARD_ACTION_REPLACEMENT_VERSION = (
    "v97_frozen_travel_en_fold2_cleanroom_user_selected_two_card_registry_v1"
)
TRAVEL_EN_V4_FOLD1_SCHEMA_ALIAS_CARD = (
    "travel_en_v4_fold1_schema_tool_name_repair_recommend_around_restaurants_v1"
)
TRAVEL_EN_V4_FOLD1_DUPLICATE_RESTAURANT_CARD = (
    "travel_en_v4_fold1_duplicate_batch_recommend_restaurants_v1"
)
TRAVEL_EN_V4_FOLD1_DUPLICATE_ROAD_CARD = (
    "travel_en_v4_fold1_duplicate_batch_query_road_route_info_v1"
)
TRAVEL_EN_V4_FOLD1_RESTAURANT_NAME_NORMALIZATION_CARD = (
    "travel_en_v4_fold1_restaurant_parenthetical_name_normalization_nonexplicit_v1"
)
TRAVEL_EN_V4_FOLD1_RESTAURANT_ANCHOR_CARD = (
    "travel_en_v4_fold1_restaurant_anchor_prior_location_nonexplicit_v1"
)
TRAVEL_EN_FOLD1_CLEANROOM_HOTEL_LOOPCUT_CARD = (
    "travel_en_fold1_cleanroom_repeated_failure_hotel_loopcut_v1"
)
TRAVEL_EN_FOLD1_CLEANROOM_DUPLICATE_SEARCH_LOCATION_CARD = (
    "travel_en_fold1_cleanroom_duplicate_batch_search_location_v1"
)
TRAVEL_EN_FOLD1_CLEANROOM_DUPLICATE_ATTRACTION_CARD = (
    "travel_en_fold1_cleanroom_duplicate_batch_recommend_attractions_v1"
)
TRAVEL_EN_FOLD2_CLEANROOM_HOTEL_LOOPCUT_CARD = (
    "travel_en_fold2_cleanroom_repeated_failure_hotel_loopcut_v1"
)
TRAVEL_EN_FOLD2_CLEANROOM_INFERRED_ALIAS_CARD = (
    "travel_en_fold2_cleanroom_inferred_restaurant_tool_alias_v1"
)
TRAVEL_EN_FOLD2_CLEANROOM_SAME_COORDINATE_CARD = (
    "travel_en_fold2_cleanroom_same_coordinate_route_drop_v1"
)
TRAVEL_EN_FOLD2_CLEANROOM_ATTRACTION_DETAILS_LOOPCUT_CARD = (
    "travel_en_fold2_cleanroom_repeated_failure_query_attraction_details_v1"
)
TRAVEL_EN_FOLD2_CLEANROOM_DUPLICATE_ATTRACTION_CARD = (
    "travel_en_fold2_cleanroom_duplicate_batch_recommend_attractions_v1"
)
TRAVEL_EN_FOLD3_CLEANROOM_DUPLICATE_RESTAURANT_CARD = (
    "travel_en_fold3_cleanroom_duplicate_batch_recommend_restaurants_v1"
)
TRAVEL_EN_FOLD3_CLEANROOM_HOTEL_LOOPCUT_CARD = (
    "travel_en_fold3_cleanroom_repeated_failure_query_hotel_info_v1"
)
TRAVEL_EN_FOLD3_CLEANROOM_INFERRED_ALIAS_CARD = (
    "travel_en_fold3_cleanroom_inferred_restaurant_tool_alias_v1"
)
TRAVEL_EN_FOLD3_CLEANROOM_SAME_COORDINATE_CARD = (
    "travel_en_fold3_cleanroom_same_coordinate_route_drop_v1"
)
TRAVEL_EN_FOLD3_CLEANROOM_DUPLICATE_ROAD_CARD = (
    "travel_en_fold3_cleanroom_duplicate_batch_query_road_route_info_v1"
)
TRAVEL_EN_FOLD3_V8_K6_DUPLICATE_AROUND_CARD = (
    "travel_en_fold3_v8_k6_duplicate_batch_recommend_around_restaurants_v1"
)
TRAVEL_EN_FOLD3_V8_K6_HOTEL_LOOPCUT_CARD = (
    "travel_en_fold3_v8_k6_repeated_failure_query_hotel_info_v1"
)
TRAVEL_EN_FOLD3_V8_K6_SAME_COORDINATE_CARD = (
    "travel_en_fold3_v8_k6_same_coordinate_route_drop_v1"
)
TRAVEL_EN_FOLD3_V8_K6_SEARCH_LOOPCUT_CARD = (
    "travel_en_fold3_v8_k6_repeated_failure_search_location_v1"
)
TRAVEL_EN_FOLD3_V8_K6_PARENTHETICAL_CARD = (
    "travel_en_fold3_v8_k6_restaurant_parenthetical_name_normalization_v1"
)
TRAVEL_EN_FOLD3_V8_K6_RESTAURANT_LOOPCUT_CARD = (
    "travel_en_fold3_v8_k6_repeated_failure_recommend_restaurants_v1"
)
TRAVEL_ZH_FOLD3_RESTAURANT_ALIAS_CARD = (
    "travel_zh_fold3_restaurant_alias_v1"
)
TRAVEL_ZH_FOLD3_RESTAURANT_PARENTHETICAL_CARD = (
    "travel_zh_fold3_restaurant_parenthetical_name_normalization_v1"
)
TRAVEL_ZH_FOLD3_FLIGHT_REPEAT_LOOPCUT_CARD = (
    "travel_zh_fold3_flight_repeated_failure_loopcut_v1"
)
TRAVEL_ZH_FOLD3_DUPLICATE_RESTAURANT_CARD = (
    "travel_zh_fold3_duplicate_batch_recommend_restaurants_pruning_v1"
)
THREE_DAY_BATCH_ATTRACTION_ACTION_REPLACEMENT_VERSION = (
    "v46_tau_action_replacement_three_day_batch_attraction_tool_family_v1"
)
THREE_DAY_BATCH_ATTRACTION_CARD = (
    "travel_three_day_batch_attraction_tool_family_repair_v1"
)
ENGLISH_ACTION_REPLACEMENT_VERSIONS = {
    LEGACY_ENGLISH_ACTION_REPLACEMENT_VERSION,
    IMMEDIATE_ENGLISH_ACTION_REPLACEMENT_VERSION,
    SCHEMA_ENGLISH_ACTION_REPLACEMENT_VERSION,
    GROUNDED_ENGLISH_ACTION_REPLACEMENT_VERSION,
    SHORT_TRIP_ENGLISH_ACTION_REPLACEMENT_VERSION,
    TWO_DAY_ENGLISH_ACTION_REPLACEMENT_VERSION,
    ENGLISH_ACTION_REPLACEMENT_VERSION,
    THREE_STRIKE_RESTAURANT_ALIAS_ACTION_REPLACEMENT_VERSION,
    SPECIAL_SERVICE_RESTAURANT_ALIAS_FOLLOWUP_ACTION_REPLACEMENT_VERSION,
}
ACTION_REPLACEMENT_VERSIONS = {
    LEGACY_ACTION_REPLACEMENT_VERSION,
    ACTION_REPLACEMENT_VERSION,
    ENGLISH_HOTEL_ACTION_REPLACEMENT_VERSION,
    ATTRACTION_GROUNDING_ACTION_REPLACEMENT_VERSION,
    LOCATION_CANONICALIZATION_ACTION_REPLACEMENT_VERSION,
    ATTRACTION_TOOL_FAMILY_ACTION_REPLACEMENT_VERSION,
    ENGLISH_HOTEL_SERVICE_ACTION_REPLACEMENT_VERSION,
    JI_FOUR_STAR_HOTEL_ACTION_REPLACEMENT_VERSION,
    ATTRACTION_READY_JI_HOTEL_ACTION_REPLACEMENT_VERSION,
    IMMEDIATE_ATTRACTION_TOOL_FAMILY_ACTION_REPLACEMENT_VERSION,
    FUZZY_LOCATION_CANONICALIZATION_ACTION_REPLACEMENT_VERSION,
    BUDGET_SAFE_FUZZY_LOCATION_CANONICALIZATION_ACTION_REPLACEMENT_VERSION,
    IMMEDIATE_FUZZY_LOCATION_CANONICALIZATION_ACTION_REPLACEMENT_VERSION,
    RESTAURANT_GROUNDED_COORDINATE_ACTION_REPLACEMENT_VERSION,
    TRAVEL_EN_FOLD1_DECISION_FORK_COORDINATE_ACTION_REPLACEMENT_VERSION,
    TRAVEL_EN_FOLD2_DECISION_FORK_SAME_ROUTE_ACTION_REPLACEMENT_VERSION,
    RESTAURANT_ANCHOR_PREREQUISITE_ACTION_REPLACEMENT_VERSION,
    RESTAURANT_ANCHOR_RECOMMENDATION_ACTION_REPLACEMENT_VERSION,
    EXPLICIT_BUDGET_FUZZY_LOCATION_ACTION_REPLACEMENT_VERSION,
    SPECIAL_SERVICE_ATTRACTION_TOOL_FAMILY_ACTION_REPLACEMENT_VERSION,
    THREE_DAY_BATCH_ATTRACTION_ACTION_REPLACEMENT_VERSION,
    TRAVEL_MULTICARD_ACTION_REPLACEMENT_VERSION,
    TRAVEL_FOUR_CARD_ACTION_REPLACEMENT_VERSION,
    TRAVEL_STRICT_POSITIVE_TWO_CARD_ACTION_REPLACEMENT_VERSION,
    TRAVEL_ZH_FOLD1_FLIGHT_SCHEMA_ACTION_REPLACEMENT_VERSION,
    TRAVEL_ZH_FOLD1_TWO_CARD_ACTION_REPLACEMENT_VERSION,
    TRAVEL_ZH_FOLD2_FLIGHT_SCHEMA_ACTION_REPLACEMENT_VERSION,
    TRAVEL_ZH_FOLD2_HOTEL_EMPTY_BRAND_ACTION_REPLACEMENT_VERSION,
    TRAVEL_ZH_FOLD3_ROUND2_TRAIN_ONLY_ACTION_REPLACEMENT_VERSION,
    TRAVEL_ZH_FOLD3_FIXED_POSITIVE_ACTION_REPLACEMENT_VERSION,
    TRAVEL_ZH_FOLD3_RESTAURANT_ALIAS_ONLY_ACTION_REPLACEMENT_VERSION,
    TRAVEL_ZH_FOLD3_RESTAURANT_PARENTHETICAL_ONLY_ACTION_REPLACEMENT_VERSION,
    TRAVEL_ZH_FOLD3_FLIGHT_REPEAT_LOOPCUT_ACTION_REPLACEMENT_VERSION,
    TRAVEL_ZH_FOLD3_DUPLICATE_RESTAURANT_ACTION_REPLACEMENT_VERSION,
    TRAVEL_ZH_FOLD3_USER_SELECTED_THREE_CARD_ACTION_REPLACEMENT_VERSION,
    TRAVEL_EN_FOLD1_SINGLE_VICINITY_ALIAS_ACTION_REPLACEMENT_VERSION,
    TRAVEL_EN_FOLD2_LATE_ALIAS_ACTION_REPLACEMENT_VERSION,
    TRAVEL_EN_FOLD3_ANCHOR_ACTION_REPLACEMENT_VERSION,
    TRAVEL_EN_FOLD3_STAGNATION_ANCHOR_ACTION_REPLACEMENT_VERSION,
    TRAVEL_EN_V2_FOLD1_ACTION_REPLACEMENT_VERSION,
    TRAVEL_EN_V4_FOLD1_SCHEMA_ALIAS_ACTION_REPLACEMENT_VERSION,
    TRAVEL_EN_V4_FOLD1_DUPLICATE_RESTAURANT_ACTION_REPLACEMENT_VERSION,
    TRAVEL_EN_V4_FOLD1_DUPLICATE_ROAD_ACTION_REPLACEMENT_VERSION,
    TRAVEL_EN_V4_FOLD1_COMBINED_ACTION_REPLACEMENT_VERSION,
    TRAVEL_EN_V4_FOLD1_RESTAURANT_NAME_NORMALIZATION_ACTION_REPLACEMENT_VERSION,
    TRAVEL_EN_V4_FOLD1_RESTAURANT_ANCHOR_ACTION_REPLACEMENT_VERSION,
    TRAVEL_EN_V4_FOLD1_ROUND2_COMBINED_ACTION_REPLACEMENT_VERSION,
    TRAVEL_EN_FOLD1_CLEANROOM_HOTEL_LOOPCUT_ACTION_REPLACEMENT_VERSION,
    TRAVEL_EN_FOLD1_CLEANROOM_THREE_CARD_ACTION_REPLACEMENT_VERSION,
    TRAVEL_EN_FOLD2_CLEANROOM_HOTEL_LOOPCUT_ACTION_REPLACEMENT_VERSION,
    TRAVEL_EN_FOLD2_CLEANROOM_INFERRED_ALIAS_ACTION_REPLACEMENT_VERSION,
    TRAVEL_EN_FOLD2_CLEANROOM_SAME_COORDINATE_ACTION_REPLACEMENT_VERSION,
    TRAVEL_EN_FOLD2_CLEANROOM_THREE_CARD_ACTION_REPLACEMENT_VERSION,
    TRAVEL_EN_FOLD2_CLEANROOM_ATTRACTION_DETAILS_LOOPCUT_ACTION_REPLACEMENT_VERSION,
    TRAVEL_EN_FOLD2_CLEANROOM_DUPLICATE_ATTRACTION_ACTION_REPLACEMENT_VERSION,
    TRAVEL_EN_FOLD2_CLEANROOM_ROUND3_TWO_CARD_ACTION_REPLACEMENT_VERSION,
    TRAVEL_EN_FOLD3_CLEANROOM_DUPLICATE_RESTAURANT_ACTION_REPLACEMENT_VERSION,
    TRAVEL_EN_FOLD3_CLEANROOM_HOTEL_LOOPCUT_ACTION_REPLACEMENT_VERSION,
    TRAVEL_EN_FOLD3_CLEANROOM_TWO_CARD_ACTION_REPLACEMENT_VERSION,
    TRAVEL_EN_FOLD3_CLEANROOM_ROUND2_FOUR_CARD_ACTION_REPLACEMENT_VERSION,
    TRAVEL_EN_FOLD3_CLEANROOM_ROUND3_STRICT_FOUR_CARD_ACTION_REPLACEMENT_VERSION,
    TRAVEL_EN_FOLD3_V8_K6_DUPLICATE_AROUND_ACTION_REPLACEMENT_VERSION,
    TRAVEL_EN_FOLD3_V8_K6_HOTEL_LOOPCUT_ACTION_REPLACEMENT_VERSION,
    TRAVEL_EN_FOLD3_V8_K6_TWO_CARD_ACTION_REPLACEMENT_VERSION,
    TRAVEL_EN_FOLD3_V8_K6_SAME_COORDINATE_ACTION_REPLACEMENT_VERSION,
    TRAVEL_EN_FOLD3_V8_K6_SEARCH_LOOPCUT_ACTION_REPLACEMENT_VERSION,
    TRAVEL_EN_FOLD3_V8_K6_PARENTHETICAL_ACTION_REPLACEMENT_VERSION,
    TRAVEL_EN_FOLD3_V8_K6_ROUND2_THREE_CARD_ACTION_REPLACEMENT_VERSION,
    TRAVEL_EN_FOLD3_V8_K6_RESTAURANT_LOOPCUT_ACTION_REPLACEMENT_VERSION,
    TRAVEL_EN_FOLD3_V8_K6_USER_FROZEN_FOUR_CARD_ACTION_REPLACEMENT_VERSION,
    TRAVEL_EN_FOLD2_CLEANROOM_USER_FROZEN_TWO_CARD_ACTION_REPLACEMENT_VERSION,
    *ENGLISH_ACTION_REPLACEMENT_VERSIONS,
}
SUPPORTED_HOTEL_BRANDS = ("全季", "亚朵", "万豪", "希尔顿", "如家", "锦江之星", "汉庭", "桔子")
INVALID_HOTEL_BRAND_PLACEHOLDERS = {"无", "其他", "不限", "不限制", "任意"}
ENGLISH_REMOVABLE_HOTEL_BRANDS = {
    "all seasons hotel",
    "atour hotel",
    "hanting hotel",
    "hilton",
    "home inn",
    "ji hotel",
    "jinjiang inn",
    "marriott",
    "orange hotel",
    "pullman",
    "quanji hotel",
}
ENGLISH_HOTEL_BRAND_PLACEHOLDERS = {"none", "other", "any", "no preference"}
ENGLISH_HOTEL_BRAND_ALIASES = {
    "all seasons hotel": {"all seasons hotel", "ji hotel", "quanji hotel"},
    "ji hotel": {"all seasons hotel", "ji hotel", "quanji hotel"},
    "quanji hotel": {"all seasons hotel", "ji hotel", "quanji hotel"},
    "home inn": {"home inn", "homeinn"},
}


def get_field(value: object, name: str, default: object = None) -> object:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def canonical_json(value: object) -> str:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value.strip()
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def completed_tool_events(messages: list[dict]) -> list[dict[str, str]]:
    events: list[dict[str, str]] = []
    pending: list[dict[str, str]] = []
    for message in messages:
        role = get_field(message, "role")
        tool_calls = get_field(message, "tool_calls") or []
        if role == "assistant" and tool_calls:
            pending = []
            for call in tool_calls:
                function = get_field(call, "function") or {}
                name = str(get_field(function, "name") or "")
                arguments = canonical_json(get_field(function, "arguments") or {})
                event = {
                    "name": name,
                    "signature": f"{name}|{arguments}",
                    "result": "",
                }
                events.append(event)
                pending.append(event)
        elif role == "tool" and pending:
            pending.pop(0)["result"] = canonical_json(get_field(message, "content") or "")
    return events


def is_persist_marker(message: object) -> bool:
    return bool(
        get_field(message, "role") == "system"
        and str(get_field(message, "content") or "").startswith(PERSIST_ACE_PREFIX)
    )


def completed_single_tool_pairs(messages: list[dict]) -> list[dict[str, object]]:
    pairs: list[dict[str, object]] = []
    for index, message in enumerate(messages[:-1]):
        if get_field(message, "role") != "assistant":
            continue
        tool_calls = get_field(message, "tool_calls") or []
        if len(tool_calls) != 1:
            continue
        result_message = messages[index + 1]
        if get_field(result_message, "role") != "tool":
            continue
        function = get_field(tool_calls[0], "function") or {}
        name = str(get_field(function, "name") or "")
        arguments = canonical_json(get_field(function, "arguments") or {})
        pairs.append(
            {
                "assistant_index": index,
                "tool_index": index + 1,
                "name": name,
                "signature": f"{name}|{arguments}",
                "result": canonical_json(get_field(result_message, "content") or ""),
            }
        )
    return pairs


def prepare_persist_context_dedup(messages: list[dict]) -> tuple[list[dict], dict | None]:
    visible = [message for message in messages if not is_persist_marker(message)]
    markers = [message for message in messages if is_persist_marker(message)]
    pairs = completed_single_tool_pairs(messages)
    if len(pairs) < 4 or len(pairs) < 2 or len(markers) >= 2:
        return visible, None
    left, right = pairs[-2:]
    latest_non_marker_index = max(
        (index for index, message in enumerate(messages) if not is_persist_marker(message)),
        default=-1,
    )
    recoverable = bool(
        str(left["name"]).startswith("query_")
        and left["signature"] == right["signature"]
        and right["tool_index"] == latest_non_marker_index
        and left["result"]
        and left["result"] == right["result"]
        and sum(len(str(get_field(message, "content") or "")) for message in messages) <= 70000
    )
    if not recoverable:
        return visible, None
    removed = {int(right["assistant_index"]), int(right["tool_index"])}
    visible = [
        message
        for index, message in enumerate(messages)
        if index not in removed and not is_persist_marker(message)
    ]
    event = "typed_ace" if not markers else "post_action_guard"
    payload = {
        "event": event,
        "version": "v3_context_dedup",
        "family": "exact_duplicate_evidence_reuse",
        "tool": right["name"],
        "call_index": len(pairs),
        "action": "model_visible_context_dedup",
        "removed_messages": 2,
        "model_visible_only": True,
    }
    marker = {
        "role": "system",
        "content": f"{PERSIST_ACE_PREFIX} {json.dumps(payload, ensure_ascii=False, sort_keys=True)}",
    }
    return visible, marker


def normalize_date(year: str, month: str, day: str) -> str:
    return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"


def travel_requirement_ledger(messages: list[dict]) -> dict[str, object]:
    user_query = next(
        (str(get_field(message, "content") or "") for message in messages if get_field(message, "role") == "user"),
        "",
    )
    route = re.search(
        r"从([\u4e00-\u9fff]{2,8}?)(?:出发)?去([\u4e00-\u9fff]{2,8}?)(?:玩|旅游|出发|待|，|。)",
        user_query,
    )
    dates = []
    inherited_year = None
    for year, month, day in re.findall(r"(?:(20\d{2})年)?(\d{1,2})月(\d{1,2})[日号]?", user_query):
        if year:
            inherited_year = year
        if inherited_year:
            dates.append(normalize_date(inherited_year, month, day))
    if len(dates) < 2 and dates:
        day_only = re.search(r"(\d{1,2})[号日](?:就|再)?(?:回来|返回|回)", user_query)
        if day_only:
            departure_year, departure_month, _ = dates[0].split("-")
            dates.append(normalize_date(departure_year, departure_month, day_only.group(1)))
    return_context = " ".join(
        match.group(1)
        for match in re.finditer(r"(?:回程|回来|再回)([^。！？\n]{0,32})", user_query)
    )
    return_mode = None
    if re.search(r"火车|高铁|列车", return_context):
        return_mode = "train"
    elif re.search(r"飞机|航班", return_context):
        return_mode = "flight"
    required_families = []
    for family, pattern in (
        ("hotel", r"酒店|住宿|住的地方"),
        ("attraction", r"景点|景区|博物院|遗址"),
        ("restaurant", r"餐厅|吃饭|美食|用餐"),
    ):
        if re.search(pattern, user_query):
            required_families.append(family)
    if re.search(r"火车|高铁|列车", user_query):
        required_families.append("train")
    if re.search(r"飞机|航班", user_query):
        required_families.append("flight")
    return {
        "origin": route.group(1) if route else None,
        "destination": route.group(2) if route else None,
        "departure_date": dates[0] if dates else None,
        "return_date": dates[1] if len(dates) > 1 else None,
        "return_mode": return_mode,
        "required_families": sorted(set(required_families)),
    }


def first_user_query(messages: list[dict]) -> str:
    return next(
        (str(get_field(message, "content") or "") for message in messages if get_field(message, "role") == "user"),
        "",
    )


def travel_calendar_days(user_query: str) -> int | None:
    matches = list(re.finditer(r"(?:(20\d{2})年)?(\d{1,2})月(\d{1,2})[号日]?", user_query))
    if not matches:
        duration_match = re.search(
            r"\b(one|two|three|four|five|six|seven|eight|nine|ten|\d{1,2})[- ]day\b",
            user_query,
            flags=re.IGNORECASE,
        ) or re.search(
            r"\b(?:stay(?:ing)?\s+for\s+)(one|two|three|four|five|six|seven|eight|nine|ten|\d{1,2})\s+days?\b",
            user_query,
            flags=re.IGNORECASE,
        )
        if duration_match is not None:
            token = duration_match.group(1).lower()
            words = {
                "one": 1,
                "two": 2,
                "three": 3,
                "four": 4,
                "five": 5,
                "six": 6,
                "seven": 7,
                "eight": 8,
                "nine": 9,
                "ten": 10,
            }
            days = words.get(token, int(token) if token.isdigit() else 0)
            return days if 1 <= days <= 31 else None
        months = {
            name.lower(): index
            for index, name in enumerate(
                (
                    "January",
                    "February",
                    "March",
                    "April",
                    "May",
                    "June",
                    "July",
                    "August",
                    "September",
                    "October",
                    "November",
                    "December",
                ),
                start=1,
            )
        }
        english_matches = list(
            re.finditer(
                r"\b(" + "|".join(months) + r")\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s*(20\d{2}))?",
                user_query,
                flags=re.IGNORECASE,
            )
        )
        if not english_matches:
            return None
        first = english_matches[0]
        year = int(first.group(3) or 0)
        if not year:
            return None
        month = months[first.group(1).lower()]
        try:
            departure = datetime(year, month, int(first.group(2))).date()
        except ValueError:
            return None
        return_date_match = next(
            (
                candidate
                for candidate in english_matches[1:]
                if re.search(
                    r"(?:return|until)",
                    user_query[max(first.end(), candidate.start() - 48) : candidate.start()],
                    flags=re.IGNORECASE,
                )
            ),
            None,
        )
        if return_date_match is not None:
            second = return_date_match
            return_year = int(second.group(3) or year)
            return_month = months[second.group(1).lower()]
            return_day = int(second.group(2))
        else:
            return_match = re.search(
                r"(?:return(?:ing)?|until)[^.!?\n]{0,24}?(?:the\s+)?(\d{1,2})(?:st|nd|rd|th)\b",
                user_query[first.end() :],
                flags=re.IGNORECASE,
            )
            if return_match is None:
                return None
            return_year = year
            return_month = month
            return_day = int(return_match.group(1))
        try:
            return_date = datetime(return_year, return_month, return_day).date()
        except ValueError:
            return None
        days = (return_date - departure).days + 1
        return days if 1 <= days <= 31 else None
    first = matches[0]
    year = int(first.group(1) or 0)
    if not year:
        return None
    try:
        departure = datetime(year, int(first.group(2)), int(first.group(3))).date()
    except ValueError:
        return None
    return_match = re.search(
        r"(?:(20\d{2})年)?(?:(\d{1,2})月)?(\d{1,2})[号日](?:[^。！？\n]{0,12})(?:回|返)",
        user_query[first.end() :],
    )
    if return_match is not None:
        return_year = int(return_match.group(1) or year)
        return_month = int(return_match.group(2) or first.group(2))
        return_day = int(return_match.group(3))
    elif len(matches) >= 2:
        second = matches[1]
        return_year = int(second.group(1) or year)
        return_month = int(second.group(2))
        return_day = int(second.group(3))
    else:
        return None
    try:
        return_date = datetime(return_year, return_month, return_day).date()
    except ValueError:
        return None
    days = (return_date - departure).days + 1
    return days if 1 <= days <= 31 else None


def has_explicit_travel_budget(user_query: str) -> bool:
    return bool(
        re.search(
            r"\b(?:total\s+)?budget\b[^.!?\n]{0,80}\d[\d,]*(?:\.\d+)?\s*(?:yuan|rmb)\b",
            user_query,
            flags=re.IGNORECASE,
        )
        or re.search(
            r"(?:¥|￥)\s*\d[\d,]*(?:\.\d+)?[^.!?\n]{0,40}\b(?:budget|limit)\b",
            user_query,
            flags=re.IGNORECASE,
        )
    )


def normalized_english_phrase(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).split())


def english_hotel_brand_components(value: str) -> list[str]:
    return [
        normalized_english_phrase(component)
        for component in re.split(r"[,;/|]+", str(value or ""))
        if normalized_english_phrase(component)
    ]


def english_hotel_brand_is_removable(value: str) -> bool:
    components = english_hotel_brand_components(value)
    allowed = ENGLISH_REMOVABLE_HOTEL_BRANDS | ENGLISH_HOTEL_BRAND_PLACEHOLDERS
    return bool(components) and all(component in allowed for component in components)


def english_hotel_brand_explicitly_requested(value: str, user_query: str) -> bool:
    normalized_query = f" {normalized_english_phrase(user_query)} "
    for component in english_hotel_brand_components(value):
        aliases = ENGLISH_HOTEL_BRAND_ALIASES.get(component, {component})
        if any(f" {alias} " in normalized_query for alias in aliases):
            return True
    return False


def hotel_stay_calendar_days(arguments: dict) -> int | None:
    try:
        checkin = datetime.strptime(str(arguments.get("checkinDate") or ""), "%Y-%m-%d").date()
        checkout = datetime.strptime(str(arguments.get("checkoutDate") or ""), "%Y-%m-%d").date()
    except ValueError:
        return None
    days = (checkout - checkin).days + 1
    return days if 1 <= days <= 31 else None


def has_explicit_robot_hotel_service(user_query: str) -> bool:
    return bool(
        re.search(
            r"\brobot(?:ic)?\b[^.!?\n]{0,36}\b(?:service|delivery|room|meal)\b",
            user_query,
            flags=re.IGNORECASE,
        )
        or re.search(
            r"\b(?:service|delivery|room|meal)\b[^.!?\n]{0,36}\brobot(?:ic)?\b",
            user_query,
            flags=re.IGNORECASE,
        )
    )


def travel_semantic_gate_evidence(messages: list[dict]) -> dict[str, object]:
    user_query = first_user_query(messages)
    requested_hotel_services = [
        service
        for service in ("洗衣机", "烘干机", "健身房", "游泳池", "停车场", "早餐", "接送机")
        if service in user_query
    ]
    restaurant_highest_rated = bool(
        re.search(r"(?:评分|评价)\s*最高(?:的)?[^。！？\n]{0,12}餐厅", user_query)
        or re.search(r"餐厅[^。！？\n]{0,12}(?:评分|评价)\s*最高", user_query)
    )
    explicit_budget = bool(
        re.search(r"(?:(?:总)?预算|总开销)[^。！？\n]{0,20}\d{2,}\s*元", user_query)
        or re.search(r"\d{2,}\s*元[^。！？\n]{0,12}(?:以内|以下|之内|预算|开销)", user_query)
        or re.search(r"每晚\s*\d{2,}\s*(?:到|至|[-~～])\s*\d{2,}\s*元", user_query)
    )
    trip_days = travel_calendar_days(user_query)
    return {
        "restaurant_highest_rated": restaurant_highest_rated,
        "explicit_budget": explicit_budget,
        "explicit_hotel_service": bool(requested_hotel_services),
        "requested_hotel_services": requested_hotel_services,
        "trip_days": trip_days,
        "low_complexity_trip": bool(trip_days is not None and trip_days <= 4),
    }


def apply_travel_semantic_recoverability_gate(messages: list[dict], decision: dict | None) -> dict | None:
    if decision is None:
        return None
    payload = decision["payload"]
    reason = payload.get("reason")
    version = payload.get("version")
    evidence = travel_semantic_gate_evidence(messages)
    if reason == "return_transport_contract_violation":
        return None
    if reason == "unavailable_tool_alias":
        if not evidence["restaurant_highest_rated"] or not evidence["low_complexity_trip"]:
            return None
        gate_reason = "ranked_restaurant_evidence_low_complexity"
        if version == "v14_tau_aligned_safe_bundle":
            decision["instruction"] += (
                " After obtaining a real candidate, schedule it only when visible opening-hours evidence and "
                "travel-time evidence fit the itinerary. Otherwise leave that meal unresolved. Recompute the "
                "budget once from visible prices; do not preserve stale restaurant costs."
            )
        elif version in {
            "v15_tau_source_preserving",
            "v16_tau_serializer_constraint_only",
            "v17_tau_personalized_hotel_backtracking",
            "v18_tau_no_compaction_source_preserving",
            "v19_tau_selective_no_compaction",
        }:
            return None
    elif reason == "repeated_failed_hotel_overconstraint":
        if evidence["explicit_budget"]:
            gate_reason = "explicit_budget_feasibility"
        elif (
            version in {
                "v17_tau_personalized_hotel_backtracking",
                "v18_tau_no_compaction_source_preserving",
                "v19_tau_selective_no_compaction",
            }
            and evidence["explicit_hotel_service"]
        ):
            gate_reason = "explicit_hotel_service_feasibility"
            decision["instruction"] += (
                " Preserve every hotel service explicitly requested by the user. Keep hotelBrands only when the "
                "original user request names that brand; otherwise omit hotelBrands entirely. After the corrected "
                "lookup, select a candidate only when its visible services include all requested services."
            )
        else:
            return None
    elif reason == "restaurant_detail_not_in_visible_candidates":
        if version in {
            "v14_tau_aligned_safe_bundle",
            "v15_tau_source_preserving",
            "v16_tau_serializer_constraint_only",
            "v17_tau_personalized_hotel_backtracking",
            "v18_tau_no_compaction_source_preserving",
            "v19_tau_selective_no_compaction",
        }:
            return None
        gate_reason = "visible_candidate_grounding"
    elif reason == "exact_read_only_action_and_effect_stagnation":
        gate_reason = "exact_signature_exact_effect_no_progress"
    elif reason == "bounded_read_only_family_stagnation":
        gate_reason = "same_scope_family_no_evidence_growth"
    else:
        return None
    payload["recoverability_gate"] = "execute"
    payload["recoverability_gate_reason"] = gate_reason
    payload["recoverability_gate_evidence"] = evidence
    return decision


def tool_family(name: str) -> str:
    if "hotel" in name:
        return "hotel"
    if "restaurant" in name:
        return "restaurant"
    if "attraction" in name:
        return "attraction"
    if "train" in name:
        return "train"
    if "flight" in name:
        return "flight"
    if name == "search_location":
        return "location"
    return name


def tool_result_failed(result: str) -> bool:
    text = str(result or "").strip()
    lower = text.lower()
    if not text or any(
        marker in lower
        for marker in (
            '"error"',
            "未找到",
            "not found",
            "no result",
            "no recommended",
            "no hotel information found",
        )
    ):
        return True
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return False
    return parsed in ([], None) or bool(isinstance(parsed, dict) and parsed.get("error"))


def proposed_calls(response: object) -> list[dict[str, str]]:
    choices = get_field(response, "choices") or []
    if not choices:
        return []
    message = get_field(choices[0], "message") or {}
    calls = []
    for call in get_field(message, "tool_calls") or []:
        function = get_field(call, "function") or {}
        calls.append(
            {
                "name": str(get_field(function, "name") or ""),
                "arguments": canonical_json(get_field(function, "arguments") or {}),
            }
        )
    return calls


def parse_arguments(call: dict[str, str]) -> dict:
    try:
        parsed = json.loads(call.get("arguments") or "{}")
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def visible_message_snapshot(messages: list[dict]) -> list[dict[str, object]]:
    snapshot: list[dict[str, object]] = []
    for message in messages:
        if is_persist_marker(message):
            continue
        tool_calls = []
        for call in get_field(message, "tool_calls") or []:
            function = get_field(call, "function") or {}
            tool_calls.append(
                {
                    "id": str(get_field(call, "id") or ""),
                    "name": str(get_field(function, "name") or ""),
                    "arguments": canonical_json(get_field(function, "arguments") or {}),
                }
            )
        snapshot.append(
            {
                "role": str(get_field(message, "role") or ""),
                "content": str(get_field(message, "content") or ""),
                "tool_call_id": str(get_field(message, "tool_call_id") or ""),
                "tool_calls": tool_calls,
            }
        )
    return snapshot


def pre_action_state_hash(messages: list[dict], call: dict[str, str]) -> str:
    state = {
        "history": visible_message_snapshot(messages),
        "proposal": {
            "name": call["name"],
            "arguments": canonical_json(call["arguments"]),
        },
    }
    encoded = json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def pre_action_batch_state_hash(messages: list[dict], calls: list[dict[str, str]]) -> str:
    state = {
        "history": visible_message_snapshot(messages),
        "proposal": [
            {"name": call["name"], "arguments": canonical_json(call["arguments"])}
            for call in calls
        ],
    }
    encoded = json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def travel_hotel_action_replacement_decision(
    messages: list[dict],
    calls: list[dict[str, str]],
    version: str = ACTION_REPLACEMENT_VERSION,
) -> dict | None:
    if version not in ACTION_REPLACEMENT_VERSIONS or len(calls) != 1:
        return None
    card = {
        ACTION_REPLACEMENT_VERSION: HOTEL_CONSTRAINT_CARD,
        LEGACY_ACTION_REPLACEMENT_VERSION: LEGACY_HOTEL_CONSTRAINT_CARD,
        ENGLISH_HOTEL_ACTION_REPLACEMENT_VERSION: ENGLISH_HOTEL_CONSTRAINT_CARD,
        ENGLISH_HOTEL_SERVICE_ACTION_REPLACEMENT_VERSION: ENGLISH_HOTEL_SERVICE_CARD,
        JI_FOUR_STAR_HOTEL_ACTION_REPLACEMENT_VERSION: JI_FOUR_STAR_HOTEL_CARD,
        ATTRACTION_READY_JI_HOTEL_ACTION_REPLACEMENT_VERSION: (
            ATTRACTION_READY_JI_HOTEL_CARD
        ),
    }.get(version)
    if card is None:
        return None
    call = calls[0]
    if call["name"] != "query_hotel_info":
        return None
    if any(
        is_persist_marker(message)
        and f'"card": "{card}"' in str(get_field(message, "content") or "")
        for message in messages
    ):
        return None

    arguments = parse_arguments(call)
    required = ("destination", "checkinDate", "checkoutDate")
    if any(not str(arguments.get(key) or "").strip() for key in required):
        return None
    proposed_brand = str(arguments.get("hotelBrands") or "").strip()
    user_query = first_user_query(messages)
    if version in {
        ENGLISH_HOTEL_ACTION_REPLACEMENT_VERSION,
        ENGLISH_HOTEL_SERVICE_ACTION_REPLACEMENT_VERSION,
        JI_FOUR_STAR_HOTEL_ACTION_REPLACEMENT_VERSION,
        ATTRACTION_READY_JI_HOTEL_ACTION_REPLACEMENT_VERSION,
    }:
        if not english_hotel_brand_is_removable(proposed_brand):
            return None
        if english_hotel_brand_explicitly_requested(proposed_brand, user_query):
            return None
        if has_explicit_travel_budget(user_query):
            return None
        trip_days = hotel_stay_calendar_days(arguments)
        if trip_days is None or trip_days > 4:
            return None
        if version == ENGLISH_HOTEL_SERVICE_ACTION_REPLACEMENT_VERSION:
            if len(english_hotel_brand_components(proposed_brand)) != 1:
                return None
            if not has_explicit_robot_hotel_service(user_query):
                return None
        if version in {
            JI_FOUR_STAR_HOTEL_ACTION_REPLACEMENT_VERSION,
            ATTRACTION_READY_JI_HOTEL_ACTION_REPLACEMENT_VERSION,
        }:
            if (
                normalized_english_phrase(proposed_brand) != "ji hotel"
                or str(arguments.get("hotelStar") or "").strip() != "4"
                or trip_days != 4
            ):
                return None
        if version == ATTRACTION_READY_JI_HOTEL_ACTION_REPLACEMENT_VERSION:
            normalized_query = normalized_english_phrase(user_query)
            explicit_attraction_requirement = bool(
                "attraction" in normalized_query
                and any(
                    phrase in normalized_query
                    for phrase in ("top three", "highest rated", "must see", "iconic")
                )
            )
            events = completed_tool_events(messages)
            attraction_evidence_ready = any(
                event["name"] == "recommend_attractions"
                and not tool_result_failed(event["result"])
                for event in events
            )
            if not explicit_attraction_requirement or not attraction_evidence_ready:
                return None
    else:
        if proposed_brand not in INVALID_HOTEL_BRAND_PLACEHOLDERS:
            return None
        if any(brand in user_query for brand in SUPPORTED_HOTEL_BRANDS):
            return None
        trip_days = travel_calendar_days(user_query)
        if version == ACTION_REPLACEMENT_VERSION and (trip_days is None or trip_days > 2):
            return None

    signature = f"{call['name']}|{canonical_json(arguments)}"
    events = completed_tool_events(messages)
    matching_positions = [
        index
        for index, event in enumerate(events)
        if event["signature"] == signature and tool_result_failed(event["result"])
    ]
    if len(matching_positions) < 2:
        return None
    prior_positions = matching_positions[-2:]
    prior_results = [events[index]["result"] for index in prior_positions]
    if not prior_results[0] or prior_results[0] != prior_results[1]:
        return None
    if any(
        tool_family(event["name"]) == "hotel" and not tool_result_failed(event["result"])
        for event in events[prior_positions[0] + 1 :]
    ):
        return None

    replacement_arguments = dict(arguments)
    del replacement_arguments["hotelBrands"]
    original_action = {"kind": "tool_call", "tool": call["name"], "arguments": arguments}
    replacement_action = {
        "kind": "tool_call",
        "tool": call["name"],
        "arguments": replacement_arguments,
    }
    payload = {
        "event": "action_replacement",
        "version": version,
        "card": card,
        "decision": "switch_to_assistant_tool_call",
        "reason": (
            "repeated_failed_inferred_english_hotel_brand"
            if version
            in {
                ENGLISH_HOTEL_ACTION_REPLACEMENT_VERSION,
                ENGLISH_HOTEL_SERVICE_ACTION_REPLACEMENT_VERSION,
                JI_FOUR_STAR_HOTEL_ACTION_REPLACEMENT_VERSION,
                ATTRACTION_READY_JI_HOTEL_ACTION_REPLACEMENT_VERSION,
            }
            else "repeated_failed_invalid_hotel_brand_placeholder"
        ),
        "pre_action_state_hash": pre_action_state_hash(messages, call),
        "original_proposal": original_action,
        "replacement_action": replacement_action,
        "failure_signature": hashlib.sha256(prior_results[0].encode("utf-8")).hexdigest(),
        "effect_unchanged": True,
        "evidence_delta": "none",
        "productive_retry": False,
        "removed_argument": {"hotelBrands": proposed_brand},
        "risk_envelope": {
            "trip_days": trip_days,
            "max_trip_days": (
                4
                if version
                in {
                    ENGLISH_HOTEL_ACTION_REPLACEMENT_VERSION,
                    ENGLISH_HOTEL_SERVICE_ACTION_REPLACEMENT_VERSION,
                    JI_FOUR_STAR_HOTEL_ACTION_REPLACEMENT_VERSION,
                    ATTRACTION_READY_JI_HOTEL_ACTION_REPLACEMENT_VERSION,
                }
                else 2 if version == ACTION_REPLACEMENT_VERSION else None
            ),
            "explicit_budget": has_explicit_travel_budget(user_query),
            "single_inferred_brand": len(english_hotel_brand_components(proposed_brand)) == 1,
            "explicit_robot_hotel_service": has_explicit_robot_hotel_service(user_query),
        },
        "trigger_count": 1,
        "max_triggers_per_trajectory": 1,
        "write_action_injected": False,
    }
    return {
        "decision": "switch_to_assistant_tool_call",
        "payload": payload,
        "concrete_action": replacement_action,
    }


def travel_restaurant_alias_action_replacement_decision(
    messages: list[dict],
    calls: list[dict[str, str]],
    version: str = ENGLISH_ACTION_REPLACEMENT_VERSION,
) -> dict | None:
    if version not in ENGLISH_ACTION_REPLACEMENT_VERSIONS or len(calls) != 1:
        return None
    call = calls[0]
    if call["name"] != "recommend_around_restaurants":
        return None
    card = {
        LEGACY_ENGLISH_ACTION_REPLACEMENT_VERSION: LEGACY_RESTAURANT_ALIAS_CARD,
        IMMEDIATE_ENGLISH_ACTION_REPLACEMENT_VERSION: IMMEDIATE_RESTAURANT_ALIAS_CARD,
        SCHEMA_ENGLISH_ACTION_REPLACEMENT_VERSION: SCHEMA_RESTAURANT_ALIAS_CARD,
        GROUNDED_ENGLISH_ACTION_REPLACEMENT_VERSION: GROUNDED_RESTAURANT_ALIAS_CARD,
        SHORT_TRIP_ENGLISH_ACTION_REPLACEMENT_VERSION: SHORT_TRIP_RESTAURANT_ALIAS_CARD,
        TWO_DAY_ENGLISH_ACTION_REPLACEMENT_VERSION: TWO_DAY_RESTAURANT_ALIAS_CARD,
        ENGLISH_ACTION_REPLACEMENT_VERSION: RESTAURANT_ALIAS_CARD,
        THREE_STRIKE_RESTAURANT_ALIAS_ACTION_REPLACEMENT_VERSION: (
            THREE_STRIKE_RESTAURANT_ALIAS_CARD
        ),
        SPECIAL_SERVICE_RESTAURANT_ALIAS_FOLLOWUP_ACTION_REPLACEMENT_VERSION: (
            SPECIAL_SERVICE_RESTAURANT_ALIAS_FOLLOWUP_CARD
        ),
    }[version]
    if any(
        is_persist_marker(message)
        and f'"card": "{card}"' in str(get_field(message, "content") or "")
        for message in messages
    ):
        return None

    arguments = parse_arguments(call)
    required = ("latitude", "longitude")
    if any(not str(arguments.get(key) or "").strip() for key in required):
        return None
    if set(arguments) != set(required):
        return None

    signature = f"{call['name']}|{canonical_json(arguments)}"
    events = completed_tool_events(messages)
    grounding_event = grounded_search_location_event(events, arguments)
    grounded_versions = {
        GROUNDED_ENGLISH_ACTION_REPLACEMENT_VERSION,
        SHORT_TRIP_ENGLISH_ACTION_REPLACEMENT_VERSION,
        TWO_DAY_ENGLISH_ACTION_REPLACEMENT_VERSION,
        ENGLISH_ACTION_REPLACEMENT_VERSION,
    }
    if version in grounded_versions and grounding_event is None:
        return None
    trip_days = travel_calendar_days(first_user_query(messages))
    max_trip_days = {
        SHORT_TRIP_ENGLISH_ACTION_REPLACEMENT_VERSION: 3,
        TWO_DAY_ENGLISH_ACTION_REPLACEMENT_VERSION: 2,
        ENGLISH_ACTION_REPLACEMENT_VERSION: 2,
        THREE_STRIKE_RESTAURANT_ALIAS_ACTION_REPLACEMENT_VERSION: 2,
        SPECIAL_SERVICE_RESTAURANT_ALIAS_FOLLOWUP_ACTION_REPLACEMENT_VERSION: 5,
    }.get(version)
    if max_trip_days is not None and (
        trip_days is None or trip_days > max_trip_days
    ):
        return None
    explicit_budget = has_explicit_travel_budget(first_user_query(messages))
    if version in {
        ENGLISH_ACTION_REPLACEMENT_VERSION,
        THREE_STRIKE_RESTAURANT_ALIAS_ACTION_REPLACEMENT_VERSION,
    } and explicit_budget:
        return None
    if version == SPECIAL_SERVICE_RESTAURANT_ALIAS_FOLLOWUP_ACTION_REPLACEMENT_VERSION:
        if explicit_budget or trip_days is None or trip_days < 4:
            return None
        if not any(
            is_persist_marker(message)
            and f'"card": "{SPECIAL_SERVICE_ATTRACTION_TOOL_FAMILY_CARD}"'
            in str(get_field(message, "content") or "")
            for message in messages
        ):
            return None
    last_failure = None
    failure_result = None
    schema_versions = {
        SCHEMA_ENGLISH_ACTION_REPLACEMENT_VERSION,
        GROUNDED_ENGLISH_ACTION_REPLACEMENT_VERSION,
        SHORT_TRIP_ENGLISH_ACTION_REPLACEMENT_VERSION,
        TWO_DAY_ENGLISH_ACTION_REPLACEMENT_VERSION,
        ENGLISH_ACTION_REPLACEMENT_VERSION,
        SPECIAL_SERVICE_RESTAURANT_ALIAS_FOLLOWUP_ACTION_REPLACEMENT_VERSION,
    }
    if version not in schema_versions:
        matching_positions = [
            index
            for index, event in enumerate(events)
            if event["signature"] == signature
            and "tool 'recommend_around_restaurants' not found" in event["result"].lower()
        ]
        if not matching_positions:
            return None
        if (
            version == THREE_STRIKE_RESTAURANT_ALIAS_ACTION_REPLACEMENT_VERSION
            and len(matching_positions) < 3
        ):
            return None
        last_failure = matching_positions[-1]
        if (
            version == IMMEDIATE_ENGLISH_ACTION_REPLACEMENT_VERSION
            and last_failure != len(events) - 1
        ):
            return None
        if any(
            tool_family(event["name"]) == "restaurant"
            and not tool_result_failed(event["result"])
            for event in events[last_failure + 1 :]
        ):
            return None
        failure_result = events[last_failure]["result"]

    original_action = {"kind": "tool_call", "tool": call["name"], "arguments": arguments}
    replacement_action = {
        "kind": "tool_call",
        "tool": "recommend_restaurants",
        "arguments": dict(arguments),
    }
    schema_repair = version in schema_versions
    payload = {
        "event": "action_replacement",
        "version": version,
        "card": card,
        "decision": "switch_to_assistant_tool_call",
        "reason": (
            "grounded_two_day_no_budget_unregistered_restaurant_tool_alias_schema_repair"
            if version == ENGLISH_ACTION_REPLACEMENT_VERSION
            else "three_strike_authoritative_restaurant_tool_alias_error"
            if version == THREE_STRIKE_RESTAURANT_ALIAS_ACTION_REPLACEMENT_VERSION
            else "post_special_service_grounded_restaurant_tool_alias_schema_repair"
            if version
            == SPECIAL_SERVICE_RESTAURANT_ALIAS_FOLLOWUP_ACTION_REPLACEMENT_VERSION
            else "grounded_two_day_unregistered_restaurant_tool_alias_schema_repair"
            if version == TWO_DAY_ENGLISH_ACTION_REPLACEMENT_VERSION
            else "grounded_short_trip_unregistered_restaurant_tool_alias_schema_repair"
            if version == SHORT_TRIP_ENGLISH_ACTION_REPLACEMENT_VERSION
            else "grounded_unregistered_restaurant_tool_alias_schema_repair"
            if version == GROUNDED_ENGLISH_ACTION_REPLACEMENT_VERSION
            else "unregistered_restaurant_tool_alias_schema_repair"
            if version == SCHEMA_ENGLISH_ACTION_REPLACEMENT_VERSION
            else "repeated_authoritative_restaurant_tool_alias_error"
        ),
        "pre_action_state_hash": pre_action_state_hash(messages, call),
        "original_proposal": original_action,
        "replacement_action": replacement_action,
        "failure_signature": (
            hashlib.sha256(failure_result.encode("utf-8")).hexdigest()
            if failure_result is not None
            else None
        ),
        "effect_unchanged": None if schema_repair else True,
        "evidence_delta": (
            "visible_coordinate_grounding"
            if version in grounded_versions
            else "pre_execution_schema_validation"
            if schema_repair
            else "none"
        ),
        "productive_retry": False,
        "authoritative_error": failure_result,
        "coordinate_grounding": (
            {
                "tool": grounding_event["name"],
                "signature": grounding_event["signature"],
            }
            if grounding_event is not None
            else None
        ),
        "risk_envelope": {
            "read_only": True,
            "argument_preserving": True,
            "source_tool_unavailable": True,
            "source_tool_unregistered": schema_repair,
            "target_tool": "recommend_restaurants",
            "trip_days": trip_days,
            "max_trip_days": max_trip_days,
            "explicit_budget": explicit_budget,
            "explicit_budget_allowed": version
            not in {
                ENGLISH_ACTION_REPLACEMENT_VERSION,
                THREE_STRIKE_RESTAURANT_ALIAS_ACTION_REPLACEMENT_VERSION,
            },
            "minimum_authoritative_failures": (
                3
                if version == THREE_STRIKE_RESTAURANT_ALIAS_ACTION_REPLACEMENT_VERSION
                else None
            ),
            "requires_special_service_attraction_card": (
                version
                == SPECIAL_SERVICE_RESTAURANT_ALIAS_FOLLOWUP_ACTION_REPLACEMENT_VERSION
            ),
        },
        "trigger_count": 1,
        "max_triggers_per_trajectory": 1,
        "write_action_injected": False,
    }
    return {
        "decision": "switch_to_assistant_tool_call",
        "payload": payload,
        "concrete_action": replacement_action,
    }


def travel_en_fold1_single_vicinity_alias_action_replacement_decision(
    messages: list[dict],
    calls: list[dict[str, str]],
    version: str = TRAVEL_EN_FOLD1_SINGLE_VICINITY_ALIAS_ACTION_REPLACEMENT_VERSION,
) -> dict | None:
    if version != TRAVEL_EN_FOLD1_SINGLE_VICINITY_ALIAS_ACTION_REPLACEMENT_VERSION:
        return None
    if any(
        is_persist_marker(message)
        and f'"card": "{TRAVEL_EN_FOLD1_SINGLE_VICINITY_ALIAS_CARD}"'
        in str(get_field(message, "content") or "")
        for message in messages
    ):
        return None

    user_query = first_user_query(messages)
    vicinity_count = len(
        re.findall(r"\b(?:nearby|near|around)\b", user_query, flags=re.IGNORECASE)
    )
    if vicinity_count > 1:
        return None

    events = completed_tool_events(messages)
    if any(
        event["name"] == "search_location" and not tool_result_failed(event["result"])
        for event in events
    ):
        return None

    eligible: list[tuple[int, dict[str, str], dict, int, str]] = []
    for call_index, call in enumerate(calls):
        if call["name"] != "recommend_around_restaurants":
            continue
        arguments = parse_arguments(call)
        if set(arguments) != {"latitude", "longitude"}:
            continue
        if any(not str(arguments.get(key) or "").strip() for key in ("latitude", "longitude")):
            continue
        signature = f"{call['name']}|{canonical_json(arguments)}"
        matching_positions = [
            index
            for index, event in enumerate(events)
            if event["signature"] == signature
            and "tool 'recommend_around_restaurants' not found" in event["result"].lower()
        ]
        if not matching_positions:
            continue
        last_failure = matching_positions[-1]
        if any(
            tool_family(event["name"]) == "restaurant"
            and not tool_result_failed(event["result"])
            for event in events[last_failure + 1 :]
        ):
            continue
        eligible.append(
            (call_index, call, arguments, len(matching_positions), events[last_failure]["result"])
        )
    if len(eligible) != 1:
        return None

    call_index, call, arguments, failure_count, failure_result = eligible[0]
    original_action = {"kind": "tool_call", "tool": call["name"], "arguments": arguments}
    replacement_action = {
        "kind": "tool_call",
        "tool": "recommend_restaurants",
        "arguments": dict(arguments),
    }
    payload = {
        "event": "action_replacement",
        "version": version,
        "card": TRAVEL_EN_FOLD1_SINGLE_VICINITY_ALIAS_CARD,
        "decision": "switch_to_assistant_tool_call",
        "reason": "single_vicinity_failed_restaurant_alias_before_successful_location_lookup",
        "pre_action_state_hash": pre_action_state_hash(messages, call),
        "original_proposal": original_action,
        "replacement_action": replacement_action,
        "target_tool_call_index": call_index,
        "proposal_batch_size": len(calls),
        "collapse_proposal_batch": False,
        "failure_signature": hashlib.sha256(failure_result.encode("utf-8")).hexdigest(),
        "evidence_delta": "authoritative_tool_schema_repair",
        "productive_retry": False,
        "risk_envelope": {
            "read_only": True,
            "argument_preserving": True,
            "vicinity_requirement_count": vicinity_count,
            "successful_search_location_before_proposal": False,
            "authoritative_failure_count": failure_count,
            "unique_eligible_call": True,
        },
        "trigger_count": 1,
        "max_triggers_per_trajectory": 1,
        "write_action_injected": False,
    }
    return {
        "decision": "switch_to_assistant_tool_call",
        "payload": payload,
        "concrete_action": replacement_action,
    }


def travel_en_fold2_late_alias_action_replacement_decision(
    messages: list[dict],
    calls: list[dict[str, str]],
    version: str = TRAVEL_EN_FOLD2_LATE_ALIAS_ACTION_REPLACEMENT_VERSION,
) -> dict | None:
    if version != TRAVEL_EN_FOLD2_LATE_ALIAS_ACTION_REPLACEMENT_VERSION:
        return None
    if any(
        is_persist_marker(message)
        and f'"card": "{TRAVEL_EN_FOLD2_LATE_ALIAS_CARD}"'
        in str(get_field(message, "content") or "")
        for message in messages
    ):
        return None

    events = completed_tool_events(messages)
    eligible: list[tuple[int, dict[str, str], dict, int, str]] = []
    for call_index, call in enumerate(calls):
        if call["name"] != "recommend_around_restaurants":
            continue
        arguments = parse_arguments(call)
        if set(arguments) != {"latitude", "longitude"}:
            continue
        if any(not str(arguments.get(key) or "").strip() for key in ("latitude", "longitude")):
            continue
        signature = f"{call['name']}|{canonical_json(arguments)}"
        matching_positions = [
            index
            for index, event in enumerate(events)
            if event["signature"] == signature
            and "tool 'recommend_around_restaurants' not found" in event["result"].lower()
        ]
        if len(matching_positions) < 2:
            continue
        last_failure = matching_positions[-1]
        if any(
            tool_family(event["name"]) == "restaurant"
            and not tool_result_failed(event["result"])
            for event in events[last_failure + 1 :]
        ):
            continue
        eligible.append(
            (call_index, call, arguments, len(matching_positions), events[last_failure]["result"])
        )
    if len(eligible) != 1:
        return None

    call_index, call, arguments, failure_count, failure_result = eligible[0]
    original_action = {"kind": "tool_call", "tool": call["name"], "arguments": arguments}
    replacement_action = {
        "kind": "tool_call",
        "tool": "recommend_restaurants",
        "arguments": dict(arguments),
    }
    payload = {
        "event": "action_replacement",
        "version": version,
        "card": TRAVEL_EN_FOLD2_LATE_ALIAS_CARD,
        "decision": "switch_to_assistant_tool_call",
        "reason": "replace_repeated_failed_restaurant_alias_with_schema_valid_tool",
        "pre_action_state_hash": pre_action_state_hash(messages, call),
        "original_proposal": original_action,
        "replacement_action": replacement_action,
        "target_tool_call_index": call_index,
        "proposal_batch_size": len(calls),
        "collapse_proposal_batch": False,
        "failure_signature": hashlib.sha256(failure_result.encode("utf-8")).hexdigest(),
        "evidence_delta": "authoritative_tool_schema_repair_after_two_exact_failures",
        "productive_retry": False,
        "risk_envelope": {
            "read_only": True,
            "argument_preserving": True,
            "minimum_prior_exact_failures": 2,
            "authoritative_failure_count": failure_count,
            "unique_eligible_call": True,
            "fold2_train_induced": True,
            "prior_fold_gate_imported": False,
        },
        "trigger_count": 1,
        "max_triggers_per_trajectory": 1,
        "write_action_injected": False,
    }
    return {
        "decision": "switch_to_assistant_tool_call",
        "payload": payload,
        "concrete_action": replacement_action,
    }


def travel_en_fold3_anchor_action_replacement_decision(
    messages: list[dict],
    calls: list[dict[str, str]],
    version: str = TRAVEL_EN_FOLD3_ANCHOR_ACTION_REPLACEMENT_VERSION,
) -> dict | None:
    if version not in {
        TRAVEL_EN_FOLD3_ANCHOR_ACTION_REPLACEMENT_VERSION,
        TRAVEL_EN_FOLD3_STAGNATION_ANCHOR_ACTION_REPLACEMENT_VERSION,
    }:
        return None
    strict_stagnation = version == TRAVEL_EN_FOLD3_STAGNATION_ANCHOR_ACTION_REPLACEMENT_VERSION
    card = (
        TRAVEL_EN_FOLD3_STAGNATION_ANCHOR_CARD
        if strict_stagnation
        else TRAVEL_EN_FOLD3_ANCHOR_CARD
    )
    if any(
        is_persist_marker(message)
        and f'"card": "{card}"'
        in str(get_field(message, "content") or "")
        for message in messages
    ):
        return None

    events = completed_tool_events(messages)
    coordinates: list[tuple[str, str]] = []
    for event in events:
        if event["name"] != "search_location" or tool_result_failed(event["result"]):
            continue
        try:
            result = json.loads(event["result"])
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(result, dict):
            continue
        latitude = result.get("latitude") or result.get("lat")
        longitude = result.get("longitude") or result.get("lng") or result.get("lon")
        if latitude is not None and longitude is not None:
            coordinates.append((str(latitude), str(longitude)))
    if len(coordinates) != 1:
        return None
    latitude, longitude = coordinates[0]
    replacement_arguments = {"latitude": latitude, "longitude": longitude}
    replacement_signature = f"recommend_restaurants|{canonical_json(replacement_arguments)}"
    if any(
        event["signature"] == replacement_signature and tool_result_failed(event["result"])
        for event in events
    ):
        return None

    candidates = []
    for index, call in enumerate(calls):
        if call["name"] != "query_restaurant_details":
            continue
        arguments = parse_arguments(call)
        if set(arguments) != {"restaurant_name"} or not str(arguments["restaurant_name"]).strip():
            continue
        signature = f"query_restaurant_details|{canonical_json(arguments)}"
        failed_events = [
            event
            for event in events
            if event["signature"] == signature and tool_result_failed(event["result"])
        ]
        if not failed_events:
            continue
        failed_alias_events = [
            event
            for event in events
            if event["name"] == "recommend_around_restaurants"
            and tool_result_failed(event["result"])
        ]
        if strict_stagnation and (
            len(failed_events) < 4 or len(failed_alias_events) < 6
        ):
            continue
        original_action = {"kind": "tool_call", "tool": call["name"], "arguments": arguments}
        replacement_action = {
            "kind": "tool_call",
            "tool": "recommend_restaurants",
            "arguments": replacement_arguments,
        }
        payload = {
            "event": "action_replacement",
            "version": version,
            "card": card,
            "decision": "switch_to_assistant_tool_call",
            "reason": "restaurant_detail_failed_switch_to_recommend_restaurants_at_unique_prior_location",
            "pre_action_state_hash": pre_action_batch_state_hash(messages, calls),
            "original_proposal": original_action,
            "replacement_action": replacement_action,
            "failure_signature": hashlib.sha256(failed_events[-1]["result"].encode("utf-8")).hexdigest(),
            "evidence_delta": "unique_visible_location_coordinates",
            "productive_retry": False,
            "risk_envelope": {
                "prior_detail_failure": True,
                "unique_visible_location_coordinates": True,
                "prior_replacement_failure": False,
                "minimum_prior_exact_detail_failures": 4 if strict_stagnation else 1,
                "observed_prior_exact_detail_failures": len(failed_events),
                "minimum_prior_alias_failures": 6 if strict_stagnation else 0,
                "observed_prior_alias_failures": len(failed_alias_events),
                "fold3_train_round3_gate": strict_stagnation,
            },
            "trigger_count": 1,
            "max_triggers_per_trajectory": 1,
            "target_tool_call_index": index,
            "proposal_batch_size": len(calls),
            "write_action_injected": False,
        }
        candidates.append(
            {
                "decision": "switch_to_assistant_tool_call",
                "payload": payload,
                "concrete_action": replacement_action,
            }
        )
    return candidates[0] if len(candidates) == 1 else None


def travel_en_v2_fold1_anchor_action_replacement_decision(
    messages: list[dict],
    calls: list[dict[str, str]],
    version: str = TRAVEL_EN_V2_FOLD1_ACTION_REPLACEMENT_VERSION,
) -> dict | None:
    if version != TRAVEL_EN_V2_FOLD1_ACTION_REPLACEMENT_VERSION:
        return None
    card = TRAVEL_EN_V2_FOLD1_RESTAURANT_ANCHOR_CARD
    if any(
        is_persist_marker(message)
        and card in str(get_field(message, "content") or "")
        for message in messages
    ):
        return None

    events = completed_tool_events(messages)
    coordinates: list[tuple[str, str]] = []
    for event in events:
        if event["name"] != "search_location" or tool_result_failed(event["result"]):
            continue
        try:
            result = json.loads(event["result"])
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(result, dict):
            continue
        latitude = result.get("latitude") or result.get("lat")
        longitude = result.get("longitude") or result.get("lng") or result.get("lon")
        if latitude is not None and longitude is not None:
            coordinates.append((str(latitude), str(longitude)))
    if len(coordinates) != 1:
        return None

    latitude, longitude = coordinates[0]
    replacement_arguments = {"latitude": latitude, "longitude": longitude}
    replacement_signature = f"recommend_restaurants|{canonical_json(replacement_arguments)}"
    if any(
        event["signature"] == replacement_signature and tool_result_failed(event["result"])
        for event in events
    ):
        return None

    normalized_user_query = f" {normalized_english_phrase(first_user_query(messages))} "
    candidates = []
    for call_index, call in enumerate(calls):
        if call["name"] != "query_restaurant_details":
            continue
        arguments = parse_arguments(call)
        if set(arguments) != {"restaurant_name"}:
            continue
        restaurant_name = str(arguments.get("restaurant_name") or "").strip()
        normalized_restaurant_name = normalized_english_phrase(restaurant_name)
        if not normalized_restaurant_name:
            continue
        if f" {normalized_restaurant_name} " in normalized_user_query:
            continue
        signature = f"query_restaurant_details|{canonical_json(arguments)}"
        failed_events = [
            event
            for event in events
            if event["signature"] == signature and tool_result_failed(event["result"])
        ]
        if not failed_events:
            continue
        original_action = {
            "kind": "tool_call",
            "tool": call["name"],
            "arguments": arguments,
        }
        replacement_action = {
            "kind": "tool_call",
            "tool": "recommend_restaurants",
            "arguments": replacement_arguments,
        }
        payload = {
            "event": "action_replacement",
            "version": version,
            "card": card,
            "decision": "switch_to_assistant_tool_call",
            "reason": "replace_failed_nonexplicit_restaurant_detail_with_unique_prior_location_recommendation",
            "pre_action_state_hash": pre_action_batch_state_hash(messages, calls),
            "original_proposal": original_action,
            "replacement_action": replacement_action,
            "target_tool_call_index": call_index,
            "proposal_batch_size": len(calls),
            "collapse_proposal_batch": False,
            "failure_signature": hashlib.sha256(failed_events[-1]["result"].encode("utf-8")).hexdigest(),
            "evidence_delta": "unique_visible_location_coordinates",
            "productive_retry": False,
            "task_id_or_gold_used": False,
            "train_only_design": True,
            "fold": "travel_en_v2_fold1",
            "card_role": "paper_primary",
            "primary_metric": "official_composite_and_commonsense",
            "paper_primary_tradeoff": True,
            "expected_tradeoffs": ["tool_calls_up", "broad_fl_may_increase"],
            "risk_envelope": {
                "prior_detail_failure": True,
                "unique_visible_location_coordinates": True,
                "prior_replacement_failure": False,
                "explicit_user_restaurant_protected": True,
                "argument_source": "online_visible_search_location_result",
            },
            "trigger_count": 1,
            "max_triggers_per_trajectory": 1,
            "write_action_injected": False,
        }
        candidates.append(
            {
                "decision": "switch_to_assistant_tool_call",
                "payload": payload,
                "concrete_action": replacement_action,
            }
        )
    return candidates[0] if len(candidates) == 1 else None


def travel_en_v2_fold1_search_location_loopcut_decision(
    messages: list[dict],
    calls: list[dict[str, str]],
    version: str = TRAVEL_EN_V2_FOLD1_ACTION_REPLACEMENT_VERSION,
) -> dict | None:
    if version != TRAVEL_EN_V2_FOLD1_ACTION_REPLACEMENT_VERSION:
        return None
    card = TRAVEL_EN_V2_FOLD1_SEARCH_LOCATION_LOOPCUT_CARD
    if any(
        is_persist_marker(message)
        and card in str(get_field(message, "content") or "")
        for message in messages
    ):
        return None

    events = completed_tool_events(messages)
    candidates = []
    for call_index, call in enumerate(calls):
        if call["name"] != "search_location":
            continue
        arguments = parse_arguments(call)
        if set(arguments) != {"place_name"} or not str(arguments.get("place_name") or "").strip():
            continue
        signature = f"search_location|{canonical_json(arguments)}"
        matching_positions = [
            index for index, event in enumerate(events) if event["signature"] == signature
        ]
        failures = [
            events[index]
            for index in matching_positions
            if tool_result_failed(events[index]["result"])
        ]
        if len(failures) < 2:
            continue
        later_events = events[matching_positions[-1] + 1 :]
        if any(not tool_result_failed(event["result"]) for event in later_events):
            continue
        original_action = {
            "kind": "tool_call",
            "tool": call["name"],
            "arguments": arguments,
        }
        replacement_action = {
            "kind": "tool_call",
            "tool": "__DROP_TOOL_CALL__",
            "arguments": {"drop_reason": "repeated_failure_without_new_evidence"},
        }
        payload = {
            "event": "action_replacement",
            "version": version,
            "card": card,
            "decision": "switch_to_assistant_tool_call",
            "reason": "cut_exact_repeated_search_location_failure_without_new_successful_tool_evidence",
            "pre_action_state_hash": pre_action_state_hash(messages, call),
            "original_proposal": original_action,
            "replacement_action": replacement_action,
            "target_tool_call_index": call_index,
            "proposal_batch_size": len(calls),
            "collapse_proposal_batch": False,
            "failure_signature": hashlib.sha256(failures[-1]["result"].encode("utf-8")).hexdigest(),
            "evidence_delta": "none_since_last_exact_failure",
            "productive_retry": False,
            "task_id_or_gold_used": False,
            "train_only_design": True,
            "fold": "travel_en_v2_fold1",
            "card_role": "paper_primary_and_efficiency_compensation",
            "primary_metric": "official_composite_with_tool_call_reduction",
            "paper_primary_tradeoff": True,
            "expected_tradeoffs": ["one_train_official_loss_observed_in_round1"],
            "risk_envelope": {
                "minimum_prior_exact_failures": 2,
                "observed_prior_exact_failures": len(failures),
                "new_successful_tool_evidence_since_last_exact_call": False,
                "read_only_action": True,
            },
            "trigger_count": 1,
            "max_triggers_per_trajectory": 1,
            "write_action_injected": False,
        }
        candidates.append(
            {
                "decision": "switch_to_assistant_tool_call",
                "payload": payload,
                "concrete_action": replacement_action,
            }
        )
    return candidates[0] if len(candidates) == 1 else None


def _travel_attraction_grounding_single_decision(
    messages: list[dict],
    call: dict[str, str],
    all_calls: list[dict[str, str]],
    target_tool_call_index: int,
    version: str = ATTRACTION_GROUNDING_ACTION_REPLACEMENT_VERSION,
) -> dict | None:
    if version != ATTRACTION_GROUNDING_ACTION_REPLACEMENT_VERSION:
        return None
    if call["name"] != "query_attraction_details":
        return None
    if any(
        is_persist_marker(message)
        and f'"card": "{ATTRACTION_GROUNDING_CARD}"' in str(get_field(message, "content") or "")
        for message in messages
    ):
        return None

    arguments = parse_arguments(call)
    if set(arguments) != {"attraction_name"}:
        return None
    attraction_name = str(arguments.get("attraction_name") or "").strip()
    if not attraction_name:
        return None
    user_query = first_user_query(messages)
    trip_days = travel_calendar_days(user_query)
    if trip_days is None or trip_days > 4 or has_explicit_travel_budget(user_query):
        return None

    events = completed_tool_events(messages)
    signature = f"query_attraction_details|{canonical_json(arguments)}"
    matching_positions = [
        index
        for index, event in enumerate(events)
        if event["signature"] == signature and tool_result_failed(event["result"])
    ]
    if len(matching_positions) < 2:
        return None
    prior_results = [events[index]["result"] for index in matching_positions[-2:]]
    if not prior_results[0] or prior_results[0] != prior_results[1]:
        return None
    normalized_name = normalized_english_phrase(attraction_name)
    grounded_event = next(
        (
            event
            for event in reversed(events[: matching_positions[-1] + 1])
            if event["name"] == "recommend_attractions"
            and not tool_result_failed(event["result"])
            and normalized_name in normalized_english_phrase(event["result"])
        ),
        None,
    )
    if grounded_event is None:
        return None
    if any(
        event["name"] == "search_location"
        and normalized_name
        == normalized_english_phrase(
            str(parse_arguments({"arguments": event["signature"].partition("|")[2]}).get("place_name") or "")
        )
        for event in events
    ):
        return None
    if any(
        event["signature"] == signature and not tool_result_failed(event["result"])
        for event in events
    ):
        return None

    original_action = {"kind": "tool_call", "tool": call["name"], "arguments": arguments}
    replacement_action = {
        "kind": "tool_call",
        "tool": "search_location",
        "arguments": {"place_name": attraction_name},
    }
    payload = {
        "event": "action_replacement",
        "version": version,
        "card": ATTRACTION_GROUNDING_CARD,
        "decision": "switch_to_assistant_tool_call",
        "reason": "repeated_failed_grounded_attraction_detail_lookup",
        "pre_action_state_hash": pre_action_batch_state_hash(messages, all_calls),
        "original_proposal": original_action,
        "replacement_action": replacement_action,
        "failure_signature": hashlib.sha256(prior_results[0].encode("utf-8")).hexdigest(),
        "grounding_signature": hashlib.sha256(grounded_event["result"].encode("utf-8")).hexdigest(),
        "effect_unchanged": True,
        "evidence_delta": "none",
        "productive_retry": False,
        "risk_envelope": {
            "trip_days": trip_days,
            "max_trip_days": 4,
            "explicit_budget": False,
            "exact_recommended_name": True,
            "prior_location_lookup": False,
        },
        "trigger_count": 1,
        "max_triggers_per_trajectory": 1,
        "target_tool_call_index": target_tool_call_index,
        "proposal_batch_size": len(all_calls),
        "write_action_injected": False,
    }
    return {
        "decision": "switch_to_assistant_tool_call",
        "payload": payload,
        "concrete_action": replacement_action,
    }


def travel_attraction_grounding_action_replacement_decision(
    messages: list[dict],
    calls: list[dict[str, str]],
    version: str = ATTRACTION_GROUNDING_ACTION_REPLACEMENT_VERSION,
) -> dict | None:
    if version != ATTRACTION_GROUNDING_ACTION_REPLACEMENT_VERSION or not calls:
        return None
    candidates = [
        decision
        for index, call in enumerate(calls)
        if (
            decision := _travel_attraction_grounding_single_decision(
                messages,
                call,
                calls,
                index,
                version=version,
            )
        )
        is not None
    ]
    return candidates[0] if len(candidates) == 1 else None


def recommended_attraction_names(events: list[dict[str, str]]) -> list[str]:
    names = []
    for event in events:
        if event["name"] != "recommend_attractions" or tool_result_failed(event["result"]):
            continue
        for line in str(event["result"]).splitlines():
            line = line.strip()
            if not line or line.lower().startswith("recommended attractions") or "," not in line:
                continue
            name = line.split(",", 1)[0].strip()
            if name:
                names.append(name)
    return names


def travel_location_canonicalization_action_replacement_decision(
    messages: list[dict],
    calls: list[dict[str, str]],
    version: str = LOCATION_CANONICALIZATION_ACTION_REPLACEMENT_VERSION,
) -> dict | None:
    if version != LOCATION_CANONICALIZATION_ACTION_REPLACEMENT_VERSION or len(calls) != 1:
        return None
    call = calls[0]
    if call["name"] != "search_location":
        return None
    if any(
        is_persist_marker(message)
        and f'"card": "{LOCATION_CANONICALIZATION_CARD}"'
        in str(get_field(message, "content") or "")
        for message in messages
    ):
        return None
    arguments = parse_arguments(call)
    if set(arguments) != {"place_name"}:
        return None
    alias = str(arguments.get("place_name") or "").strip()
    alias_normalized = normalized_english_phrase(alias)
    alias_tokens = set(alias_normalized.split())
    if len(alias_tokens) < 2:
        return None
    user_query = first_user_query(messages)
    normalized_query = f" {normalized_english_phrase(user_query)} "
    if f" {alias_normalized} " not in normalized_query:
        return None
    trip_days = travel_calendar_days(user_query)
    if trip_days is None or trip_days > 6 or has_explicit_travel_budget(user_query):
        return None

    events = completed_tool_events(messages)
    signature = f"search_location|{canonical_json(arguments)}"
    failed_events = [
        event
        for event in events
        if event["signature"] == signature
        and tool_result_failed(event["result"])
        and "exactly consistent with tool results" in event["result"].lower()
    ]
    if not failed_events:
        return None
    candidates = []
    for name in recommended_attraction_names(events):
        normalized_name = normalized_english_phrase(name)
        candidate_tokens = set(normalized_name.split())
        if (
            alias_tokens < candidate_tokens
            and len(candidate_tokens - alias_tokens) == 1
            and len(alias_tokens & candidate_tokens) / len(alias_tokens | candidate_tokens) >= 0.6
        ):
            candidates.append(name)
    unique_candidates = {
        normalized_english_phrase(candidate): candidate for candidate in candidates
    }
    if len(unique_candidates) != 1:
        return None
    canonical_name = next(iter(unique_candidates.values()))
    canonical_arguments = {"place_name": canonical_name}
    canonical_signature = f"search_location|{canonical_json(canonical_arguments)}"
    if any(event["signature"] == canonical_signature for event in events):
        return None

    original_action = {"kind": "tool_call", "tool": call["name"], "arguments": arguments}
    replacement_action = {
        "kind": "tool_call",
        "tool": "search_location",
        "arguments": canonical_arguments,
    }
    payload = {
        "event": "action_replacement",
        "version": version,
        "card": LOCATION_CANONICALIZATION_CARD,
        "decision": "switch_to_assistant_tool_call",
        "reason": "failed_user_alias_has_unique_authoritative_canonical_name",
        "pre_action_state_hash": pre_action_state_hash(messages, call),
        "original_proposal": original_action,
        "replacement_action": replacement_action,
        "failure_signature": hashlib.sha256(failed_events[-1]["result"].encode("utf-8")).hexdigest(),
        "effect_unchanged": True,
        "evidence_delta": "none",
        "productive_retry": False,
        "risk_envelope": {
            "trip_days": trip_days,
            "max_trip_days": 6,
            "explicit_budget": False,
            "user_alias_tokens_preserved": True,
            "added_canonical_tokens": sorted(
                set(normalized_english_phrase(canonical_name).split()) - alias_tokens
            ),
            "unique_authoritative_candidate": True,
        },
        "trigger_count": 1,
        "max_triggers_per_trajectory": 1,
        "write_action_injected": False,
    }
    return {
        "decision": "switch_to_assistant_tool_call",
        "payload": payload,
        "concrete_action": replacement_action,
    }


def _travel_fuzzy_location_canonicalization_single_decision(
    messages: list[dict],
    call: dict[str, str],
    all_calls: list[dict[str, str]],
    target_tool_call_index: int,
    version: str = FUZZY_LOCATION_CANONICALIZATION_ACTION_REPLACEMENT_VERSION,
) -> dict | None:
    if version not in {
        FUZZY_LOCATION_CANONICALIZATION_ACTION_REPLACEMENT_VERSION,
        BUDGET_SAFE_FUZZY_LOCATION_CANONICALIZATION_ACTION_REPLACEMENT_VERSION,
        IMMEDIATE_FUZZY_LOCATION_CANONICALIZATION_ACTION_REPLACEMENT_VERSION,
        EXPLICIT_BUDGET_FUZZY_LOCATION_ACTION_REPLACEMENT_VERSION,
    }:
        return None
    if call["name"] != "search_location":
        return None
    card = (
        EXPLICIT_BUDGET_FUZZY_LOCATION_CARD
        if version == EXPLICIT_BUDGET_FUZZY_LOCATION_ACTION_REPLACEMENT_VERSION
        else IMMEDIATE_FUZZY_LOCATION_CANONICALIZATION_CARD
        if version == IMMEDIATE_FUZZY_LOCATION_CANONICALIZATION_ACTION_REPLACEMENT_VERSION
        else BUDGET_SAFE_FUZZY_LOCATION_CANONICALIZATION_CARD
        if version == BUDGET_SAFE_FUZZY_LOCATION_CANONICALIZATION_ACTION_REPLACEMENT_VERSION
        else FUZZY_LOCATION_CANONICALIZATION_CARD
    )
    if any(
        is_persist_marker(message)
        and f'"card": "{card}"'
        in str(get_field(message, "content") or "")
        for message in messages
    ):
        return None
    arguments = parse_arguments(call)
    if set(arguments) != {"place_name"}:
        return None
    alias = str(arguments.get("place_name") or "").strip()
    alias_normalized = normalized_english_phrase(alias)
    if not alias_normalized:
        return None
    user_query = first_user_query(messages)
    trip_days = travel_calendar_days(user_query)
    explicit_budget = has_explicit_travel_budget(user_query)
    budget_safe_version = version in {
        BUDGET_SAFE_FUZZY_LOCATION_CANONICALIZATION_ACTION_REPLACEMENT_VERSION,
        IMMEDIATE_FUZZY_LOCATION_CANONICALIZATION_ACTION_REPLACEMENT_VERSION,
        EXPLICIT_BUDGET_FUZZY_LOCATION_ACTION_REPLACEMENT_VERSION,
    }
    explicit_budget_only = (
        version == EXPLICIT_BUDGET_FUZZY_LOCATION_ACTION_REPLACEMENT_VERSION
    )
    immediate_version = (
        version == IMMEDIATE_FUZZY_LOCATION_CANONICALIZATION_ACTION_REPLACEMENT_VERSION
    )
    max_trip_days = 7 if budget_safe_version else 6
    if trip_days is None or trip_days > max_trip_days:
        return None
    if explicit_budget and not budget_safe_version:
        return None
    if explicit_budget_only and not explicit_budget:
        return None

    events = completed_tool_events(messages)
    signature = f"search_location|{canonical_json(arguments)}"
    failed_events = [
        event
        for event in events
        if event["signature"] == signature
        and tool_result_failed(event["result"])
        and "exactly consistent with tool results" in event["result"].lower()
    ]
    if not failed_events and not immediate_version:
        return None
    candidate_names = list(
        dict.fromkeys(
            recommended_attraction_names(events)
            + collect_named_candidates(events, "restaurant")
        )
    )
    ranked = []
    alias_tokens = set(alias_normalized.split())
    for candidate in candidate_names:
        candidate_normalized = normalized_english_phrase(candidate)
        if not candidate_normalized or candidate_normalized == alias_normalized:
            continue
        candidate_tokens = set(candidate_normalized.split())
        sequence_score = difflib.SequenceMatcher(
            None, alias_normalized, candidate_normalized
        ).ratio()
        token_score = len(alias_tokens & candidate_tokens) / max(
            1, len(alias_tokens | candidate_tokens)
        )
        ranked.append((max(sequence_score, token_score), candidate))
    ranked.sort(reverse=True)
    if not ranked:
        return None
    top_score, canonical_name = ranked[0]
    second_score = ranked[1][0] if len(ranked) > 1 else 0.0
    if top_score < 0.8 or top_score - second_score < 0.2:
        return None
    canonical_arguments = {"place_name": canonical_name}
    canonical_signature = f"search_location|{canonical_json(canonical_arguments)}"
    if any(event["signature"] == canonical_signature for event in events):
        return None

    original_action = {"kind": "tool_call", "tool": call["name"], "arguments": arguments}
    replacement_action = {
        "kind": "tool_call",
        "tool": "search_location",
        "arguments": canonical_arguments,
    }
    payload = {
        "event": "action_replacement",
        "version": version,
        "card": card,
        "decision": "switch_to_assistant_tool_call",
        "reason": (
            "grounded_location_alias_has_unique_high_confidence_schema_name"
            if immediate_version
            else "failed_location_alias_has_unique_high_confidence_grounded_name"
        ),
        "pre_action_state_hash": pre_action_batch_state_hash(messages, all_calls),
        "original_proposal": original_action,
        "replacement_action": replacement_action,
        "failure_signature": (
            hashlib.sha256(failed_events[-1]["result"].encode("utf-8")).hexdigest()
            if failed_events
            else None
        ),
        "effect_unchanged": True if failed_events else None,
        "evidence_delta": "none" if failed_events else "authoritative_schema_grounding",
        "productive_retry": False,
        "risk_envelope": {
            "trip_days": trip_days,
            "max_trip_days": max_trip_days,
            "explicit_budget": explicit_budget,
            "explicit_budget_allowed": budget_safe_version,
            "explicit_budget_required": explicit_budget_only,
            "similarity_score": top_score,
            "similarity_margin": top_score - second_score,
            "unique_authoritative_candidate": True,
        },
        "trigger_count": 1,
        "max_triggers_per_trajectory": 1,
        "target_tool_call_index": target_tool_call_index,
        "proposal_batch_size": len(all_calls),
        "write_action_injected": False,
    }
    return {
        "decision": "switch_to_assistant_tool_call",
        "payload": payload,
        "concrete_action": replacement_action,
    }


def travel_fuzzy_location_canonicalization_action_replacement_decision(
    messages: list[dict],
    calls: list[dict[str, str]],
    version: str = FUZZY_LOCATION_CANONICALIZATION_ACTION_REPLACEMENT_VERSION,
) -> dict | None:
    if version not in {
        FUZZY_LOCATION_CANONICALIZATION_ACTION_REPLACEMENT_VERSION,
        BUDGET_SAFE_FUZZY_LOCATION_CANONICALIZATION_ACTION_REPLACEMENT_VERSION,
        IMMEDIATE_FUZZY_LOCATION_CANONICALIZATION_ACTION_REPLACEMENT_VERSION,
        EXPLICIT_BUDGET_FUZZY_LOCATION_ACTION_REPLACEMENT_VERSION,
    } or not calls:
        return None
    candidates = [
        decision
        for index, call in enumerate(calls)
        if (
            decision := _travel_fuzzy_location_canonicalization_single_decision(
                messages, call, calls, index, version=version
            )
        )
        is not None
    ]
    return candidates[0] if len(candidates) == 1 else None


def _travel_attraction_tool_family_single_decision(
    messages: list[dict],
    call: dict[str, str],
    all_calls: list[dict[str, str]],
    target_tool_call_index: int,
    version: str = ATTRACTION_TOOL_FAMILY_ACTION_REPLACEMENT_VERSION,
) -> dict | None:
    if version not in {
        ATTRACTION_TOOL_FAMILY_ACTION_REPLACEMENT_VERSION,
        IMMEDIATE_ATTRACTION_TOOL_FAMILY_ACTION_REPLACEMENT_VERSION,
        SPECIAL_SERVICE_ATTRACTION_TOOL_FAMILY_ACTION_REPLACEMENT_VERSION,
        THREE_DAY_BATCH_ATTRACTION_ACTION_REPLACEMENT_VERSION,
    }:
        return None
    if call["name"] != "query_restaurant_details":
        return None
    card = (
        SPECIAL_SERVICE_ATTRACTION_TOOL_FAMILY_CARD
        if version == SPECIAL_SERVICE_ATTRACTION_TOOL_FAMILY_ACTION_REPLACEMENT_VERSION
        else THREE_DAY_BATCH_ATTRACTION_CARD
        if version == THREE_DAY_BATCH_ATTRACTION_ACTION_REPLACEMENT_VERSION
        else IMMEDIATE_ATTRACTION_TOOL_FAMILY_CARD
        if version == IMMEDIATE_ATTRACTION_TOOL_FAMILY_ACTION_REPLACEMENT_VERSION
        else ATTRACTION_TOOL_FAMILY_CARD
    )
    if any(
        is_persist_marker(message)
        and f'"card": "{card}"'
        in str(get_field(message, "content") or "")
        for message in messages
    ):
        return None
    arguments = parse_arguments(call)
    if set(arguments) != {"restaurant_name"}:
        return None
    entity_name = str(arguments.get("restaurant_name") or "").strip()
    normalized_name = normalized_english_phrase(entity_name)
    if not normalized_name:
        return None
    user_query = first_user_query(messages)
    trip_days = travel_calendar_days(user_query)
    if trip_days is None or trip_days > 5 or has_explicit_travel_budget(user_query):
        return None
    special_service_version = (
        version == SPECIAL_SERVICE_ATTRACTION_TOOL_FAMILY_ACTION_REPLACEMENT_VERSION
    )
    three_day_batch_version = (
        version == THREE_DAY_BATCH_ATTRACTION_ACTION_REPLACEMENT_VERSION
    )
    if special_service_version and (
        trip_days < 4
        or not user_requests_restaurant_near_entity(user_query, entity_name)
        or not user_requests_special_restaurant_service_near_entity(
            user_query, entity_name
        )
    ):
        return None
    if three_day_batch_version and (
        trip_days != 3
        or len(all_calls) != 2
        or any(item["name"] != "query_restaurant_details" for item in all_calls)
    ):
        return None

    events = completed_tool_events(messages)
    signature = f"query_restaurant_details|{canonical_json(arguments)}"
    failed_events = [
        event
        for event in events
        if event["signature"] == signature and tool_result_failed(event["result"])
    ]
    if (
        version
        in {
            ATTRACTION_TOOL_FAMILY_ACTION_REPLACEMENT_VERSION,
            SPECIAL_SERVICE_ATTRACTION_TOOL_FAMILY_ACTION_REPLACEMENT_VERSION,
        }
        and not failed_events
    ):
        return None
    attraction_names = {
        normalized_english_phrase(name) for name in recommended_attraction_names(events)
    }
    if normalized_name not in attraction_names:
        return None
    if any(
        event["name"] == "recommend_restaurants"
        and not tool_result_failed(event["result"])
        and normalized_name in normalized_english_phrase(event["result"])
        for event in events
    ):
        return None
    if any(
        event["signature"] == signature and not tool_result_failed(event["result"])
        for event in events
    ):
        return None
    replacement_arguments = {"attraction_name": entity_name}
    replacement_signature = f"query_attraction_details|{canonical_json(replacement_arguments)}"
    if any(event["signature"] == replacement_signature for event in events):
        return None

    original_action = {"kind": "tool_call", "tool": call["name"], "arguments": arguments}
    replacement_action = {
        "kind": "tool_call",
        "tool": "query_attraction_details",
        "arguments": replacement_arguments,
    }
    payload = {
        "event": "action_replacement",
        "version": version,
        "card": card,
        "decision": "switch_to_assistant_tool_call",
        "reason": (
            "grounded_attraction_misrouted_to_restaurant_detail_tool_schema_repair"
            if version
            in {
                IMMEDIATE_ATTRACTION_TOOL_FAMILY_ACTION_REPLACEMENT_VERSION,
                THREE_DAY_BATCH_ATTRACTION_ACTION_REPLACEMENT_VERSION,
            }
            else "special_service_restaurant_anchor_misrouted_to_restaurant_detail_tool"
            if special_service_version
            else "grounded_attraction_misrouted_to_restaurant_detail_tool"
        ),
        "pre_action_state_hash": pre_action_batch_state_hash(messages, all_calls),
        "original_proposal": original_action,
        "replacement_action": replacement_action,
        "failure_signature": (
            hashlib.sha256(failed_events[-1]["result"].encode("utf-8")).hexdigest()
            if failed_events
            else None
        ),
        "effect_unchanged": True if failed_events else None,
        "evidence_delta": "none" if failed_events else "authoritative_schema_grounding",
        "productive_retry": False,
        "risk_envelope": {
            "trip_days": trip_days,
            "max_trip_days": 5,
            "explicit_budget": False,
            "exact_authoritative_attraction_name": True,
            "restaurant_evidence_conflict": False,
            "minimum_trip_days": 4 if special_service_version else None,
            "exact_trip_days": 3 if three_day_batch_version else None,
            "required_proposal_batch_size": 2 if three_day_batch_version else None,
            "explicit_nearby_restaurant_requirement": special_service_version,
            "explicit_special_service_requirement": special_service_version,
        },
        "trigger_count": 1,
        "max_triggers_per_trajectory": 1,
        "target_tool_call_index": target_tool_call_index,
        "proposal_batch_size": len(all_calls),
        "write_action_injected": False,
    }
    return {
        "decision": "switch_to_assistant_tool_call",
        "payload": payload,
        "concrete_action": replacement_action,
    }


def travel_attraction_tool_family_action_replacement_decision(
    messages: list[dict],
    calls: list[dict[str, str]],
    version: str = ATTRACTION_TOOL_FAMILY_ACTION_REPLACEMENT_VERSION,
) -> dict | None:
    if version not in {
        ATTRACTION_TOOL_FAMILY_ACTION_REPLACEMENT_VERSION,
        IMMEDIATE_ATTRACTION_TOOL_FAMILY_ACTION_REPLACEMENT_VERSION,
        SPECIAL_SERVICE_ATTRACTION_TOOL_FAMILY_ACTION_REPLACEMENT_VERSION,
        THREE_DAY_BATCH_ATTRACTION_ACTION_REPLACEMENT_VERSION,
    } or not calls:
        return None
    candidates = [
        decision
        for index, call in enumerate(calls)
        if (
            decision := _travel_attraction_tool_family_single_decision(
                messages,
                call,
                calls,
                index,
                version=version,
            )
        )
        is not None
    ]
    return candidates[0] if len(candidates) == 1 else None


def user_requests_restaurant_near_entity(user_query: str, entity_name: str) -> bool:
    raw_query = str(user_query or "")
    raw_entity = str(entity_name or "").strip()
    if raw_query and raw_entity and raw_entity in raw_query:
        position = raw_query.find(raw_entity)
        context = raw_query[max(0, position - 80) : position + len(raw_entity) + 80]
        if any(marker in context for marker in ("附近", "周边", "旁边", "周围", "那边", "那里")) and any(
            marker in context for marker in ("餐厅", "饭店", "吃饭", "用餐", "餐饮", "美食")
        ):
            return True
    normalized_query = normalized_english_phrase(user_query)
    normalized_name = normalized_english_phrase(entity_name)
    if not normalized_query or not normalized_name:
        return False
    position = normalized_query.find(normalized_name)
    if position < 0:
        return False
    context = normalized_query[
        max(0, position - 180) : position + len(normalized_name) + 180
    ]
    return bool(
        any(word in context.split() for word in ("near", "nearby", "around"))
        and any(
            word in context.split()
            for word in ("restaurant", "restaurants", "meal", "meals", "dining")
        )
    )


def user_requests_special_restaurant_service_near_entity(
    user_query: str, entity_name: str
) -> bool:
    normalized_query = normalized_english_phrase(user_query)
    normalized_name = normalized_english_phrase(entity_name)
    if not normalized_query or not normalized_name:
        return False
    position = normalized_query.find(normalized_name)
    if position < 0:
        return False
    context = normalized_query[
        max(0, position - 220) : position + len(normalized_name) + 220
    ]
    service_phrases = (
        "birthday",
        "set menu",
        "set menus",
        "queue service",
        "queue reservation",
        "virtual queue",
        "private room",
        "private dining",
        "anniversary",
    )
    return any(phrase in context for phrase in service_phrases)


def travel_multicard_action_replacement_decision(
    messages: list[dict], calls: list[dict[str, str]]
) -> dict | None:
    decisions = [
        decision
        for decision in (
            travel_restaurant_alias_action_replacement_decision(
                messages, calls, version=ENGLISH_ACTION_REPLACEMENT_VERSION
            ),
            travel_fuzzy_location_canonicalization_action_replacement_decision(
                messages,
                calls,
                version=EXPLICIT_BUDGET_FUZZY_LOCATION_ACTION_REPLACEMENT_VERSION,
            ),
            travel_attraction_tool_family_action_replacement_decision(
                messages,
                calls,
                version=SPECIAL_SERVICE_ATTRACTION_TOOL_FAMILY_ACTION_REPLACEMENT_VERSION,
            ),
        )
        if decision is not None
    ]
    return decisions[0] if len(decisions) == 1 else None


def travel_four_card_action_replacement_decision(
    messages: list[dict], calls: list[dict[str, str]]
) -> dict | None:
    decisions = [
        decision
        for decision in (
            travel_multicard_action_replacement_decision(messages, calls),
            travel_hotel_action_replacement_decision(
                messages,
                calls,
                version=ATTRACTION_READY_JI_HOTEL_ACTION_REPLACEMENT_VERSION,
            ),
        )
        if decision is not None
    ]
    return decisions[0] if len(decisions) == 1 else None


def travel_strict_positive_two_card_action_replacement_decision(
    messages: list[dict], calls: list[dict[str, str]]
) -> dict | None:
    decisions = [
        decision
        for decision in (
            travel_restaurant_alias_action_replacement_decision(
                messages, calls, version=ENGLISH_ACTION_REPLACEMENT_VERSION
            ),
            travel_attraction_tool_family_action_replacement_decision(
                messages,
                calls,
                version=SPECIAL_SERVICE_ATTRACTION_TOOL_FAMILY_ACTION_REPLACEMENT_VERSION,
            ),
        )
        if decision is not None
    ]
    return decisions[0] if len(decisions) == 1 else None


def travel_en_v4_fold1_action_replacement_decision(
    messages: list[dict],
    calls: list[dict[str, str]],
    version: str,
) -> dict | None:
    enabled_cards = {
        TRAVEL_EN_V4_FOLD1_SCHEMA_ALIAS_ACTION_REPLACEMENT_VERSION: {
            TRAVEL_EN_V4_FOLD1_SCHEMA_ALIAS_CARD,
        },
        TRAVEL_EN_V4_FOLD1_DUPLICATE_RESTAURANT_ACTION_REPLACEMENT_VERSION: {
            TRAVEL_EN_V4_FOLD1_DUPLICATE_RESTAURANT_CARD,
        },
        TRAVEL_EN_V4_FOLD1_DUPLICATE_ROAD_ACTION_REPLACEMENT_VERSION: {
            TRAVEL_EN_V4_FOLD1_DUPLICATE_ROAD_CARD,
        },
        TRAVEL_EN_V4_FOLD1_COMBINED_ACTION_REPLACEMENT_VERSION: {
            TRAVEL_EN_V4_FOLD1_SCHEMA_ALIAS_CARD,
            TRAVEL_EN_V4_FOLD1_DUPLICATE_RESTAURANT_CARD,
            TRAVEL_EN_V4_FOLD1_DUPLICATE_ROAD_CARD,
        },
        TRAVEL_EN_V4_FOLD1_RESTAURANT_NAME_NORMALIZATION_ACTION_REPLACEMENT_VERSION: {
            TRAVEL_EN_V4_FOLD1_RESTAURANT_NAME_NORMALIZATION_CARD,
        },
        TRAVEL_EN_V4_FOLD1_RESTAURANT_ANCHOR_ACTION_REPLACEMENT_VERSION: {
            TRAVEL_EN_V4_FOLD1_RESTAURANT_ANCHOR_CARD,
        },
        TRAVEL_EN_V4_FOLD1_ROUND2_COMBINED_ACTION_REPLACEMENT_VERSION: {
            TRAVEL_EN_V4_FOLD1_SCHEMA_ALIAS_CARD,
            TRAVEL_EN_V4_FOLD1_DUPLICATE_RESTAURANT_CARD,
            TRAVEL_EN_V4_FOLD1_RESTAURANT_NAME_NORMALIZATION_CARD,
            TRAVEL_EN_V4_FOLD1_RESTAURANT_ANCHOR_CARD,
        },
    }[version]
    triggered_cards = {
        card
        for message in messages
        if is_persist_marker(message)
        for card in enabled_cards
        if card in str(get_field(message, "content") or "")
    }
    decisions = []

    if (
        TRAVEL_EN_V4_FOLD1_SCHEMA_ALIAS_CARD in enabled_cards
        and TRAVEL_EN_V4_FOLD1_SCHEMA_ALIAS_CARD not in triggered_cards
    ):
        for call_index, call in enumerate(calls):
            if call["name"] != "recommend_around_restaurants":
                continue
            arguments = parse_arguments(call)
            if set(arguments) != {"latitude", "longitude"}:
                continue
            if not all(str(arguments[key]).strip() for key in ("latitude", "longitude")):
                continue
            original = {
                "kind": "tool_call",
                "tool": "recommend_around_restaurants",
                "arguments": arguments,
            }
            replacement = {
                "kind": "tool_call",
                "tool": "recommend_restaurants",
                "arguments": arguments,
            }
            decisions.append(
                {
                    "decision": "switch_to_assistant_tool_call",
                    "payload": {
                        "event": "action_replacement",
                        "version": version,
                        "card": TRAVEL_EN_V4_FOLD1_SCHEMA_ALIAS_CARD,
                        "decision": "switch_to_assistant_tool_call",
                        "reason": "v4_fold1_train_official_schema_unique_tool_name_repair",
                        "pre_action_state_hash": pre_action_state_hash(messages, call),
                        "original_proposal": original,
                        "replacement_action": replacement,
                        "target_tool_call_index": call_index,
                        "proposal_batch_size": len(calls),
                        "collapse_proposal_batch": False,
                        "task_id_or_gold_used": False,
                        "train_only_design": True,
                        "fold": "travel_en_v4_fold1",
                        "card_role": "paper_primary_and_efficiency",
                        "primary_metric": "commonsense_and_composite",
                        "paper_primary_tradeoff": True,
                        "expected_tradeoffs": ["per_trial_composite_may_vary"],
                        "risk": "low_schema_deterministic",
                        "candidate_kind": "schema_tool_name_repair",
                        "trajectory_trigger_cap": 1,
                    },
                    "concrete_action": replacement,
                }
            )

    for card, tool, version_name, drop_reason, expected_tradeoffs in (
        (
            TRAVEL_EN_V4_FOLD1_DUPLICATE_RESTAURANT_CARD,
            "recommend_restaurants",
            "v4_fold1_train_exact_duplicate_restaurant_batch_pruning",
            "duplicate_recommend_restaurants_within_same_batch",
            ["per_trial_composite_may_vary"],
        ),
        (
            TRAVEL_EN_V4_FOLD1_DUPLICATE_ROAD_CARD,
            "query_road_route_info",
            "v4_fold1_train_exact_duplicate_road_route_batch_pruning",
            "duplicate_query_road_route_info_within_same_batch",
            ["tool_calls_and_functional_loop_may_increase"],
        ),
    ):
        if card not in enabled_cards or card in triggered_cards:
            continue
        seen: dict[str, int] = {}
        for call_index, call in enumerate(calls):
            if call["name"] != tool:
                continue
            arguments = parse_arguments(call)
            signature = f"{tool}|{canonical_json(arguments)}"
            if signature not in seen:
                seen[signature] = call_index
                continue
            original = {"kind": "tool_call", "tool": tool, "arguments": arguments}
            replacement = {
                "kind": "tool_call",
                "tool": "__DROP_TOOL_CALL__",
                "arguments": {"drop_reason": drop_reason},
            }
            decisions.append(
                {
                    "decision": "switch_to_assistant_tool_call",
                    "payload": {
                        "event": "action_replacement",
                        "version": version,
                        "card": card,
                        "decision": "switch_to_assistant_tool_call",
                        "reason": version_name,
                        "pre_action_state_hash": pre_action_state_hash(messages, call),
                        "original_proposal": original,
                        "replacement_action": replacement,
                        "target_tool_call_index": call_index,
                        "proposal_batch_size": len(calls),
                        "collapse_proposal_batch": False,
                        "task_id_or_gold_used": False,
                        "train_only_design": True,
                        "fold": "travel_en_v4_fold1",
                        "card_role": "paper_primary_tradeoff",
                        "primary_metric": "delivery_commonsense_and_composite",
                        "paper_primary_tradeoff": True,
                        "expected_tradeoffs": expected_tradeoffs,
                        "risk": "low_exact_duplicate_batch_pruning",
                        "candidate_kind": "duplicate_batch_call_skip",
                        "duplicate_of_call_index": seen[signature],
                        "trajectory_trigger_cap": 1,
                    },
                    "concrete_action": replacement,
                }
            )

    events = completed_tool_events(messages)
    user_query_normalized = normalized_english_phrase(first_user_query(messages))
    if (
        TRAVEL_EN_V4_FOLD1_RESTAURANT_NAME_NORMALIZATION_CARD in enabled_cards
        and TRAVEL_EN_V4_FOLD1_RESTAURANT_NAME_NORMALIZATION_CARD not in triggered_cards
    ):
        for call_index, call in enumerate(calls):
            if call["name"] != "query_restaurant_details":
                continue
            arguments = parse_arguments(call)
            restaurant_name = str(arguments.get("restaurant_name") or "").strip()
            if set(arguments) != {"restaurant_name"} or not restaurant_name:
                continue
            restaurant_name_normalized = normalized_english_phrase(restaurant_name)
            if (
                restaurant_name_normalized
                and restaurant_name_normalized in user_query_normalized
            ):
                continue
            normalized_names = []
            for pattern in (r"\([^)]*\)", r"\[[^]]*\]"):
                normalized_name = re.sub(pattern, "", restaurant_name).strip(" ,;:-")
                if (
                    normalized_name
                    and normalized_name != restaurant_name
                    and normalized_name not in normalized_names
                ):
                    normalized_names.append(normalized_name)
            if len(normalized_names) != 1:
                continue
            normalized_name = normalized_names[0]
            normalized_name_user_form = normalized_english_phrase(normalized_name)
            if (
                normalized_name_user_form
                and normalized_name_user_form in user_query_normalized
            ):
                continue
            signature = f"query_restaurant_details|{canonical_json(arguments)}"
            if not any(
                event["signature"] == signature and tool_result_failed(event["result"])
                for event in events
            ):
                continue
            replacement_arguments = {"restaurant_name": normalized_name}
            replacement_signature = (
                f"query_restaurant_details|{canonical_json(replacement_arguments)}"
            )
            if any(
                event["signature"] == replacement_signature
                and tool_result_failed(event["result"])
                for event in events
            ):
                continue
            original = {
                "kind": "tool_call",
                "tool": "query_restaurant_details",
                "arguments": arguments,
            }
            replacement = {
                "kind": "tool_call",
                "tool": "query_restaurant_details",
                "arguments": replacement_arguments,
            }
            decisions.append(
                {
                    "decision": "switch_to_assistant_tool_call",
                    "payload": {
                        "event": "action_replacement",
                        "version": version,
                        "card": TRAVEL_EN_V4_FOLD1_RESTAURANT_NAME_NORMALIZATION_CARD,
                        "decision": "switch_to_assistant_tool_call",
                        "reason": "v4_fold1_round2_train_strip_unique_parenthetical_qualifier_after_detail_failure",
                        "pre_action_state_hash": pre_action_state_hash(messages, call),
                        "original_proposal": original,
                        "replacement_action": replacement,
                        "target_tool_call_index": call_index,
                        "proposal_batch_size": len(calls),
                        "collapse_proposal_batch": False,
                        "task_id_or_gold_used": False,
                        "train_only_design": True,
                        "fold": "travel_en_v4_fold1",
                        "card_role": "paper_primary",
                        "primary_metric": "commonsense_and_composite",
                        "paper_primary_tradeoff": True,
                        "expected_tradeoffs": [],
                        "risk": "low_schema_string_normalization",
                        "candidate_kind": "schema_string_normalization",
                        "original_name": restaurant_name,
                        "normalized_name": normalized_name,
                        "trajectory_trigger_cap": 1,
                    },
                    "concrete_action": replacement,
                }
            )

    if (
        TRAVEL_EN_V4_FOLD1_RESTAURANT_ANCHOR_CARD in enabled_cards
        and TRAVEL_EN_V4_FOLD1_RESTAURANT_ANCHOR_CARD not in triggered_cards
    ):
        visible_locations = []
        for event in events:
            if event["name"] != "search_location" or tool_result_failed(event["result"]):
                continue
            try:
                location = json.loads(event["result"])
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(location, dict):
                continue
            latitude = location.get("latitude") or location.get("lat")
            longitude = location.get("longitude") or location.get("lng") or location.get("lon")
            if latitude is not None and longitude is not None:
                visible_locations.append(
                    {"latitude": str(latitude), "longitude": str(longitude)}
                )
        if len(visible_locations) == 1:
            replacement_arguments = visible_locations[0]
            replacement_signature = (
                f"recommend_restaurants|{canonical_json(replacement_arguments)}"
            )
            replacement_observed_failed = any(
                event["signature"] == replacement_signature
                and tool_result_failed(event["result"])
                for event in events
            )
            if not replacement_observed_failed:
                for call_index, call in enumerate(calls):
                    if call["name"] != "query_restaurant_details":
                        continue
                    arguments = parse_arguments(call)
                    restaurant_name = str(arguments.get("restaurant_name") or "").strip()
                    if set(arguments) != {"restaurant_name"} or not restaurant_name:
                        continue
                    restaurant_name_normalized = normalized_english_phrase(restaurant_name)
                    if (
                        restaurant_name_normalized
                        and restaurant_name_normalized in user_query_normalized
                    ):
                        continue
                    signature = f"query_restaurant_details|{canonical_json(arguments)}"
                    if not any(
                        event["signature"] == signature
                        and tool_result_failed(event["result"])
                        for event in events
                    ):
                        continue
                    original = {
                        "kind": "tool_call",
                        "tool": "query_restaurant_details",
                        "arguments": arguments,
                    }
                    replacement = {
                        "kind": "tool_call",
                        "tool": "recommend_restaurants",
                        "arguments": replacement_arguments,
                    }
                    decisions.append(
                        {
                            "decision": "switch_to_assistant_tool_call",
                            "payload": {
                                "event": "action_replacement",
                                "version": version,
                                "card": TRAVEL_EN_V4_FOLD1_RESTAURANT_ANCHOR_CARD,
                                "decision": "switch_to_assistant_tool_call",
                                "reason": "v4_fold1_round2_train_failed_nonexplicit_detail_to_unique_visible_location_anchor",
                                "pre_action_state_hash": pre_action_state_hash(messages, call),
                                "original_proposal": original,
                                "replacement_action": replacement,
                                "target_tool_call_index": call_index,
                                "proposal_batch_size": len(calls),
                                "collapse_proposal_batch": False,
                                "task_id_or_gold_used": False,
                                "train_only_design": True,
                                "fold": "travel_en_v4_fold1",
                                "card_role": "efficiency_compensation",
                                "primary_metric": "functional_loop_and_tool_calls",
                                "paper_primary_tradeoff": False,
                                "expected_tradeoffs": ["official_composite_expected_neutral"],
                                "risk": "medium_prior_location_binding",
                                "candidate_kind": "prior_location_anchor_recommendation",
                                "grounded_coordinates": replacement_arguments,
                                "trajectory_trigger_cap": 1,
                            },
                            "concrete_action": replacement,
                        }
                    )

    if not decisions:
        return None
    priority = {
        TRAVEL_EN_V4_FOLD1_SCHEMA_ALIAS_CARD: 0,
        TRAVEL_EN_V4_FOLD1_DUPLICATE_RESTAURANT_CARD: 1,
        TRAVEL_EN_V4_FOLD1_RESTAURANT_NAME_NORMALIZATION_CARD: 2,
        TRAVEL_EN_V4_FOLD1_RESTAURANT_ANCHOR_CARD: 3,
        TRAVEL_EN_V4_FOLD1_DUPLICATE_ROAD_CARD: 4,
    }
    return min(
        decisions,
        key=lambda decision: (
            priority[decision["payload"]["card"]],
            int(decision["payload"].get("target_tool_call_index", 0)),
        ),
    )


def travel_zh_fold3_round2_action_replacement_decision(
    messages: list[dict],
    calls: list[dict[str, str]],
    version: str = TRAVEL_ZH_FOLD3_ROUND2_TRAIN_ONLY_ACTION_REPLACEMENT_VERSION,
) -> dict | None:
    if version not in {
        TRAVEL_ZH_FOLD3_ROUND2_TRAIN_ONLY_ACTION_REPLACEMENT_VERSION,
        TRAVEL_ZH_FOLD3_FIXED_POSITIVE_ACTION_REPLACEMENT_VERSION,
        TRAVEL_ZH_FOLD3_RESTAURANT_ALIAS_ONLY_ACTION_REPLACEMENT_VERSION,
        TRAVEL_ZH_FOLD3_RESTAURANT_PARENTHETICAL_ONLY_ACTION_REPLACEMENT_VERSION,
        TRAVEL_ZH_FOLD3_FLIGHT_REPEAT_LOOPCUT_ACTION_REPLACEMENT_VERSION,
        TRAVEL_ZH_FOLD3_DUPLICATE_RESTAURANT_ACTION_REPLACEMENT_VERSION,
        TRAVEL_ZH_FOLD3_USER_SELECTED_THREE_CARD_ACTION_REPLACEMENT_VERSION,
    }:
        return None
    events = completed_tool_events(messages)
    decisions: list[dict] = []
    enable_alias = version in {
        TRAVEL_ZH_FOLD3_ROUND2_TRAIN_ONLY_ACTION_REPLACEMENT_VERSION,
        TRAVEL_ZH_FOLD3_FIXED_POSITIVE_ACTION_REPLACEMENT_VERSION,
        TRAVEL_ZH_FOLD3_RESTAURANT_ALIAS_ONLY_ACTION_REPLACEMENT_VERSION,
    }
    broad_alias = version in {
        TRAVEL_ZH_FOLD3_FIXED_POSITIVE_ACTION_REPLACEMENT_VERSION,
        TRAVEL_ZH_FOLD3_RESTAURANT_ALIAS_ONLY_ACTION_REPLACEMENT_VERSION,
    }
    enable_parenthetical = version in {
        TRAVEL_ZH_FOLD3_FIXED_POSITIVE_ACTION_REPLACEMENT_VERSION,
        TRAVEL_ZH_FOLD3_RESTAURANT_PARENTHETICAL_ONLY_ACTION_REPLACEMENT_VERSION,
    }
    enable_road_route_drop = version in {
        TRAVEL_ZH_FOLD3_ROUND2_TRAIN_ONLY_ACTION_REPLACEMENT_VERSION,
        TRAVEL_ZH_FOLD3_FIXED_POSITIVE_ACTION_REPLACEMENT_VERSION,
        TRAVEL_ZH_FOLD3_USER_SELECTED_THREE_CARD_ACTION_REPLACEMENT_VERSION,
    }
    enable_flight_repeat_loopcut = version in {
        TRAVEL_ZH_FOLD3_FLIGHT_REPEAT_LOOPCUT_ACTION_REPLACEMENT_VERSION,
        TRAVEL_ZH_FOLD3_USER_SELECTED_THREE_CARD_ACTION_REPLACEMENT_VERSION,
    }
    enable_duplicate_restaurant_pruning = version in {
        TRAVEL_ZH_FOLD3_DUPLICATE_RESTAURANT_ACTION_REPLACEMENT_VERSION,
        TRAVEL_ZH_FOLD3_USER_SELECTED_THREE_CARD_ACTION_REPLACEMENT_VERSION,
    }
    alias_card = (
        TRAVEL_ZH_FOLD3_RESTAURANT_ALIAS_CARD
        if broad_alias
        else TRAVEL_ZH_FOLD3_SMALL_BATCH_RESTAURANT_ALIAS_CARD
    )
    if enable_alias and not any(
        is_persist_marker(message)
        and alias_card in str(get_field(message, "content") or "")
        for message in messages
    ):
        has_prior_location = any(
            event["name"] == "search_location" and not tool_result_failed(event["result"])
            for event in events
        )
        alias_batch_ok = (
            broad_alias
            or (2 <= len(calls) <= 3 and not has_prior_location)
        )
        if alias_batch_ok:
            for call_index, call in enumerate(calls):
                if call["name"] != "recommend_around_restaurants":
                    continue
                arguments = parse_arguments(call)
                if set(arguments) != {"latitude", "longitude"}:
                    continue
                replacement_signature = (
                    f"recommend_restaurants|{canonical_json(arguments)}"
                )
                if any(
                    event["signature"] == replacement_signature
                    and tool_result_failed(event["result"])
                    for event in events
                ):
                    continue
                original = {
                    "kind": "tool_call",
                    "tool": "recommend_around_restaurants",
                    "arguments": arguments,
                }
                replacement = {
                    "kind": "tool_call",
                    "tool": "recommend_restaurants",
                    "arguments": dict(arguments),
                }
                payload = {
                    "event": "action_replacement",
                    "version": version,
                    "card": alias_card,
                    "decision": "switch_to_assistant_tool_call",
                    "reason": (
                        "fold3_fixed_train_positive_restaurant_tool_alias"
                        if broad_alias
                        else "fold3_train_mined_small_batch_restaurant_tool_alias_without_prior_location"
                    ),
                    "pre_action_state_hash": pre_action_state_hash(messages, call),
                    "original_proposal": original,
                    "replacement_action": replacement,
                    "target_tool_call_index": call_index,
                    "proposal_batch_size": len(calls),
                    "collapse_proposal_batch": False,
                    "task_id_or_gold_used": False,
                    "train_only_design": True,
                    "fold": "travel_zh_fold3",
                    "card_role": "paper_primary",
                    "primary_metric": "report_valid_proxy_and_official_travel_metrics",
                    "paper_primary_tradeoff": True,
                    "expected_tradeoffs": ["calls_may_increase"],
                    "risk": "low_train_gated_schema_tool_alias",
                    "candidate_kind": (
                        "tool_name_alias"
                        if broad_alias
                        else "tool_name_alias_train_gated"
                    ),
                }
                decisions.append(
                    {
                        "decision": "switch_to_assistant_tool_call",
                        "payload": payload,
                        "concrete_action": replacement,
                    }
                )
    if (
        enable_parenthetical
        and not any(
            is_persist_marker(message)
            and TRAVEL_ZH_FOLD3_RESTAURANT_PARENTHETICAL_CARD
            in str(get_field(message, "content") or "")
            for message in messages
        )
    ):
        for call_index, call in enumerate(calls):
            if call["name"] != "query_restaurant_details":
                continue
            arguments = parse_arguments(call)
            restaurant_name = str(arguments.get("restaurant_name") or "").strip()
            if set(arguments) != {"restaurant_name"} or not restaurant_name:
                continue
            normalized_names = []
            for pattern in (r"\([^)]*\)", r"\[[^]]*\]", r"（[^）]*）"):
                normalized = re.sub(pattern, "", restaurant_name).strip(" ,;:-，：；")
                if normalized and normalized != restaurant_name and normalized not in normalized_names:
                    normalized_names.append(normalized)
            if len(normalized_names) != 1:
                continue
            signature = f"query_restaurant_details|{canonical_json(arguments)}"
            if not any(
                event["signature"] == signature and tool_result_failed(event["result"])
                for event in events
            ):
                continue
            replacement_arguments = {"restaurant_name": normalized_names[0]}
            replacement_signature = (
                f"query_restaurant_details|{canonical_json(replacement_arguments)}"
            )
            if any(
                event["signature"] == replacement_signature
                and tool_result_failed(event["result"])
                for event in events
            ):
                continue
            original = {
                "kind": "tool_call",
                "tool": "query_restaurant_details",
                "arguments": arguments,
            }
            replacement = {
                "kind": "tool_call",
                "tool": "query_restaurant_details",
                "arguments": replacement_arguments,
            }
            payload = {
                "event": "action_replacement",
                "version": version,
                "card": TRAVEL_ZH_FOLD3_RESTAURANT_PARENTHETICAL_CARD,
                "decision": "switch_to_assistant_tool_call",
                "reason": "fold3_fixed_train_positive_strip_restaurant_parenthetical_qualifier_after_failure",
                "pre_action_state_hash": pre_action_state_hash(messages, call),
                "original_proposal": original,
                "replacement_action": replacement,
                "target_tool_call_index": call_index,
                "proposal_batch_size": len(calls),
                "collapse_proposal_batch": False,
                "task_id_or_gold_used": False,
                "train_only_design": True,
                "fold": "travel_zh_fold3",
                "card_role": "paper_primary",
                "primary_metric": "report_valid_proxy_and_official_travel_metrics",
                "paper_primary_tradeoff": True,
                "expected_tradeoffs": [],
                "risk": "low_schema_string_normalization",
                "candidate_kind": "schema_string_normalization",
                "original_name": restaurant_name,
                "normalized_name": normalized_names[0],
            }
            decisions.append(
                {
                    "decision": "switch_to_assistant_tool_call",
                    "payload": payload,
                    "concrete_action": replacement,
                }
            )
    if enable_road_route_drop and not any(
        is_persist_marker(message)
        and TRAVEL_ZH_FOLD3_DUPLICATE_ROAD_ROUTE_DROP_CARD
        in str(get_field(message, "content") or "")
        for message in messages
    ):
        seen_route_calls: dict[str, int] = {}
        for call_index, call in enumerate(calls):
            if call["name"] != "query_road_route_info":
                continue
            arguments = parse_arguments(call)
            signature = f"{call['name']}|{canonical_json(arguments)}"
            if signature not in seen_route_calls:
                seen_route_calls[signature] = call_index
                continue
            if len(calls) <= 1:
                continue
            original = {
                "kind": "tool_call",
                "tool": "query_road_route_info",
                "arguments": arguments,
            }
            replacement = {
                "kind": "tool_call",
                "tool": "__DROP_TOOL_CALL__",
                "arguments": {"drop_reason": "duplicate_road_route_within_same_batch"},
            }
            payload = {
                "event": "action_replacement",
                "version": version,
                "card": TRAVEL_ZH_FOLD3_DUPLICATE_ROAD_ROUTE_DROP_CARD,
                "decision": "switch_to_assistant_tool_call",
                "reason": "fold3_train_mined_duplicate_query_road_route_info_within_same_batch",
                "pre_action_state_hash": pre_action_state_hash(messages, call),
                "original_proposal": original,
                "replacement_action": replacement,
                "target_tool_call_index": call_index,
                "proposal_batch_size": len(calls),
                "collapse_proposal_batch": False,
                "task_id_or_gold_used": False,
                "train_only_design": True,
                "fold": "travel_zh_fold3",
                "card_role": "compensation",
                "primary_metric": "report_valid_proxy_and_calls",
                "paper_primary_tradeoff": False,
                "expected_tradeoffs": [],
                "risk": "medium_loop_cut_batch_duplicate",
                "candidate_kind": "duplicate_batch_call_skip",
                "duplicate_of_call_index": seen_route_calls[signature],
            }
            decisions.append(
                {
                    "decision": "switch_to_assistant_tool_call",
                    "payload": payload,
                    "concrete_action": replacement,
                }
            )
    if enable_flight_repeat_loopcut and not any(
        is_persist_marker(message)
        and TRAVEL_ZH_FOLD3_FLIGHT_REPEAT_LOOPCUT_CARD
        in str(get_field(message, "content") or "")
        for message in messages
    ):
        for call_index, call in enumerate(calls):
            if call["name"] != "query_flight_info":
                continue
            arguments = parse_arguments(call)
            signature = f"query_flight_info|{canonical_json(arguments)}"
            matching_positions = [
                index
                for index, event in enumerate(events)
                if event["signature"] == signature
            ]
            failures = [
                events[index]
                for index in matching_positions
                if tool_result_failed(events[index]["result"])
            ]
            if len(failures) < 2:
                continue
            later_events = events[matching_positions[-1] + 1 :]
            if any(not tool_result_failed(event["result"]) for event in later_events):
                continue
            original = {
                "kind": "tool_call",
                "tool": "query_flight_info",
                "arguments": arguments,
            }
            replacement = {
                "kind": "tool_call",
                "tool": "__DROP_TOOL_CALL__",
                "arguments": {
                    "drop_reason": "repeated_failure_without_new_evidence"
                },
            }
            payload = {
                "event": "action_replacement",
                "version": version,
                "card": TRAVEL_ZH_FOLD3_FLIGHT_REPEAT_LOOPCUT_CARD,
                "decision": "switch_to_assistant_tool_call",
                "reason": "fold3_train_mined_flight_exact_repeated_failure_without_new_successful_tool_evidence",
                "pre_action_state_hash": pre_action_state_hash(messages, call),
                "original_proposal": original,
                "replacement_action": replacement,
                "target_tool_call_index": call_index,
                "proposal_batch_size": len(calls),
                "collapse_proposal_batch": False,
                "task_id_or_gold_used": False,
                "train_only_design": True,
                "fold": "travel_zh_fold3",
                "card_role": "paper_primary",
                "primary_metric": "report_valid_proxy_and_official_travel_metrics",
                "paper_primary_tradeoff": True,
                "expected_tradeoffs": [
                    "calls_may_increase",
                    "exact_repeats_may_increase",
                ],
                "risk": "medium_bounded_loop_cut",
                "candidate_kind": "repeated_failure_loop_cut",
                "prior_exact_failures": len(failures),
                "new_successful_tool_evidence_since_last_exact_call": False,
                "trajectory_trigger_cap": 1,
            }
            decisions.append(
                {
                    "decision": "switch_to_assistant_tool_call",
                    "payload": payload,
                    "concrete_action": replacement,
                }
            )
    if enable_duplicate_restaurant_pruning and not any(
        is_persist_marker(message)
        and TRAVEL_ZH_FOLD3_DUPLICATE_RESTAURANT_CARD
        in str(get_field(message, "content") or "")
        for message in messages
    ):
        seen_restaurant_calls: dict[str, int] = {}
        for call_index, call in enumerate(calls):
            if call["name"] != "recommend_restaurants":
                continue
            arguments = parse_arguments(call)
            signature = f"recommend_restaurants|{canonical_json(arguments)}"
            if signature not in seen_restaurant_calls:
                seen_restaurant_calls[signature] = call_index
                continue
            original = {
                "kind": "tool_call",
                "tool": "recommend_restaurants",
                "arguments": arguments,
            }
            replacement = {
                "kind": "tool_call",
                "tool": "__DROP_TOOL_CALL__",
                "arguments": {
                    "drop_reason": "duplicate_restaurant_call_within_same_batch"
                },
            }
            payload = {
                "event": "action_replacement",
                "version": version,
                "card": TRAVEL_ZH_FOLD3_DUPLICATE_RESTAURANT_CARD,
                "decision": "switch_to_assistant_tool_call",
                "reason": "fold3_train_mined_exact_duplicate_recommend_restaurants_within_same_batch",
                "pre_action_state_hash": pre_action_state_hash(messages, call),
                "original_proposal": original,
                "replacement_action": replacement,
                "target_tool_call_index": call_index,
                "proposal_batch_size": len(calls),
                "collapse_proposal_batch": False,
                "task_id_or_gold_used": False,
                "train_only_design": True,
                "fold": "travel_zh_fold3",
                "card_role": "paper_primary",
                "primary_metric": "report_valid_proxy_and_official_travel_metrics",
                "paper_primary_tradeoff": True,
                "expected_tradeoffs": ["calls_may_increase"],
                "risk": "low_exact_duplicate_batch_pruning",
                "candidate_kind": "duplicate_batch_call_skip",
                "duplicate_of_call_index": seen_restaurant_calls[signature],
                "trajectory_trigger_cap": 1,
            }
            decisions.append(
                {
                    "decision": "switch_to_assistant_tool_call",
                    "payload": payload,
                    "concrete_action": replacement,
                }
            )
    if version == TRAVEL_ZH_FOLD3_USER_SELECTED_THREE_CARD_ACTION_REPLACEMENT_VERSION:
        priority = {
            TRAVEL_ZH_FOLD3_DUPLICATE_RESTAURANT_CARD: 0,
            TRAVEL_ZH_FOLD3_DUPLICATE_ROAD_ROUTE_DROP_CARD: 1,
            TRAVEL_ZH_FOLD3_FLIGHT_REPEAT_LOOPCUT_CARD: 2,
        }
        eligible = [
            decision
            for decision in decisions
            if decision.get("payload", {}).get("card") in priority
        ]
        if eligible:
            return min(
                eligible,
                key=lambda decision: (
                    priority[decision["payload"]["card"]],
                    int(decision["payload"].get("target_tool_call_index", 0)),
                ),
            )
        return None
    return decisions[0] if len(decisions) == 1 else None


def travel_zh_fold1_flight_schema_action_replacement_decision(
    messages: list[dict],
    calls: list[dict[str, str]],
    version: str = TRAVEL_ZH_FOLD1_FLIGHT_SCHEMA_ACTION_REPLACEMENT_VERSION,
) -> dict | None:
    version_to_card_and_fold = {
        TRAVEL_ZH_FOLD1_FLIGHT_SCHEMA_ACTION_REPLACEMENT_VERSION: (
            TRAVEL_ZH_FOLD1_FLIGHT_SCHEMA_CARD,
            "travel_zh_fold1",
        ),
        TRAVEL_ZH_FOLD1_TWO_CARD_ACTION_REPLACEMENT_VERSION: (
            TRAVEL_ZH_FOLD1_FLIGHT_SCHEMA_CARD,
            "travel_zh_fold1",
        ),
        TRAVEL_ZH_FOLD2_FLIGHT_SCHEMA_ACTION_REPLACEMENT_VERSION: (
            TRAVEL_ZH_FOLD2_FLIGHT_SCHEMA_CARD,
            "travel_zh_fold2",
        ),
    }
    card_and_fold = version_to_card_and_fold.get(version)
    if card_and_fold is None:
        return None
    card, fold = card_and_fold
    if any(
        is_persist_marker(message)
        and card in str(get_field(message, "content") or "")
        for message in messages
    ):
        return None
    events = completed_tool_events(messages)
    for call_index, call in enumerate(calls):
        if call["name"] != "query_flight_info":
            continue
        arguments = parse_arguments(call)
        if "seat" not in arguments:
            continue
        replacement_arguments = dict(arguments)
        removed_value = replacement_arguments.pop("seat", None)
        if not {"origin", "destination", "depDate"}.issubset(replacement_arguments):
            continue
        replacement_signature = f"query_flight_info|{canonical_json(replacement_arguments)}"
        if any(
            event["signature"] == replacement_signature
            and tool_result_failed(event["result"])
            for event in events
        ):
            continue
        original = {
            "kind": "tool_call",
            "tool": "query_flight_info",
            "arguments": arguments,
        }
        replacement = {
            "kind": "tool_call",
            "tool": "query_flight_info",
            "arguments": replacement_arguments,
        }
        payload = {
            "event": "action_replacement",
            "version": version,
            "card": card,
            "decision": "switch_to_assistant_tool_call",
            "reason": (
                "drop_train_mined_unsupported_flight_seat_field_preserving_route_date"
            ),
            "pre_action_state_hash": pre_action_state_hash(messages, call),
            "original_proposal": original,
            "replacement_action": replacement,
            "target_tool_call_index": call_index,
            "proposal_batch_size": len(calls),
            "collapse_proposal_batch": False,
            "task_id_or_gold_used": False,
            "train_only_design": True,
            "fold": fold,
            "card_role": "paper_primary",
            "primary_metric": "report_valid_proxy_and_official_travel_metrics",
            "paper_primary_tradeoff": True,
            "expected_tradeoffs": ["calls_may_increase"],
            "risk": "low_schema_field_cleanup",
            "candidate_kind": "schema_unsupported_field_drop",
            "removed_field": "seat",
            "removed_value": removed_value,
            "preserved_fields": sorted(replacement_arguments),
        }
        return {
            "decision": "switch_to_assistant_tool_call",
            "payload": payload,
            "concrete_action": replacement,
        }
    return None


def travel_en_fold1_cleanroom_hotel_loopcut_decision(
    messages: list[dict],
    calls: list[dict[str, str]],
    version: str = TRAVEL_EN_FOLD1_CLEANROOM_HOTEL_LOOPCUT_ACTION_REPLACEMENT_VERSION,
) -> dict | None:
    if version not in {
        TRAVEL_EN_FOLD1_CLEANROOM_HOTEL_LOOPCUT_ACTION_REPLACEMENT_VERSION,
        TRAVEL_EN_FOLD1_CLEANROOM_THREE_CARD_ACTION_REPLACEMENT_VERSION,
    }:
        return None
    card = TRAVEL_EN_FOLD1_CLEANROOM_HOTEL_LOOPCUT_CARD
    if any(
        is_persist_marker(message)
        and card in str(get_field(message, "content") or "")
        for message in messages
    ):
        return None
    events = completed_tool_events(messages)
    for call_index, call in enumerate(calls):
        if call["name"] != "query_hotel_info":
            continue
        arguments = parse_arguments(call)
        signature = f"query_hotel_info|{canonical_json(arguments)}"
        matching_positions = [
            index for index, event in enumerate(events) if event["signature"] == signature
        ]
        failures = [
            events[index]
            for index in matching_positions
            if tool_result_failed(events[index]["result"])
        ]
        if len(failures) < 2:
            continue
        later_events = events[matching_positions[-1] + 1 :]
        if any(not tool_result_failed(event["result"]) for event in later_events):
            continue
        original = {
            "kind": "tool_call",
            "tool": "query_hotel_info",
            "arguments": arguments,
        }
        replacement = {
            "kind": "tool_call",
            "tool": "__DROP_TOOL_CALL__",
            "arguments": {"drop_reason": "repeated_failure_without_new_evidence"},
        }
        payload = {
            "event": "action_replacement",
            "version": version,
            "card": card,
            "decision": "switch_to_assistant_tool_call",
            "reason": "drop_third_or_later_exact_failure_without_new_successful_evidence",
            "pre_action_state_hash": pre_action_state_hash(messages, call),
            "original_proposal": original,
            "replacement_action": replacement,
            "target_tool_call_index": call_index,
            "proposal_batch_size": len(calls),
            "collapse_proposal_batch": False,
            "task_id_or_gold_used": False,
            "train_only_design": True,
            "fold": "travel_en_fold1_cleanroom",
            "card_role": "paper_primary",
            "primary_metric": "delivery_and_official_travel_metrics",
            "paper_primary_tradeoff": True,
            "expected_tradeoffs": ["official_composite_may_remain_flat"],
            "risk": "medium_bounded_loop_cut",
            "candidate_kind": "repeated_failure_loop_cut",
            "prior_exact_failures": len(failures),
            "new_successful_tool_evidence_since_last_exact_call": False,
            "trajectory_trigger_cap": 1,
        }
        return {
            "decision": "switch_to_assistant_tool_call",
            "payload": payload,
            "concrete_action": replacement,
        }
    return None


def travel_en_fold2_cleanroom_hotel_loopcut_decision(
    messages: list[dict],
    calls: list[dict[str, str]],
    version: str = TRAVEL_EN_FOLD2_CLEANROOM_HOTEL_LOOPCUT_ACTION_REPLACEMENT_VERSION,
) -> dict | None:
    if version not in {
        TRAVEL_EN_FOLD2_CLEANROOM_HOTEL_LOOPCUT_ACTION_REPLACEMENT_VERSION,
        TRAVEL_EN_FOLD2_CLEANROOM_THREE_CARD_ACTION_REPLACEMENT_VERSION,
    }:
        return None
    card = TRAVEL_EN_FOLD2_CLEANROOM_HOTEL_LOOPCUT_CARD
    if any(
        is_persist_marker(message)
        and card in str(get_field(message, "content") or "")
        for message in messages
    ):
        return None
    events = completed_tool_events(messages)
    for call_index, call in enumerate(calls):
        if call["name"] != "query_hotel_info":
            continue
        arguments = parse_arguments(call)
        signature = f"query_hotel_info|{canonical_json(arguments)}"
        matching_positions = [
            index for index, event in enumerate(events) if event["signature"] == signature
        ]
        failures = [
            events[index]
            for index in matching_positions
            if tool_result_failed(events[index]["result"])
        ]
        if len(failures) < 2:
            continue
        later_events = events[matching_positions[-1] + 1 :]
        if any(not tool_result_failed(event["result"]) for event in later_events):
            continue
        original = {
            "kind": "tool_call",
            "tool": "query_hotel_info",
            "arguments": arguments,
        }
        replacement = {
            "kind": "tool_call",
            "tool": "__DROP_TOOL_CALL__",
            "arguments": {"drop_reason": "repeated_failure_without_new_evidence"},
        }
        payload = {
            "event": "action_replacement",
            "version": version,
            "card": card,
            "decision": "switch_to_assistant_tool_call",
            "reason": "drop_third_or_later_exact_failure_without_new_successful_evidence",
            "pre_action_state_hash": pre_action_state_hash(messages, call),
            "original_proposal": original,
            "replacement_action": replacement,
            "target_tool_call_index": call_index,
            "proposal_batch_size": len(calls),
            "collapse_proposal_batch": False,
            "task_id_or_gold_used": False,
            "train_only_design": True,
            "fold": "travel_en_fold2_cleanroom",
            "card_role": "paper_primary",
            "primary_metric": "delivery_and_official_travel_metrics",
            "paper_primary_tradeoff": True,
            "expected_tradeoffs": ["official_composite_may_decrease"],
            "risk": "medium_bounded_loop_cut",
            "candidate_kind": "repeated_failure_loop_cut",
            "prior_exact_failures": len(failures),
            "new_successful_tool_evidence_since_last_exact_call": False,
            "trajectory_trigger_cap": 1,
        }
        return {
            "decision": "switch_to_assistant_tool_call",
            "payload": payload,
            "concrete_action": replacement,
        }
    return None


def travel_en_fold2_cleanroom_round2_decision(
    messages: list[dict],
    calls: list[dict[str, str]],
    version: str,
) -> dict | None:
    enabled_cards = {
        TRAVEL_EN_FOLD2_CLEANROOM_INFERRED_ALIAS_ACTION_REPLACEMENT_VERSION: {
            TRAVEL_EN_FOLD2_CLEANROOM_INFERRED_ALIAS_CARD,
        },
        TRAVEL_EN_FOLD2_CLEANROOM_SAME_COORDINATE_ACTION_REPLACEMENT_VERSION: {
            TRAVEL_EN_FOLD2_CLEANROOM_SAME_COORDINATE_CARD,
        },
        TRAVEL_EN_FOLD2_CLEANROOM_THREE_CARD_ACTION_REPLACEMENT_VERSION: {
            TRAVEL_EN_FOLD2_CLEANROOM_INFERRED_ALIAS_CARD,
            TRAVEL_EN_FOLD2_CLEANROOM_SAME_COORDINATE_CARD,
            TRAVEL_EN_FOLD2_CLEANROOM_HOTEL_LOOPCUT_CARD,
        },
    }.get(version)
    if enabled_cards is None:
        return None
    triggered_cards = {
        card
        for message in messages
        if is_persist_marker(message)
        for card in enabled_cards
        if card in str(get_field(message, "content") or "")
    }
    decisions = []
    if (
        TRAVEL_EN_FOLD2_CLEANROOM_INFERRED_ALIAS_CARD in enabled_cards
        and TRAVEL_EN_FOLD2_CLEANROOM_INFERRED_ALIAS_CARD not in triggered_cards
    ):
        for call_index, call in enumerate(calls):
            if call["name"] != "recommend_around_restaurants":
                continue
            arguments = parse_arguments(call)
            if set(arguments) != {"latitude", "longitude"}:
                continue
            if not all(str(arguments[key]).strip() for key in ("latitude", "longitude")):
                continue
            original = {
                "kind": "tool_call",
                "tool": "recommend_around_restaurants",
                "arguments": arguments,
            }
            replacement = {
                "kind": "tool_call",
                "tool": "recommend_restaurants",
                "arguments": arguments,
            }
            decisions.append(
                {
                    "decision": "switch_to_assistant_tool_call",
                    "payload": {
                        "event": "action_replacement",
                        "version": version,
                        "card": TRAVEL_EN_FOLD2_CLEANROOM_INFERRED_ALIAS_CARD,
                        "decision": "switch_to_assistant_tool_call",
                        "reason": "fold2_train_inferred_unique_near_schema_tool_alias",
                        "pre_action_state_hash": pre_action_state_hash(messages, call),
                        "original_proposal": original,
                        "replacement_action": replacement,
                        "target_tool_call_index": call_index,
                        "proposal_batch_size": len(calls),
                        "collapse_proposal_batch": False,
                        "task_id_or_gold_used": False,
                        "train_only_design": True,
                        "fold": "travel_en_fold2_cleanroom",
                        "card_role": "paper_primary",
                        "primary_metric": "commonsense_and_composite",
                        "paper_primary_tradeoff": True,
                        "expected_tradeoffs": ["delivery_and_calls_may_vary"],
                        "risk": "low_schema_deterministic",
                        "candidate_kind": "inferred_tool_alias",
                        "trajectory_trigger_cap": 1,
                    },
                    "concrete_action": replacement,
                }
            )
    if (
        TRAVEL_EN_FOLD2_CLEANROOM_SAME_COORDINATE_CARD in enabled_cards
        and TRAVEL_EN_FOLD2_CLEANROOM_SAME_COORDINATE_CARD not in triggered_cards
    ):
        for call_index, call in enumerate(calls):
            if call["name"] != "query_road_route_info":
                continue
            arguments = parse_arguments(call)
            origin = str(arguments.get("origin") or "").strip()
            destination = str(arguments.get("destination") or "").strip()
            if not origin or origin != destination:
                continue
            original = {
                "kind": "tool_call",
                "tool": "query_road_route_info",
                "arguments": arguments,
            }
            replacement = {
                "kind": "tool_call",
                "tool": "__DROP_TOOL_CALL__",
                "arguments": {"drop_reason": "same_origin_destination"},
            }
            decisions.append(
                {
                    "decision": "switch_to_assistant_tool_call",
                    "payload": {
                        "event": "action_replacement",
                        "version": version,
                        "card": TRAVEL_EN_FOLD2_CLEANROOM_SAME_COORDINATE_CARD,
                        "decision": "switch_to_assistant_tool_call",
                        "reason": "fold2_train_same_origin_destination_route_drop",
                        "pre_action_state_hash": pre_action_state_hash(messages, call),
                        "original_proposal": original,
                        "replacement_action": replacement,
                        "target_tool_call_index": call_index,
                        "proposal_batch_size": len(calls),
                        "collapse_proposal_batch": False,
                        "task_id_or_gold_used": False,
                        "train_only_design": True,
                        "fold": "travel_en_fold2_cleanroom",
                        "card_role": "paper_primary_and_compensation",
                        "primary_metric": "delivery_and_report_valid",
                        "paper_primary_tradeoff": True,
                        "expected_tradeoffs": ["official_composite_may_remain_flat"],
                        "risk": "low_noop_route_pruning",
                        "candidate_kind": "same_coordinate_route_drop",
                        "trajectory_trigger_cap": 1,
                    },
                    "concrete_action": replacement,
                }
            )
    if decisions:
        priority = {
            TRAVEL_EN_FOLD2_CLEANROOM_INFERRED_ALIAS_CARD: 0,
            TRAVEL_EN_FOLD2_CLEANROOM_SAME_COORDINATE_CARD: 1,
        }
        return min(
            decisions,
            key=lambda decision: (
                priority[decision["payload"]["card"]],
                int(decision["payload"].get("target_tool_call_index", 0)),
            ),
        )
    if (
        version == TRAVEL_EN_FOLD2_CLEANROOM_THREE_CARD_ACTION_REPLACEMENT_VERSION
        and TRAVEL_EN_FOLD2_CLEANROOM_HOTEL_LOOPCUT_CARD not in triggered_cards
    ):
        return travel_en_fold2_cleanroom_hotel_loopcut_decision(
            messages,
            calls,
            version=version,
        )
    return None


def travel_en_fold2_cleanroom_round3_decision(
    messages: list[dict],
    calls: list[dict[str, str]],
    version: str,
) -> dict | None:
    enabled_cards = {
        TRAVEL_EN_FOLD2_CLEANROOM_ATTRACTION_DETAILS_LOOPCUT_ACTION_REPLACEMENT_VERSION: {
            TRAVEL_EN_FOLD2_CLEANROOM_ATTRACTION_DETAILS_LOOPCUT_CARD,
        },
        TRAVEL_EN_FOLD2_CLEANROOM_DUPLICATE_ATTRACTION_ACTION_REPLACEMENT_VERSION: {
            TRAVEL_EN_FOLD2_CLEANROOM_DUPLICATE_ATTRACTION_CARD,
        },
        TRAVEL_EN_FOLD2_CLEANROOM_ROUND3_TWO_CARD_ACTION_REPLACEMENT_VERSION: {
            TRAVEL_EN_FOLD2_CLEANROOM_ATTRACTION_DETAILS_LOOPCUT_CARD,
            TRAVEL_EN_FOLD2_CLEANROOM_DUPLICATE_ATTRACTION_CARD,
        },
    }.get(version)
    if enabled_cards is None:
        return None
    triggered_cards = {
        card
        for message in messages
        if is_persist_marker(message)
        for card in enabled_cards
        if card in str(get_field(message, "content") or "")
    }
    events = completed_tool_events(messages)
    attraction_loopcut_available = False
    if (
        TRAVEL_EN_FOLD2_CLEANROOM_ATTRACTION_DETAILS_LOOPCUT_CARD in enabled_cards
        and TRAVEL_EN_FOLD2_CLEANROOM_ATTRACTION_DETAILS_LOOPCUT_CARD
        not in triggered_cards
    ):
        for call in calls:
            if call["name"] != "query_attraction_details":
                continue
            signature = (
                f"query_attraction_details|{canonical_json(parse_arguments(call))}"
            )
            if (
                sum(
                    1
                    for event in events
                    if event["signature"] == signature
                    and tool_result_failed(event["result"])
                )
                >= 2
            ):
                attraction_loopcut_available = True
                break
    if (
        TRAVEL_EN_FOLD2_CLEANROOM_DUPLICATE_ATTRACTION_CARD in enabled_cards
        and TRAVEL_EN_FOLD2_CLEANROOM_DUPLICATE_ATTRACTION_CARD not in triggered_cards
        and not attraction_loopcut_available
    ):
        seen_signatures: set[str] = set()
        for call_index, call in enumerate(calls):
            if call["name"] != "recommend_attractions":
                continue
            arguments = parse_arguments(call)
            signature = canonical_json(arguments)
            if signature not in seen_signatures:
                seen_signatures.add(signature)
                continue
            original = {
                "kind": "tool_call",
                "tool": "recommend_attractions",
                "arguments": arguments,
            }
            replacement = {
                "kind": "tool_call",
                "tool": "__DROP_TOOL_CALL__",
                "arguments": {"drop_reason": "exact_duplicate_in_current_batch"},
            }
            return {
                "decision": "switch_to_assistant_tool_call",
                "payload": {
                    "event": "action_replacement",
                    "version": version,
                    "card": TRAVEL_EN_FOLD2_CLEANROOM_DUPLICATE_ATTRACTION_CARD,
                    "decision": "switch_to_assistant_tool_call",
                    "reason": "fold2_train_exact_duplicate_recommend_attractions_pruning",
                    "pre_action_state_hash": pre_action_state_hash(messages, call),
                    "original_proposal": original,
                    "replacement_action": replacement,
                    "target_tool_call_index": call_index,
                    "proposal_batch_size": len(calls),
                    "collapse_proposal_batch": False,
                    "task_id_or_gold_used": False,
                    "train_only_design": True,
                    "fold": "travel_en_fold2_cleanroom",
                    "card_role": "paper_primary",
                    "primary_metric": "delivery_and_report_valid",
                    "paper_primary_tradeoff": True,
                    "expected_tradeoffs": [
                        "calls_may_increase",
                        "functional_loop_metrics_may_increase",
                    ],
                    "risk": "medium_batch_duplicate_pruning",
                    "candidate_kind": "duplicate_batch_call_skip",
                    "trajectory_trigger_cap": 1,
                },
                "concrete_action": replacement,
            }
    if (
        TRAVEL_EN_FOLD2_CLEANROOM_ATTRACTION_DETAILS_LOOPCUT_CARD in enabled_cards
        and TRAVEL_EN_FOLD2_CLEANROOM_ATTRACTION_DETAILS_LOOPCUT_CARD
        not in triggered_cards
    ):
        for call_index, call in enumerate(calls):
            if call["name"] != "query_attraction_details":
                continue
            arguments = parse_arguments(call)
            signature = f"query_attraction_details|{canonical_json(arguments)}"
            matching_positions = [
                index
                for index, event in enumerate(events)
                if event["signature"] == signature
            ]
            failures = [
                events[index]
                for index in matching_positions
                if tool_result_failed(events[index]["result"])
            ]
            if len(failures) < 2:
                continue
            later_events = events[matching_positions[-1] + 1 :]
            if any(not tool_result_failed(event["result"]) for event in later_events):
                continue
            original = {
                "kind": "tool_call",
                "tool": "query_attraction_details",
                "arguments": arguments,
            }
            replacement = {
                "kind": "tool_call",
                "tool": "__DROP_TOOL_CALL__",
                "arguments": {"drop_reason": "repeated_failure_without_new_evidence"},
            }
            return {
                "decision": "switch_to_assistant_tool_call",
                "payload": {
                    "event": "action_replacement",
                    "version": version,
                    "card": TRAVEL_EN_FOLD2_CLEANROOM_ATTRACTION_DETAILS_LOOPCUT_CARD,
                    "decision": "switch_to_assistant_tool_call",
                    "reason": (
                        "drop_third_or_later_exact_attraction_detail_failure_"
                        "without_new_successful_evidence"
                    ),
                    "pre_action_state_hash": pre_action_state_hash(messages, call),
                    "original_proposal": original,
                    "replacement_action": replacement,
                    "target_tool_call_index": call_index,
                    "proposal_batch_size": len(calls),
                    "collapse_proposal_batch": False,
                    "task_id_or_gold_used": False,
                    "train_only_design": True,
                    "fold": "travel_en_fold2_cleanroom",
                    "card_role": "paper_primary_and_compensation",
                    "primary_metric": "commonsense_composite_and_fl",
                    "paper_primary_tradeoff": False,
                    "expected_tradeoffs": ["tail_repetition_may_increase"],
                    "risk": "medium_bounded_loop_cut",
                    "candidate_kind": "repeated_failure_loop_cut",
                    "prior_exact_failures": len(failures),
                    "new_successful_tool_evidence_since_last_exact_call": False,
                    "trajectory_trigger_cap": 1,
                },
                "concrete_action": replacement,
            }
    return None


def travel_en_fold2_cleanroom_user_frozen_two_card_decision(
    messages: list[dict],
    calls: list[dict[str, str]],
    version: str,
) -> dict | None:
    attraction_decision = travel_en_fold2_cleanroom_round3_decision(
        messages,
        calls,
        version=TRAVEL_EN_FOLD2_CLEANROOM_ATTRACTION_DETAILS_LOOPCUT_ACTION_REPLACEMENT_VERSION,
    )
    if attraction_decision is not None:
        attraction_decision["payload"]["version"] = version
        return attraction_decision
    coordinate_decision = travel_en_fold2_cleanroom_round2_decision(
        messages,
        calls,
        version=TRAVEL_EN_FOLD2_CLEANROOM_SAME_COORDINATE_ACTION_REPLACEMENT_VERSION,
    )
    if coordinate_decision is not None:
        coordinate_decision["payload"]["version"] = version
        return coordinate_decision
    return None


def travel_en_fold3_cleanroom_round1_decision(
    messages: list[dict],
    calls: list[dict[str, str]],
    version: str,
) -> dict | None:
    enabled_cards = {
        TRAVEL_EN_FOLD3_CLEANROOM_DUPLICATE_RESTAURANT_ACTION_REPLACEMENT_VERSION: {
            TRAVEL_EN_FOLD3_CLEANROOM_DUPLICATE_RESTAURANT_CARD,
        },
        TRAVEL_EN_FOLD3_CLEANROOM_HOTEL_LOOPCUT_ACTION_REPLACEMENT_VERSION: {
            TRAVEL_EN_FOLD3_CLEANROOM_HOTEL_LOOPCUT_CARD,
        },
        TRAVEL_EN_FOLD3_CLEANROOM_TWO_CARD_ACTION_REPLACEMENT_VERSION: {
            TRAVEL_EN_FOLD3_CLEANROOM_DUPLICATE_RESTAURANT_CARD,
            TRAVEL_EN_FOLD3_CLEANROOM_HOTEL_LOOPCUT_CARD,
        },
    }.get(version)
    if enabled_cards is None:
        return None
    triggered_cards = {
        card
        for message in messages
        if is_persist_marker(message)
        for card in enabled_cards
        if card in str(get_field(message, "content") or "")
    }
    if (
        TRAVEL_EN_FOLD3_CLEANROOM_DUPLICATE_RESTAURANT_CARD in enabled_cards
        and TRAVEL_EN_FOLD3_CLEANROOM_DUPLICATE_RESTAURANT_CARD
        not in triggered_cards
    ):
        seen_signatures: set[str] = set()
        for call_index, call in enumerate(calls):
            if call["name"] != "recommend_restaurants":
                continue
            arguments = parse_arguments(call)
            signature = canonical_json(arguments)
            if signature not in seen_signatures:
                seen_signatures.add(signature)
                continue
            original = {
                "kind": "tool_call",
                "tool": "recommend_restaurants",
                "arguments": arguments,
            }
            replacement = {
                "kind": "tool_call",
                "tool": "__DROP_TOOL_CALL__",
                "arguments": {"drop_reason": "exact_duplicate_in_current_batch"},
            }
            return {
                "decision": "switch_to_assistant_tool_call",
                "payload": {
                    "event": "action_replacement",
                    "version": version,
                    "card": TRAVEL_EN_FOLD3_CLEANROOM_DUPLICATE_RESTAURANT_CARD,
                    "decision": "switch_to_assistant_tool_call",
                    "reason": "fold3_train_exact_duplicate_recommend_restaurants_pruning",
                    "pre_action_state_hash": pre_action_state_hash(messages, call),
                    "original_proposal": original,
                    "replacement_action": replacement,
                    "target_tool_call_index": call_index,
                    "proposal_batch_size": len(calls),
                    "collapse_proposal_batch": False,
                    "task_id_or_gold_used": False,
                    "train_only_design": True,
                    "fold": "travel_en_fold3_cleanroom",
                    "card_role": "paper_primary_and_compensation",
                    "primary_metric": "commonsense_composite_and_fl",
                    "paper_primary_tradeoff": False,
                    "expected_tradeoffs": ["calls_may_vary"],
                    "risk": "low_exact_batch_duplicate_pruning",
                    "candidate_kind": "duplicate_batch_call_skip",
                    "trajectory_trigger_cap": 1,
                },
                "concrete_action": replacement,
            }
    if (
        TRAVEL_EN_FOLD3_CLEANROOM_HOTEL_LOOPCUT_CARD in enabled_cards
        and TRAVEL_EN_FOLD3_CLEANROOM_HOTEL_LOOPCUT_CARD not in triggered_cards
    ):
        events = completed_tool_events(messages)
        for call_index, call in enumerate(calls):
            if call["name"] != "query_hotel_info":
                continue
            arguments = parse_arguments(call)
            signature = f"query_hotel_info|{canonical_json(arguments)}"
            matching_positions = [
                index
                for index, event in enumerate(events)
                if event["signature"] == signature
            ]
            failures = [
                events[index]
                for index in matching_positions
                if tool_result_failed(events[index]["result"])
            ]
            if len(failures) < 2:
                continue
            later_events = events[matching_positions[-1] + 1 :]
            if any(not tool_result_failed(event["result"]) for event in later_events):
                continue
            original = {
                "kind": "tool_call",
                "tool": "query_hotel_info",
                "arguments": arguments,
            }
            replacement = {
                "kind": "tool_call",
                "tool": "__DROP_TOOL_CALL__",
                "arguments": {"drop_reason": "repeated_failure_without_new_evidence"},
            }
            return {
                "decision": "switch_to_assistant_tool_call",
                "payload": {
                    "event": "action_replacement",
                    "version": version,
                    "card": TRAVEL_EN_FOLD3_CLEANROOM_HOTEL_LOOPCUT_CARD,
                    "decision": "switch_to_assistant_tool_call",
                    "reason": (
                        "drop_third_or_later_exact_hotel_failure_without_"
                        "new_successful_evidence"
                    ),
                    "pre_action_state_hash": pre_action_state_hash(messages, call),
                    "original_proposal": original,
                    "replacement_action": replacement,
                    "target_tool_call_index": call_index,
                    "proposal_batch_size": len(calls),
                    "collapse_proposal_batch": False,
                    "task_id_or_gold_used": False,
                    "train_only_design": True,
                    "fold": "travel_en_fold3_cleanroom",
                    "card_role": "paper_primary",
                    "primary_metric": "delivery",
                    "paper_primary_tradeoff": True,
                    "expected_tradeoffs": ["composite_may_decrease"],
                    "risk": "medium_bounded_loop_cut",
                    "candidate_kind": "repeated_failure_loop_cut",
                    "prior_exact_failures": len(failures),
                    "new_successful_tool_evidence_since_last_exact_call": False,
                    "trajectory_trigger_cap": 1,
                },
                "concrete_action": replacement,
            }
    return None


def travel_en_fold3_cleanroom_round2_decision(
    messages: list[dict],
    calls: list[dict[str, str]],
    version: str,
) -> dict | None:
    if version not in {
        TRAVEL_EN_FOLD3_CLEANROOM_ROUND2_FOUR_CARD_ACTION_REPLACEMENT_VERSION,
        TRAVEL_EN_FOLD3_CLEANROOM_ROUND3_STRICT_FOUR_CARD_ACTION_REPLACEMENT_VERSION,
    }:
        return None
    cards = {
        TRAVEL_EN_FOLD3_CLEANROOM_INFERRED_ALIAS_CARD,
        TRAVEL_EN_FOLD3_CLEANROOM_SAME_COORDINATE_CARD,
        TRAVEL_EN_FOLD3_CLEANROOM_DUPLICATE_ROAD_CARD,
        TRAVEL_EN_FOLD3_CLEANROOM_DUPLICATE_RESTAURANT_CARD,
    }
    triggered_cards = {
        card
        for message in messages
        if is_persist_marker(message)
        for card in cards
        if card in str(get_field(message, "content") or "")
    }
    completed_events = completed_tool_events(messages)
    decisions = []
    seen_signatures: dict[tuple[str, str], int] = {}
    for call_index, call in enumerate(calls):
        tool_name = call["name"]
        arguments = parse_arguments(call)
        signature = (tool_name, canonical_json(arguments))

        if (
            tool_name == "recommend_around_restaurants"
            and TRAVEL_EN_FOLD3_CLEANROOM_INFERRED_ALIAS_CARD
            not in triggered_cards
            and set(arguments) == {"latitude", "longitude"}
            and all(
                str(arguments[key]).strip() for key in ("latitude", "longitude")
            )
        ):
            original_signature = (
                f"recommend_around_restaurants|{canonical_json(arguments)}"
            )
            replacement_signature = f"recommend_restaurants|{canonical_json(arguments)}"
            if (
                version
                == TRAVEL_EN_FOLD3_CLEANROOM_ROUND3_STRICT_FOUR_CARD_ACTION_REPLACEMENT_VERSION
                and (
                    not any(
                        event["signature"] == original_signature
                        and tool_result_failed(event["result"])
                        for event in completed_events
                    )
                    or any(
                        event["signature"] == replacement_signature
                        and tool_result_failed(event["result"])
                        for event in completed_events
                    )
                )
            ):
                continue
            replacement = {
                "kind": "tool_call",
                "tool": "recommend_restaurants",
                "arguments": arguments,
            }
            decisions.append(
                {
                    "priority": 0,
                    "decision": "switch_to_assistant_tool_call",
                    "payload": {
                        "event": "action_replacement",
                        "version": version,
                        "card": TRAVEL_EN_FOLD3_CLEANROOM_INFERRED_ALIAS_CARD,
                        "decision": "switch_to_assistant_tool_call",
                        "reason": (
                            "fold3_train_failed_unsupported_tool_then_inferred_"
                            "unique_near_schema_alias"
                            if version
                            == TRAVEL_EN_FOLD3_CLEANROOM_ROUND3_STRICT_FOUR_CARD_ACTION_REPLACEMENT_VERSION
                            else "fold3_train_inferred_unique_near_schema_tool_alias"
                        ),
                        "pre_action_state_hash": pre_action_state_hash(messages, call),
                        "original_proposal": {
                            "kind": "tool_call",
                            "tool": tool_name,
                            "arguments": arguments,
                        },
                        "replacement_action": replacement,
                        "target_tool_call_index": call_index,
                        "proposal_batch_size": len(calls),
                        "collapse_proposal_batch": False,
                        "task_id_or_gold_used": False,
                        "train_only_design": True,
                        "fold": "travel_en_fold3_cleanroom",
                        "card_role": "paper_primary_and_compensation",
                        "primary_metric": "commonsense_composite_and_fl",
                        "paper_primary_tradeoff": True,
                        "expected_tradeoffs": ["delivery_and_calls_may_vary"],
                        "risk": (
                            "low_schema_deterministic_after_observed_failure"
                            if version
                            == TRAVEL_EN_FOLD3_CLEANROOM_ROUND3_STRICT_FOUR_CARD_ACTION_REPLACEMENT_VERSION
                            else "low_schema_deterministic"
                        ),
                        "candidate_kind": "inferred_tool_alias",
                        "trajectory_trigger_cap": 1,
                    },
                    "concrete_action": replacement,
                }
            )

        duplicate_card = {
            "recommend_restaurants": (
                TRAVEL_EN_FOLD3_CLEANROOM_DUPLICATE_RESTAURANT_CARD,
                1,
            ),
            "query_road_route_info": (
                TRAVEL_EN_FOLD3_CLEANROOM_DUPLICATE_ROAD_CARD,
                2,
            ),
        }.get(tool_name)
        if duplicate_card is not None:
            card, priority = duplicate_card
            if signature in seen_signatures and card not in triggered_cards:
                replacement = {
                    "kind": "tool_call",
                    "tool": "__DROP_TOOL_CALL__",
                    "arguments": {"drop_reason": "exact_duplicate_in_current_batch"},
                }
                decisions.append(
                    {
                        "priority": priority,
                        "decision": "switch_to_assistant_tool_call",
                        "payload": {
                            "event": "action_replacement",
                            "version": version,
                            "card": card,
                            "decision": "switch_to_assistant_tool_call",
                            "reason": "fold3_train_exact_duplicate_batch_pruning",
                            "pre_action_state_hash": pre_action_state_hash(
                                messages, call
                            ),
                            "original_proposal": {
                                "kind": "tool_call",
                                "tool": tool_name,
                                "arguments": arguments,
                            },
                            "replacement_action": replacement,
                            "target_tool_call_index": call_index,
                            "proposal_batch_size": len(calls),
                            "collapse_proposal_batch": False,
                            "task_id_or_gold_used": False,
                            "train_only_design": True,
                            "fold": "travel_en_fold3_cleanroom",
                            "card_role": "paper_primary_and_compensation",
                            "primary_metric": "commonsense_composite_and_fl",
                            "paper_primary_tradeoff": True,
                            "expected_tradeoffs": ["calls_may_vary"],
                            "risk": "low_exact_batch_duplicate_pruning",
                            "candidate_kind": "duplicate_batch_call_skip",
                            "duplicate_of_call_index": seen_signatures[signature],
                            "trajectory_trigger_cap": 1,
                        },
                        "concrete_action": replacement,
                    }
                )
            seen_signatures.setdefault(signature, call_index)

        if (
            tool_name == "query_road_route_info"
            and TRAVEL_EN_FOLD3_CLEANROOM_SAME_COORDINATE_CARD
            not in triggered_cards
        ):
            origin = str(arguments.get("origin") or "").strip()
            destination = str(arguments.get("destination") or "").strip()
            if origin and origin == destination:
                replacement = {
                    "kind": "tool_call",
                    "tool": "__DROP_TOOL_CALL__",
                    "arguments": {"drop_reason": "same_origin_destination"},
                }
                decisions.append(
                    {
                        "priority": 3,
                        "decision": "switch_to_assistant_tool_call",
                        "payload": {
                            "event": "action_replacement",
                            "version": version,
                            "card": TRAVEL_EN_FOLD3_CLEANROOM_SAME_COORDINATE_CARD,
                            "decision": "switch_to_assistant_tool_call",
                            "reason": "fold3_train_same_origin_destination_route_drop",
                            "pre_action_state_hash": pre_action_state_hash(
                                messages, call
                            ),
                            "original_proposal": {
                                "kind": "tool_call",
                                "tool": tool_name,
                                "arguments": arguments,
                            },
                            "replacement_action": replacement,
                            "target_tool_call_index": call_index,
                            "proposal_batch_size": len(calls),
                            "collapse_proposal_batch": False,
                            "task_id_or_gold_used": False,
                            "train_only_design": True,
                            "fold": "travel_en_fold3_cleanroom",
                            "card_role": "paper_primary_and_compensation",
                            "primary_metric": "commonsense_composite_and_fl",
                            "paper_primary_tradeoff": True,
                            "expected_tradeoffs": ["calls_may_increase"],
                            "risk": "low_noop_route_pruning",
                            "candidate_kind": "same_coordinate_route_drop",
                            "trajectory_trigger_cap": 1,
                        },
                        "concrete_action": replacement,
                    }
                )
    if not decisions:
        return None
    selected = min(
        decisions,
        key=lambda decision: (
            int(decision["priority"]),
            int(decision["payload"]["target_tool_call_index"]),
        ),
    )
    selected.pop("priority")
    return selected


def travel_en_fold3_v8_k6_round1_decision(
    messages: list[dict],
    calls: list[dict[str, str]],
    version: str,
) -> dict | None:
    enabled_cards = {
        TRAVEL_EN_FOLD3_V8_K6_DUPLICATE_AROUND_ACTION_REPLACEMENT_VERSION: {
            TRAVEL_EN_FOLD3_V8_K6_DUPLICATE_AROUND_CARD,
        },
        TRAVEL_EN_FOLD3_V8_K6_HOTEL_LOOPCUT_ACTION_REPLACEMENT_VERSION: {
            TRAVEL_EN_FOLD3_V8_K6_HOTEL_LOOPCUT_CARD,
        },
        TRAVEL_EN_FOLD3_V8_K6_TWO_CARD_ACTION_REPLACEMENT_VERSION: {
            TRAVEL_EN_FOLD3_V8_K6_DUPLICATE_AROUND_CARD,
            TRAVEL_EN_FOLD3_V8_K6_HOTEL_LOOPCUT_CARD,
        },
        TRAVEL_EN_FOLD3_V8_K6_USER_FROZEN_FOUR_CARD_ACTION_REPLACEMENT_VERSION: {
            TRAVEL_EN_FOLD3_V8_K6_DUPLICATE_AROUND_CARD,
            TRAVEL_EN_FOLD3_V8_K6_HOTEL_LOOPCUT_CARD,
        },
    }.get(version)
    if enabled_cards is None:
        return None
    triggered_cards = {
        card
        for message in messages
        if is_persist_marker(message)
        for card in enabled_cards
        if card in str(get_field(message, "content") or "")
    }
    if (
        TRAVEL_EN_FOLD3_V8_K6_DUPLICATE_AROUND_CARD in enabled_cards
        and TRAVEL_EN_FOLD3_V8_K6_DUPLICATE_AROUND_CARD not in triggered_cards
    ):
        seen_signatures: dict[str, int] = {}
        for call_index, call in enumerate(calls):
            if call["name"] != "recommend_around_restaurants":
                continue
            arguments = parse_arguments(call)
            signature = canonical_json(arguments)
            if signature not in seen_signatures:
                seen_signatures[signature] = call_index
                continue
            replacement = {
                "kind": "tool_call",
                "tool": "__DROP_TOOL_CALL__",
                "arguments": {"drop_reason": "exact_duplicate_in_current_batch"},
            }
            return {
                "decision": "switch_to_assistant_tool_call",
                "payload": {
                    "event": "action_replacement",
                    "version": version,
                    "card": TRAVEL_EN_FOLD3_V8_K6_DUPLICATE_AROUND_CARD,
                    "decision": "switch_to_assistant_tool_call",
                    "reason": (
                        "fold3_v8_k6_train_exact_duplicate_recommend_around_"
                        "restaurants_pruning"
                    ),
                    "pre_action_state_hash": pre_action_state_hash(messages, call),
                    "original_proposal": {
                        "kind": "tool_call",
                        "tool": "recommend_around_restaurants",
                        "arguments": arguments,
                    },
                    "replacement_action": replacement,
                    "target_tool_call_index": call_index,
                    "proposal_batch_size": len(calls),
                    "collapse_proposal_batch": False,
                    "task_id_or_gold_used": False,
                    "train_only_design": True,
                    "fold": "travel_en_fold3_v8_k6_cleanroom",
                    "card_role": "paper_primary_and_compensation",
                    "primary_metric": "commonsense_composite_and_fl",
                    "paper_primary_tradeoff": False,
                    "expected_tradeoffs": ["report_delivery_may_vary"],
                    "risk": "low_exact_batch_duplicate_pruning",
                    "candidate_kind": "duplicate_batch_call_skip",
                    "duplicate_of_call_index": seen_signatures[signature],
                    "trajectory_trigger_cap": 1,
                },
                "concrete_action": replacement,
            }
    if (
        TRAVEL_EN_FOLD3_V8_K6_HOTEL_LOOPCUT_CARD in enabled_cards
        and TRAVEL_EN_FOLD3_V8_K6_HOTEL_LOOPCUT_CARD not in triggered_cards
    ):
        events = completed_tool_events(messages)
        for call_index, call in enumerate(calls):
            if call["name"] != "query_hotel_info":
                continue
            arguments = parse_arguments(call)
            signature = f"query_hotel_info|{canonical_json(arguments)}"
            matching_positions = [
                index
                for index, event in enumerate(events)
                if event["signature"] == signature
            ]
            failures = [
                events[index]
                for index in matching_positions
                if tool_result_failed(events[index]["result"])
            ]
            if len(failures) < 2:
                continue
            later_events = events[matching_positions[-1] + 1 :]
            if any(not tool_result_failed(event["result"]) for event in later_events):
                continue
            replacement = {
                "kind": "tool_call",
                "tool": "__DROP_TOOL_CALL__",
                "arguments": {"drop_reason": "repeated_failure_without_new_evidence"},
            }
            return {
                "decision": "switch_to_assistant_tool_call",
                "payload": {
                    "event": "action_replacement",
                    "version": version,
                    "card": TRAVEL_EN_FOLD3_V8_K6_HOTEL_LOOPCUT_CARD,
                    "decision": "switch_to_assistant_tool_call",
                    "reason": (
                        "fold3_v8_k6_drop_third_or_later_exact_hotel_failure_"
                        "without_new_successful_evidence"
                    ),
                    "pre_action_state_hash": pre_action_state_hash(messages, call),
                    "original_proposal": {
                        "kind": "tool_call",
                        "tool": "query_hotel_info",
                        "arguments": arguments,
                    },
                    "replacement_action": replacement,
                    "target_tool_call_index": call_index,
                    "proposal_batch_size": len(calls),
                    "collapse_proposal_batch": False,
                    "task_id_or_gold_used": False,
                    "train_only_design": True,
                    "fold": "travel_en_fold3_v8_k6_cleanroom",
                    "card_role": "paper_primary",
                    "primary_metric": "delivery",
                    "paper_primary_tradeoff": True,
                    "expected_tradeoffs": ["official_composite_may_decrease"],
                    "risk": "medium_bounded_loop_cut",
                    "candidate_kind": "repeated_failure_loop_cut",
                    "prior_exact_failures": len(failures),
                    "new_successful_tool_evidence_since_last_exact_call": False,
                    "trajectory_trigger_cap": 1,
                },
                "concrete_action": replacement,
            }
    return None


def travel_en_fold3_v8_k6_round2_decision(
    messages: list[dict],
    calls: list[dict[str, str]],
    version: str,
) -> dict | None:
    enabled_cards = {
        TRAVEL_EN_FOLD3_V8_K6_SAME_COORDINATE_ACTION_REPLACEMENT_VERSION: {
            TRAVEL_EN_FOLD3_V8_K6_SAME_COORDINATE_CARD,
        },
        TRAVEL_EN_FOLD3_V8_K6_SEARCH_LOOPCUT_ACTION_REPLACEMENT_VERSION: {
            TRAVEL_EN_FOLD3_V8_K6_SEARCH_LOOPCUT_CARD,
        },
        TRAVEL_EN_FOLD3_V8_K6_PARENTHETICAL_ACTION_REPLACEMENT_VERSION: {
            TRAVEL_EN_FOLD3_V8_K6_PARENTHETICAL_CARD,
        },
        TRAVEL_EN_FOLD3_V8_K6_ROUND2_THREE_CARD_ACTION_REPLACEMENT_VERSION: {
            TRAVEL_EN_FOLD3_V8_K6_SAME_COORDINATE_CARD,
            TRAVEL_EN_FOLD3_V8_K6_SEARCH_LOOPCUT_CARD,
            TRAVEL_EN_FOLD3_V8_K6_PARENTHETICAL_CARD,
        },
        TRAVEL_EN_FOLD3_V8_K6_USER_FROZEN_FOUR_CARD_ACTION_REPLACEMENT_VERSION: {
            TRAVEL_EN_FOLD3_V8_K6_SAME_COORDINATE_CARD,
        },
    }.get(version)
    if enabled_cards is None:
        return None
    triggered_cards = {
        card
        for message in messages
        if is_persist_marker(message)
        for card in enabled_cards
        if card in str(get_field(message, "content") or "")
    }
    events = completed_tool_events(messages)
    decisions = []

    if (
        TRAVEL_EN_FOLD3_V8_K6_SAME_COORDINATE_CARD in enabled_cards
        and TRAVEL_EN_FOLD3_V8_K6_SAME_COORDINATE_CARD not in triggered_cards
    ):
        for call_index, call in enumerate(calls):
            if call["name"] != "query_road_route_info":
                continue
            arguments = parse_arguments(call)
            origin = str(arguments.get("origin") or "").strip()
            destination = str(arguments.get("destination") or "").strip()
            if not origin or origin != destination:
                continue
            replacement = {
                "kind": "tool_call",
                "tool": "__DROP_TOOL_CALL__",
                "arguments": {"drop_reason": "same_origin_destination"},
            }
            decisions.append(
                {
                    "priority": 0,
                    "decision": "switch_to_assistant_tool_call",
                    "payload": {
                        "event": "action_replacement",
                        "version": version,
                        "card": TRAVEL_EN_FOLD3_V8_K6_SAME_COORDINATE_CARD,
                        "decision": "switch_to_assistant_tool_call",
                        "reason": "fold3_v8_k6_train_same_origin_destination_route_drop",
                        "pre_action_state_hash": pre_action_state_hash(messages, call),
                        "original_proposal": {
                            "kind": "tool_call",
                            "tool": "query_road_route_info",
                            "arguments": arguments,
                        },
                        "replacement_action": replacement,
                        "target_tool_call_index": call_index,
                        "proposal_batch_size": len(calls),
                        "collapse_proposal_batch": False,
                        "task_id_or_gold_used": False,
                        "train_only_design": True,
                        "fold": "travel_en_fold3_v8_k6_cleanroom",
                        "card_role": "paper_primary_and_compensation",
                        "primary_metric": "commonsense_composite_and_fl",
                        "paper_primary_tradeoff": False,
                        "expected_tradeoffs": ["calls_may_vary"],
                        "risk": "low_noop_route_pruning",
                        "candidate_kind": "same_coordinate_route_drop",
                        "trajectory_trigger_cap": 1,
                    },
                    "concrete_action": replacement,
                }
            )

    if (
        TRAVEL_EN_FOLD3_V8_K6_SEARCH_LOOPCUT_CARD in enabled_cards
        and TRAVEL_EN_FOLD3_V8_K6_SEARCH_LOOPCUT_CARD not in triggered_cards
    ):
        for call_index, call in enumerate(calls):
            if call["name"] != "search_location":
                continue
            arguments = parse_arguments(call)
            signature = f"search_location|{canonical_json(arguments)}"
            matching_positions = [
                index
                for index, event in enumerate(events)
                if event["signature"] == signature
            ]
            failures = [
                events[index]
                for index in matching_positions
                if tool_result_failed(events[index]["result"])
            ]
            if len(failures) < 2:
                continue
            later_events = events[matching_positions[-1] + 1 :]
            if any(not tool_result_failed(event["result"]) for event in later_events):
                continue
            replacement = {
                "kind": "tool_call",
                "tool": "__DROP_TOOL_CALL__",
                "arguments": {"drop_reason": "repeated_failure_without_new_evidence"},
            }
            decisions.append(
                {
                    "priority": 1,
                    "decision": "switch_to_assistant_tool_call",
                    "payload": {
                        "event": "action_replacement",
                        "version": version,
                        "card": TRAVEL_EN_FOLD3_V8_K6_SEARCH_LOOPCUT_CARD,
                        "decision": "switch_to_assistant_tool_call",
                        "reason": (
                            "fold3_v8_k6_drop_third_or_later_exact_search_"
                            "failure_without_new_successful_evidence"
                        ),
                        "pre_action_state_hash": pre_action_state_hash(messages, call),
                        "original_proposal": {
                            "kind": "tool_call",
                            "tool": "search_location",
                            "arguments": arguments,
                        },
                        "replacement_action": replacement,
                        "target_tool_call_index": call_index,
                        "proposal_batch_size": len(calls),
                        "collapse_proposal_batch": False,
                        "task_id_or_gold_used": False,
                        "train_only_design": True,
                        "fold": "travel_en_fold3_v8_k6_cleanroom",
                        "card_role": "paper_primary_and_compensation",
                        "primary_metric": "commonsense_composite_and_fl",
                        "paper_primary_tradeoff": False,
                        "expected_tradeoffs": ["broad_fl_may_vary"],
                        "risk": "medium_bounded_loop_cut",
                        "candidate_kind": "repeated_failure_loop_cut",
                        "prior_exact_failures": len(failures),
                        "new_successful_tool_evidence_since_last_exact_call": False,
                        "trajectory_trigger_cap": 1,
                    },
                    "concrete_action": replacement,
                }
            )

    if (
        TRAVEL_EN_FOLD3_V8_K6_PARENTHETICAL_CARD in enabled_cards
        and TRAVEL_EN_FOLD3_V8_K6_PARENTHETICAL_CARD not in triggered_cards
    ):
        user_query_normalized = normalized_english_phrase(first_user_query(messages))
        for call_index, call in enumerate(calls):
            if call["name"] != "query_restaurant_details":
                continue
            arguments = parse_arguments(call)
            restaurant_name = str(arguments.get("restaurant_name") or "").strip()
            if set(arguments) != {"restaurant_name"} or not restaurant_name:
                continue
            restaurant_name_normalized = normalized_english_phrase(restaurant_name)
            if (
                restaurant_name_normalized
                and restaurant_name_normalized in user_query_normalized
            ):
                continue
            normalized_names = []
            for pattern in (r"\([^)]*\)", r"\[[^]]*\]"):
                normalized_name = re.sub(pattern, "", restaurant_name).strip(" ,;:-")
                if (
                    normalized_name
                    and normalized_name != restaurant_name
                    and normalized_name not in normalized_names
                ):
                    normalized_names.append(normalized_name)
            if len(normalized_names) != 1:
                continue
            normalized_name = normalized_names[0]
            normalized_user_form = normalized_english_phrase(normalized_name)
            if normalized_user_form and normalized_user_form in user_query_normalized:
                continue
            original_signature = (
                f"query_restaurant_details|{canonical_json(arguments)}"
            )
            if not any(
                event["signature"] == original_signature
                and tool_result_failed(event["result"])
                for event in events
            ):
                continue
            replacement_arguments = {"restaurant_name": normalized_name}
            replacement_signature = (
                f"query_restaurant_details|{canonical_json(replacement_arguments)}"
            )
            if any(
                event["signature"] == replacement_signature
                and tool_result_failed(event["result"])
                for event in events
            ):
                continue
            replacement = {
                "kind": "tool_call",
                "tool": "query_restaurant_details",
                "arguments": replacement_arguments,
            }
            decisions.append(
                {
                    "priority": 2,
                    "decision": "switch_to_assistant_tool_call",
                    "payload": {
                        "event": "action_replacement",
                        "version": version,
                        "card": TRAVEL_EN_FOLD3_V8_K6_PARENTHETICAL_CARD,
                        "decision": "switch_to_assistant_tool_call",
                        "reason": (
                            "fold3_v8_k6_strip_unique_parenthetical_qualifier_"
                            "after_detail_failure"
                        ),
                        "pre_action_state_hash": pre_action_state_hash(messages, call),
                        "original_proposal": {
                            "kind": "tool_call",
                            "tool": "query_restaurant_details",
                            "arguments": arguments,
                        },
                        "replacement_action": replacement,
                        "target_tool_call_index": call_index,
                        "proposal_batch_size": len(calls),
                        "collapse_proposal_batch": False,
                        "task_id_or_gold_used": False,
                        "train_only_design": True,
                        "fold": "travel_en_fold3_v8_k6_cleanroom",
                        "card_role": "paper_primary",
                        "primary_metric": "commonsense_and_composite",
                        "paper_primary_tradeoff": False,
                        "expected_tradeoffs": ["calls_may_increase"],
                        "risk": "low_schema_string_normalization",
                        "candidate_kind": "schema_string_normalization",
                        "explicit_user_entity_protected": True,
                        "trajectory_trigger_cap": 1,
                    },
                    "concrete_action": replacement,
                }
            )

    if not decisions:
        return None
    selected = min(
        decisions,
        key=lambda decision: (
            int(decision["priority"]),
            int(decision["payload"]["target_tool_call_index"]),
        ),
    )
    selected.pop("priority")
    return selected


def travel_en_fold3_v8_k6_round3_decision(
    messages: list[dict],
    calls: list[dict[str, str]],
    version: str,
) -> dict | None:
    if version not in {
        TRAVEL_EN_FOLD3_V8_K6_RESTAURANT_LOOPCUT_ACTION_REPLACEMENT_VERSION,
        TRAVEL_EN_FOLD3_V8_K6_USER_FROZEN_FOUR_CARD_ACTION_REPLACEMENT_VERSION,
    }:
        return None
    if any(
        is_persist_marker(message)
        and TRAVEL_EN_FOLD3_V8_K6_RESTAURANT_LOOPCUT_CARD
        in str(get_field(message, "content") or "")
        for message in messages
    ):
        return None

    events = completed_tool_events(messages)
    for call_index, call in enumerate(calls):
        if call["name"] != "recommend_restaurants":
            continue
        arguments = parse_arguments(call)
        signature = f"recommend_restaurants|{canonical_json(arguments)}"
        matching_positions = [
            index
            for index, event in enumerate(events)
            if event["signature"] == signature
        ]
        failures = [
            events[index]
            for index in matching_positions
            if tool_result_failed(events[index]["result"])
        ]
        if len(failures) < 2:
            continue
        later_events = events[matching_positions[-1] + 1 :]
        if any(not tool_result_failed(event["result"]) for event in later_events):
            continue
        replacement = {
            "kind": "tool_call",
            "tool": "__DROP_TOOL_CALL__",
            "arguments": {"drop_reason": "repeated_failure_without_new_evidence"},
        }
        return {
            "decision": "switch_to_assistant_tool_call",
            "payload": {
                "event": "action_replacement",
                "version": version,
                "card": TRAVEL_EN_FOLD3_V8_K6_RESTAURANT_LOOPCUT_CARD,
                "decision": "switch_to_assistant_tool_call",
                "reason": (
                    "fold3_v8_k6_drop_third_or_later_exact_restaurant_"
                    "recommendation_failure_without_new_successful_evidence"
                ),
                "pre_action_state_hash": pre_action_state_hash(messages, call),
                "original_proposal": {
                    "kind": "tool_call",
                    "tool": "recommend_restaurants",
                    "arguments": arguments,
                },
                "replacement_action": replacement,
                "target_tool_call_index": call_index,
                "proposal_batch_size": len(calls),
                "collapse_proposal_batch": False,
                "task_id_or_gold_used": False,
                "train_only_design": True,
                "fold": "travel_en_fold3_v8_k6_cleanroom",
                "card_role": "compensation",
                "primary_metric": "exact_repeats_and_broad_fl",
                "paper_primary_tradeoff": False,
                "expected_tradeoffs": ["delivery_and_composite_expected_flat"],
                "risk": "medium_bounded_loop_cut",
                "candidate_kind": "repeated_failure_loop_cut",
                "prior_exact_failures": len(failures),
                "new_successful_tool_evidence_since_last_exact_call": False,
                "trajectory_trigger_cap": 1,
            },
            "concrete_action": replacement,
        }
    return None


def travel_en_fold3_v8_k6_user_frozen_four_card_decision(
    messages: list[dict],
    calls: list[dict[str, str]],
    version: str,
) -> dict | None:
    decision = travel_en_fold3_v8_k6_round2_decision(messages, calls, version)
    if decision is not None:
        return decision
    decision = travel_en_fold3_v8_k6_round1_decision(messages, calls, version)
    if decision is not None:
        return decision
    return travel_en_fold3_v8_k6_round3_decision(messages, calls, version)


def travel_en_fold1_cleanroom_three_card_decision(
    messages: list[dict],
    calls: list[dict[str, str]],
    version: str = TRAVEL_EN_FOLD1_CLEANROOM_THREE_CARD_ACTION_REPLACEMENT_VERSION,
) -> dict | None:
    if version != TRAVEL_EN_FOLD1_CLEANROOM_THREE_CARD_ACTION_REPLACEMENT_VERSION:
        return None
    hotel_decision = travel_en_fold1_cleanroom_hotel_loopcut_decision(
        messages,
        calls,
        version=version,
    )
    if hotel_decision is not None:
        return hotel_decision

    cards = {
        "search_location": TRAVEL_EN_FOLD1_CLEANROOM_DUPLICATE_SEARCH_LOCATION_CARD,
        "recommend_attractions": TRAVEL_EN_FOLD1_CLEANROOM_DUPLICATE_ATTRACTION_CARD,
    }
    for tool_name, card in cards.items():
        if any(
            is_persist_marker(message)
            and card in str(get_field(message, "content") or "")
            for message in messages
        ):
            continue
        seen_signatures: dict[str, int] = {}
        for call_index, call in enumerate(calls):
            if call["name"] != tool_name:
                continue
            arguments = parse_arguments(call)
            signature = f"{tool_name}|{canonical_json(arguments)}"
            if signature not in seen_signatures:
                seen_signatures[signature] = call_index
                continue
            original = {
                "kind": "tool_call",
                "tool": tool_name,
                "arguments": arguments,
            }
            replacement = {
                "kind": "tool_call",
                "tool": "__DROP_TOOL_CALL__",
                "arguments": {"drop_reason": "duplicate_in_same_batch"},
            }
            payload = {
                "event": "action_replacement",
                "version": version,
                "card": card,
                "decision": "switch_to_assistant_tool_call",
                "reason": "drop_exact_duplicate_call_in_same_proposal_batch",
                "pre_action_state_hash": pre_action_state_hash(messages, call),
                "original_proposal": original,
                "replacement_action": replacement,
                "target_tool_call_index": call_index,
                "proposal_batch_size": len(calls),
                "collapse_proposal_batch": False,
                "task_id_or_gold_used": False,
                "train_only_design": True,
                "fold": "travel_en_fold1_cleanroom",
                "card_role": "paper_primary",
                "primary_metric": "delivery_and_official_travel_metrics",
                "paper_primary_tradeoff": True,
                "expected_tradeoffs": [
                    "tool_calls_or_failed_calls_may_increase_after_continuation"
                ],
                "risk": "medium_loop_cut_batch_duplicate",
                "candidate_kind": "duplicate_batch_call_skip",
                "duplicate_of_call_index": seen_signatures[signature],
                "trajectory_trigger_cap": 1,
            }
            return {
                "decision": "switch_to_assistant_tool_call",
                "payload": payload,
                "concrete_action": replacement,
            }
    return None


def travel_zh_fold1_restaurant_repeat_loopcut_decision(
    messages: list[dict],
    calls: list[dict[str, str]],
    version: str = TRAVEL_ZH_FOLD1_TWO_CARD_ACTION_REPLACEMENT_VERSION,
) -> dict | None:
    if version != TRAVEL_ZH_FOLD1_TWO_CARD_ACTION_REPLACEMENT_VERSION:
        return None
    card = TRAVEL_ZH_FOLD1_RESTAURANT_REPEAT_LOOPCUT_CARD
    if any(
        is_persist_marker(message)
        and card in str(get_field(message, "content") or "")
        for message in messages
    ):
        return None
    events = completed_tool_events(messages)
    for call_index, call in enumerate(calls):
        if call["name"] != "query_restaurant_details":
            continue
        arguments = parse_arguments(call)
        signature = f"query_restaurant_details|{canonical_json(arguments)}"
        matching_positions = [
            index for index, event in enumerate(events) if event["signature"] == signature
        ]
        failures = [
            events[index]
            for index in matching_positions
            if tool_result_failed(events[index]["result"])
        ]
        if len(failures) < 2:
            continue
        later_events = events[matching_positions[-1] + 1 :]
        if any(not tool_result_failed(event["result"]) for event in later_events):
            continue
        original = {
            "kind": "tool_call",
            "tool": "query_restaurant_details",
            "arguments": arguments,
        }
        replacement = {
            "kind": "tool_call",
            "tool": "__DROP_TOOL_CALL__",
            "arguments": {"drop_reason": "repeated_failure_without_new_evidence"},
        }
        payload = {
            "event": "action_replacement",
            "version": version,
            "card": card,
            "decision": "switch_to_assistant_tool_call",
            "reason": "cut_exact_repeated_failure_without_new_successful_tool_evidence",
            "pre_action_state_hash": pre_action_state_hash(messages, call),
            "original_proposal": original,
            "replacement_action": replacement,
            "target_tool_call_index": call_index,
            "proposal_batch_size": len(calls),
            "collapse_proposal_batch": False,
            "task_id_or_gold_used": False,
            "train_only_design": True,
            "fold": "travel_zh_fold1",
            "card_role": "efficiency_compensation",
            "primary_metric": "calls_and_functional_loop_proxy",
            "paper_primary_tradeoff": False,
            "expected_tradeoffs": ["report_valid_may_change"],
            "risk": "medium_bounded_loop_cut",
            "candidate_kind": "repeated_failure_loop_cut",
            "prior_exact_failures": len(failures),
            "new_successful_tool_evidence_since_last_exact_call": False,
            "trajectory_trigger_cap": 1,
        }
        return {
            "decision": "switch_to_assistant_tool_call",
            "payload": payload,
            "concrete_action": replacement,
        }
    return None


def travel_zh_fold2_hotel_empty_brand_action_replacement_decision(
    messages: list[dict],
    calls: list[dict[str, str]],
    version: str = TRAVEL_ZH_FOLD2_HOTEL_EMPTY_BRAND_ACTION_REPLACEMENT_VERSION,
) -> dict | None:
    if version != TRAVEL_ZH_FOLD2_HOTEL_EMPTY_BRAND_ACTION_REPLACEMENT_VERSION:
        return None
    if any(
        is_persist_marker(message)
        and TRAVEL_ZH_FOLD2_HOTEL_EMPTY_BRAND_CARD
        in str(get_field(message, "content") or "")
        for message in messages
    ):
        return None
    placeholder_values = {"无", "其他", "不限", "不限品牌", "任意", ""}
    events = completed_tool_events(messages)
    for call_index, call in enumerate(calls):
        if call["name"] != "query_hotel_info":
            continue
        arguments = parse_arguments(call)
        brand = str(arguments.get("hotelBrands") or "").strip()
        if brand not in placeholder_values:
            continue
        if not {"destination", "checkinDate", "checkoutDate"}.issubset(arguments):
            continue
        has_prior_no_result = False
        for event in events:
            if event["name"] != "query_hotel_info" or not tool_result_failed(event["result"]):
                continue
            try:
                event_arguments = json.loads(event["signature"].split("|", 1)[1])
            except Exception:
                continue
            if not isinstance(event_arguments, dict):
                continue
            if (
                str(event_arguments.get("destination") or "")
                == str(arguments.get("destination") or "")
                and str(event_arguments.get("checkinDate") or "")
                == str(arguments.get("checkinDate") or "")
                and str(event_arguments.get("checkoutDate") or "")
                == str(arguments.get("checkoutDate") or "")
            ):
                has_prior_no_result = True
                break
        if not has_prior_no_result:
            continue
        replacement_arguments = dict(arguments)
        removed_value = replacement_arguments.pop("hotelBrands", None)
        replacement_signature = f"query_hotel_info|{canonical_json(replacement_arguments)}"
        if any(
            event["signature"] == replacement_signature
            and tool_result_failed(event["result"])
            for event in events
        ):
            continue
        original = {
            "kind": "tool_call",
            "tool": "query_hotel_info",
            "arguments": arguments,
        }
        replacement = {
            "kind": "tool_call",
            "tool": "query_hotel_info",
            "arguments": replacement_arguments,
        }
        payload = {
            "event": "action_replacement",
            "version": version,
            "card": TRAVEL_ZH_FOLD2_HOTEL_EMPTY_BRAND_CARD,
            "decision": "switch_to_assistant_tool_call",
            "reason": (
                "after_prior_hotel_no_result_drop_empty_hotelBrands_placeholder"
            ),
            "pre_action_state_hash": pre_action_state_hash(messages, call),
            "original_proposal": original,
            "replacement_action": replacement,
            "target_tool_call_index": call_index,
            "proposal_batch_size": len(calls),
            "collapse_proposal_batch": False,
            "task_id_or_gold_used": False,
            "train_only_design": True,
            "fold": "travel_zh_fold2",
            "card_role": "paper_primary",
            "primary_metric": "report_valid_proxy_and_official_travel_metrics",
            "paper_primary_tradeoff": True,
            "expected_tradeoffs": ["hotel_constraints_may_relax_empty_placeholder_only"],
            "risk": "medium_schema_placeholder_cleanup",
            "candidate_kind": "hotel_empty_brand_drop_zh",
            "removed_field": "hotelBrands",
            "removed_value": removed_value,
            "preserved_fields": sorted(replacement_arguments),
        }
        return {
            "decision": "switch_to_assistant_tool_call",
            "payload": payload,
            "concrete_action": replacement,
        }
    return None


def _travel_restaurant_anchor_prerequisite_single_decision(
    messages: list[dict],
    call: dict[str, str],
    all_calls: list[dict[str, str]],
    target_tool_call_index: int,
    version: str = RESTAURANT_ANCHOR_PREREQUISITE_ACTION_REPLACEMENT_VERSION,
) -> dict | None:
    if version != RESTAURANT_ANCHOR_PREREQUISITE_ACTION_REPLACEMENT_VERSION:
        return None
    if call["name"] != "query_restaurant_details":
        return None
    if any(
        is_persist_marker(message)
        and f'"card": "{RESTAURANT_ANCHOR_PREREQUISITE_CARD}"'
        in str(get_field(message, "content") or "")
        for message in messages
    ):
        return None
    arguments = parse_arguments(call)
    if set(arguments) != {"restaurant_name"}:
        return None
    entity_name = str(arguments.get("restaurant_name") or "").strip()
    normalized_name = normalized_english_phrase(entity_name)
    if not normalized_name:
        return None
    user_query = first_user_query(messages)
    trip_days = travel_calendar_days(user_query)
    if trip_days is None or trip_days > 5 or has_explicit_travel_budget(user_query):
        return None
    if not user_requests_restaurant_near_entity(user_query, entity_name):
        return None

    events = completed_tool_events(messages)
    signature = f"query_restaurant_details|{canonical_json(arguments)}"
    failed_events = [
        event
        for event in events
        if event["signature"] == signature and tool_result_failed(event["result"])
    ]
    if not failed_events:
        return None
    attraction_names = {
        normalized_english_phrase(name) for name in recommended_attraction_names(events)
    }
    if normalized_name not in attraction_names:
        return None
    if any(
        event["name"] == "recommend_restaurants"
        and not tool_result_failed(event["result"])
        and normalized_name in normalized_english_phrase(event["result"])
        for event in events
    ):
        return None
    replacement_arguments = {"place_name": entity_name}
    replacement_signature = f"search_location|{canonical_json(replacement_arguments)}"
    if any(event["signature"] == replacement_signature for event in events):
        return None

    original_action = {"kind": "tool_call", "tool": call["name"], "arguments": arguments}
    replacement_action = {
        "kind": "tool_call",
        "tool": "search_location",
        "arguments": replacement_arguments,
    }
    payload = {
        "event": "action_replacement",
        "version": version,
        "card": RESTAURANT_ANCHOR_PREREQUISITE_CARD,
        "decision": "switch_to_assistant_tool_call",
        "reason": "failed_restaurant_detail_is_grounded_nearby_attraction_anchor_missing_location",
        "pre_action_state_hash": pre_action_batch_state_hash(messages, all_calls),
        "original_proposal": original_action,
        "replacement_action": replacement_action,
        "failure_signature": hashlib.sha256(
            failed_events[-1]["result"].encode("utf-8")
        ).hexdigest(),
        "effect_unchanged": True,
        "evidence_delta": "none",
        "productive_retry": False,
        "risk_envelope": {
            "trip_days": trip_days,
            "max_trip_days": 5,
            "explicit_budget": False,
            "exact_authoritative_attraction_name": True,
            "explicit_nearby_restaurant_requirement": True,
            "prior_exact_location_lookup": False,
        },
        "trigger_count": 1,
        "max_triggers_per_trajectory": 1,
        "target_tool_call_index": target_tool_call_index,
        "proposal_batch_size": len(all_calls),
        "write_action_injected": False,
    }
    return {
        "decision": "switch_to_assistant_tool_call",
        "payload": payload,
        "concrete_action": replacement_action,
    }


def travel_restaurant_anchor_prerequisite_action_replacement_decision(
    messages: list[dict],
    calls: list[dict[str, str]],
    version: str = RESTAURANT_ANCHOR_PREREQUISITE_ACTION_REPLACEMENT_VERSION,
) -> dict | None:
    if version != RESTAURANT_ANCHOR_PREREQUISITE_ACTION_REPLACEMENT_VERSION or not calls:
        return None
    candidates = [
        decision
        for index, call in enumerate(calls)
        if (
            decision := _travel_restaurant_anchor_prerequisite_single_decision(
                messages, call, calls, index, version=version
            )
        )
        is not None
    ]
    return candidates[0] if len(candidates) == 1 else None


def _travel_restaurant_anchor_recommendation_single_decision(
    messages: list[dict],
    call: dict[str, str],
    all_calls: list[dict[str, str]],
    target_tool_call_index: int,
    version: str = RESTAURANT_ANCHOR_RECOMMENDATION_ACTION_REPLACEMENT_VERSION,
) -> dict | None:
    if version != RESTAURANT_ANCHOR_RECOMMENDATION_ACTION_REPLACEMENT_VERSION:
        return None
    if call["name"] != "query_restaurant_details":
        return None
    if any(
        is_persist_marker(message)
        and f'"card": "{RESTAURANT_ANCHOR_RECOMMENDATION_CARD}"'
        in str(get_field(message, "content") or "")
        for message in messages
    ):
        return None
    arguments = parse_arguments(call)
    if set(arguments) != {"restaurant_name"}:
        return None
    entity_name = str(arguments.get("restaurant_name") or "").strip()
    normalized_name = normalized_english_phrase(entity_name)
    if not normalized_name:
        return None
    user_query = first_user_query(messages)
    trip_days = travel_calendar_days(user_query)
    if trip_days is None or trip_days > 5 or has_explicit_travel_budget(user_query):
        return None
    if not user_requests_restaurant_near_entity(user_query, entity_name):
        return None

    events = completed_tool_events(messages)
    signature = f"query_restaurant_details|{canonical_json(arguments)}"
    failed_events = [
        event
        for event in events
        if event["signature"] == signature and tool_result_failed(event["result"])
    ]
    if not failed_events:
        return None
    attraction_names = {
        normalized_english_phrase(name) for name in recommended_attraction_names(events)
    }
    if normalized_name not in attraction_names:
        return None
    location_signature = f"search_location|{canonical_json({'place_name': entity_name})}"
    coordinates = {}
    grounding_event = None
    for event in events:
        if event["signature"] != location_signature or tool_result_failed(event["result"]):
            continue
        try:
            result = json.loads(event["result"])
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(result, dict):
            continue
        latitude = str(result.get("latitude") or "").strip()
        longitude = str(result.get("longitude") or "").strip()
        if latitude and longitude:
            coordinates[(latitude, longitude)] = event
    if len(coordinates) != 1:
        return None
    (latitude, longitude), grounding_event = next(iter(coordinates.items()))
    replacement_arguments = {"latitude": latitude, "longitude": longitude}
    replacement_signature = f"recommend_restaurants|{canonical_json(replacement_arguments)}"
    if any(event["signature"] == replacement_signature for event in events):
        return None

    original_action = {"kind": "tool_call", "tool": call["name"], "arguments": arguments}
    replacement_action = {
        "kind": "tool_call",
        "tool": "recommend_restaurants",
        "arguments": replacement_arguments,
    }
    payload = {
        "event": "action_replacement",
        "version": version,
        "card": RESTAURANT_ANCHOR_RECOMMENDATION_CARD,
        "decision": "switch_to_assistant_tool_call",
        "reason": "failed_attraction_as_restaurant_detail_has_unique_grounded_nearby_anchor",
        "pre_action_state_hash": pre_action_batch_state_hash(messages, all_calls),
        "original_proposal": original_action,
        "replacement_action": replacement_action,
        "failure_signature": hashlib.sha256(
            failed_events[-1]["result"].encode("utf-8")
        ).hexdigest(),
        "grounding_signature": hashlib.sha256(
            grounding_event["result"].encode("utf-8")
        ).hexdigest(),
        "effect_unchanged": True,
        "evidence_delta": "unique_visible_location_coordinates",
        "productive_retry": False,
        "risk_envelope": {
            "trip_days": trip_days,
            "max_trip_days": 5,
            "explicit_budget": False,
            "exact_authoritative_attraction_name": True,
            "explicit_nearby_restaurant_requirement": True,
            "unique_grounded_anchor_coordinates": True,
            "prior_exact_restaurant_recommendation": False,
        },
        "trigger_count": 1,
        "max_triggers_per_trajectory": 1,
        "target_tool_call_index": target_tool_call_index,
        "proposal_batch_size": len(all_calls),
        "write_action_injected": False,
    }
    return {
        "decision": "switch_to_assistant_tool_call",
        "payload": payload,
        "concrete_action": replacement_action,
    }


def travel_restaurant_anchor_recommendation_action_replacement_decision(
    messages: list[dict],
    calls: list[dict[str, str]],
    version: str = RESTAURANT_ANCHOR_RECOMMENDATION_ACTION_REPLACEMENT_VERSION,
) -> dict | None:
    if version != RESTAURANT_ANCHOR_RECOMMENDATION_ACTION_REPLACEMENT_VERSION or not calls:
        return None
    candidates = [
        decision
        for index, call in enumerate(calls)
        if (
            decision := _travel_restaurant_anchor_recommendation_single_decision(
                messages, call, calls, index, version=version
            )
        )
        is not None
    ]
    return candidates[0] if len(candidates) == 1 else None


def grounded_search_location_event(
    events: list[dict[str, str]], arguments: dict
) -> dict[str, str] | None:
    latitude = str(arguments.get("latitude") or "").strip()
    longitude = str(arguments.get("longitude") or "").strip()
    if not latitude or not longitude:
        return None
    for event in reversed(events):
        if event["name"] != "search_location" or tool_result_failed(event["result"]):
            continue
        try:
            payload = json.loads(event["result"])
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(payload, dict):
            continue
        if (
            str(payload.get("latitude") or "").strip() == latitude
            and str(payload.get("longitude") or "").strip() == longitude
        ):
            return event
    return None


def travel_restaurant_grounded_coordinate_action_replacement_decision(
    messages: list[dict],
    calls: list[dict[str, str]],
    version: str = RESTAURANT_GROUNDED_COORDINATE_ACTION_REPLACEMENT_VERSION,
) -> dict | None:
    if version != RESTAURANT_GROUNDED_COORDINATE_ACTION_REPLACEMENT_VERSION:
        return None
    if len(calls) < 2:
        return None
    if any(
        is_persist_marker(message)
        and f'"card": "{RESTAURANT_GROUNDED_COORDINATE_CARD}"'
        in str(get_field(message, "content") or "")
        for message in messages
    ):
        return None
    restaurant_indices = [
        index for index, call in enumerate(calls) if call["name"] == "recommend_restaurants"
    ]
    if len(restaurant_indices) != 1:
        return None
    target_index = restaurant_indices[0]
    arguments = parse_arguments(calls[target_index])
    if set(arguments) != {"latitude", "longitude"}:
        return None
    events = completed_tool_events(messages)
    call_signatures = [
        f"{call['name']}|{canonical_json(parse_arguments(call))}" for call in calls
    ]
    if any(
        not any(event["signature"] == call_signature for event in events)
        for call_signature in call_signatures
    ):
        return None
    signature = f"recommend_restaurants|{canonical_json(arguments)}"
    failed_positions = [
        index
        for index, event in enumerate(events)
        if event["signature"] == signature and tool_result_failed(event["result"])
    ]
    if not failed_positions or grounded_search_location_event(events, arguments) is not None:
        return None
    grounded_after_failure = []
    for index, event in enumerate(events):
        if index <= failed_positions[-1]:
            continue
        if event["name"] != "search_location" or tool_result_failed(event["result"]):
            continue
        try:
            result = json.loads(event["result"])
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(result, dict):
            continue
        latitude = str(result.get("latitude") or "").strip()
        longitude = str(result.get("longitude") or "").strip()
        if latitude and longitude:
            grounded_after_failure.append((latitude, longitude, event))
    unique_coordinates = {
        (latitude, longitude): event
        for latitude, longitude, event in grounded_after_failure
    }
    if len(unique_coordinates) != 1:
        return None
    (latitude, longitude), grounding_event = next(iter(unique_coordinates.items()))
    replacement_arguments = {"latitude": latitude, "longitude": longitude}
    replacement_signature = (
        f"recommend_restaurants|{canonical_json(replacement_arguments)}"
    )
    if any(event["signature"] == replacement_signature for event in events):
        return None
    original_action = {
        "kind": "tool_call",
        "tool": "recommend_restaurants",
        "arguments": arguments,
    }
    replacement_action = {
        "kind": "tool_call",
        "tool": "recommend_restaurants",
        "arguments": replacement_arguments,
    }
    payload = {
        "event": "action_replacement",
        "version": version,
        "card": RESTAURANT_GROUNDED_COORDINATE_CARD,
        "decision": "switch_to_assistant_tool_call",
        "reason": "repeated_ungrounded_restaurant_coordinates_have_unique_visible_prerequisite",
        "pre_action_state_hash": pre_action_batch_state_hash(messages, calls),
        "original_proposal": original_action,
        "replacement_action": replacement_action,
        "failure_signature": hashlib.sha256(
            events[failed_positions[-1]]["result"].encode("utf-8")
        ).hexdigest(),
        "grounding_signature": hashlib.sha256(
            grounding_event["result"].encode("utf-8")
        ).hexdigest(),
        "effect_unchanged": True,
        "evidence_delta": "unique_visible_location_coordinates",
        "productive_retry": False,
        "risk_envelope": {
            "original_batch_size": len(calls),
            "all_original_calls_previously_executed": True,
            "original_coordinates_grounded": False,
            "unique_new_grounded_coordinates": True,
        },
        "trigger_count": 1,
        "max_triggers_per_trajectory": 1,
        "target_tool_call_index": target_index,
        "proposal_batch_size": len(calls),
        "replacement_batch_size": 1,
        "replacement_target_tool_call_index": 0,
        "collapse_proposal_batch": True,
        "write_action_injected": False,
    }
    return {
        "decision": "switch_to_assistant_tool_call",
        "payload": payload,
        "concrete_action": replacement_action,
    }


def travel_en_fold1_decision_fork_coordinate_rebinding_decision(
    messages: list[dict],
    calls: list[dict[str, str]],
    version: str = TRAVEL_EN_FOLD1_DECISION_FORK_COORDINATE_ACTION_REPLACEMENT_VERSION,
) -> dict | None:
    if version != TRAVEL_EN_FOLD1_DECISION_FORK_COORDINATE_ACTION_REPLACEMENT_VERSION:
        return None
    if any(
        is_persist_marker(message)
        and f'"card": "{TRAVEL_EN_FOLD1_DECISION_FORK_COORDINATE_CARD}"'
        in str(get_field(message, "content") or "")
        for message in messages
    ):
        return None
    restaurant_indices = [
        index for index, call in enumerate(calls) if call["name"] == "recommend_restaurants"
    ]
    if len(restaurant_indices) != 1:
        return None
    target_index = restaurant_indices[0]
    arguments = parse_arguments(calls[target_index])
    if set(arguments) != {"latitude", "longitude"}:
        return None
    events = completed_tool_events(messages)
    signature = f"recommend_restaurants|{canonical_json(arguments)}"
    failed_positions = [
        index
        for index, event in enumerate(events)
        if event["signature"] == signature and tool_result_failed(event["result"])
    ]
    if not failed_positions:
        return None
    grounded_after_failure: dict[tuple[str, str], dict[str, str]] = {}
    for index, event in enumerate(events):
        if index <= failed_positions[-1]:
            continue
        if event["name"] != "search_location" or tool_result_failed(event["result"]):
            continue
        try:
            result = json.loads(event["result"])
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(result, dict):
            continue
        latitude = str(result.get("latitude") or result.get("lat") or "").strip()
        longitude = str(
            result.get("longitude") or result.get("lng") or result.get("lon") or ""
        ).strip()
        if latitude and longitude:
            grounded_after_failure[(latitude, longitude)] = event
    if len(grounded_after_failure) != 1:
        return None
    (latitude, longitude), grounding_event = next(iter(grounded_after_failure.items()))
    replacement_arguments = {"latitude": latitude, "longitude": longitude}
    if replacement_arguments == arguments:
        return None
    replacement_signature = (
        f"recommend_restaurants|{canonical_json(replacement_arguments)}"
    )
    if any(event["signature"] == replacement_signature for event in events):
        return None
    original_action = {
        "kind": "tool_call",
        "tool": "recommend_restaurants",
        "arguments": arguments,
    }
    replacement_action = {
        "kind": "tool_call",
        "tool": "recommend_restaurants",
        "arguments": replacement_arguments,
    }
    payload = {
        "event": "action_replacement",
        "version": version,
        "card": TRAVEL_EN_FOLD1_DECISION_FORK_COORDINATE_CARD,
        "decision": "switch_to_assistant_tool_call",
        "reason": "failed_restaurant_coordinates_rebound_to_unique_new_visible_location",
        "pre_action_state_hash": pre_action_batch_state_hash(messages, calls),
        "original_proposal": original_action,
        "replacement_action": replacement_action,
        "failure_signature": hashlib.sha256(
            events[failed_positions[-1]]["result"].encode("utf-8")
        ).hexdigest(),
        "grounding_signature": hashlib.sha256(
            grounding_event["result"].encode("utf-8")
        ).hexdigest(),
        "effect_unchanged": True,
        "evidence_delta": "unique_visible_location_coordinates_after_failure",
        "productive_retry": False,
        "risk_envelope": {
            "prior_card_registry_imported": False,
            "gold_or_task_id_used": False,
            "original_coordinates_failed": True,
            "unique_new_grounded_coordinates": True,
            "user_hard_constraint_changed": False,
        },
        "trigger_count": 1,
        "max_triggers_per_trajectory": 1,
        "target_tool_call_index": target_index,
        "proposal_batch_size": len(calls),
        "replacement_batch_size": len(calls),
        "replacement_target_tool_call_index": target_index,
        "collapse_proposal_batch": False,
        "write_action_injected": False,
    }
    return {
        "decision": "switch_to_assistant_tool_call",
        "payload": payload,
        "concrete_action": replacement_action,
    }


def travel_en_fold2_decision_fork_same_route_drop_decision(
    messages: list[dict],
    calls: list[dict[str, str]],
    version: str = TRAVEL_EN_FOLD2_DECISION_FORK_SAME_ROUTE_ACTION_REPLACEMENT_VERSION,
) -> dict | None:
    if version != TRAVEL_EN_FOLD2_DECISION_FORK_SAME_ROUTE_ACTION_REPLACEMENT_VERSION:
        return None
    if any(
        is_persist_marker(message)
        and f'"card": "{TRAVEL_EN_FOLD2_DECISION_FORK_SAME_ROUTE_CARD}"'
        in str(get_field(message, "content") or "")
        for message in messages
    ):
        return None
    for call_index, call in enumerate(calls):
        if call["name"] != "query_road_route_info":
            continue
        arguments = parse_arguments(call)
        origin = str(arguments.get("origin") or "").strip()
        destination = str(arguments.get("destination") or "").strip()
        if not origin or not destination or origin != destination:
            continue
        original_action = {
            "kind": "tool_call",
            "tool": "query_road_route_info",
            "arguments": arguments,
        }
        replacement_action = {
            "kind": "tool_call",
            "tool": "__DROP_TOOL_CALL__",
            "arguments": {"drop_reason": "same_origin_destination"},
        }
        payload = {
            "event": "action_replacement",
            "version": version,
            "card": TRAVEL_EN_FOLD2_DECISION_FORK_SAME_ROUTE_CARD,
            "decision": "switch_to_assistant_tool_call",
            "reason": "same_origin_destination_route_query_is_noop",
            "pre_action_state_hash": pre_action_batch_state_hash(messages, calls),
            "original_proposal": original_action,
            "replacement_action": replacement_action,
            "effect_unchanged": True,
            "evidence_delta": "proposal_arguments_show_identical_route_endpoints",
            "productive_retry": False,
            "risk_envelope": {
                "prior_card_registry_imported": False,
                "gold_or_task_id_used": False,
                "origin_destination_exact_match": True,
                "user_hard_constraint_changed": False,
                "paper_primary_tradeoff": True,
                "expected_tradeoffs": ["calls_may_increase"],
            },
            "trigger_count": 1,
            "max_triggers_per_trajectory": 1,
            "target_tool_call_index": call_index,
            "proposal_batch_size": len(calls),
            "replacement_batch_size": max(0, len(calls) - 1),
            "collapse_proposal_batch": False,
            "write_action_injected": False,
        }
        return {
            "decision": "switch_to_assistant_tool_call",
            "payload": payload,
            "concrete_action": replacement_action,
        }
    return None


def set_field(value: object, name: str, field_value: object) -> None:
    if isinstance(value, dict):
        value[name] = field_value
    else:
        setattr(value, name, field_value)


def apply_travel_action_replacement(response: object, decision: dict) -> None:
    choices = get_field(response, "choices") or []
    if not choices:
        raise ValueError("Cannot replace a proposal without a response choice")
    message = get_field(choices[0], "message") or {}
    tool_calls = get_field(message, "tool_calls") or []
    target_index = int(decision["payload"].get("target_tool_call_index", 0))
    expected_batch_size = int(decision["payload"].get("proposal_batch_size", 1))
    if len(tool_calls) != expected_batch_size or not 0 <= target_index < len(tool_calls):
        raise ValueError("Travel action replacement proposal batch mismatch")
    function = get_field(tool_calls[target_index], "function") or {}
    action = decision["concrete_action"]
    original_tool = decision["payload"]["original_proposal"]["tool"]
    if str(get_field(function, "name") or "") != original_tool:
        raise ValueError("Replacement source tool does not match the proposed tool")
    if action["tool"] == "__DROP_TOOL_CALL__":
        set_field(
            message,
            "tool_calls",
            [tool_call for index, tool_call in enumerate(tool_calls) if index != target_index],
        )
        return
    set_field(function, "name", action["tool"])
    set_field(function, "arguments", canonical_json(action["arguments"]))
    if decision["payload"].get("collapse_proposal_batch"):
        set_field(message, "tool_calls", [tool_calls[target_index]])


def action_replacement_marker(decision: dict) -> dict:
    return {
        "role": "system",
        "content": f"{PERSIST_ACE_PREFIX} {json.dumps(decision['payload'], ensure_ascii=False, sort_keys=True)}",
    }


def collect_named_candidates(events: list[dict[str, str]], family: str) -> list[str]:
    names: list[str] = []
    for event in events:
        if tool_family(event["name"]) != family or tool_result_failed(event["result"]):
            continue
        try:
            payload = json.loads(event["result"])
        except (json.JSONDecodeError, TypeError):
            continue
        stack = [payload]
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                candidate = value.get("restaurant_name") or value.get("name")
                if isinstance(candidate, str) and candidate.strip() and candidate.strip() not in names:
                    names.append(candidate.strip())
                stack.extend(value.values())
            elif isinstance(value, list):
                stack.extend(value)
    return names


REPEAT_SCOPE_KEYS = (
    "origin",
    "destination",
    "city",
    "depDate",
    "date",
    "checkInDate",
    "checkOutDate",
    "restaurant_name",
    "hotel_name",
    "attraction_name",
    "latitude",
    "longitude",
)


def event_arguments(event: dict[str, str]) -> dict:
    _, separator, arguments = event.get("signature", "").partition("|")
    if not separator:
        return {}
    return parse_arguments({"arguments": arguments})


def repeat_scope(name: str, arguments: dict) -> tuple[str, tuple[tuple[str, str], ...]] | None:
    anchors = tuple(
        (key, canonical_json(arguments[key]))
        for key in REPEAT_SCOPE_KEYS
        if key in arguments and arguments[key] not in (None, "")
    )
    if not anchors:
        return None
    return tool_family(name), anchors


def v15_bounded_family_stagnation(
    events: list[dict[str, str]],
    call: dict[str, str],
) -> dict[str, object] | None:
    if not call["name"].startswith("query_") or len(events) < 3:
        return None
    proposed_scope = repeat_scope(call["name"], parse_arguments(call))
    if proposed_scope is None:
        return None
    recent = events[-3:]
    recent_scopes = [repeat_scope(event["name"], event_arguments(event)) for event in recent]
    if any(scope != proposed_scope for scope in recent_scopes):
        return None
    results = [event["result"] for event in recent]
    if not results[0] or any(result != results[0] for result in results[1:]):
        return None
    return {
        "repeat_count": 3,
        "effect_unchanged": True,
        "same_scope": True,
        "family": proposed_scope[0],
        "scope": dict(proposed_scope[1]),
        "prior_signatures": [event["signature"] for event in recent],
    }


def v15_repeat_recovery_decision(
    messages: list[dict],
    calls: list[dict[str, str]],
    version: str = "v15_tau_source_preserving",
) -> dict | None:
    if len(calls) != 1:
        return None
    markers = [
        message
        for message in messages
        if is_persist_marker(message)
        and f'"version": "{version}"' in str(get_field(message, "content") or "")
    ]
    if len(markers) >= 2:
        return None
    call = calls[0]
    stagnation = v15_bounded_family_stagnation(completed_tool_events(messages), call)
    if stagnation is None:
        return None
    payload = {
        "event": "typed_ace" if not markers else "post_action_guard",
        "version": version,
        "blocked_tool": call["name"],
        "blocked_signature": f"{call['name']}|{call['arguments']}",
        "write_action_injected": False,
        "card": "functional_progress_repeat_guard",
        "reason": "bounded_read_only_family_stagnation",
        "productive_retry_protected": True,
        **stagnation,
    }
    return make_travel_decision(
        payload,
        (
            "[PERSIST-ACE Guard: Bounded Functional-Progress Repeat] Three consecutive read-only calls "
            "within the same user-visible query scope returned exactly the same result, so another call "
            "in that scope is blocked even if optional filter arguments changed. Reuse the visible result "
            "and continue with a different requirement. A different city, date, route, named entity, or "
            "coordinate scope remains allowed. Do not invent evidence and do not finalize solely because "
            "this call was blocked."
        ),
        version,
    )


def make_travel_decision(payload: dict, instruction: str, version: str) -> dict:
    payload["version"] = version
    if version in {
        "v5_shopping_aligned_completion_guard",
        "v6_shopping_aligned_route_safe",
        "v7_shopping_aligned_bounded_retry",
        "v8_shopping_aligned_batch_guard",
        "v9_shopping_aligned_structure_safe",
        "v10_shopping_aligned_extractor_safe",
        "v11_shopping_aligned_draft_aware",
        "v12_shopping_aligned_serializer_safe",
        "v13_shopping_aligned_semantic_gate",
        "v14_tau_aligned_safe_bundle",
        "v15_tau_source_preserving",
        "v16_tau_serializer_constraint_only",
        "v17_tau_personalized_hotel_backtracking",
        "v18_tau_no_compaction_source_preserving",
        "v19_tau_selective_no_compaction",
    }:
        if version == "v18_tau_no_compaction_source_preserving" or (
            version == "v19_tau_selective_no_compaction"
            and payload.get("reason") == "repeated_failed_hotel_overconstraint"
        ):
            return {"payload": payload, "instruction": instruction}
        if version in {
            "v9_shopping_aligned_structure_safe",
            "v10_shopping_aligned_extractor_safe",
            "v11_shopping_aligned_draft_aware",
            "v12_shopping_aligned_serializer_safe",
            "v13_shopping_aligned_semantic_gate",
            "v14_tau_aligned_safe_bundle",
            "v15_tau_source_preserving",
            "v16_tau_serializer_constraint_only",
            "v17_tau_personalized_hotel_backtracking",
            "v19_tau_selective_no_compaction",
        }:
            instruction += (
                " [PERSIST context-budget finalization] When eventually producing the final answer, keep the complete "
                "plan under 2400 Chinese characters and close the </plan> tag. Before </plan>, include exactly one "
                "compact line beginning `预算汇总：` with transport, accommodation, meals, tickets, and total; do not "
                "enumerate repeated addition terms. Do not finalize solely because of this guard."
            )
        else:
            instruction += (
                " [PERSIST context-budget finalization] When eventually producing the final answer, keep the complete "
                "plan under 3200 Chinese characters, close the </plan> tag, and state any budget summary once without "
                "enumerating repeated addition terms. Do not finalize solely because of this guard."
            )
    return {"payload": payload, "instruction": instruction}


def travel_recovery_candidate(
    messages: list[dict],
    calls: list[dict[str, str]],
    version: str = "v4_shopping_aligned",
) -> dict | None:
    if not calls:
        return None
    same_unavailable_restaurant_batch = len(calls) > 1 and all(
        call["name"] == "recommend_around_restaurants" for call in calls
    )
    if len(calls) > 1 and not same_unavailable_restaurant_batch:
        return None
    markers = [
        message
        for message in messages
        if is_persist_marker(message)
        and f'"version": "{version}"' in str(get_field(message, "content") or "")
    ]
    if len(markers) >= 2:
        return None
    call = calls[0]
    name = call["name"]
    arguments = parse_arguments(call)
    ledger = travel_requirement_ledger(messages)
    events = completed_tool_events(messages)
    signature = f"{name}|{call['arguments']}"
    event = "typed_ace" if not markers else "post_action_guard"
    payload = {
        "event": event,
        "version": version,
        "blocked_tool": name,
        "blocked_signature": signature,
        "model_visible_ledger": ledger,
        "write_action_injected": False,
    }
    if same_unavailable_restaurant_batch:
        payload["blocked_batch_size"] = len(calls)

    if name in {"query_train_info", "query_flight_info"} and ledger.get("return_date"):
        is_return = arguments.get("depDate") == ledger["return_date"]
        expected_tool = {
            "train": "query_train_info",
            "flight": "query_flight_info",
        }.get(str(ledger.get("return_mode") or ""))
        expected_origin = ledger.get("destination")
        expected_destination = ledger.get("origin")
        wrong_mode = bool(is_return and expected_tool and name != expected_tool)
        wrong_direction = bool(
            is_return
            and expected_origin
            and expected_destination
            and (
                arguments.get("origin") != expected_origin
                or arguments.get("destination") != expected_destination
            )
        )
        if wrong_mode or wrong_direction:
            payload["card"] = "constraint_backtracking_card"
            payload["reason"] = "return_transport_contract_violation"
            return make_travel_decision(
                payload,
                (
                    "[Requirement-Ledger PERSIST-ACE: Return Constraint Backtracking] Block the proposed return "
                    f"transport lookup. The user-visible ledger requires return travel on {ledger['return_date']} "
                    f"from {expected_origin} to {expected_destination} using {expected_tool}. Re-read those original "
                    "constraints and issue at most one corrected read-only lookup. Do not switch transport mode, "
                    "reverse the route, invent availability, or finalize solely because this action was blocked."
                ),
                version,
            )

    if name == "recommend_around_restaurants":
        payload["card"] = "tool_prerequisite_repair_card"
        payload["reason"] = "unavailable_tool_alias"
        return make_travel_decision(
            payload,
            (
                "[Requirement-Ledger PERSIST-ACE: Tool Prerequisite Repair] Every proposed call in this batch uses "
                "an unavailable restaurant tool name and the entire batch is blocked. "
                "Use the available recommend_restaurants tool with coordinates returned by search_location. "
                "Issue one read-only lookup and wait for its real candidates before requesting restaurant details."
            ),
            version,
        )

    matching_events = [event_row for event_row in events if event_row["signature"] == signature]
    failed_matches = [event_row for event_row in matching_events if tool_result_failed(event_row["result"])]
    if name == "query_hotel_info" and len(failed_matches) >= 2:
        payload["card"] = "constraint_backtracking_card"
        payload["reason"] = "repeated_failed_hotel_overconstraint"
        return make_travel_decision(
            payload,
            (
                "[Requirement-Ledger PERSIST-ACE: Hotel Constraint Backtracking] The exact hotel query has already "
                "failed twice and is blocked. Preserve destination, check-in, check-out, and the user-requested star "
                "level. Remove only an unsupported hotelBrands placeholder such as 无 or 其他, then issue at most one "
                "relaxed hotel lookup. If that also returns no candidates, record the amenity/property constraint as "
                "unverified and continue the remaining transport, attraction, and restaurant requirements."
            ),
            version,
        )

    if name == "query_restaurant_details":
        restaurant_name = str(arguments.get("restaurant_name") or "")
        candidates = collect_named_candidates(events, "restaurant")
        if candidates and restaurant_name not in candidates:
            payload["card"] = "grounded_evidence_card"
            payload["reason"] = "restaurant_detail_not_in_visible_candidates"
            payload["visible_candidates"] = candidates[:5]
            return make_travel_decision(
                payload,
                (
                    "[Requirement-Ledger PERSIST-ACE: Grounded Restaurant Evidence] The proposed restaurant detail "
                    "lookup is not grounded in a name returned by the recommendation tool and is blocked. Choose at "
                    f"most one exact visible candidate name from {candidates[:5]}, or continue without details. "
                    "Do not invent or paraphrase restaurant names."
                ),
                version,
            )
    if (
        version == "v14_tau_aligned_safe_bundle"
        and name.startswith("query_")
        and len(events) >= 2
        and events[-2]["signature"] == signature
        and events[-1]["signature"] == signature
        and events[-2]["result"] == events[-1]["result"]
    ):
        payload["card"] = "exact_effect_repeat_safe_guard"
        payload["reason"] = "exact_read_only_action_and_effect_stagnation"
        payload["repeat_count"] = 2
        payload["effect_unchanged"] = True
        payload["productive_retry_protected"] = False
        return make_travel_decision(
            payload,
            (
                "[PERSIST-ACE Guard: Exact-Effect Repeat Safe] The same read-only tool call has already "
                "completed twice with exactly the same result, so a third identical call is blocked. Reuse "
                "the visible result. Continue with a materially different constraint or the next unmet "
                "requirement; if neither is available, record the requirement as unresolved. Do not invent "
                "new evidence and do not finalize solely because this call was blocked."
            ),
            version,
        )
    if version in {
        "v15_tau_source_preserving",
        "v17_tau_personalized_hotel_backtracking",
        "v18_tau_no_compaction_source_preserving",
        "v19_tau_selective_no_compaction",
    }:
        return v15_repeat_recovery_decision(messages, calls, version=version)
    return None


def travel_recovery_decision(
    messages: list[dict],
    calls: list[dict[str, str]],
    version: str = "v4_shopping_aligned",
) -> dict | None:
    decision = travel_recovery_candidate(messages, calls, version=version)
    if version in {
        "v13_shopping_aligned_semantic_gate",
        "v14_tau_aligned_safe_bundle",
        "v15_tau_source_preserving",
        "v16_tau_serializer_constraint_only",
        "v17_tau_personalized_hotel_backtracking",
        "v18_tau_no_compaction_source_preserving",
        "v19_tau_selective_no_compaction",
    }:
        gated = apply_travel_semantic_recoverability_gate(messages, decision)
        if gated is None and version in {
            "v15_tau_source_preserving",
            "v17_tau_personalized_hotel_backtracking",
            "v18_tau_no_compaction_source_preserving",
            "v19_tau_selective_no_compaction",
        }:
            fallback = v15_repeat_recovery_decision(messages, calls, version=version)
            return apply_travel_semantic_recoverability_gate(messages, fallback)
        return gated
    return decision


def recovery_marker(decision: dict) -> dict:
    return {
        "role": "system",
        "content": (
            f"{PERSIST_ACE_PREFIX} {json.dumps(decision['payload'], ensure_ascii=False, sort_keys=True)}\n"
            f"{decision['instruction']}"
        ),
    }


def bounded_finalization_retry_reason(messages: list[dict], response: object, version: str) -> str | None:
    if version not in {
        "v7_shopping_aligned_bounded_retry",
        "v8_shopping_aligned_batch_guard",
        "v9_shopping_aligned_structure_safe",
        "v10_shopping_aligned_extractor_safe",
        "v11_shopping_aligned_draft_aware",
        "v12_shopping_aligned_serializer_safe",
        "v13_shopping_aligned_semantic_gate",
        "v14_tau_aligned_safe_bundle",
    }:
        return None
    if proposed_calls(response):
        return None
    markers = [
        message
        for message in messages
        if is_persist_marker(message)
        and f'"version": "{version}"' in str(get_field(message, "content") or "")
    ]
    bounded_reasons = {
        "incomplete_plan_generation_retry",
        "missing_budget_summary_retry",
    }
    if any(
        any(f'"reason": "{reason}"' in str(get_field(message, "content") or "") for reason in bounded_reasons)
        for message in markers
    ):
        return None
    try:
        content = str(response.choices[0].message.content or "")
    except (AttributeError, IndexError, TypeError):
        return None
    lower = content.lower()
    if "<plan>" in lower and "</plan>" not in lower:
        if markers or version == "v14_tau_aligned_safe_bundle":
            return "incomplete_plan_generation_retry"
        return None
    if not markers:
        return None
    if version in {
        "v9_shopping_aligned_structure_safe",
        "v10_shopping_aligned_extractor_safe",
        "v11_shopping_aligned_draft_aware",
        "v12_shopping_aligned_serializer_safe",
        "v13_shopping_aligned_semantic_gate",
        "v14_tau_aligned_safe_bundle",
    }:
        plan_match = re.search(r"<plan>(.*?)</plan>", content, flags=re.DOTALL | re.IGNORECASE)
        if plan_match is not None:
            extracted_plan = plan_match.group(1)
            if "预算汇总" not in extracted_plan and "budget summary" not in extracted_plan.lower():
                return "missing_budget_summary_retry"
    return None


def needs_bounded_finalization_retry(messages: list[dict], response: object, version: str) -> bool:
    return bounded_finalization_retry_reason(messages, response, version) is not None


def move_existing_budget_inside_plan(content: str) -> tuple[str, bool]:
    match = re.match(
        r"(?is)^(.*?<plan>.*?)(</plan>)(\s*(?:预算汇总|budget summary)\s*[:：].*?)\s*$",
        str(content or ""),
    )
    if match is None:
        return content, False
    plan_prefix, closing_tag, budget_suffix = match.groups()
    if "预算汇总" in plan_prefix or "budget summary" in plan_prefix.lower():
        return content, False
    repaired = f"{plan_prefix.rstrip()}\n{budget_suffix.strip()}\n{closing_tag}"
    return repaired, True


def source_preserving_plan_repair(messages: list[dict], content: str) -> tuple[str, str | None]:
    moved_content, move_applied = move_existing_budget_inside_plan(content)
    if move_applied:
        return moved_content, "budget_summary_moved_inside_plan"
    text = str(content or "")
    lower = text.lower()
    if "<plan>" not in lower:
        return text, None
    plan_start = lower.find("<plan>")
    closing_start = lower.find("</plan>", plan_start)
    plan_end = closing_start if closing_start >= 0 else len(text)
    plan = text[plan_start:plan_end].rstrip()
    budget_match = re.search(
        r"(?im)^\s*(?:\*\*)?(?:预算汇总|budget summary)(?:\*\*)?\s*[:：]",
        plan,
    )
    repeated_arithmetic = None
    if budget_match is not None:
        repeated_arithmetic = re.search(
            r"(?:\+\s*\d+(?:\.\d+)?){8,}\s*$",
            plan[budget_match.start():],
        )
    if closing_start >= 0:
        if repeated_arithmetic is None:
            return text, None
        repaired_plan = f"{plan[:budget_match.start()].rstrip()}\n预算汇总：现有草稿未形成可靠合计。\n</plan>"
        suffix = text[closing_start + len("</plan>"):]
        return f"{repaired_plan}{suffix}", "repeated_arithmetic_budget_trimmed"
    observed_days = {int(value) for value in re.findall(r"(?im)^\s*Day\s+(\d+)\s*:", plan)}
    user_query = next(
        (str(get_field(message, "content") or "") for message in messages if get_field(message, "role") == "user"),
        "",
    )
    expected_days = travel_calendar_days(user_query)
    day_coverage_complete = bool(observed_days) and (
        expected_days is None or set(range(1, expected_days + 1)).issubset(observed_days)
    )
    if not day_coverage_complete:
        return text, None
    if repeated_arithmetic is not None:
        plan = plan[:budget_match.start()].rstrip()
        plan = f"{plan}\n预算汇总：现有草稿未形成可靠合计。"
        return f"{plan}\n</plan>", "repeated_arithmetic_tail_trimmed_and_closed"
    return f"{plan}\n</plan>", "complete_day_plan_closed"


def source_preserving_finalizer_marker(version: str, reason: str) -> dict:
    payload = {
        "card": "source_preserving_deterministic_serializer",
        "event": "post_action_guard",
        "reason": reason,
        "source_facts_only": True,
        "deterministic_text_repair": True,
        "version": version,
        "write_action_injected": False,
    }
    return {
        "role": "system",
        "content": f"{PERSIST_ACE_PREFIX} {json.dumps(payload, ensure_ascii=False, sort_keys=True)}",
    }


def extractor_safe_move_marker(version: str) -> dict:
    payload = {
        "card": "extractor_safe_serializer",
        "event": "post_action_guard",
        "reason": "budget_summary_moved_inside_plan",
        "source_text_only": True,
        "text_rewrite_applied": True,
        "version": version,
        "write_action_injected": False,
    }
    return {
        "role": "system",
        "content": f"{PERSIST_ACE_PREFIX} {json.dumps(payload, ensure_ascii=False, sort_keys=True)}",
    }


def bounded_finalization_retry_marker(
    version: str,
    reason: str = "incomplete_plan_generation_retry",
) -> dict:
    payload = {
        "card": "bounded_finalization_card",
        "event": "post_action_guard",
        "reason": reason,
        "version": version,
        "write_action_injected": False,
    }
    if reason == "missing_budget_summary_retry":
        instruction = (
            "[PERSIST-ACE extractor-safe finalization repair] The immediately preceding assistant draft contains a "
            "complete <plan> block but the benchmark extractor would lose its budget summary. Copy that same itinerary "
            "without adding days, tools, places, or availability. Return only one corrected <plan>...</plan> block and "
            "insert exactly one compact line beginning `预算汇总：` immediately before </plan>. Reuse only prices in "
            "the draft; if an exact total cannot be derived, label it as an estimate rather than omitting the line. "
            "Do not call tools."
        )
    else:
        instruction = (
            "[PERSIST-ACE bounded finalization repair] The previous final-plan generation was incomplete before its "
            "closing tag and is discarded. Using only evidence already visible in this trajectory, rewrite the complete "
            "itinerary once. Do not call tools. Include every required travel day, but keep each day compact: one line "
            "per transport, hotel, meal, and attraction item. Keep the entire answer under 2600 Chinese characters, "
            "emit exactly one <plan>...</plan> block, close </plan>, and put exactly one compact line beginning "
            "`预算汇总：` immediately before </plan>. Do not invent unavailable facts."
        )
    return {
        "role": "system",
        "content": f"{PERSIST_ACE_PREFIX} {json.dumps(payload, ensure_ascii=False, sort_keys=True)}\n{instruction}",
    }


def persist_ace_message(messages: list[dict], version: str = "v2_safe_escape") -> dict | None:
    markers = [
        message
        for message in messages
        if get_field(message, "role") == "system"
        and str(get_field(message, "content") or "").startswith(PERSIST_ACE_PREFIX)
    ]
    events = completed_tool_events(messages)
    if len(events) < 4 or len(events) < 2:
        return None
    left, right = events[-2:]
    recoverable = bool(
        left["name"].startswith("query_")
        and left["signature"] == right["signature"]
        and left["result"]
        and left["result"] == right["result"]
        and sum(len(str(get_field(message, "content") or "")) for message in messages) <= 70000
    )
    if not recoverable:
        return None
    if not markers:
        payload = {
            "event": "typed_ace",
            "version": version,
            "family": "exact_duplicate_evidence_reuse",
            "tool": right["name"],
            "call_index": len(events),
            "guard": "one_shot",
        }
        if version == "v1_finish_or_escape":
            instruction = (
                "The immediately repeated read-only query returned exactly the same evidence. "
                "Reuse that cached evidence and do not issue the same call again. If a requirement "
                "is still unresolved, change a constraint or switch to a different relevant query tool; "
                "otherwise finish the travel plan. Do not invent unavailable facts."
            )
        else:
            instruction = (
                "The immediately repeated read-only query returned exactly the same evidence. "
                "Reuse that cached evidence and do not issue the same call again. If a requirement "
                "is still unresolved, change a constraint or switch to a different relevant read-only "
                "query tool. Continue normal planning and do not finalize solely because of this reminder. "
                "When eventually writing the final plan, state each budget subtotal once and do not repeat "
                "arithmetic terms. Do not invent unavailable facts."
            )
    elif len(markers) == 1:
        payload = {
            "event": "post_action_guard",
            "version": version,
            "family": "exact_duplicate_evidence_reuse",
            "tool": right["name"],
            "call_index": len(events),
            "guard": "stop_exact_repeat",
        }
        if version == "v1_finish_or_escape":
            instruction = (
                "Post-action guard: the same read-only call repeated again after recovery advice. "
                "Stop this exact query now. Use already observed evidence, make one materially different "
                "read-only query if necessary, or finalize the plan."
            )
        else:
            instruction = (
                "Post-action guard: the same read-only call repeated again after recovery advice. "
                "Stop this exact query now. Reuse observed evidence or make one materially different "
                "read-only query, then continue normal planning. Do not force an immediate final answer."
            )
    else:
        return None
    return {
        "role": "system",
        "content": f"{PERSIST_ACE_PREFIX} {json.dumps(payload, ensure_ascii=False, sort_keys=True)}\n{instruction}",
    }


def install_persist_ace_patch(version: str) -> list[str]:
    module = sys.modules.get("agent.tools_fn_agent")
    if module is None:
        raise RuntimeError("Travel agent module is not loaded")
    patched = []
    for name, candidate in vars(module).items():
        if not isinstance(candidate, type) or not hasattr(candidate, "_call_llm"):
            continue
        original = candidate._call_llm

        def patched_call(self, messages, tools=None, _original=original):
            if version in ACTION_REPLACEMENT_VERSIONS:
                visible_messages = [message for message in messages if not is_persist_marker(message)]
                response = _original(self, visible_messages, tools)
                calls = proposed_calls(response)
                if version == TRAVEL_STRICT_POSITIVE_TWO_CARD_ACTION_REPLACEMENT_VERSION:
                    decision = travel_strict_positive_two_card_action_replacement_decision(
                        messages, calls
                    )
                elif version in {
                    TRAVEL_ZH_FOLD1_FLIGHT_SCHEMA_ACTION_REPLACEMENT_VERSION,
                    TRAVEL_ZH_FOLD2_FLIGHT_SCHEMA_ACTION_REPLACEMENT_VERSION,
                }:
                    decision = travel_zh_fold1_flight_schema_action_replacement_decision(
                        messages,
                        calls,
                        version=version,
                    )
                elif version == TRAVEL_ZH_FOLD1_TWO_CARD_ACTION_REPLACEMENT_VERSION:
                    decision = travel_zh_fold1_flight_schema_action_replacement_decision(
                        messages,
                        calls,
                        version=version,
                    ) or travel_zh_fold1_restaurant_repeat_loopcut_decision(
                        messages,
                        calls,
                        version=version,
                    )
                elif version == TRAVEL_ZH_FOLD2_HOTEL_EMPTY_BRAND_ACTION_REPLACEMENT_VERSION:
                    decision = travel_zh_fold2_hotel_empty_brand_action_replacement_decision(
                        messages,
                        calls,
                        version=version,
                    )
                elif version in {
                    TRAVEL_ZH_FOLD3_ROUND2_TRAIN_ONLY_ACTION_REPLACEMENT_VERSION,
                    TRAVEL_ZH_FOLD3_FIXED_POSITIVE_ACTION_REPLACEMENT_VERSION,
                    TRAVEL_ZH_FOLD3_RESTAURANT_ALIAS_ONLY_ACTION_REPLACEMENT_VERSION,
                    TRAVEL_ZH_FOLD3_RESTAURANT_PARENTHETICAL_ONLY_ACTION_REPLACEMENT_VERSION,
                    TRAVEL_ZH_FOLD3_FLIGHT_REPEAT_LOOPCUT_ACTION_REPLACEMENT_VERSION,
                    TRAVEL_ZH_FOLD3_DUPLICATE_RESTAURANT_ACTION_REPLACEMENT_VERSION,
                    TRAVEL_ZH_FOLD3_USER_SELECTED_THREE_CARD_ACTION_REPLACEMENT_VERSION,
                }:
                    decision = travel_zh_fold3_round2_action_replacement_decision(
                        messages,
                        calls,
                        version=version,
                    )
                elif version == TRAVEL_EN_FOLD1_SINGLE_VICINITY_ALIAS_ACTION_REPLACEMENT_VERSION:
                    decision = travel_en_fold1_single_vicinity_alias_action_replacement_decision(
                        messages,
                        calls,
                        version=version,
                    )
                elif version == TRAVEL_EN_FOLD2_LATE_ALIAS_ACTION_REPLACEMENT_VERSION:
                    decision = travel_en_fold2_late_alias_action_replacement_decision(
                        messages,
                        calls,
                        version=version,
                    )
                elif version in {
                    TRAVEL_EN_FOLD3_ANCHOR_ACTION_REPLACEMENT_VERSION,
                    TRAVEL_EN_FOLD3_STAGNATION_ANCHOR_ACTION_REPLACEMENT_VERSION,
                }:
                    decision = travel_en_fold3_anchor_action_replacement_decision(
                        messages,
                        calls,
                        version=version,
                    )
                elif version in {
                    TRAVEL_EN_V4_FOLD1_SCHEMA_ALIAS_ACTION_REPLACEMENT_VERSION,
                    TRAVEL_EN_V4_FOLD1_DUPLICATE_RESTAURANT_ACTION_REPLACEMENT_VERSION,
                    TRAVEL_EN_V4_FOLD1_DUPLICATE_ROAD_ACTION_REPLACEMENT_VERSION,
                    TRAVEL_EN_V4_FOLD1_COMBINED_ACTION_REPLACEMENT_VERSION,
                    TRAVEL_EN_V4_FOLD1_RESTAURANT_NAME_NORMALIZATION_ACTION_REPLACEMENT_VERSION,
                    TRAVEL_EN_V4_FOLD1_RESTAURANT_ANCHOR_ACTION_REPLACEMENT_VERSION,
                    TRAVEL_EN_V4_FOLD1_ROUND2_COMBINED_ACTION_REPLACEMENT_VERSION,
                }:
                    decision = travel_en_v4_fold1_action_replacement_decision(
                        messages,
                        calls,
                        version=version,
                    )
                elif (
                    version
                    == TRAVEL_EN_FOLD1_CLEANROOM_HOTEL_LOOPCUT_ACTION_REPLACEMENT_VERSION
                ):
                    decision = travel_en_fold1_cleanroom_hotel_loopcut_decision(
                        messages,
                        calls,
                        version=version,
                    )
                elif (
                    version
                    == TRAVEL_EN_FOLD1_CLEANROOM_THREE_CARD_ACTION_REPLACEMENT_VERSION
                ):
                    decision = travel_en_fold1_cleanroom_three_card_decision(
                        messages,
                        calls,
                        version=version,
                    )
                elif (
                    version
                    == TRAVEL_EN_FOLD2_CLEANROOM_HOTEL_LOOPCUT_ACTION_REPLACEMENT_VERSION
                ):
                    decision = travel_en_fold2_cleanroom_hotel_loopcut_decision(
                        messages,
                        calls,
                        version=version,
                    )
                elif version in {
                    TRAVEL_EN_FOLD2_CLEANROOM_INFERRED_ALIAS_ACTION_REPLACEMENT_VERSION,
                    TRAVEL_EN_FOLD2_CLEANROOM_SAME_COORDINATE_ACTION_REPLACEMENT_VERSION,
                    TRAVEL_EN_FOLD2_CLEANROOM_THREE_CARD_ACTION_REPLACEMENT_VERSION,
                }:
                    decision = travel_en_fold2_cleanroom_round2_decision(
                        messages,
                        calls,
                        version=version,
                    )
                elif version in {
                    TRAVEL_EN_FOLD2_CLEANROOM_ATTRACTION_DETAILS_LOOPCUT_ACTION_REPLACEMENT_VERSION,
                    TRAVEL_EN_FOLD2_CLEANROOM_DUPLICATE_ATTRACTION_ACTION_REPLACEMENT_VERSION,
                    TRAVEL_EN_FOLD2_CLEANROOM_ROUND3_TWO_CARD_ACTION_REPLACEMENT_VERSION,
                }:
                    decision = travel_en_fold2_cleanroom_round3_decision(
                        messages,
                        calls,
                        version=version,
                    )
                elif (
                    version
                    == TRAVEL_EN_FOLD2_CLEANROOM_USER_FROZEN_TWO_CARD_ACTION_REPLACEMENT_VERSION
                ):
                    decision = travel_en_fold2_cleanroom_user_frozen_two_card_decision(
                        messages,
                        calls,
                        version=version,
                    )
                elif version in {
                    TRAVEL_EN_FOLD3_CLEANROOM_DUPLICATE_RESTAURANT_ACTION_REPLACEMENT_VERSION,
                    TRAVEL_EN_FOLD3_CLEANROOM_HOTEL_LOOPCUT_ACTION_REPLACEMENT_VERSION,
                    TRAVEL_EN_FOLD3_CLEANROOM_TWO_CARD_ACTION_REPLACEMENT_VERSION,
                }:
                    decision = travel_en_fold3_cleanroom_round1_decision(
                        messages,
                        calls,
                        version=version,
                    )
                elif version in {
                    TRAVEL_EN_FOLD3_CLEANROOM_ROUND2_FOUR_CARD_ACTION_REPLACEMENT_VERSION,
                    TRAVEL_EN_FOLD3_CLEANROOM_ROUND3_STRICT_FOUR_CARD_ACTION_REPLACEMENT_VERSION,
                }:
                    decision = travel_en_fold3_cleanroom_round2_decision(
                        messages,
                        calls,
                        version=version,
                    )
                elif version in {
                    TRAVEL_EN_FOLD3_V8_K6_DUPLICATE_AROUND_ACTION_REPLACEMENT_VERSION,
                    TRAVEL_EN_FOLD3_V8_K6_HOTEL_LOOPCUT_ACTION_REPLACEMENT_VERSION,
                    TRAVEL_EN_FOLD3_V8_K6_TWO_CARD_ACTION_REPLACEMENT_VERSION,
                }:
                    decision = travel_en_fold3_v8_k6_round1_decision(
                        messages,
                        calls,
                        version=version,
                    )
                elif version in {
                    TRAVEL_EN_FOLD3_V8_K6_SAME_COORDINATE_ACTION_REPLACEMENT_VERSION,
                    TRAVEL_EN_FOLD3_V8_K6_SEARCH_LOOPCUT_ACTION_REPLACEMENT_VERSION,
                    TRAVEL_EN_FOLD3_V8_K6_PARENTHETICAL_ACTION_REPLACEMENT_VERSION,
                    TRAVEL_EN_FOLD3_V8_K6_ROUND2_THREE_CARD_ACTION_REPLACEMENT_VERSION,
                }:
                    decision = travel_en_fold3_v8_k6_round2_decision(
                        messages,
                        calls,
                        version=version,
                    )
                elif version == TRAVEL_EN_FOLD3_V8_K6_RESTAURANT_LOOPCUT_ACTION_REPLACEMENT_VERSION:
                    decision = travel_en_fold3_v8_k6_round3_decision(
                        messages,
                        calls,
                        version=version,
                    )
                elif version == TRAVEL_EN_FOLD3_V8_K6_USER_FROZEN_FOUR_CARD_ACTION_REPLACEMENT_VERSION:
                    decision = travel_en_fold3_v8_k6_user_frozen_four_card_decision(
                        messages,
                        calls,
                        version=version,
                    )
                elif version == TRAVEL_FOUR_CARD_ACTION_REPLACEMENT_VERSION:
                    decision = travel_four_card_action_replacement_decision(
                        messages, calls
                    )
                elif version == TRAVEL_MULTICARD_ACTION_REPLACEMENT_VERSION:
                    decision = travel_multicard_action_replacement_decision(
                        messages, calls
                    )
                elif version == RESTAURANT_ANCHOR_RECOMMENDATION_ACTION_REPLACEMENT_VERSION:
                    decision = travel_restaurant_anchor_recommendation_action_replacement_decision(
                        messages,
                        calls,
                        version=version,
                    )
                elif version == RESTAURANT_ANCHOR_PREREQUISITE_ACTION_REPLACEMENT_VERSION:
                    decision = travel_restaurant_anchor_prerequisite_action_replacement_decision(
                        messages,
                        calls,
                        version=version,
                    )
                elif version == RESTAURANT_GROUNDED_COORDINATE_ACTION_REPLACEMENT_VERSION:
                    decision = travel_restaurant_grounded_coordinate_action_replacement_decision(
                        messages,
                        calls,
                        version=version,
                    )
                elif (
                    version
                    == TRAVEL_EN_FOLD1_DECISION_FORK_COORDINATE_ACTION_REPLACEMENT_VERSION
                ):
                    decision = travel_en_fold1_decision_fork_coordinate_rebinding_decision(
                        messages,
                        calls,
                        version=version,
                    )
                elif (
                    version
                    == TRAVEL_EN_FOLD2_DECISION_FORK_SAME_ROUTE_ACTION_REPLACEMENT_VERSION
                ):
                    decision = travel_en_fold2_decision_fork_same_route_drop_decision(
                        messages,
                        calls,
                        version=version,
                    )
                elif version in {
                    ATTRACTION_TOOL_FAMILY_ACTION_REPLACEMENT_VERSION,
                    IMMEDIATE_ATTRACTION_TOOL_FAMILY_ACTION_REPLACEMENT_VERSION,
                    SPECIAL_SERVICE_ATTRACTION_TOOL_FAMILY_ACTION_REPLACEMENT_VERSION,
                }:
                    decision = travel_attraction_tool_family_action_replacement_decision(
                        messages,
                        calls,
                        version=version,
                    )
                elif version in {
                    FUZZY_LOCATION_CANONICALIZATION_ACTION_REPLACEMENT_VERSION,
                    BUDGET_SAFE_FUZZY_LOCATION_CANONICALIZATION_ACTION_REPLACEMENT_VERSION,
                    IMMEDIATE_FUZZY_LOCATION_CANONICALIZATION_ACTION_REPLACEMENT_VERSION,
                    EXPLICIT_BUDGET_FUZZY_LOCATION_ACTION_REPLACEMENT_VERSION,
                }:
                    decision = travel_fuzzy_location_canonicalization_action_replacement_decision(
                        messages,
                        calls,
                        version=version,
                    )
                elif version == LOCATION_CANONICALIZATION_ACTION_REPLACEMENT_VERSION:
                    decision = travel_location_canonicalization_action_replacement_decision(
                        messages,
                        calls,
                        version=version,
                    )
                elif version == ATTRACTION_GROUNDING_ACTION_REPLACEMENT_VERSION:
                    decision = travel_attraction_grounding_action_replacement_decision(
                        messages,
                        calls,
                        version=version,
                    )
                elif version in ENGLISH_ACTION_REPLACEMENT_VERSIONS:
                    decision = travel_restaurant_alias_action_replacement_decision(
                        messages,
                        calls,
                        version=version,
                    )
                else:
                    decision = travel_hotel_action_replacement_decision(
                        messages,
                        calls,
                        version=version,
                    )
                if decision is not None:
                    apply_travel_action_replacement(response, decision)
                    messages.append(action_replacement_marker(decision))
                return response
            if version in {
                "v4_shopping_aligned",
                "v5_shopping_aligned_completion_guard",
                "v6_shopping_aligned_route_safe",
                "v7_shopping_aligned_bounded_retry",
                "v8_shopping_aligned_batch_guard",
                "v9_shopping_aligned_structure_safe",
                "v10_shopping_aligned_extractor_safe",
                "v11_shopping_aligned_draft_aware",
                "v12_shopping_aligned_serializer_safe",
                "v13_shopping_aligned_semantic_gate",
                "v14_tau_aligned_safe_bundle",
                "v15_tau_source_preserving",
                "v16_tau_serializer_constraint_only",
                "v17_tau_personalized_hotel_backtracking",
                "v18_tau_no_compaction_source_preserving",
                "v19_tau_selective_no_compaction",
            }:
                response = _original(self, messages, tools)
                for _ in range(2):
                    decision = travel_recovery_decision(
                        messages,
                        proposed_calls(response),
                        version=version,
                    )
                    if decision is None:
                        break
                    messages.append(recovery_marker(decision))
                    response = _original(self, messages, tools)
                if version in {
                    "v15_tau_source_preserving",
                    "v16_tau_serializer_constraint_only",
                    "v17_tau_personalized_hotel_backtracking",
                    "v18_tau_no_compaction_source_preserving",
                    "v19_tau_selective_no_compaction",
                }:
                    repaired_content, repair_reason = source_preserving_plan_repair(
                        messages,
                        str(response.choices[0].message.content or ""),
                    )
                    if repair_reason is not None:
                        messages.append(source_preserving_finalizer_marker(version, repair_reason))
                        response.choices[0].message.content = repaired_content
                    return response
                finalization_reason = bounded_finalization_retry_reason(messages, response, version)
                if finalization_reason is not None:
                    if (
                        version in {
                            "v12_shopping_aligned_serializer_safe",
                            "v13_shopping_aligned_semantic_gate",
                            "v14_tau_aligned_safe_bundle",
                        }
                        and finalization_reason == "missing_budget_summary_retry"
                    ):
                        moved_content, move_applied = move_existing_budget_inside_plan(
                            str(response.choices[0].message.content or "")
                        )
                        if move_applied:
                            messages.append(extractor_safe_move_marker(version))
                            response.choices[0].message.content = moved_content
                            return response
                    if (
                        version in {
                            "v11_shopping_aligned_draft_aware",
                            "v12_shopping_aligned_serializer_safe",
                            "v13_shopping_aligned_semantic_gate",
                            "v14_tau_aligned_safe_bundle",
                        }
                        and finalization_reason == "missing_budget_summary_retry"
                    ):
                        messages.append(response.choices[0].message)
                    messages.append(bounded_finalization_retry_marker(version, finalization_reason))
                    response = _original(self, messages, None)
                return response
            if version == "v3_context_dedup":
                visible_messages, marker = prepare_persist_context_dedup(messages)
                if marker is not None:
                    messages.append(marker)
                return _original(self, visible_messages, tools)
            recovery = persist_ace_message(messages, version=version)
            if recovery is not None:
                messages.append(recovery)
            return _original(self, messages, tools)

        candidate._call_llm = patched_call
        patched.append(name)
    if not patched:
        raise RuntimeError("No Travel agent class accepted the PERSIST-ACE patch")
    return patched


def load_travel_runner(travel_root: Path) -> ModuleType:
    ensure_qwen_agent_shim()
    travel_root = travel_root.resolve()
    if str(travel_root) not in sys.path:
        sys.path.insert(0, str(travel_root))
    spec = importlib.util.spec_from_file_location("deepplanning_travel_run", travel_root / "run.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load Travel Planning runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install_request_seed(seed: int | None) -> dict | None:
    if seed is None:
        return None
    module = sys.modules.get("agent.call_llm")
    if module is None or not hasattr(module, "load_model_config"):
        raise RuntimeError("Travel call_llm module is not loaded")
    original = module.load_model_config

    def seeded_load_model_config(model_name: str):
        config = dict(original(model_name))
        extra_body = dict(config.get("extra_body") or {})
        configured_seed = extra_body.get("seed")
        if configured_seed is not None and int(configured_seed) != seed:
            raise RuntimeError(
                f"Configured request seed {configured_seed} conflicts with CLI seed {seed}"
            )
        extra_body["seed"] = seed
        config["extra_body"] = extra_body
        return config

    module.load_model_config = seeded_load_model_config
    return {"seed": seed, "injection": "openai_extra_body_top_level_merge"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--travel-root", type=Path, default=DEFAULT_TRAVEL_ROOT)
    parser.add_argument("--model", default=PRIMARY_MODEL_CONFIG)
    parser.add_argument("--language", choices=("zh", "en"), default="zh")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-llm-calls", type=int, default=400)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--rerun-ids")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--persist-ace", action="store_true")
    parser.add_argument(
        "--persist-ace-version",
        choices=(
            "v1_finish_or_escape",
            "v2_safe_escape",
            "v3_context_dedup",
            "v4_shopping_aligned",
            "v5_shopping_aligned_completion_guard",
            "v6_shopping_aligned_route_safe",
            "v7_shopping_aligned_bounded_retry",
            "v8_shopping_aligned_batch_guard",
            "v9_shopping_aligned_structure_safe",
            "v10_shopping_aligned_extractor_safe",
            "v11_shopping_aligned_draft_aware",
            "v12_shopping_aligned_serializer_safe",
            "v13_shopping_aligned_semantic_gate",
            "v14_tau_aligned_safe_bundle",
            "v15_tau_source_preserving",
            "v16_tau_serializer_constraint_only",
            "v17_tau_personalized_hotel_backtracking",
            "v18_tau_no_compaction_source_preserving",
            "v19_tau_selective_no_compaction",
            LEGACY_ACTION_REPLACEMENT_VERSION,
            ACTION_REPLACEMENT_VERSION,
            ENGLISH_ACTION_REPLACEMENT_VERSION,
            EXPLICIT_BUDGET_FUZZY_LOCATION_ACTION_REPLACEMENT_VERSION,
            SPECIAL_SERVICE_ATTRACTION_TOOL_FAMILY_ACTION_REPLACEMENT_VERSION,
            SPECIAL_SERVICE_RESTAURANT_ALIAS_FOLLOWUP_ACTION_REPLACEMENT_VERSION,
            THREE_STRIKE_RESTAURANT_ALIAS_ACTION_REPLACEMENT_VERSION,
            THREE_DAY_BATCH_ATTRACTION_ACTION_REPLACEMENT_VERSION,
            JI_FOUR_STAR_HOTEL_ACTION_REPLACEMENT_VERSION,
            ATTRACTION_READY_JI_HOTEL_ACTION_REPLACEMENT_VERSION,
            TRAVEL_MULTICARD_ACTION_REPLACEMENT_VERSION,
            TRAVEL_FOUR_CARD_ACTION_REPLACEMENT_VERSION,
            TRAVEL_STRICT_POSITIVE_TWO_CARD_ACTION_REPLACEMENT_VERSION,
            TRAVEL_ZH_FOLD1_FLIGHT_SCHEMA_ACTION_REPLACEMENT_VERSION,
            TRAVEL_ZH_FOLD1_TWO_CARD_ACTION_REPLACEMENT_VERSION,
            TRAVEL_ZH_FOLD2_FLIGHT_SCHEMA_ACTION_REPLACEMENT_VERSION,
            TRAVEL_ZH_FOLD2_HOTEL_EMPTY_BRAND_ACTION_REPLACEMENT_VERSION,
            TRAVEL_ZH_FOLD3_ROUND2_TRAIN_ONLY_ACTION_REPLACEMENT_VERSION,
            TRAVEL_ZH_FOLD3_FIXED_POSITIVE_ACTION_REPLACEMENT_VERSION,
            TRAVEL_ZH_FOLD3_RESTAURANT_ALIAS_ONLY_ACTION_REPLACEMENT_VERSION,
            TRAVEL_ZH_FOLD3_RESTAURANT_PARENTHETICAL_ONLY_ACTION_REPLACEMENT_VERSION,
            TRAVEL_ZH_FOLD3_FLIGHT_REPEAT_LOOPCUT_ACTION_REPLACEMENT_VERSION,
            TRAVEL_ZH_FOLD3_DUPLICATE_RESTAURANT_ACTION_REPLACEMENT_VERSION,
            TRAVEL_ZH_FOLD3_USER_SELECTED_THREE_CARD_ACTION_REPLACEMENT_VERSION,
            TRAVEL_EN_FOLD1_SINGLE_VICINITY_ALIAS_ACTION_REPLACEMENT_VERSION,
            TRAVEL_EN_FOLD2_LATE_ALIAS_ACTION_REPLACEMENT_VERSION,
            TRAVEL_EN_FOLD3_ANCHOR_ACTION_REPLACEMENT_VERSION,
            TRAVEL_EN_FOLD3_STAGNATION_ANCHOR_ACTION_REPLACEMENT_VERSION,
            TRAVEL_EN_V4_FOLD1_SCHEMA_ALIAS_ACTION_REPLACEMENT_VERSION,
            TRAVEL_EN_V4_FOLD1_DUPLICATE_RESTAURANT_ACTION_REPLACEMENT_VERSION,
            TRAVEL_EN_V4_FOLD1_DUPLICATE_ROAD_ACTION_REPLACEMENT_VERSION,
            TRAVEL_EN_V4_FOLD1_COMBINED_ACTION_REPLACEMENT_VERSION,
            TRAVEL_EN_V4_FOLD1_RESTAURANT_NAME_NORMALIZATION_ACTION_REPLACEMENT_VERSION,
            TRAVEL_EN_V4_FOLD1_RESTAURANT_ANCHOR_ACTION_REPLACEMENT_VERSION,
            TRAVEL_EN_V4_FOLD1_ROUND2_COMBINED_ACTION_REPLACEMENT_VERSION,
            TRAVEL_EN_FOLD1_CLEANROOM_HOTEL_LOOPCUT_ACTION_REPLACEMENT_VERSION,
            TRAVEL_EN_FOLD1_CLEANROOM_THREE_CARD_ACTION_REPLACEMENT_VERSION,
            TRAVEL_EN_FOLD2_CLEANROOM_HOTEL_LOOPCUT_ACTION_REPLACEMENT_VERSION,
            TRAVEL_EN_FOLD2_CLEANROOM_INFERRED_ALIAS_ACTION_REPLACEMENT_VERSION,
            TRAVEL_EN_FOLD2_CLEANROOM_SAME_COORDINATE_ACTION_REPLACEMENT_VERSION,
            TRAVEL_EN_FOLD2_CLEANROOM_THREE_CARD_ACTION_REPLACEMENT_VERSION,
            TRAVEL_EN_FOLD2_CLEANROOM_ATTRACTION_DETAILS_LOOPCUT_ACTION_REPLACEMENT_VERSION,
            TRAVEL_EN_FOLD2_CLEANROOM_DUPLICATE_ATTRACTION_ACTION_REPLACEMENT_VERSION,
            TRAVEL_EN_FOLD2_CLEANROOM_ROUND3_TWO_CARD_ACTION_REPLACEMENT_VERSION,
            TRAVEL_EN_FOLD3_CLEANROOM_DUPLICATE_RESTAURANT_ACTION_REPLACEMENT_VERSION,
            TRAVEL_EN_FOLD3_CLEANROOM_HOTEL_LOOPCUT_ACTION_REPLACEMENT_VERSION,
            TRAVEL_EN_FOLD3_CLEANROOM_TWO_CARD_ACTION_REPLACEMENT_VERSION,
            TRAVEL_EN_FOLD3_CLEANROOM_ROUND2_FOUR_CARD_ACTION_REPLACEMENT_VERSION,
            TRAVEL_EN_FOLD3_CLEANROOM_ROUND3_STRICT_FOUR_CARD_ACTION_REPLACEMENT_VERSION,
            TRAVEL_EN_FOLD3_V8_K6_DUPLICATE_AROUND_ACTION_REPLACEMENT_VERSION,
            TRAVEL_EN_FOLD3_V8_K6_HOTEL_LOOPCUT_ACTION_REPLACEMENT_VERSION,
            TRAVEL_EN_FOLD3_V8_K6_TWO_CARD_ACTION_REPLACEMENT_VERSION,
            TRAVEL_EN_FOLD3_V8_K6_SAME_COORDINATE_ACTION_REPLACEMENT_VERSION,
            TRAVEL_EN_FOLD3_V8_K6_SEARCH_LOOPCUT_ACTION_REPLACEMENT_VERSION,
            TRAVEL_EN_FOLD3_V8_K6_PARENTHETICAL_ACTION_REPLACEMENT_VERSION,
            TRAVEL_EN_FOLD3_V8_K6_ROUND2_THREE_CARD_ACTION_REPLACEMENT_VERSION,
            TRAVEL_EN_FOLD3_V8_K6_RESTAURANT_LOOPCUT_ACTION_REPLACEMENT_VERSION,
            TRAVEL_EN_FOLD3_V8_K6_USER_FROZEN_FOUR_CARD_ACTION_REPLACEMENT_VERSION,
            TRAVEL_EN_FOLD2_CLEANROOM_USER_FROZEN_TWO_CARD_ACTION_REPLACEMENT_VERSION,
            TRAVEL_EN_FOLD1_DECISION_FORK_COORDINATE_ACTION_REPLACEMENT_VERSION,
            TRAVEL_EN_FOLD2_DECISION_FORK_SAME_ROUTE_ACTION_REPLACEMENT_VERSION,
        ),
        default="v2_safe_escape",
    )
    args = parser.parse_args()

    model_identity = resolve_model_identity(args.travel_root.parent / "models_config.json", args.model)
    runner = load_travel_runner(args.travel_root)
    seed_patch = install_request_seed(args.seed)
    patched_classes = install_persist_ace_patch(args.persist_ace_version) if args.persist_ace else []
    run_args = argparse.Namespace(
        model=args.model,
        language=args.language,
        workers=args.workers,
        max_llm_calls=args.max_llm_calls,
        output_dir=None,
        _user_output_dir=args.output_root,
        save_intermediate=True,
        start_from="inference",
        rerun_ids=args.rerun_ids,
        verbose=False,
        debug=True,
    )
    runner.setup_paths(run_args)
    reports_dir = run_args.output_dir / "reports"
    trajectories_dir = run_args.output_dir / "trajectories"
    reports_before = len(list(reports_dir.glob("id_*.txt")))
    trajectories_before = len(list(trajectories_dir.glob("*.json")))

    success, result = runner.run_step_inference(run_args)
    reports_after = len(list(reports_dir.glob("id_*.txt")))
    trajectories_after = len(list(trajectories_dir.glob("*.json")))
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "api_model": model_identity["api_model"],
        "model_config_sha256": model_identity["model_config_sha256"],
        "language": args.language,
        "workers": args.workers,
        "max_llm_calls": args.max_llm_calls,
        "request_seed": args.seed,
        "seed_patch": seed_patch,
        "inference_only": True,
        "persist_ace": args.persist_ace,
        "persist_ace_version": args.persist_ace_version if args.persist_ace else None,
        "patched_classes": patched_classes,
        "conversion_run": False,
        "evaluation_run": False,
        "success": success,
        "result": result,
        "reports_before": reports_before,
        "reports_after": reports_after,
        "trajectories_before": trajectories_before,
        "trajectories_after": trajectories_after,
        "missing_report_ids": runner.detect_missing_ids(reports_dir, "id_*.txt", total_ids=120),
        "output_dir": str(run_args.output_dir),
    }
    manifest_path = run_args.output_dir / "raw_inference_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
