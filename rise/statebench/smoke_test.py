"""StateTrace-EDS-ECA 核心模块的无网络 smoke test。"""

from state_trace_eds_ec import listwise_selector_tool
from state_trace_event_scaling import argument_shape


assert argument_shape("reservation_id", "opaque") == "identifier"
assert argument_shape("start_date", "2026-01-01") == "temporal"
tool = listwise_selector_tool(2)
assert tool["name"] == "select_listwise_trajectory"
assert tool["parameters"]["properties"]["index"]["enum"] == [0, 1]
assert tool["parameters"]["required"] == ["index", "analysis"]
print("PASS: StateTrace-EDS-ECA paper-aligned selector contract")
