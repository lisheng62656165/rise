# Reporting Avg. Cost Per Task

Average cost per task is one of the four headline metrics. Reporting cost is **optional** but strongly encouraged — without reported cost, your submission will show `$0` cost in `metrics.json` and cannot be compared on the cost axis. Other metrics (completion, reliability, UX) still score correctly.

## Report cost from your agent

Providers and SDKs report token buckets differently, so STATE-Bench does not compute cost from token counts. Your custom agent should calculate cost using the accounting model that is correct for your provider, then report the dollar amount:

```python
self.add_cost_usd(cost_usd)
```

Use categories when you want a breakdown in metrics:

```python
self.add_cost_usd(agent_call_cost_usd, category="agent_turn")
self.add_cost_usd(memory_build_cost_usd, category="memory_ingestion")
self.add_cost_usd(memory_lookup_cost_usd, category="memory_retrieval")
```

## Where token counts come from

### OOTB StateBenchAgent (Azure AI Foundry/OpenAI)

Token counts come straight from the Responses API usage metadata. No extra code is required; the agent records usage automatically.

### Custom client + agent

Token reporting is optional telemetry. If your provider exposes useful counts, call `self.add_token_usage(...)` after each provider LLM call returns:

```python
self.add_token_usage(
    input_tokens=input_tokens,
    output_tokens=output_tokens,
    cached_input_tokens=cached_input_tokens,
)
```

- `input_tokens` and `output_tokens` are required for usage telemetry to be recorded.
- `cached_input_tokens` is optional.
- If either input or output tokens are missing, STATE-Bench skips usage telemetry for that call.
- Token counts do not affect cost. Use `self.add_cost_usd(...)` for cost.

See the full signature in [`state_bench/agents/base.py`](../../state_bench/agents/base.py).
