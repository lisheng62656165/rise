# Sources and Third-Party Notices

## STATE-Bench

- Upstream: https://github.com/microsoft/STATE-Bench
- Bundled source: `benchmark/`, version 0.8.1, upstream revision `5644b18`.
- License: MIT, Copyright (c) 2026 STATE-Bench contributors. Original license is preserved at `benchmark/LICENSE`.
- Includes domain tools, task definitions, task-specific initial environments, simulator, judges and UX v27 scorer. The project-specific per-domain 70/30 train/dev split is included alongside the official 50-task Test split.
- The top-level OpenAI-compatible agent adapter implements the bundled benchmark agent interface. We do not claim Microsoft endorses the StateTrace methods or their results.

## OAgents / Scaling Test-time Compute for LLM Agents

- Reference implementation: https://github.com/OPPO-PersonalAI/OAgents
- Source revision: `027f2c4579ee7e7767bfe54c66df48a902d43e98`.
- Official prompt path: `OAgents/src/oagents/prompts/ORM_list_wise.yaml`; original bytes are bundled at `third_party/OAgents/ORM_list_wise.yaml`.
- OAgents is Apache-2.0 licensed. The complete upstream license is at `third_party/OAgents/LICENSE`; it is not relicensed under the top-level MIT license.
- The prompt and its evaluated string are unchanged. `oagents_official_orm.py` uses the official trajectory wrapper and final `you can start!`, fixed candidate order, plain JSON response, temperature 0 and max_tokens 2048. No OAgents installation is needed.
- The local runner, public conversation serialization, provider adapter, strict JSON validation and bounded same-input retries are StateBench integration code. They match this project's corrected official ORM reselection, not the upstream CodeAgent runtime or the paper's original benchmark tables.
- The former `OAGENTS_BON_LISTWISE_INSTRUCTION` helper remains unused by the new Best-of-4 entrypoint. Old selected outputs must not be reused as official ORM results.

## StateTrace Experiment Source Mapping

The release removes historical runners and machine-specific launchers while retaining the core functions from the project experiment implementation:

| Released file | Experimental responsibility preserved |
| --- | --- |
| `run_statetrace_eds_eca_paper.py` | Minimal A/B'/C'/D' driver corresponding to the EDS-EC path in `run_state_trace_iterative_dsr.py` |
| `state_trace_eds_ec.py` | Paper-aligned binary selection instruction, deterministic credit decision, credit state and frontier functions; not the original MiMo joint-credit protocol |
| `state_trace_event_scaling.py`, `state_trace_eds_r.py` | Public event analysis, hierarchical event representation and generation guidance |
| `state_trace_sdcr.py`, `state_trace_dor.py`, `trace_ir.py` | Public transactions, disagreement and trace normalization |
| `event_shapes.py` | Original argument-shape helper, extracted without the unrelated v2 method |
| `state_trace_pool_select.py` | Public evidence packet for EDS binary selection; old Best4 helper not invoked |
| `oagents_official_orm.py` | `select_oagents_official_best4.py` build_messages, parse_choice and same-input retry protocol |

Algorithm functions and their dependencies were extracted from the experimental files; CLI packaging, reliable transport, checkpoint/resume and reporting were made portable. Existing output files, credentials and research scores are not bundled. This mapping is not a claim that historical results have been reproduced using the release on a new model.
