# Licenses and provenance

- Original research runner and adapters: Apache-2.0 (see LICENSE).
- APPWorld: https://github.com/stonybrooknlp/appworld, bundled PyPI wheel
  `appworld==0.1.3.post1`, the version used by the research environment.
- Official data bundle: https://s3.us-west-2.amazonaws.com/appworld.dev/data-0.1.0.bundle
  is included unchanged in `assets/`. Dataset/app code is protected material:
  Apache-2.0 WITH an additional requirement to redistribute protected content
  and derivatives ONLY in encrypted form. The wheel contains encrypted apps.
  Do not publish `runtime/`, extracted app code, test task descriptions,
  raw evaluation trajectories, or decrypted datasets. Read the unpacked
  `runtime/data/LICENSE` and `README_BEFORE_SHARING.md` before sharing derivatives.
  This is not an assertion that the entire package has an unqualified Apache license.
- OAgents: https://github.com/OPPO-PersonalAI/OAgents, Apache-2.0.
  `third_party/OAgents/ORM_list_wise.yaml` is the upstream prompt; the local
  selector embeds its `prompt` value. The upstream license is included there.
  Upstream path: `OAgents/src/oagents/prompts/ORM_list_wise.yaml`.
  Downloaded 2026-09-21. The equality is checked by a unit test.
- Research files were copied from the existing APPWorld experiment runner,
  not rewritten from the paper. `src/appworld_state_trace_eds_eca.py` and
  `src/appworld_state_trace_eds_eca_adapter.py` preserve Faithful v3.

## Release changes relative to the research workspace

Relative paths replace private server paths. A bundled data installer, unified
driver, common-task metrics and tests were added. OpenAI request parameter
normalization was added, without changing the Faithful v3 prompts. Candidate
JSON writes are atomic to avoid partial files after interruption. Malformed
OAgents responses still select candidate zero, but are now correctly marked
fallback.

The dagger on Best-of-4 means an APPWorld adaptation: four independent fresh
ReAct runs plus the official OAgents full-trajectory list-wise prompt, using
this project's ReAct execution loop and text serializer. This is NOT a claim
of bit-for-bit reproduction of OAgents' complete planning/execution framework.
There is no oracle selector and no selection using evaluator labels.
