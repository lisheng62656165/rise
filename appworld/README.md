# Faithful v3 and Shared-A Best-of-4 on APPWorld

This folder is a self-contained release for three directly comparable methods:
Vanilla, StateTrace-EDS-ECA Faithful v3, and the modified official OAgents
parallel Best-of-4 selector with the same Vanilla candidate A. The main method
is Faithful v3; OAgents is included only for the final comparison. It includes
the encrypted APPWorld bundle, the APPWorld wheel, and Linux/Python 3.12
dependency wheels. It does not need the research repository, old server paths,
old outputs, or a separate dataset download.

You need Python 3.12 with `venv`, an OpenAI-compatible API key, model access and API budget. A ChatGPT web subscription or Codex login is not an API key. Linux/WSL2 x86_64 is the validated target. No GPU is required.

## Codex instruction

Give another Codex session this folder and say:

> Read README.md and AGENTS.md. Use only the APPWorld data, wheel, and dependency files inside this folder; do not clone another repository or download another dataset. Use the ChatGPT/OpenAI-compatible model I specify with OPENAI_API_KEY from the environment. Run the offline smoke checks, then run the complete test_normal and test_challenge with `run.py --method both` and a fresh output directory. This must compare Vanilla, Faithful v3, and modified official OAgents parallel Best-of-4 with shared Vanilla A. Do not use evaluator labels in prompts, do not call a partial run complete, and run `summarize.py` at the end. Report exactly Test-N TGC, Test-N SGC, Test-C TGC, and Test-C SGC for all three methods plus pairwise deltas.

## Install

```bash
python3.12 bootstrap.py
.venv/bin/python -m pytest -q
.venv/bin/python scripts/smoke_environment.py
# Optional fake-API integration test; no API cost.
.venv/bin/python tests/integration_smoke.py
```

On validated Linux x86_64/Python 3.12, dependencies install offline from
`assets/wheels-linux-py312`; the encrypted dataset is already in this folder
and is unpacked locally into `runtime/`. No project, dependency, or dataset
download is needed on that target. Python itself and an API credential/model
account are system prerequisites. Do not publish `runtime/`. The checks
confirm test_normal=168 tasks/56 scenarios and test_challenge=417 tasks/139
scenarios.

## Model configuration

```bash
export OPENAI_API_KEY='your-api-key'
export OPENAI_BASE_URL='https://api.openai.com/v1'
export MODEL_NAME='gpt-4.1'
export MAX_COMPLETION_TOKENS=4096
export REQUEST_TIMEOUT=180
export MAX_RPM=6
export PYTHONHASHSEED=0
```

Use any OpenAI-compatible model ID; gpt-4.1 is only an example. Actor and
selector use the same model and endpoint. NVIDIA-compatible and other
OpenAI-compatible endpoints work by changing OPENAI_BASE_URL and MODEL_NAME.
For reasoning models, the client uses max_completion_tokens and removes
unsupported sampling/seed fields. Each actor/selector request receives the
recorded per-stage seed when the provider supports seeds; local candidate
ordering also uses a derived seed. Set OPENAI_REASONING=1 if needed. Never
inherit another experiment's .env.

## Full test run

First validate the local environment with the offline checks, then validate the
API with one real task:

```bash
.venv/bin/python run.py --method faithful-v3 --split both --limit 1 --workers 1 --output outputs/faithful-smoke
```

Then run both official test splits and both methods:

```bash
.venv/bin/python run.py --method both --split both --workers 4 --output outputs/full-comparison
```

### Test-C main configuration

The reported APPWorld `test_challenge` main experiment is recorded in
`configs/test_challenge_main.json`. It uses DeepSeek Flash through the
OpenAI-compatible endpoint, `max_steps=50`, selector budget 2048, and these
default seeds:

```text
shared Vanilla Candidate A / OAgents A: 53403
Faithful proposal candidates B/C/D: 64639, 64640, 64641
Faithful selector base: 77113
OAgents selector base: 53403
```

Reproduce the same protocol with a fresh output directory:

```bash
.venv/bin/python run.py \
  --config configs/test_challenge_main.json \
  --method both \
  --workers 4 \
  --output outputs/test_challenge_main
```

The command performs a fresh run and does not promise bit-for-bit reproduction
of the historical table. The table below is the recorded full-coverage
reference result from the source experiment; it is not silently recomputed or
claimed as a new run by this release package. The protocol keeps Vanilla A
shared between Faithful v3 and OAgents Best-of-4, while OAgents B/C/D remain
independent fresh candidates.

| Method | Test-C TGC | Test-C SGC |
|---|---:|---:|
| Vanilla (Faithful Candidate A) | 366/417 = 87.8% | 105/139 = 75.5% |
| OAgents parallel Best-of-4 (shared A; selector=53403) | 367/417 = 88.0% | 108/139 = 77.7% |
| Faithful v3 | 374/417 = 89.7% | 114/139 = 82.0% |
| Faithful v3 - Vanilla | +1.9pp | +6.5pp |
| OAgents Best-of-4 - Vanilla | +0.2pp | +2.2pp |
| Faithful v3 - OAgents Best-of-4 | +1.7pp | +4.3pp |

The reference result covers all 417 Test-C tasks and all 139 scenarios. The
values are supplied by the experiment record associated with source thread
`01a0c4ab-cd33-7aa0-ae88-c01056189e8d`; credentials and raw trajectories are
not part of this release. Any runner may override the seeds with
`--seed-a`, `--proposal-seeds`, `--faithful-selector-seed`, and
`--oagents-selector-seed`.

This runs 168 Test-N tasks and 417 Test-C tasks. Omit `--limit` for the full
run. The same command resumes incomplete tasks by reusing candidates and
checkpoints. It exits with code 2 when files remain incomplete; inspect
`error.json` and repeat the command after fixing the provider or network issue.

To run only the main method, use:

```bash
.venv/bin/python run.py --method faithful-v3 --split both --workers 4 --output outputs/faithful-only
```

In the comparison run, OAgents B/C/D are independent fresh rollouts and never
use Faithful B/C/D. Its selector uses the upstream ORM list-wise prompt in
`third_party/OAgents/ORM_list_wise.yaml`. This is an APPWORLD adaptation of the
official selection prompt plus this package's ReAct loop, not a claim of
bit-for-bit reproduction of the entire OAgents framework.

Workers parallelize tasks. Faithful A/B/C/D stages within one task are sequential because proposals use the selected incumbent. OAgents candidates are executed sequentially within a task by this runner. MAX_RPM is per worker and does not increase an account quota.

## Faithful v3 algorithm

```text
A = fresh Vanilla ReAct trajectory
incumbent = A
for B, C, D:
    compile incumbent public API events
    construct the EDS public-event frontier
    generate a fresh proposal using structural event credit
    compare incumbent and proposal with a binary public-evidence selector
    keep one trajectory and build preserve/avoid ECA credit
return the final incumbent
```

Implementation: `scripts/run_appworld_state_trace_eds_eca.py` is orchestration; `src/appworld_state_trace_eds_eca_adapter.py` is the APPWORLD event/EDS adapter; `src/appworld_state_trace_eds_eca.py` contains selector, ECA credit and proposal prompts; `src/appworld_mimo.py` contains the ReAct loop; `src/llm_client.py` contains the API client.

The selector receives only the task instruction and public trajectory evidence. Evaluator success, hidden state, gold actions and reward are used only after a trajectory finishes for reporting.

## Metrics

```bash
.venv/bin/python summarize.py --output outputs/full-comparison
```

The output contains `metrics.md` and `metrics.json` with:

| Method | Test-N TGC | Test-N SGC | Test-C TGC | Test-C SGC |
|---|---:|---:|---:|---:|
| Vanilla | computed | computed | computed | computed |
| Faithful v3 | computed | computed | computed | computed |
| OAgents parallel Best-of-4 (shared A) | computed | computed | computed | computed |
| Faithful v3 - Vanilla | pp | pp | pp | pp |
| OAgents Best-of-4 - Vanilla | pp | pp | pp | pp |
| Faithful v3 - OAgents Best-of-4 | pp | pp | pp | pp |

The comparison output contains Vanilla, Faithful v3, and OAgents values. The
four requested columns are Test-N TGC, Test-N SGC, Test-C TGC, and Test-C SGC.
TGC is successful official evaluated tasks divided by tasks. SGC is scenarios
whose three official tasks all succeed divided by complete scenarios. Incomplete
scenarios are excluded and never treated as successes. Full coverage is
Test-N 168/168 tasks and 56/56 scenarios; Test-C 417/417 tasks and 139/139
scenarios. Deltas use unrounded ratios. It also prints all three pairwise
deltas. When a run is incomplete, every method is scored only on the common
completed task IDs, and missing scenarios are omitted from SGC.

For a complete run, `metrics.md` is the report to copy into the paper table;
it contains exactly these four columns for Vanilla, Faithful v3, and OAgents
parallel Best-of-4 (shared A), followed by the three pairwise delta rows.

## Reproducibility and release rules

The main Test-C defaults are A=53403, proposals=64639,64640,64641,
Faithful selector=77113, OAgents selector=53403; max steps=50 and selector
budget=2048. The manifest records model, endpoint, seed and task IDs but never
the API key. Changing model, seeds or budget requires a new output directory.
The benchmark order is the official order; OAgents display order is derived
from its recorded selector seed.

APPWORLD data is distributed as an encrypted bundle under its additional sharing requirement. Do not publish runtime/, extracted app source, task descriptions, raw trajectories, .env, .venv, outputs or keys. Keep the encrypted bundle and wheel. See NOTICE.md and VALIDATION.md.
