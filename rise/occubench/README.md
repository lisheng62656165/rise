# OccuBench: StateBench-EDS-ECA vs OAgents Best-of-4

This folder is a self-contained OccuBench release for the 382-task benchmark.
It contains the benchmark data, Language World Model environment, verifier,
the StateBench EDS-ECA adapter, and the OAgents list-wise Best-of-4 adapter.
All benchmark-specific files are inside this folder.

## Reported Results

All complete rows below use the same 382-task cohort and the same paired
verifier snapshot (deepseek-v4.1-flash).

| Method | Tasks | Passed | Pass@1 |
|---|---:|---:|---:|
| **StateBench-EDS-ECA Original FINAL** | 382 | 242 | **63.35%** |
| **OAgents Best-of-4 adapted** | 382 | 243 | **63.61%** |
| **Vanilla-A** | 382 | 190 | **49.74%** |

All three methods have valid verifier labels for the complete 382-task cohort.
Invalid API, timeout, or malformed verifier responses were retried and are not
counted as failures. The released Vanilla-A score is computed from the same A
anchor trajectories and the same verifier configuration as the paired EDS-ECA
and Best-of-4 scores.

These are the reported evaluation results. Per-task run outputs are generated
under `results/` when the experiment is run and are intentionally not checked
into this source repository.

## Category Breakdown

| Method | Avg | Agri | Biz | Comm | Edu | Hlth | Ind | Pub | Sci | Tech | Trans |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| StateBench-EDS-ECA Original FINAL | 242/382 (63.35%) | 22/31 (70.97%) | 42/63 (66.67%) | 25/36 (69.44%) | 14/32 (43.75%) | 16/21 (76.19%) | 30/55 (54.55%) | 13/25 (52.00%) | 13/16 (81.25%) | 31/50 (62.00%) | 36/53 (67.92%) |
| OAgents Best-of-4 adapted | 243/382 (63.61%) | 23/31 (74.19%) | 45/63 (71.43%) | 20/36 (55.56%) | 17/32 (53.12%) | 13/21 (61.90%) | 29/55 (52.73%) | 17/25 (68.00%) | 12/16 (75.00%) | 34/50 (68.00%) | 33/53 (62.26%) |
| Vanilla-A | 190/382 (49.74%) | 18/31 (58.06%) | 35/63 (55.56%) | 15/36 (41.67%) | 12/32 (37.50%) | 11/21 (52.38%) | 25/55 (45.45%) | 13/25 (52.00%) | 8/16 (50.00%) | 27/50 (54.00%) | 26/53 (49.06%) |

The full-precision machine-readable category report and Markdown table are
generated under the selected run's `scores/` directory. Recompute them from a
completed run with:

~~~bash
python aggregate_occubench_metrics.py --data-root data --scores-dir results/full_comparison/scores --out results/full_comparison/scores/category_metrics.json
~~~

## Codex Run Instructions

In Codex, first quote this README and then use the following instruction:

> Work only inside this OccuBench folder. The complete 382-task dataset is
> already in data/; do not download another benchmark.
> Use the model and OpenAI-compatible endpoint supplied through environment
> variables. Run run_full_comparison.py to execute StateBench-EDS-ECA
> Original, OAgents Best-of-4, and Vanilla on the same 382 tasks. Keep the API
> key in the environment, never print it, and keep invalid verifier/API
> responses pending rather than converting them to failures. Run the category
> aggregation after scoring.

The model is not bundled. A user must provide a model endpoint and API key;
that is the only external service required for a live evaluation. Python 3.12
itself must already be installed because an application folder cannot include
the Python interpreter. No benchmark, world-model data, verifier code, or
Python package download is needed for the supported Windows/Linux Python 3.12
setups. A ChatGPT web subscription alone is not an API credential: use a model
ID enabled for API access, its API endpoint, and a valid API key. Other
OpenAI-compatible providers can be used by supplying their model ID and `/v1`
base URL.

## Setup

The included data and source code are sufficient; no dataset download is
needed. Offline dependency wheels are included for Linux x86_64/Python 3.12
and Windows x86_64/Python 3.12:

~~~bash
python3.12 bootstrap.py
source .venv/bin/activate
~~~

On Windows PowerShell:

~~~powershell
py -3.12 bootstrap.py
.\.venv\Scripts\Activate.ps1
~~~

For another platform or Python version, use Python 3.10+ and install
requirements.txt with pip; this is the only setup path that may need package
index access.

## Full 382-Task Run

Set the endpoint and key through environment variables:

~~~bash
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_API_KEY="REDACTED"
~~~

Then run:

~~~bash
python run_full_comparison.py \
  --model gpt-4o \
  --base-url "$OPENAI_BASE_URL" \
  --api-key-env OPENAI_API_KEY \
  --seed 53403 \
  --selector-seed 77113 \
  --workers 8 \
  --selector-workers 8 \
  --max-tokens 16384 \
  --limit 382
~~~

On Windows PowerShell, set the same variables and run:

~~~powershell
$env:OPENAI_BASE_URL = "https://api.openai.com/v1"
$env:OPENAI_API_KEY = "REDACTED"
python run_full_comparison.py --model gpt-4o --base-url $env:OPENAI_BASE_URL `
  --api-key-env OPENAI_API_KEY --seed 53403 --selector-seed 77113 `
  --workers 8 --selector-workers 8 --max-tokens 16384 --limit 382
~~~

The endpoint may be any OpenAI-compatible service. For a local vLLM server,
use its /v1 URL and the corresponding model name. The runner stores only
redacted configuration metadata.

The run creates results/full_comparison/ with eds_eca, oagents_bon4, and
scores subdirectories. The score directory contains scores.jsonl,
vanilla_scores.jsonl, category_metrics.json, and category_metrics.md. The
summary table is regenerated only after all three methods have valid labels;
pending verifier responses remain pending and are not converted to failures.

The default seeds are `53403` for cohort/rollout sampling and `77113` for
selector ordering/API calls; pass them explicitly when reproducing a reported
run. The run is resumable: EDS-ECA uses stage checkpoints and resume, while the
Best-of-4 runner skips task directories that already have result.json. Never
mix output directories from different models, endpoints, or sampling
configurations.

## Algorithm Definitions

### StateBench-EDS-ECA Original

The first trajectory A is generated as a fresh Vanilla rollout. Each later
stage generates a fresh proposal in a fresh environment. The public trace is
converted into StateBench event records; the selector compares the accepted
incumbent and proposal using only public task text, actions, observations, and
responses. It returns an event-credit decision and the accepted incumbent is
carried forward to guide the next stage. Verifier labels, reward, gold state,
and is_correct are not selector inputs. The final output is the accepted
incumbent after A/B/C/D.

### OAgents Best-of-4

The method reuses the same stage-A anchor and generates three fresh,
independent proposals. An OAgents ORM list-wise selector sees the visible task
instruction and the four public trajectories and selects exactly one
candidate. It does not receive verifier labels, reward, hidden state, or EDS
event credit. The selector prompt is vendored at
vendor/oagents/ORM_list_wise.yaml.

Vanilla in the comparison is the exact same A trajectory used as the
EDS-ECA anchor and one of the four Best-of-4 candidates. A and the accepted
EDS-ECA FINAL are scored together by the paired verifier; Best-of-4's selected
trajectory is scored with that same verifier configuration.

## Public-Information Boundary

Selector packets contain only the visible agent instruction, public agent
actions, public environment observations, and public agent responses.
They do not contain verification_plan, is_correct, verifier feedback, reward,
gold state, candidate source names, or hidden evaluator metadata.
The LWM may expose information through its public observations because that is
the benchmark environment interface; this is not silently replaced with
verifier information.

## Project Layout

~~~text
data/                         # bundled 382-task OccuBench data
occubench/                    # LWM, agent loop, verifier, evaluation CLI
vendor/statebench/            # StateBench event/EDS-ECA core
vendor/oagents/               # OAgents list-wise selector prompt
run_occubench_eds_eca_mimo.py # StateBench EDS-ECA adapter
run_occubench_bestof4.py      # OAgents Best-of-4 adapter
run_full_comparison.py        # complete 382-task entry point
score_bestof4.py              # deferred Best-of-4 verifier scoring
aggregate_occubench_metrics.py# ten-category aggregation
~~~

## Tests and Smoke Checks

Run the unit tests before a live batch (the release was checked with 38 tests):

~~~bash
python -m pytest -q tests
~~~

To inspect all available CLI arguments without calling a model:

~~~bash
python run_occubench_eds_eca_mimo.py --help
python run_occubench_bestof4.py --help
python run_full_comparison.py --help
~~~

## Citation

~~~bibtex
@article{hu2026occubench,
  title={OccuBench: Evaluating AI Agents on Real-World Professional Tasks via Language World Models},
  author={Xiaomeng Hu and Yinger Zhang and Fei Huang and Jianhong Tu and Yang Su and Lianghao Deng and Yuxuan Liu and Yantao Liu and Dayiheng Liu and Tsung-Yi Ho},
  journal={arXiv preprint arXiv:2604.10866},
  year={2026}
}
~~~

## Licenses

OccuBench files retain the included OccuBench Apache-2.0 license.
StateBench-derived modules and OAgents materials retain their respective
license files under vendor/statebench/ and vendor/oagents/.
