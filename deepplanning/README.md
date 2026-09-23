# DeepPlanning: Vanilla, OAgents Best-of-4, and StateTrace-EDS-ECA

This folder is a self-contained reproduction package for the DeepPlanning benchmark. It includes the benchmark data, local tool databases, official evaluators, agent code, StateTrace-EDS-ECA, and the OAgents-style Best-of-4 implementation.

The full benchmark is used as **test**: Shopping has 120 cases (50/50/20), and Travel has 120 Chinese plus 120 English cases. No train/dev/test split is created in this package.

Verify the bundled data before running:

```powershell
python verify_release.py
```

## Methods

| Name | Definition |
|---|---|
| `vanilla` | One independent agent rollout per task |
| `oagents_best4` | Four independent rollouts, then list-wise public-trajectory selection |
| `statetrace_dsr` | StateTrace-EDS-ECA with public evidence selection; DeepPlanning schema evidence is used for Travel |

Best-of-4 follows the paper protocol: four independent trajectories are generated first, then a selector chooses one. Official evaluator output is never passed to the selector.

## Requirements

The benchmark data, local tool databases, and official evaluators are already included. You do not need to download DeepPlanning or copy anything from another directory. The only runtime prerequisites are Python 3.10+, the packages in `requirements.txt`, and an API key for the model provider. The package does not bundle or require any private server path.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:DEEPSEEK_API_KEY = "your-key"
$env:DEEPPLANNING_OPENAI_BASE_URL = "https://api.deepseek.com/v1"
```

Set `DEEPPLANNING_MODEL` and `DEEPPLANNING_API_MODEL` to an OpenAI-compatible model. The bundled config contains `deepseek-v4.1-flash` as an example. Never commit an API key. If your network requires a proxy, set `DEEPPLANNING_HTTP_PROXY`; it is optional.

## Run The Full Test

The commands are resumable: existing case outputs are retained. Start with a moderate worker count and increase it only within the provider rate limit.

### Vanilla

Run Shopping levels 1, 2, and 3 with `run_deepplanning_shopping_subset.py`, and run Travel once for each language with `run_deepplanning_travel_inference_only.py`. The complete cohort is 50 + 50 + 20 Shopping cases and 120 cases per Travel language.

For a one-command WSL/Linux run, set the provider key and execute. The script uses only paths relative to this folder:

```bash
export DEEPSEEK_API_KEY='your-key'
./launch_deepseek_vanilla_full.sh
```

On Windows, the same full cohort can be run directly from PowerShell (the `.sh` launchers are optional):

```powershell
$env:DEEPSEEK_API_KEY = "your-key"
$env:DEEPPLANNING_OPENAI_BASE_URL = "https://api.deepseek.com/v1"
python run_deepplanning_shopping_subset.py --shopping-root .\shoppingplanning --model deepseek-v4.1-flash --level 1 --case-ids (1..50) --run-name deepseek_vanilla_L1 --workers 20 --max-llm-calls 400 --trial 1 --orchestration-seed 53403 --allow-inference-failures
python run_deepplanning_shopping_subset.py --shopping-root .\shoppingplanning --model deepseek-v4.1-flash --level 2 --case-ids (1..50) --run-name deepseek_vanilla_L2 --workers 20 --max-llm-calls 400 --trial 1 --orchestration-seed 53403 --allow-inference-failures
python run_deepplanning_shopping_subset.py --shopping-root .\shoppingplanning --model deepseek-v4.1-flash --level 3 --case-ids (1..20) --run-name deepseek_vanilla_L3 --workers 10 --max-llm-calls 400 --trial 1 --orchestration-seed 53403 --allow-inference-failures
python run_deepplanning_travel_inference_only.py --travel-root .\travelplanning --model deepseek-v4.1-flash --language zh --workers 20 --max-llm-calls 400 --seed 53403 --output-root .\travel_runs\deepseek_vanilla
python run_deepplanning_travel_inference_only.py --travel-root .\travelplanning --model deepseek-v4.1-flash --language en --workers 20 --max-llm-calls 400 --seed 53403 --output-root .\travel_runs\deepseek_vanilla
```

### StateTrace-EDS-ECA

Run `run_deepplanning_eds_eca.py` after the Vanilla anchor exists. The supplied launcher already points to the relative Vanilla artifact and passes the model directory correctly. For Travel the adapter is enabled with:

```text
--anchor-model-slug deepseek-v4.1-flash --deepplanning-adapter
```

The core selector remains StateTrace-EDS-ECA. The adapter exposes only public DeepPlanning structure and grounding evidence; it does not read evaluator labels, gold state, or hidden requirements.

### OAgents Best-of-4

Generate four independent candidate directories with distinct request seeds, then select:

```powershell
python select_best_of_4_deepplanning.py --root . --cohort travel-zh --workers 20
python select_best_of_4_deepplanning.py --root . --cohort travel-en --workers 20
python select_best_of_4_deepplanning.py --root . --cohort shopping --workers 20
```

The included `launch_deepseek_best4_full.sh` shows the four-candidate layout and is usable under WSL/Linux. The Python entry points are portable across Windows, Linux, and macOS.

```bash
export DEEPSEEK_API_KEY='your-key'
./launch_deepseek_best4_full.sh
python select_best_of_4_deepplanning.py --root . --cohort shopping --workers 20
```

After the Vanilla anchor is complete, run the StateTrace pipeline:

```bash
./launch_deepseek_eds_eca_full.sh
```

## Official Evaluation And Metrics

Put official summaries in this layout:

```text
results/<method>/shopping/**/summary_report.json
results/<method>/travel_zh/evaluation_summary.json
results/<method>/travel_en/evaluation_summary.json
```

Use method names `vanilla`, `oagents_best4`, and `statetrace_dsr`, then run:

```powershell
python report_metrics.py --results .\results --output .\results\metrics.json --plot .\results\deepplanning_metrics.png
```

The report follows the provided figure exactly. The comparison rows are `vanilla`, `oagents_best4`, and `statetrace_dsr`; the columns are Travel `CS Score`, `PS Score`, `Comp Score`, `Case Acc.` and Shopping `Match Score`, `Case Acc.`. Travel values are averaged over Chinese and English cohorts. Shopping values are aggregated from official matched-product and successful-case counts.

Generate both the JSON result, the six-column CSV table, and the grouped figure:

```powershell
python report_metrics.py --results .\results --output .\results\metrics.json --csv .\results\metrics.csv --plot .\results\deepplanning_metrics.png
```

The three method directories must contain official summaries before this command is run:

```text
results/vanilla/shopping/**/summary_report.json
results/vanilla/travel_zh/evaluation_summary.json
results/vanilla/travel_en/evaluation_summary.json
results/oagents_best4/...
results/statetrace_dsr/...
```

The script intentionally fails when a method or cohort is missing, so an incomplete run cannot be reported as a complete comparison.

## Reproducibility Rules

- Keep model, request seed, candidate count, selector seed, and worker policy fixed across methods.
- Record API timeout, quota, and transport failures separately; do not count them as algorithm failures.
- Never use official evaluator output during selection.
- This package treats the full cohort as test-only; do not tune prompts or rules from final test scores.
