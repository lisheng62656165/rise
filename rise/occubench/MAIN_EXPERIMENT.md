# Default Main Experiment

This is the adopted OccuBench experiment configuration, not a claim of exact
bitwise reproduction from a new model API run. `run_full_comparison.py` loads
`configs/main_results.json` by default. Explicit CLI options take precedence.
Use a new `--output` directory when changing models or seeds.

| Setting | Adopted value |
|---|---|
| Dataset | Bundled OccuBench `eval_benchmark_solvable.jsonl`, all 382 unique tasks |
| Agent, world model, selector, verifier | `deepseek-v4.1-flash` |
| Environment | E0, full initial context, max 200 steps per trajectory |
| Agent token limit | 16384 per request |
| Agent/world extra body | Empty in recorded source config |
| Agent sampling | No explicit API seed, temperature or top_p; provider defaults |
| Cohort/Python seed | 53403 (`--seed`) |
| EDS-ECA policy/design | `statebench-eds-eca` / `original` |
| EDS-ECA stages | Vanilla A, then fresh B/C/D proposals with accepted incumbent and event credit |
| EDS-ECA selector | Two-way structured `select_and_credit_events`, max_tokens=1200 |
| EDS-ECA selector base seed | 77113 (`--selector-seed`) |
| EDS-ECA candidate ordering seed | base + task_id * 3 + stage_index |
| EDS-ECA request seed | ordering seed + attempt; B/C/D stage_index=1/2/3; first attempt=0 |
| Best-of-4 candidates | Same Vanilla A plus 3 independent raw proposals; no EDS event guidance |
| Best-of-4 selector | Bundled OAgents ORM list-wise prompt, visible task and public traces only |
| Best-of-4 candidate ordering | Deterministic shuffle from cohort seed + task_id; fixed across selector retries |
| Best-of-4 selector request | temperature=0, max_tokens=2048, stream=false, thinking disabled |
| Best-of-4 selector API seed | Fixed 53403 on every request and retry (`--oagents-selector-seed`) |
| Verification | One vote, same model and rubric; invalid responses stay pending |
| Main metric | Pass@1 of the selected trajectory, not oracle pass@4 |
| Avg | Total passed / total valid tasks (micro average), not mean of category percentages |
| Columns | Avg, Agri, Biz, Comm, Edu, Hlth, Ind, Pub, Sci, Tech, Trans |

The `--seed` controls task selection (when fewer than 382 are requested),
Python-side randomness and Best-of-4 candidate ordering. Agent/world generation
requests do not explicitly send an API sampling seed or temperature, so those
outputs can vary by provider. The concurrency defaults of 8 workers and 8 active
selector requests are portable operational defaults, not a claim about
historical batch concurrency.
Change them to match provider quotas. No private endpoint or credential is bundled.
The endpoint must support the configured model, tool calls, selector seed,
and JSON responses. A provider may accept a seed without deterministic output.

## Adopted Results and Provenance

| Method | Passed / valid | Pass@1 |
|---|---:|---:|
| Vanilla (shared A) | 190/382 | 49.74% |
| StateBench-EDS-ECA Original FINAL | 242/382 | 63.35% |
| OAgents Best-of-4, selector seed=53403 | 236/382 | 61.78% |

Source experiment IDs (names only; no dependency on the original machine):

- EDS/A: `occubench-statebench-eds-eca-deepseek-v4.1-flash-382-v1`.
- Four-candidate source: `occubench-oagents-bon4-deepseek-statebench-eds-eca-382-v18-matched`.
- Adopted re-selection: `occubench-oagents-bon4-deepseek-382-selector-seed53403`.

The last run reused the existing four candidates; only selection and necessary
verification ran again. All 382 pairs are valid: EDS wins 58, loses 52, ties
272 against the adopted Best-of-4, a +1.57 percentage point difference.
The earlier Best-of-4 result, 243/382 (63.61%), did not explicitly send a
selector API seed. It remains a historical result and must not be mixed into
this main table. The seed53403 re-selection changed 154 selected candidates;
API and verifier variability mean that change cannot be attributed only to seed.

The public package contains code and data, not historical trajectory outputs.
Running the default entry point regenerates trajectories under this recipe.
Changing seeds is permitted, but creates a new experiment rather than the
historical snapshot. Local archived scores are in
`results/main_selector53403/scores`; the previous local snapshot remains in
`results/occubench_382`.

## Commands

After setup in README.md, with OPENAI_BASE_URL and OPENAI_API_KEY configured:

```bash
python run_full_comparison.py --base-url "$OPENAI_BASE_URL"
```

The default model must exist at that endpoint. To change models or seeds:

```bash
python run_full_comparison.py --model YOUR_MODEL --base-url "$OPENAI_BASE_URL" --seed 42 --selector-seed 43 --oagents-selector-seed 44 --output results/other_model_seed42
```

The final reports are `scores/overall_results.json`, `scores/overall_results.md`,
`scores/category_metrics.json`, and `scores/category_metrics.md` in the output
directory. `experiment_config.json` records effective model/seed settings
without credentials. The runner also writes `cohort_task_ids.txt`, which fixes
the selected task set for resume. Standalone Best-of-4 re-selection supports
`--candidate-source` plus `--selector-seed 53403`; always choose a separate
`--out` directory to preserve the earlier selection.
