# Reproduction Checklist

Use the commands in [README.md](README.md); do not substitute historical score files.

1. Install Python 3.12 and `requirements.txt` in a fresh environment.
2. Supply an authorized API key, Chat Completions model ID, and base URL.
3. Run `python -X utf8 -B -m pytest -q` (offline fixtures, not model results).
4. Run `run_experiment.py --tasks-per-domain 1` with a separate smoke output directory.
5. Run the full pipeline without `--tasks-per-domain`; Test must contain 150 tasks per method per seed.
6. Use five distinct run seeds for `pass^5`. For example: 42 142 242 342 442. Candidate seeds within a run are documented in README.
7. Keep identical simulator, selector, judge and provider settings across methods. Judge seed defaults to 88002.
8. Resume the same command after transient failures. Never treat invalid selection as a completed final output.
9. Read `comparison.md`, per-method JSON, and each seed's `score_summary.json`. Do not mix methods as repeated seeds in the aggregator.
10. Archive configuration alongside results. API seed support does not guarantee deterministic cloud inference.

This is a portable implementation of the project's StateBench-adapted protocols, not a claim that an unrun API model reproduces paper numbers. No Test outcomes enter online ECA credit or selectors.

## Upgrading Old Best-of-4 Outputs

The old shuffled, tool-call selector is no longer the baseline. To reselect
existing candidates without generating them again:

1. Create a NEW Best4 output directory. Copy only the old `candidates/` tree
   (containing `candidate_0` through `candidate_3`) into it. Do not copy
   `final/`, scores or reports. Keep the old experiment unchanged.
2. Configure the same provider/model variables as the original run, then run:

```bash
python -X utf8 run_oagents_best_of4.py --output-dir outputs/best4_official --split test --seed 42 --selector-seed 77113 --workers 10 --selector-workers 10
python -X utf8 phase0_score.py --input-dir outputs/best4_official/final --output-dir outputs/best4_official_scores --split test --workers 10 --judge-seed 88002 --with-ux
```

Use the actual original seeds instead of the example values. The runner skips
all existing candidates and selects missing finals with the bundled official
ORM prompt. All four candidates must be present per task to avoid generation
of missing candidates. Old custom-selector finals are rejected, not silently
relabelled. New selected trajectories need new scoring; no old Best4 score
is automatically attached to a different selected trajectory.

## Protocol Boundaries

- EDS: binary model selector and deterministic public event credit; no joint
  model-produced credit and no global four-way EDS selection.
- Best4: verbatim official ORM prompt, fixed c1-c4 order, temperature 0,
  max_tokens 2048, plain JSON, no domain system prompt or event features.
- A model-specific switch omitting temperature/seed or using another token
  parameter is explicit and recorded; disclose this provider adaptation.
- `pass^5` requires five complete repeated Test runs. One run reports only
  pass@1 and UX, with pass^5 marked N/A. Do not report internal Best4 oracle
  selection as pass^5.
