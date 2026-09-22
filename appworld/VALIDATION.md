# Validation record

Date: 2026-09-21. Platform: Linux x86_64, Python 3.12.

## Executed successfully

- Copied the release artifact to a standalone directory outside the research
  repository. Created a new venv; no research packages were imported from the
  old venv (only its Python interpreter was used to create a new environment).
- Installed APPWorld 0.1.3.post1 from the bundled wheel and all dependencies.
- Repeated installation in a second standalone directory with `PIP_NO_INDEX=1`
  and pip `--no-index --find-links assets/wheels-linux-py312`. All dependencies
  were installed locally. No dataset/PyPI download was used for this second run.
- Unpacked the official encrypted local data bundle: Test-N 168 task IDs and
  Test-C 417 task IDs. No LFS pointer files or external data paths are needed.
- Ran a real AppWorld public API operation and `world.evaluate()` on one task
  from each split. Both environment/evaluator smoke tests passed.
- `python -m pytest -q`: **15 passed**. Covers event compiler, hierarchy,
  documentation/error distinction, event credit projection,
  selector evidence ownership, no hidden evaluation in selector inputs,
  binary Faithful selection, OpenAI reasoning request normalization, exact OAgents
  upstream prompt equality, shared Vanilla identity and incomplete SGC groups.
- Ran `tests/integration_smoke.py`: two methods, both splits, one task per
  split, real AppWorld/evaluator, fake local API responses, max_steps=1.
  Verified four candidate artifacts, final selectors, exact shared Vanilla A,
  four-column summary, then repeated the same run to verify resume.

## Not claimed

- Fake API integration results are NOT model performance measurements.
- No paid OpenAI/ChatGPT-model full run was launched for this packaging task.
- No new 585-task performance replication was performed. Historical Nemotron
  results are not asserted as the output of this release or of another model.
- Native Windows/macOS and other Python/CPU platforms were not validated.
- Provider availability, quota, long-context support and deterministic seed
  behavior are not guaranteed. Run a paid one-task smoke with your chosen model.
- Passing small execution tests does not guarantee all model-generated programs
  finish without error. The driver preserves failed-task diagnostics and resumes.

## Known protocol details retained

- StateTrace's disagreement detector compares public structural signatures;
  it does not evaluate hidden goal satisfaction.
- On malformed Faithful evidence, fallback selects displayed A, which is not always
  the incumbent because display order alternates. This is retained from the
  source method, not silently changed in the release.
- Four independent OAgents candidates use the research ReAct runner, not the
  entire upstream planning framework; this is why the comparison has a dagger.
- Parallelism is across tasks. OAgents' four within-task independent candidates
  are scheduled sequentially; do not claim upstream parallel wall-clock cost.
- Shared A is reused in the two-method comparison. Neither method sees evaluator
  success when proposing or selecting a trajectory.
