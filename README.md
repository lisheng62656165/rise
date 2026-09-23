# RISE

RISE is organized by benchmark. Current releases include self-contained
StateBench and OccuBench implementations:

```text
rise/
  statebench/    # StateTrace-EDS-ECA, official ORM Best-of-4, and Vanilla
  occubench/     # StateBench-EDS-ECA, OAgents Best-of-4, and Vanilla
  <future-dataset>/
```

## Run OccuBench from Codex

For OccuBench, reference this README and
[`rise/occubench/README.md`](rise/occubench/README.md), then run the commands
from `rise/occubench`. The folder includes the complete 382-task dataset,
world-model configurations, verifier, methods, scoring utilities, and offline
Python wheels for Windows/Linux x86_64 with Python 3.12. No separate OccuBench
checkout or dataset download is required. A Python 3.12 interpreter and access
to a model API endpoint/key are still required; a ChatGPT web subscription by
itself is not API access. The OccuBench README documents
the full EDS-ECA, OAgents Best-of-4, and Vanilla comparison and reports the
`Avg, Agri, Biz, Comm, Edu, Hlth, Ind, Pub, Sci, Tech, Trans` metrics.

## Run StateBench from Codex

When using Codex, reference this README and
[`rise/statebench/README.md`](rise/statebench/README.md), then run the
StateBench experiment from the `rise/statebench` directory. The package
contains the StateBench tasks, environments, scoring code, prompts, and the
bundled official ORM prompt. You do **not** need to clone StateBench or OAgents
or download benchmark data.

The complete Test split has 150 tasks: customer support, shopping assistant,
and travel, with 50 tasks per domain. The final comparison is exactly:

1. Vanilla
2. Paper-aligned StateTrace-EDS-ECA
3. Modified official ORM Best-of-4

Use a concrete API model ID, such as `gpt-4.1`, or another model exposed by an
OpenAI-compatible Chat Completions endpoint. “ChatGPT” by itself is not an API
model ID. Keep the API key in an environment variable or a private file
outside this repository; never paste or commit it.

From a fresh checkout on Windows PowerShell:

```powershell
cd rise\statebench
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:OPENAI_API_KEY = "YOUR_API_KEY"
```

Run a three-task smoke test first:

```powershell
python -X utf8 run_experiment.py `
  --model gpt-4.1 `
  --base-url https://api.openai.com/v1 `
  --seeds 42 `
  --tasks-per-domain 1 `
  --workers 3 `
  --output-dir outputs/smoke
```

Then run the complete 150-task Test split. Five seeds are required to report
`pass^5`:

```powershell
python -X utf8 run_experiment.py `
  --model gpt-4.1 `
  --base-url https://api.openai.com/v1 `
  --seeds 42 142 242 342 442 `
  --workers 10 `
  --output-dir outputs/test_five
```

Replace the model ID, base URL, and API-key environment variable for another
OpenAI-compatible provider. The base URL should end at `/v1`, not at
`/chat/completions`. `--workers 20` is possible when the provider allows it;
the service quota and connection limits determine the safe value.

The command writes `comparison.md` and `comparison.json`, plus one metrics
file for each method, under `outputs/test_five`. It resumes successful work
when the same output directory is reused. Do not start two processes against
the same output directory.

## Metrics

- **Task Completion pass@1**: the mean completion rate over the final selected
  trajectory for every scored task. With five complete runs this is over 750
  task-run rows.
- **pass^5**: the fraction of the 150 task keys that succeed in all five
  independent runs. It is not `pass@5` and is not an oracle over Best-of-4
  candidates.
- **UX**: the mean StateBench `ux_score` over the same scored final
  trajectories.

A single complete Test run reports `pass^5 = N/A`; it does not invent that
metric from one run. API failures are reported as incomplete execution and
must be repaired and resumed before claiming a complete comparison.

For the algorithm and scoring details, read
[`rise/statebench/README.md`](rise/statebench/README.md). For a local check
without paid API calls, run:

```powershell
cd rise\statebench
python -X utf8 -B -m pytest -q
python -X utf8 -B verify_release.py
```

The repository does not include API keys, experiment outputs, or private
endpoints. Future datasets should be added as sibling directories under
`rise/`, each with its own README and reproduction instructions.
