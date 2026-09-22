# Running this artifact with Codex

Read README.md first. Use run.py, not the internal research scripts, as the
public entry point. Work only inside this directory. Never read unrelated
credential files, inherit another experiment's .env, or print API keys.

When asked to run with ChatGPT, use an OpenAI API model ID specified by the
user, OPENAI_API_KEY from their environment, and api.openai.com/v1. A ChatGPT
web subscription is not an API credential. If the model is unspecified, ask
which API model and budget they intend before launching all 585 tasks.

Use Linux/WSL2 with Python 3.12. Run bootstrap.py, pytest, and the environment
smoke first. Then run a separate --limit 1 trial. Do not call that a complete
benchmark. Use a different output directory for the full run. Never silently
replace a failed provider/model with another model. Resume with the same
command; changing --workers is permitted. Report remaining tasks and errors.

Faithful v3 preserves event hierarchy, structural ECA, fresh proposals and
binary selection. OAgents uses independent candidates and the full-trajectory
upstream ORM list-wise selector. Do not substitute the unused generic selector
in run_appworld_best_of4.py. Do not use Faithful B/C/D as OAgents candidates.
Vanilla A is shared when running --method both.

Run summarize.py for the four requested columns. Report full split coverage
and complete scenario counts; never treat missing evaluations as successes
or claim partial results are the official full-set result. Do not tune the
method using test labels. Keep credentials, extracted protected data and
outputs out of Git. Changes of model or budget require a new --output.
