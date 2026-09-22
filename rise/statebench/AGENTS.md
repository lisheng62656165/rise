# Running This Research Package

- Read README.md before executing. This folder is self-contained for StateBench, not AppWorld.
- Use Python 3.12, requirements.txt and run_experiment.py. Do not download another benchmark repository.
- The user supplies the actual API model ID and credential source. Never print secrets or commit credentials, outputs, environments, or private logs.
- If the user only says "ChatGPT", reuse explicitly configured model/provider settings or ask for the missing API model ID and credential source. Do not assume a web subscription supplies API access. Do not clone external repositories, download benchmark data/model weights, or use the original author's local paths. Installing requirements.txt is expected.
- Use a separate three-task smoke directory before a full run. A full Test has 150 tasks; five repeats have 750 scored rows per method.
- Preserve EDS-ECA: Vanilla A, three fresh event-guided proposals, pairwise selection against the accepted incumbent, deterministic public event credit. Never replace it with a four-way selector or keyword score.
- Preserve Best-of-4: four independent fresh trajectories, fixed c1-c4 order, bundled verbatim OAgents ORM prompt, plain JSON index/analysis, temperature 0, max_tokens 2048. No EDS features, domain system prompt, shuffling, or selector tools. Identify it as official-ORM StateBench adaptation, not exact official-runtime reproduction. Never reuse old custom-selector finals or scores.
- Hidden requirements and evaluator output belong only to the harness/offline scoring. Do not use them for online generation, event credit, selection, or prompt tuning on Test.
- Resume the same command/output directory. Do not regenerate successful proposals just because the selector failed. Never silently count API errors as successful completion of the pipeline.
- pass^5 means all five repeated results succeed for a task. Aggregate seeds of the same method only; never mix Vanilla/EDS/Best4 as repeats.
- Five complete runs are required for pass^5; one requested run remains one run with pass^5=N/A. Do not add paid batches without user authorization. A request for all three metrics may use the README's five-run command once model/access are specified.
- Report paid API usage and actual completion honestly; offline fixtures are not scientific results. Do not guarantee improvement or cloud API determinism.
