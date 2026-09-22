# Release Verification

## Executed Checks

Updated 2026-09-22 for the corrected official ORM selector.

Test platform: Windows, Python 3.12.14. A new virtual environment was created and
installed using only this package's `requirements.txt`. The package was copied
to a separate directory, and the full test suite was run there with the new
interpreter, without importing the original research repository.

| Check | Observed result |
| --- | --- |
| Fresh dependency installation | Successful |
| `python -m pip check` | No broken requirements found |
| `python -X utf8 -B -m pytest -q` in release directory | **40 passed**, 298.31 seconds (includes five-run 750-task local fixture) |
| `python -X utf8 -B smoke_test.py` | Passed |
| Bundled tasks and task environments | All **450** loaded; each domain split is 70/30/50 |
| EDS core source comparison | **70 function/class ASTs match** in 7 core modules against current experiment source |
| Official ORM integration comparison | `build_messages`, `parse_choice`, `select_with_retries` ASTs exactly match the completed official reselection implementation |
| Bundled ORM prompt | Byte-identical to upstream prompt in the recorded OAgents source revision |
| `verify_release.py` | 150 Test tasks and 70/30/50 splits validated; no API calls |

The source comparison covers extracted core definitions, not a claim of
byte-identical legacy launchers or identical scientific outcomes. Portable
drivers, transport retries and reporting are separately implemented and tested.

## End-to-End Test Scope

`test_pipeline.py` runs BOTH a three-task smoke and the complete 150-task
Test command-line pipeline from an unrelated
working directory. A local HTTP fixture supplies synthetic Chat Completions
responses through the real OpenAI SDK. The test executes:

- All three bundled StateBench domain environments and real tool handlers.
- Vanilla A, three EDS proposals, binary tool-call selectors, independent
  Best-of-4 and its plain-JSON official ORM selector.
- README-style `OPENAI_API_KEY`, `--model`, and `--base-url` configuration.
- Exact official ORM system prompt, fixed four-candidate input, temperature 0
  and 2048 output-token budget; no selector tools or domain system prompt.
- The bundled official Task and UX scoring code, then result aggregation.
- Public selector packets without hidden state diffs or task-completion scores.
- A second identical pipeline invocation with **zero additional HTTP requests**.

The synthetic full-coverage metrics test checks `749/750` pass@1 and `149/150`
pass^5 on constructed labels; removing a task prevents a formal full-coverage
report. Another five-run test checks all-row UX averaging and distinguishes
80% pass@1 from 0% pass^5. Those values are test fixtures, **not measured model performance**.

Provider tests also cover explicit seed/temperature omission and the
`max_completion_tokens` conversion across a transient request retry.

## Static Audit Limitations

The external `test-research-code/repro_check.py` audit reports README, ENV,
ENTRYPOINT and DATA as OK, but reports SEED as MISSING and RESULTS_CMD as WARN.
These were inspected rather than hidden:

- Its seed regular expression does not recognize `random.Random(seed)` or
  client/API `seed` assignment. This package uses both, with documented offsets.
- Its recursive first-README selection picks `benchmark/README.md` on Windows,
  not the package README. The top-level README passes its command detector.

The static warnings do not replace execution evidence, and fixed API seeds do
not guarantee deterministic responses from a cloud provider.

## Not Claimed

No paid external API experiment or real-model 150-task Test was launched for
this packaging verification. No fresh GPT, MIMO, Nemotron or DeepSeek accuracy
or UX result is claimed. A provider-specific three-task smoke is still needed
before a paid full experiment. Linux/macOS commands are documented but were
not executed on those operating systems in this verification.

Release scanning found no real API-key patterns, original user-specific paths,
private experiment endpoint/server addresses, or external symlinks. Virtual
environments, nested Git metadata and experiment outputs are not included.
Local `__pycache__`/`.pytest_cache` directories are ignored by Git and are not
needed at runtime. Keep credentials, future outputs, virtual environments
and private API logs excluded when publishing.
