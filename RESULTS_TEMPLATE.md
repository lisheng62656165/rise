# Results template

Run the same command for five independently seeded runs, then fill this table
from `outputs/test_five/<method>_metrics.json` (the pipeline also creates comparison.md):

| Method | Task Completion pass@1 | pass^5 | UX | State Req. | Task Req. |
| --- | ---: | ---: | ---: | ---: | ---: |
| Vanilla | | | | | |
| OAgents official ORM Best-of-4 (StateBench adaptation) | | | | | |
| Paper-aligned StateTrace-EDS-ECA | | | | | |

Do not copy historical numbers from another model, endpoint, judge seed, or
task cohort into this table. Record those settings beside the generated report.

Each method needs exactly 5 x 150 valid task scores for pass^5. State Req. and
Task Req. in the JSON are means, not raw counts. UX uses the bundled official
v27 scorer; its model and seed must be reported, not inferred from the agent.
