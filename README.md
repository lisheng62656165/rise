# RISE

This repository is organized by benchmark:

```text
rise/
  statebench/    # StateTrace-EDS-ECA and official ORM Best-of-4 on StateBench
  <future-dataset>/
```

The current release is the self-contained StateBench implementation in
[`rise/statebench/`](rise/statebench/README.md). Read its README for setup,
the complete 150-task Test command, model configuration, resume behavior, and
the `pass@1`, `pass^5`, and UX reports.

The repository does not include API keys, experiment outputs, or private
endpoints. Future datasets should be added as sibling directories under
`rise/` with their own README and reproduction instructions.
