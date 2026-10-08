# Contributing to this optimized fork

Keep original upstream authorship, licensing and source provenance intact.
Changes to the optimized implementation belong in focused commits with the
reason for the change and relevant validation results in the commit body.

## Codex coauthorship

The repository owner requests Codex coauthorship on commits created with
Codex's assistance. End each such commit message with a blank line followed by:

```text
Co-authored-by: Codex <267193182+codex@users.noreply.github.com>
```

The numeric ID and login identify the official [Codex account](https://github.com/codex).
They were checked against GitHub's public user API on 2026-10-08. The human
author/committer remains the account making the commit. Do not add Codex
authorship to inherited upstream commits or work to which it did not contribute.

GitHub requires an email associated with the coauthor's account for linked
attribution; see [GitHub's coauthor documentation](https://docs.github.com/en/pull-requests/how-tos/commit-changes/creating-a-commit-with-multiple-authors).
Verifying the trailer alone does not verify the rendered account/author list.
After publication, inspect the commit page as well.

The first 43 fork commits, through `b8b921b5cabc808fc9ccab0157f7cc7030fa43f7`,
all contain `Co-authored-by: Codex <codex@openai.com>`. An audit confirmed that
GitHub's rendered author list on that last commit contained only `MichaDit`.
The repository's `.mailmap` normalizes this old identity for mailmap-aware Git
clients; it does not rewrite the original commit objects. Historical web-host
attribution must be checked independently and must not be assumed to follow
the mailmap.

## Performance and numerical changes

Measure complete API wall time with reproducible inputs, separate instrumentation
from ordinary timings, and keep all repetitions. Serialize performance runs on
a shared host. Preserve frequency plans, windows, segment starts and the
documented inherited statistics unless an estimator change is explicitly agreed.
Check dtype, normalization, DC plus small signals, tones, spectral nulls and
nonfinite/overflow behavior alongside ordinary noise. See
[the numerical test guide](test_fast/README.md) and
[the FFTW comparison](docs/fftw-comparison.md).
