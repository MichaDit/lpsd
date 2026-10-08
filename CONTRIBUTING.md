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

GitHub's commit API preserves trailing message whitespace. A controlled check
with the same parent and file tree found that an extra blank paragraph after
the canonical trailer (`\n\n`) prevented the linked coauthor from appearing;
removing those final newlines restored it. Read messages from `git cat-file commit`,
not formatted `git show --format=%B` output, which adds another newline.
Before API publication, normalize only the final line endings with
`message.rstrip("\r\n")` and verify that the canonical trailer ends the message.
The local audit rejects trailing blank paragraphs while allowing an ordinary
single final newline. This formatting check supplements the rendered-author
check; Git's trailer parser alone accepts the problematic blank paragraph.

The first 43 fork commits, through `b8b921b5cabc808fc9ccab0157f7cc7030fa43f7`,
originally contained `Co-authored-by: Codex <codex@openai.com>`. An audit confirmed that
GitHub's rendered author list on that last commit contained only `MichaDit`.
The repository's `.mailmap` normalizes this old identity for mailmap-aware Git
clients, but a subsequent web check showed that the alias did not repair the
historical GitHub author list.

The original sequence is preserved on
[`archive/pre-linked-coauthors-2026-10-08`](https://github.com/MichaDit/lpsd/tree/archive/pre-linked-coauthors-2026-10-08).
The 43 affected trailers and the following attribution-policy commit were
recreated with linked Codex attribution. Every file tree and the original
upstream ancestor are identical. Recreating the commits changed their IDs and
timestamps; the [old/new mapping and original timestamps](docs/coauthor-repair.json)
make that explicit. The [44-commit web audit](docs/coauthor-audit.json) confirms
that GitHub actually rendered `codex` on each recreated commit page.

The 15 subsequent fast.3 commits needed the terminal-blank-paragraph repair
described above. Their original objects remain on
[`archive/fast3-before-trailer-normalization-2026-10-08`](https://github.com/MichaDit/lpsd/tree/archive/fast3-before-trailer-normalization-2026-10-08).
The [second metadata-only mapping](docs/coauthor-trailer-repair.json) records
their unchanged trees and refreshed timestamps. The
[59-commit fast.3 web audit](docs/coauthor-audit-fast3.json) covers the complete
corrected history through the release commit; later commits must pass the
same audit after publication.

To repeat both the local and public GitHub checks from a complete checkout:

```sh
python tools/audit_coauthors.py --github --output .validation/coauthors.json
```

Omit `--github` for a local trailer and message-format check. The default range
begins after the pinned upstream v1.0.6 commit; upstream history is not required to carry
Codex trailers. A changed page format or failed request fails the web audit
visibly instead of being treated as successful attribution.

## Performance and numerical changes

Measure complete API wall time with reproducible inputs, separate instrumentation
from ordinary timings, and keep all repetitions. Serialize performance runs on
a shared host. Preserve frequency plans, windows, segment starts and the
documented inherited statistics unless an estimator change is explicitly agreed.
Check dtype, normalization, DC plus small signals, tones, spectral nulls and
nonfinite/overflow behavior alongside ordinary noise. See
[the numerical test guide](test_fast/README.md) and
[the FFTW comparison](docs/fftw-comparison.md).
