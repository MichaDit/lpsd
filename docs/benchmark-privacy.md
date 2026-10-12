# Benchmark privacy and public reporting

**Default: local results only.** Publishing is a separate, explicit choice.
A public GitHub issue is associated with the GitHub account that submits it.
Removing hostnames and paths from a JSON file does not make that issue anonymous
and does not by itself establish compliance with the GDPR/DSGVO.

This document describes the implemented data boundary and the responsibilities
that remain with the operator of a public benchmark collection. It accompanies
the [portable benchmark instructions](portable-benchmark.md).

## Purpose, destination and choice

The purpose of a benchmark report is to compare the numerical results,
execution time and resource use of LPSD and FFTW across software versions and
machine classes. The program generates its own fixed synthetic signals. It
does not read a participant's measurement data or search their documents.

The only implemented result-publication destination is one new public issue
and, when needed for a larger report, up to 15 comments in that same issue in
[MichaDit/lpsd](https://github.com/MichaDit/lpsd/issues). There is no project
upload server, email upload, public paste service or automatic push to the
repository. GitHub allows a user with read access to create an issue;
repository write access is not a requirement for ordinary outside reports.
Repository issue settings and GitHub's own account restrictions still apply.
See the [GitHub create-issue API documentation](https://docs.github.com/en/rest/issues/issues#create-an-issue).

Use the program without a GitHub account when you only want local results.
`run --upload-github --accept-public-upload` selects publication explicitly.
`submit RESULTS.json` alone prepares the exact public body or numbered parts locally;
`submit RESULTS.json --accept-public-upload` additionally attempts publication.
`--github-login` is a separate bootstrap choice that requests an interactive
GitHub browser/device authorization flow. Downloading `gh` is not consent to
start that login flow.

## The public data schema

The uploader constructs a new object using an allowlist. It does not send the
original JSON and try to remove a few suspicious field names. Allowed values
also have numeric bounds, restricted version syntax, fixed enums or SHA256
format checks. Unknown fields and arbitrary text are not carried through.

| Included in the public report | Purpose |
|---|---|
| Schema version, source commit when known, source/binary hashes and clean/modified-source status | Identify which implementation was measured |
| OS family and separate process/host architecture enums | Distinguish native and emulated execution |
| Available CPU/thread counts, CPU quota, total RAM and admitted budgets | Explain resource constraints without collecting the CPU marketing name |
| Restricted Python/NumPy/pandas/SciPy/pyFFTW/LPSD and actual FFTW versions | Keep dependencies and runtime libraries distinguishable |
| Synthetic preset, sample count, seed, sample rate, window/averaging settings | Reproduce the benchmark input and estimator configuration |
| Requested and achieved point counts, storage/FFT layout, thread counts | Describe the work actually performed |
| Explicit smoothing request, admitted limit, executor capacity, task-range/point counts and closed backend/policy enums | Identify the selected matched-FFTW execution configuration without claiming measured active-thread counts |
| Smoothing runtime-reserve bytes and a closed empirical model identifier | Explain extra headroom charged inside the existing allocation budget; not measured RSS |
| Optional native working request in MiB, automatic/admitted/applied byte limits and closed policy | Distinguish requested and actually used native allowances; keep explicit-cap forecast profiles separate without collecting paths or free-form text |
| Wall/CPU times, available phases, input setup, process memory and numeric working-set metadata | Compare completed work and resource use |
| Native scratch-pool initial/final/peak byte capacities, a replacement Boolean and actual plan indices | Describe observed buffer lifecycle without including policy text or claiming RSS savings |
| Closed disk power-reduction option, native ABI, bounded group/energy/endpoint counts and state bytes | Identify the experimental Direct-log route without exposing the FFT matrix or power arrays |
| Live-memory observation Boolean and bounded sample/missing-sample/RSS-fallback counts | Distinguish actual live observations from a later process high-water mark and explicitly count macOS disk probes that conservatively use RSS; one observed sample does not imply continuous coverage or an observed private-memory value |
| Closed status/reason categories, forecasts and timeouts | Preserve incomplete or infeasible cases without disclosing exception text |
| Numeric quality checks, noise roughness, peak widths, comparison summaries and spectrum hashes | Check whether speed results remain numerically comparable |

**Excluded** are hostname, username, account names, email, serial numbers,
machine UUIDs, network addresses, IP/MAC addresses, exact OS build strings,
file/directory names and paths, environment variables, access tokens, command
history, raw logs, exception messages, arbitrary hardware descriptions and
user-supplied measurement data. Raw sample arrays and optional saved spectra
are also excluded. Hashes refer to project code, loaded numerical libraries
and synthetic data; the program does not hash private identity values as a
substitute for removing them.

Smoothing controls use bounded integer fields and closed enums. The optional
`requested_smoothing_workers` records the positive user request;
`smoothing_workers_limit` records the limit after the admitted CPU cap.
`smoothing_pool_capacity` and `smoothing_task_ranges` describe the configured
executor and work partition, not observed simultaneous thread activity.
`smoothing_runtime_reserve_bytes` with model
`incremental_9mib_above_cap4_v1` is an empirical incremental admission value.
It is not a measured physical-memory increase or a universal stack bound.
No paths, thread names or scheduler logs are added. Older reports with missing
smoothing fields keep them absent; the sanitizer does not infer that four
workers were actually used in an older measurement. These fields extend the
existing allowlist without changing its schema identifier or upload consent.

The macOS `live_memory_rss_fallback_samples` field is retained only when both
it and `live_memory_samples` are valid bounded integers and the fallback
count does not exceed the live count. Missing or malformed coverage is not
converted to zero or treated as evidence of continuous monitoring. Direct-log
metadata likewise contains only the selected route, ABI, counts, state bytes
and power-vector Boolean, without FFT or spectral arrays.

Hardware class, timing and software combinations can still be distinctive,
especially when associated with a public account or information posted
elsewhere. The schema is data-minimizing, not a mathematical anonymity claim.

The authoritative schema is
[`lpsd_benchmark/privacy.py`](../lpsd_benchmark/privacy.py). Public output is
size-limited. The existing single-body format is retained whenever it fits.
A larger report uses deterministic gzip plus base64 and, if necessary, up to
16 indexed parts: one issue body and comments in that same issue. Each body is
limited to 60,000 bytes. The 16 MiB input limit and 512-case schema limit are
unchanged; the uploader rejects reports that exceed the part limit before its
noninteractive authentication check or any result transfer. No cases or fields
are silently dropped. Every part identifies
the same full-payload SHA256 and its index/total; concatenating the chunks in
order reconstructs exactly the same allowlisted JSON. It adds no spectra, logs
or private files. The SHA256 detects byte changes; this hash is not proof that an
untrusted contributor ran the claimed experiment on the claimed hardware.

## Local files and installation traffic

`results.json` is the local report. With `--save-spectra`, it retains synthetic
spectra for plotting and each checkpoint also creates a compact
`results.public.json` directly from the allowlist, without spectra. Both files
are replaced atomically, separately. Automatic submission explicitly uses the
compact file; manual submission uses only the file you select, with no silent
sibling-file fallback. The existing 16 MiB reader limit remains in place, so
select `.public.json` when a spectra-bearing local report exceeds it. Submission
reapplies the sanitizer and retains the issue-body size limit. Creating the
compact local file does not authorize upload.

Both input names prepare `results.github-issue.md`, the first proposed public
body, plus numbered `*.part-NN-of-TT.md` files if needed. Review the complete
printed list before manual publication, or run locally first if you want that
review step before opting into automatic upload. The accompanying
`*.request.json` files wrap only each public comment body for GitHub's API.
The uploader prepares all parts and requests before its noninteractive
authentication check or any result transfer. Installation downloads and an
explicitly requested `--github-login` are separate earlier bootstrap steps;
they can occur before the measurements and report preparation.

A multipart transfer is not atomic. A local `*.submission.json` file records
the full payload digest, part count, confirmed issue/comment URLs and the last
part whose acceptance is uncertain. It includes no account details or raw
transport responses. The first claim is exclusive; later checkpoints are
atomic replacements. Existing receipts are retained and never silently reset.
Repeated submission of the same payload reports the saved state without
recreating or continuing the transfer. Check the issue and its part markers
before manually adding missing parts. No uncertain request is retried
automatically. The original explicit public-upload choice also covers these
comments; no extra data category or additional destination is introduced.

Synthetic dataset and FFT workspace files use a private directory under the
original OS/user temporary directory by default. An explicit `--work-dir`
selects their parent instead. The runner removes its owned case directories
and default work directory; it does not remove an explicitly selected parent
or unrelated files. The retained installation and report stay in the printed
benchmark directory. Select local storage outside synchronized folders for
large work files; their paths are not included in the public payload.

The installation also retains a private environment, source tree and
`bootstrap-local.json`. Build sidecars and terminal diagnostics may contain
local compiler paths or other environment-specific details. **Do not attach
the whole installation directory or terminal log to a public issue.** Those
files are outside the public report boundary. If browser login was requested,
the private GitHub configuration may contain credentials when the operating
system credential store is unavailable.

Local-only benchmarking still downloads software. HTTPS requests to GitHub,
Python/uv distributions, PyPI and optionally Zig's distribution service expose
ordinary connection/request metadata to those services. The benchmark's JSON
sanitizer does not control their network logs. The pinned GitHub CLI telemetry
and update checks are explicitly disabled in program-controlled GitHub
subprocesses (`GH_TELEMETRY=0`, `DO_NOT_TRACK=1`, update notifiers disabled);
inherited `GH_DEBUG` is removed to avoid API debug logging. These controls do
not disable GitHub's ordinary authentication, security or service logs. See
the [GitHub CLI environment documentation](https://cli.github.com/manual/gh_help_environment).

## Authentication and submitting without repository write access

An existing `gh` login can submit an issue through the participant's own
account. When both upload flags are present and `gh` is missing, the bootstrap
can download a SHA-pinned private GitHub CLI. It changes the child process's
PATH, not the user's global installation. An explicitly requested new login
uses a private `GH_CONFIG_DIR`; possible Git-credential configuration changes
are redirected to a private file. No `gh auth setup-git`, Git push or SSH-key
upload is needed for reporting.

GitHub CLI's browser login normally stores credentials in the OS credential
store. Its documented fallback is a plaintext configuration file if that
store cannot be used. The benchmark does not force insecure storage. It also
does not request a broad personal access token just for a benchmark report.
See [`gh auth login`](https://cli.github.com/manual/gh_auth_login). Existing
authentication environment variables may be used by `gh`; token values are
not inserted into command arguments, result fields or uploader diagnostics.

The uploader probes login without interactive prompts, suppresses the raw
command output, and sends the prepared body with `gh issue create --body-file`.
It accepts a success receipt only when the returned URL identifies an issue
in the intended repository. Missing credentials or software leave a local
draft with instructions. A network failure after requesting creation can
leave the outcome uncertain; check the repository before retrying. There is
no automatic repeated issue creation.

You can instead sign in to GitHub in your own browser, open
[New issue](https://github.com/MichaDit/lpsd/issues/new), and paste the reviewed
draft. The benchmark performs no authentication or publication for that manual
route. The maintainer can collect or label public reports without granting
contributors code-write permission. A code change belongs in a reviewed PR;
a participant's result belongs in an issue.

## GitHub's separate processing

GitHub processes account and connection data separately, including service
usage, device and IP information described in its
[General Privacy Statement](https://docs.github.com/en/site-policy/privacy-policies/github-general-privacy-statement).
A public issue has an author, a publication time, a URL and potentially
comments or links from other users. Its contents can be viewed, copied,
indexed and redistributed by others. The program does not promise that
editing or deleting an issue removes notifications, caches or independent
copies.

To withdraw a public report, edit or remove content you control and contact
the repository maintainer through an appropriate existing project contact
channel for deletion of the issue or derived records. Do not post additional
sensitive information in a public deletion request. GitHub provides its own
privacy-rights contact and procedures in its privacy statement. Revoking
GitHub CLI authorization is a separate account-security action; deleting a
local benchmark directory does not necessarily revoke a token stored by
GitHub or the OS credential store.

## Operator duties under the GDPR/DSGVO

The [European Data Protection Board's data-protection basics](https://www.edpb.europa.eu/sme/learn-the-basics/data-protection-basics_en)
explain that data can be personal when a person is directly or indirectly
identifiable. Purpose limitation, data minimization, transparency, security
and retention therefore remain relevant even though this technical payload
omits obvious direct identifiers. Public account association is a concrete
reason not to describe the collected issues as anonymous.

The software implements technical measures and an affirmative upload choice.
It cannot determine the collection operator's legal identity, legal basis,
applicable jurisdiction, retention period or downstream processing. Before
advertising an organized collection as GDPR-compliant, its operator must
publish the applicable controller/contact information, purpose and legal
basis, recipients and international-transfer information where relevant,
retention/deletion practice and data-subject-rights information. GitHub's
privacy statement does not replace the operator's own obligations for a
separately organized collection or later exported database.

If consent is the chosen legal basis, the operator must ensure that it is
freely given, specific, informed, unambiguous and withdrawable, and handle
withdrawal and any further processing appropriately. The upload flags alone
are not a complete legal assessment. See the
[EDPB guidance on lawful processing and consent](https://www.edpb.europa.eu/sme/be-compliant/process-personal-data-lawfully_en).

This repository intentionally supplies no claim of blanket “DSGVO certification”.
For a small volunteer project, keeping results local by default, using an
explicit public issue workflow, collecting only necessary structured values
and documenting the remaining operator duties provides a concrete, reviewable
starting point. Any additional collection endpoint, new identifying field or
automated export requires its own review and updated notice before use.
