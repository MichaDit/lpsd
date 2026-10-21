# Local builds and regression tests

GitHub Actions workflow definitions have been removed from `main`. Use a
complete source checkout and an existing Python environment with the package's
`test` dependencies and a working supported C compiler. Optional FFTW/CUDA
runtimes govern the corresponding tests; a skip does not qualify a runtime.
The launcher does not install packages, upload reports or run remote jobs.

```sh
python scripts/validate_local.py
```

The same command works in a configured Windows PowerShell environment. Use
the Python executable belonging to the intended environment. `CC` and the
existing native-build settings remain available; the launcher does not select
or install a compiler for you. Record the resulting native build metadata.

An explicit local report parent and per-stage timeout are optional:

```sh
python scripts/validate_local.py --output-parent ./local-results --stage-timeout 3600
```

Each invocation creates a fresh retained child directory; it does not
overwrite a preceding report. The stages are the reference build, fast build,
full `test_fast` suite, and upstream `test` suite. A failed or timed-out stage
stops the launcher with a nonzero exit code. Test JUnit files distinguish
passes, failures, errors, skips and expected failures. A zero exit code is a
result for that host, not a claim about other operating systems or CPUs.

The equivalent manual commands, with dependencies already installed, are:

```sh
python build_native.py
python -m lpsd_fast.build
python -m pytest -o addopts= -o junit_family=legacy -q -ra --tb=short test_fast
python -m pytest -o addopts= -o junit_family=legacy -q -ra --tb=short test
```

`junit_family=legacy` preserves the existing per-test recorded properties
without the incompatible xUnit2 warning. This does not relax an assertion.
Short tracebacks preserve the original error and JUnit when a failing test
has already closed a memory mapping: formatting its array contents in a
long traceback can otherwise crash the test process. The launcher has a
small subprocess regression for this failure-reporting path.
Build/runtime changes and unit-test results belong to the same source and
binary hashes. Do not label a missing dependency, partial run or expected
failure as an executed passing test.

The [additional local platform validation](cross-platform-validation-20261021.md)
records execution under Wine and QEMU, cross-compilation for macOS and
Windows ARM64, and the limits of those checks. Emulated timings do not
replace measurements on the physical target hardware.

## Physical benchmark follow-up

Run the portable benchmark separately, after regression validation. Use the
existing allocation/dataset budgets; do not inherit an unrelated small test
supervisor ceiling. For the affected ARM cases these remain 0.375/0.75 GiB
RAM and 2 GiB disk respectively, with the requested two-thread baseline.
Windows retains its separate 32-GiB allocation and 16-GiB dataset budgets.
The old 1B restart case is still excluded from the routine continuation.

From the current checkout, a small local comparison is:

```sh
python -m lpsd_benchmark run --statistics corrected --sizes 100000 --point-factors 1 2 --storage both --profile performance --threads 2 --max-hours 0.5
```

Append the existing device-specific `--ram-gib` and `--disk-gib` values.
All requested cases must be examined: an admission skip stays a skip. The
SBC's small effective budget may not admit all disk smoke cases. This is not
fixed by reporting a skip as successful.

For controlled comparisons of 1/2/4 workers, use separate runs with identical
source, library, input, output count, statistics, storage and budgets. Do not
infer a device-wide optimum from a single 1M measurement or override an
explicit thread cap. Compare both compute time and whole-process time.

Logs, JUnit and the local summary may contain environment, test or filesystem
information and remain local. They are **not** the portable report's privacy
allowlist. Return the already sanitized portable `results.json`, or
`results.public.json` when using `--save-spectra`, for public benchmark exchange.
See [maintenance findings and exact validation scope](maintenance-20261015.md).
