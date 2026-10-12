# Linux process-tree memory guard follow-up

## Recovered publication

The interrupted conversation had already published fast.9 on `main` through
[`c2d97ef`](https://github.com/MichaDit/lpsd/commit/c2d97efbeb24b54022d9f9caecc85bf6d5063365).
The recovery review verified both contributing PR histories, all 60 source
digests in the final qualification, and the hashes and case counts of all seven
physical-device reports. The complete published tree also passed the later
[eight native/build jobs](https://github.com/MichaDit/lpsd/actions/runs/38248210122)
and [four portable jobs](https://github.com/MichaDit/lpsd/actions/runs/38248210176).
These observations establish recovery of the published work; they do not
establish recovery of any inaccessible, uncommitted temporary files.

The [fast.9 performance report](performance-fast9.md) retains the measured
packed-phase improvements and direct logarithmic power reduction. Its source
revisions, physical-device failures, interruptions and numerical gates remain
historical evidence. This follow-up concerns the benchmark supervisor.

## Additional fault found during recovery

The Linux memory reader could accept a partial process-tree sum as a successful
live observation. It skipped an unreadable child status and treated a missing
child RSS counter as zero. A positive parent reading then satisfied the
supervisor's availability check.

A second part of the same fault affected procfs mounts without
`/proc/PID/task/TID/children`. The old fallback found the worker's namespace PID
through parent relationships, but subsequent probes did not enumerate its
descendants. This condition was observed in the recovery environment, rather
than inferred from a hypothetical older kernel. A live reproducer used a
child with a 64 MiB allocation: the old reader reported only the parent's
memory, omitting the child's allocation and interpreter memory. Selectively
denying access to a known live child's status reproduced the partial-sum fault
independently.

## Corrected reader

The reader distinguishes a readable empty children list, an unavailable
observation, and an existing task without the optional children interface.
It verifies PID/parent relationships and the required counters before
accepting the observed tree. An unreadable live descendant invalidates the
current sample instead of contributing zero bytes.

When the children interface is absent, one procfs inventory supplies verified
parent relationships and reusable status values for the whole probe. The
reader does not rescan every process for each descendant. On systems exposing
the children files, the normal path remains restricted to the owned tree.
Verified process disappearances are distinguished from unreadable live
processes. A zombie/dead state is treated as having released its memory only
when the status also verifies `Threads == 1`. A zombie group leader can still
have live threads: an independent live review reproduced `State: Z`,
`Threads: 2` and a surviving thread holding a 64 MiB allocation. Such a group,
or an unknown thread count, makes the observation unavailable and uses the
existing grace/termination policy. The reader does not invent zero memory
or attempt a separate thread-level measurement for that uncommon state.
Missing anonymous-memory counters on older kernels retain the conservative
RSS fallback.

The supervisor's existing one-second observation grace, RAM/disk budgets,
timeouts and process cleanup policy apply. No estimator, numerical kernel,
frequency plan, precision, output schema or automatic worker limit changes.

These are sampled observations. The Linux kernel documents that
[children enumeration is not an atomic snapshot](https://docs.kernel.org/filesystems/proc.html#proc-pid-task-tid-children-information-about-task-children)
and that status RSS accounting can be asynchronous. The correction rejects
observed incomplete data and fixes missing descendant enumeration; it does
not turn the polling supervisor into an OS-enforced memory sandbox or prove
that every transient peak is observed.

## Validation

The corrected source is
[`09d5385f262988885478a4d47151fcd1dfd39e9c`](https://github.com/MichaDit/lpsd/commit/09d5385f262988885478a4d47151fcd1dfd39e9c),
with tree `479925f5a9cdeaefd9ed350c0eac2d1100bb0711`. Its parent is the
recovered publication `c2d97ef`; the commit retains the Codex coauthor trailer.
The numerical libraries and bootstrap sources are unchanged from that parent.

Local Linux x86-64 validation on the final reader passed **148 targeted tests**
with one intentional macOS-only skip. These cover the new reader, existing
memory-availability behavior and portable benchmark supervision. A separate
set of **14 owned-file and disk-budget tests** also passed. The new checks
exercise both procfs routes, nested descendants, access denials, missing and
invalid counters, PID namespaces, verified process/thread exits, zombie
leaders with surviving threads, and a real owned 64 MiB allocation.

A final supervised smoke completed all **six cases**: simple FFTW, matched
FFTW and native LPSD at 100,000 samples, with 469 and 938 actual output points,
two workers and 1 GiB resource budgets. All recorded result invariants passed.
The supervisor recorded 89 valid live-memory observations and one temporarily
unavailable observation; every case completed with observed live-memory
coverage. These were RAM cases with synthetic white noise, not a new large-data
or disk-performance campaign.

The source-bound [validation record](../benchmarks/results/linux-memory-guard-validation-20261012.json)
keeps the local test counts, final six-case portable smoke and platform CI
separate from the earlier fast.9 performance measurements. The exact source
is tested by the [native/build workflow](https://github.com/MichaDit/lpsd/actions/runs/38284527956)
and [portable workflow](https://github.com/MichaDit/lpsd/actions/runs/38284527904).
All **eight native/build jobs and four portable jobs passed**. Each job's
actual initial checkout was verified as the source commit above.

| Regression environment | Passed | Skipped | Expected failures |
|---|---:|---:|---:|
| Linux x86-64 GCC/Python 3.10, GCC/Python 3.12 and Clang/Python 3.12, each | 2,320 | 5 | 2 |
| Linux ARM64 GCC/Python 3.12 | 2,320 | 5 | 2 |
| macOS ARM64 Clang/Python 3.12 | 1,918 | 407 | 2 |
| Windows x64 private-bootstrap regression subset | 409 | 2 | 0 |

The five native numerical jobs also passed 18 upstream tests each. The three
wheel jobs verified the installed package outside the checkout. Each of the
four portable platforms completed 12 RAM cases and its direct/packed disk
correctness steps. Windows explicitly executed and passed all 24 required
frozen-estimator comparisons, one Bluestein comparison and 28 instrumented
CSD cases. Focused subsets overlap the full suites and are not added as
distinct tests. Platform and optional skips remain recorded; Windows's
private-bootstrap subset is not the complete Linux-oriented suite.

The reproduction uses only owned synthetic processes and allocations. No
physical-device rerun or new spectral speedup is inferred from the supervision
tests. Updated launch commands are in the
[portable guide](portable-benchmark.md#start-from-a-pinned-revision).

## Recovered final device evidence

The concurrent [device publication](https://github.com/MichaDit/lpsd/commit/e193080134a9578f9dd0ca1b1b809e8f64e470e2)
added the complete fast.8 instrument results while this follow-up was being
qualified. Both parent histories and the complete report are retained.
The [physical-device report](device-tests-20261012.md) now has:

| Physical fast.8 campaign | Retained result |
|---|---|
| Windows x86-64 | 131 successful unique cases out of 132; final doubled-point 1B LPSD case interrupted by the host crash |
| ARM64 SBC | Complete matrix: 108 successes, no failures, 24 resource skips |
| ARM64 instrument | Complete matrix: 116 successes, three memory-limit failures, 13 skips |

The instrument's completed prerequisite-corrected regression run passed
2,101 tests, failed two, skipped six and retained two expected failures.
The two actual failures are the already recorded deadline fixture and extreme
CSD comparison, addressed in the later fast.9 source. The recovered cleanup
record reports release at 11:42:57 UTC and fresh verification at 17:15:12 UTC
on 12 October, with test supervisors and owned test-rootfs mounts absent.
All seven current report hashes, case totals and the regression-summary hash
were checked against the updated manifest.

These recovered records qualify the **fast.8** device cohort. They do not
establish a physical-instrument rerun of fast.9. The Windows host bugcheck and
the older invalid doubled-point 1B result still lack enough diagnostic evidence
for a root-cause claim. The completed code corrections do not relabel those
original failed or interrupted results.
