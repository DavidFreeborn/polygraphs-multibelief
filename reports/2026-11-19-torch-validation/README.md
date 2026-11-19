# PyTorch validation and CPU timings

The new optional operators execute exact Bayesian inference inside PolyGraphs
using PyTorch tensors. Existing reference operations and historical study
results are unchanged. The C++ worker used below was recovered from the current
`multibelief-results-2026-11-19.zip` archive, version 2. The same bytes are
already published as `reports/current/worker.cpp`; the verification script
uses that preserved source directly and does not add a redundant copy.

## Correctness

The full suite passed: **144 tests passed; 34 skipped**. The skips comprise 32
R/bnlearn checks because R was unavailable and two CUDA checks because there was
no CUDA hardware. CPU inference was executed using Python 3.9.6, PyTorch 2.2.2,
NumPy 1.26.4 and DGL 1.1.2 on macOS arm64.

New tests compare all finite-model joints, marginals and payoffs against both
per-agent and batched exact NumPy implementations with controlled common
evidence. They cover the report's Baseline, Converging, Polarizing and
Factionalizing models, the earlier 32-state example, hypothesis and payoff
decisions, arbitrary state cardinalities, heterogeneous agent CPTs, and
calibrated dependent initial joints. Trust checks compare the full H-source
pairs with the exact reference, including dogmatic priors and deterministic
report likelihoods. Other checks cover directed and duplicate edges,
self-loops, isolated agents, honest private/report identity, reproducible
PyTorch sampling, absence of NumPy round trips in the simulation loop,
Bala-Goyal decisions and trajectories, operator discovery, actual simulations,
metadata, HDF5 snapshots and all five new configurations.

The existing NumPy reference also has independent enumeration tests for the
trust factorization; these were included in the executed full suite.

### Comparison with the archived C++ worker

The unmodified worker regenerated **108 simulations**: all 54 experimental
conditions, two seeds, 60 agents and the full 500-round horizon. The conditions
cover all five models, all three graph families and all three environments,
including both trust-prior variants. The replay restores each saved initial
joint, including the matched-P(G) calibration, and supplies the recorded private
observations and outgoing reports to the actual PyTorch/DGL operators.

All **3,240,000 experiment decisions matched**. Maximum absolute differences:

| Comparison | Maximum error |
| --- | ---: |
| Final posterior probability | 2.407e-13 |
| Final log posterior | 2.729e-12 |
| Sampled P(G)/P(H) means and population standard deviations | 2.853e-14 |
| Sampled source-trust and Brier summaries | 1.816e-14 |

These are checks of final posteriors and sampled population summaries, plus
all individual experiment decisions. They do not claim that every individual
intermediate posterior was exported by the C++ worker. Separate unit tests
check complete intermediate joints and H-source pairs over controlled sequences.

C++ source SHA-256:
`06bc1a53041f45f46510d2a340ea158703c3bf72efaee09cff99a7d145e5c889`.
The archived and published sources match, and source bytes were checked
unchanged after compilation and execution. No existing C++ source was edited.

One semantic difference from the older finite NumPy operations is intentional:
the study's C++ source keeps private observations genuine and corrupts only
outgoing reports. The new tensor operator follows the C++ source. The finite
NumPy comparisons therefore use honest/common evidence, while corrupted-report
semantics are verified both against the existing `ReportedMultiBeliefOp` and
by replaying the C++ runs in all environments. Existing operators retain their
previous behavior.

## CPU performance

The table gives microseconds per round of inference, including publication of
graph marginals, with identical fixed honest observations on bidirectional cycle
graphs. Timings are medians of three batches after five warmup rounds, with one
PyTorch thread; construction and sampling are measured separately in
`benchmark.json`. This benchmark was run separately from the test suite and
C++ regeneration. These are local measurements, not general hardware guarantees.

| Agents | Model | Per-agent NumPy | Batched NumPy | PyTorch | Speedup over per-agent reference |
| ---: | --- | ---: | ---: | ---: | ---: |
| 60 | Baseline | 2503.0 | 104.1 | 108.6 | 23.0x |
| 60 | 32-state factionalization | 3357.3 | 207.7 | 181.0 | 18.6x |
| 60 | Source trust | 6174.1 | 149.2 | 176.7 | 34.9x |
| 1,000 | Baseline | 40092.2 | 342.7 | 222.1 | 180.5x |
| 1,000 | 32-state factionalization | 52478.8 | 949.4 | 431.8 | 121.5x |
| 1,000 | Source trust | 99553.6 | 413.2 | 371.1 | 268.3x |

At 60 agents, batching already removes most interpreter overhead; PyTorch is
slightly slower than batched NumPy for Baseline and trust on this machine, and
slightly faster for the 32-state model. At 1,000 agents, its evidence reduction
also avoids the dense N-by-N matrix used by the finite batched NumPy operator.
No GPU timing or CPU-versus-GPU speedup has been established.

The preserved C++ worker generated all 108 full 500-round simulations in
10.73 seconds in the recorded validation run, including diagnostics, gzip
records and JSON output. This is a full-worker timing, whereas the table above
measures inference only. It does not establish a like-for-like speed ratio
between C++ and PyTorch; no claim of a PyTorch speed advantage over C++ is made.

## Reproduction

From the repository root, using the dependencies in the multibelief environment:

```sh
PYTHONPATH=. python -m pytest tests -q
PYTHONPATH=. python scripts/validate_torch_cpp.py --output /tmp/cpp-validation.json
PYTHONPATH=. python scripts/benchmark_torch_batched.py --output /tmp/benchmark.json
```

The C++ check requires `clang++` with C++17 support and zlib. Pass
`--source /path/to/worker.cpp` to check another preserved copy. Regenerated raw
records are temporary and are deleted after verification. It does not rerun or
replace the existing 100-repetition results archive.

The new operators use PyTorch RNGs for simulation draws and keep NumPy only for
initial configuration/prior compilation. Matching numerical seeds across
backends does not produce matching random streams. Exact replay against common
evidence is used instead. CUDA tests are conditional; execution remains
unverified until run on matching CUDA builds of PyTorch and DGL.

See `tests.json`, `cpp-validation.json` and `benchmark.json` for machine-readable
results, and the [operator guide](../../docs/guide/ops/torch-multibelief.md) for
supported settings, the exact factorization and evidence semantics.
