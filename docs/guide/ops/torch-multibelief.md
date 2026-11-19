# Tensor Bayesian operators

`TorchMultiBeliefOp` and `TorchBayesianReliabilityOp` add native PyTorch
inference to this fork. They use the existing DGL graphs, simulation loop,
termination hooks, metadata and HDF5 snapshots. The original Python/R and
batched NumPy operators remain available.

Select an operator in a configuration file:

```yaml
op: TorchMultiBeliefOp
device: cpu
epsilon: 0.05
trials: 1
multibelief.case: baseline
multibelief.stop_on_consensus: false
simulation.steps: 500
```

The five complete examples are `torch-baseline.yaml`, `torch-converging.yaml`,
`torch-polarizing.yaml`, `torch-factionalizing.yaml` and `torch-trust.yaml` in
`examples/multibelief`. Run one using the normal PolyGraphs CLI:

```bash
python -m polygraphs.run -f examples/multibelief/torch-factionalizing.yaml
```

These configurations illustrate the models. Their independently drawn root
priors do not reproduce the report's calibration of initial P(G) across models.
That calibration introduces dependencies and requires restoring the complete
joint, as described below.

## Inference and evidence

The finite operator compiles the configuration and initial joint distributions
once with the independent exact NumPy implementation. It then stores the joint
for all agents as a float64 tensor, including dependencies between beliefs.
Observation likelihoods and marginal indicator matrices are compiled once.
Every simulation step samples, aggregates evidence, normalizes posteriors and
publishes marginals using PyTorch. There are no NumPy conversions in these
steps. Saving a snapshot or metadata can transfer tensors to the CPU.

If an agent receives k successes in n fresh trials, its update is

\[
\log P_{t+1}(z) = \log P_t(z) + k\log p(z)
  + (n-k)\log(1-p(z)) - \log Z.
\]

Here z is a full assignment of the belief variables, p(z) is the observation
probability in that assignment, and Z normalizes that agent's joint. The
state-independent binomial coefficient cancels. No independence assumption is
introduced between an agent's belief variables.

`TorchMultiBeliefOp` supports `baseline`, `meta`, custom finite networks,
and all three `networked-factionalization` cases. This includes the five-node,
32-state example as well as the four-node Factionalizing model in the latest
study. Custom state cardinalities and per-agent CPTs are supported. The
observation probabilities must lie strictly inside (0, 1). Shared evidence and
propaganda configurations continue to use their reference operators.

The source mechanism matches the archived C++ study: every experimenting agent
receives a genuine private observation. Honest agents forward that same
observation. Unreliable sources replace only their outgoing report, using .5
for uninformative reports or 1 minus the true probability for opposite reports.
The agent uses its private evidence once and each distinct incoming source's
report once. Self-loops and parallel edges do not duplicate evidence. The
older finite NumPy operators instead change unreliable agents' own observations;
they are consequently equivalent in the honest-source condition. The existing
`ReportedMultiBeliefOp` and the C++ replay provide the comparisons for corrupted
outgoing reports.

`payoffs` stores outgoing reports; `private_payoffs` stores private observations.
Both columns use the order (successes, trials). `received` and
`cumulative_received` count received trials, including private trials. Experiment
decisions use the existing strict thresholds: configured target probability > .5 for `hypothesis`, or
expected payoff > `safe_payoff` for `expected_payoff`. Results exactly at a
threshold can differ between implementations through floating-point rounding.
The C++ replay checks every decision rather than inferring decision agreement
from close posterior probabilities.

## Exact source trust

`TorchBayesianReliabilityOp` stores

\[
P(H,R_1,\ldots,R_d)=P(H)\prod_{j=1}^d P(R_j\mid H).
\]

This factorization is exact for the configured model: initial source
reliabilities are independent of H and one another, private observations
depend only on H, and each report depends only on H and that source's
persistent reliability. Multiplying the associated likelihoods preserves the
factorization. It retains the correlations between H and each R; replacing
these conditional factors with independent marginal trust values would be
incorrect.

For each directed source-recipient pair, the implementation stores a 2 by 2
conditional table. Cumulative private and report counts are sufficient
statistics. Reconstructing from the initial priors and these counts avoids
counting old reports twice. Runtime storage is O(N+E), where E counts distinct
directed social relationships, rather than exponential in an agent's degree.
The existing NumPy network representation is used only to validate and compile
these factors. Dogmatic priors, empty edge sets, and deterministic unreliable
report probabilities are handled explicitly. Impossible evidence raises
`ImpossibleEvidence`.

## Devices and reproducibility

Inference and sampling support `cpu` and CUDA devices with float64 support.
For CUDA, install mutually compatible CUDA builds of PyTorch and DGL, and set
`device: cuda:0` before constructing the operator. Apple's MPS backend is not
supported because it does not implement the required float64 operations.
Only CPU execution has been verified on the available machine. Conditional
CUDA comparison tests are included, but were skipped because CUDA hardware
was unavailable. GPU performance has not been established.

The class selects PyTorch inference. A private copy of the configuration uses
NumPy for initialization, even when the general `multibelief.backend` field is
`bnlearn`; the caller's configuration is unchanged and no R process is started.
Metadata records both backends and the actual device and precision.

Initial priors and source identities retain the reference initialization
rules. Simulation draws use the PyTorch RNG. Use `polygraphs.random(seed)` to
seed the relevant generators. Identical numerical seeds do not imply identical
draws across NumPy, PyTorch, CUDA and C++. Comparisons therefore replay common
evidence and restore common initial priors.

To restore a calibrated or otherwise dependent initial finite joint, use
`operator.set_initial_log_joint(graph, values)` before processing evidence.
`values` has shape (agents, joint states) and contains log probabilities in the
operator's C-order assignments. The method normalizes each row and publishes
the result. Metadata includes the actual initial joint as probabilities.

## Executed verification

The [validation record](../../../reports/2026-11-19-torch-validation/README.md)
contains the test results, exact C++ source checksum, numerical errors and CPU
timings. Reproduce the checks from the repository root:

```bash
PYTHONPATH=. python -m pytest tests -q
PYTHONPATH=. python scripts/validate_torch_cpp.py \
  --output /tmp/cpp-validation.json
PYTHONPATH=. python scripts/benchmark_torch_batched.py --output /tmp/benchmark.json
```

The C++ script compiles the published `reports/current/worker.cpp` without changing it
(or an alternative source passed with `--source`),
regenerates 54 conditions with two seeds and 500 rounds, replays every private
observation and report through the actual DGL operators, and checks decisions,
final posteriors, sampled P(G)/P(H) trajectories, population standard deviations
and trust summaries. It writes all regenerated per-run records to a temporary
directory and verifies the source bytes are unchanged.
