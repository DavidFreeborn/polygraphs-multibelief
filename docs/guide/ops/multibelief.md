# Multibelief Networks

Two operations extend PolyGraphs with Bayesian belief networks inside its agents:

- `MultiBeliefOp`: a user-specified discrete network for each agent, with evidence
  propagated through all its beliefs.
- `BayesianReliabilityOp`: a hypothesis and persistent beliefs about the reliability
  of individual neighbours, updated together by Bayes' rule.

They use the normal PolyGraphs graph constructors, configuration files, simulation
runner and snapshots. Existing operator implementations, including `BalaGoyalOp`
and `OConnorWeatherallOp`, are unchanged. The low-level `simulate_()` defaults now
use the same 0.99/0.5 stopping thresholds as configuration-driven simulations;
upstream had these two defaults reversed.

## Installation

Use Python 3.10 and install the fork from its repository root:

```bash
python -m pip install -r requirements-multibelief.txt
```

The constraints select NumPy 1.26.4, PyTorch 2.2.2 and DGL 1.1.2. They avoid the
GraphBolt dependency requirements of newer DGL releases. The operations require CPU;
upstream operators retain their own device support.

On Linux, installing the CPU-only PyTorch wheel first avoids downloading CUDA
libraries that these operations do not use:

```bash
python -m pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-multibelief.txt
```

The default backend is Marco Scutari's **R bnlearn**, with gRain for exact
inference. The similarly named Python package is a different project.

```r
install.packages(c("bnlearn", "gRain", "jsonlite"))
```

`Rscript` must be on `PATH`, or provided through `POLYGRAPHS_RSCRIPT` or
`multibelief.rscript`. For example, in PowerShell:

```powershell
$env:POLYGRAPHS_RSCRIPT = 'C:\Program Files\R\R-4.6.2\bin\Rscript.exe'
```

Use the path for the R version actually installed. A persistent R subprocess is
shared by an operation's agents and closed after the simulation. The backend builds
`bn.fit` objects with `custom.fit()`, converts them with `as.grain()`, conditions
virtual evidence nodes and queries the posterior with `querygrain()`. It does not
estimate CPTs from the agents' observations. In particular, `bn.fit(method="bayes")`
is parameter estimation, not the belief updating needed here.

Set `multibelief.backend: numpy` to use the independent exact Python backend. Missing
R dependencies cause an explicit error; there is no silent change of backend.

## Evidence and repeated observations

Let `X` contain an agent's persistent belief variables and `D_t` a fresh observation.
The update is

\[
P_t(X) \propto P_{t-1}(X)P(D_t\mid X)
             \propto P_0(X)\prod_{s=1}^{t}P(D_s\mid X).
\]

The Python backend updates the **joint** log posterior incrementally. It also keeps
accumulated log likelihood factors; `BeliefState.replay()` reconstructs the result
from the original network. The R backend uses that original network and accumulated
factors on each update. Factors with the same scope are combined, so the number of
virtual nodes does not grow with the number of observations. A new evidence batch
is applied atomically within each agent; zero-probability evidence raises
`ImpossibleEvidence` rather than inventing a posterior.

For `k` successes in `n` independent Bernoulli trials, the likelihood factor is
`p(X)**k * (1-p(X))**(n-k)`. The binomial coefficient cancels across hypotheses.
This is equivalent to adding a separate observed node for every trial. Repeatedly
setting the value of a *single persistent data variable* is a different model.

Conditioning a collider such as `H -> D <- S` usually makes `H` and `S` dependent.
Replacing their root priors by their posterior marginals and retaining the old DAG
would erase this dependency. Both backends preserve it.

The general operation stores the full discrete joint, with a default limit of
65,536 states per agent (`multibelief.max_states`). Its cost is exponential in the
number of belief variables, not in the number of observations. The reliability
operation exploits a smaller exact factorization described below. All inference
uses floating-point arithmetic. The log-space Python backend is preferable for
extreme likelihoods for which gRain's probability-space calculations underflow.

## Multibelief bandits

`case: baseline` represents the draft's Model 1, with persistent hypothesis `H`.
`case: meta` represents Model 2, adding evidence-quality belief `S`. With state
order `[0, 1]`, the probability of a successful observation is:

| H | S | P(D=1) |
|---|---|---|
| 0 | 0 | 0.5 - doubtful_epsilon |
| 0 | 1 | 0.5 - epsilon |
| 1 | 0 | 0.5 + doubtful_epsilon |
| 1 | 1 | 0.5 + epsilon |

`doubtful_epsilon` defaults to `epsilon / 2`. Actual trials use
`multibelief.true_probability`, defaulting to `0.5 + epsilon`.

In each round all agents choose an action from their pre-round beliefs, collect
fresh observations, share those observations with outgoing neighbours, then update.
Own observations are included exactly once, whether or not the graph has self-loops.
Parallel edges do not duplicate observations. Evidence is shared only in the round
in which it is generated; received reports are not forwarded as fresh evidence.
The graph topology is fixed for the lifetime of the operation.

With `decision: expected_payoff`, an agent tests B if
`E[P(D=1 | H,S)] > safe_payoff`, where the expectation uses its joint posterior and
`safe_payoff` defaults to 0.5. Ties select A, matching the existing PolyGraphs
operator. With `decision: hypothesis`, the agent tests if `P(H=1) > 0.5`. This is
the decision-threshold control. The two rules coincide in the baseline model but
need not coincide after `H` and `S` become dependent. The control does not implement
an unspecified numerical adjustment to rewards.

Optional propaganda is independent across agents and rounds. On a selected round,
an agent receives likelihood evidence on the configured node. The examples use
`likelihood: [1.0, 0.5]` on `S`, which halves its prior odds before propagation
through the joint distribution. A matched example targets `H` instead. The draft
does not specify signal strength or an integer trial count; **10 trials and these
signal likelihoods are explicit configurable choices**, not established paper
parameters. The draft's tentative results are not assumed by the implementation.

## Factionalization

`case: networked-factionalization1`, `networked-factionalization2` and
`networked-factionalization3` put the belief structures from Freeborn (2024),
section 4, inside agents that share fresh observations with their neighbours:

| Case | Network | Figure population |
|---|---|---|
| 1 | `H1 -> H2` | 15 |
| 2 | `H1 -> H2 <- H3` | 40 |
| 3 | `H4 -> H3 <- H5`, `H1 -> H2 <- H3` | 60 |

Case 1 uses `P(H2=1 | H1) = [0.1, 0.9]`. Each two-parent CPT in cases 2 and 3
uses `[[0.9, 0.1], [0.1, 0.9]]`, following the diagrams and tables. Figure 4's
mislabelled table header is interpreted using its diagram and section 4.2.

The main `factionalization-1.yaml` to `factionalization-3.yaml` examples use social
cycles. Each active agent conducts Bernoulli trials with likelihoods
`P(D=1 | H2=0)=0.35` and `P(D=1 | H2=1)=0.65`; the true success rate is 0.65.
It tests B when its joint-derived expected success rate exceeds 0.5, otherwise
chooses A. It updates on its own and incoming neighbours' fresh counts once per
round. Complete graphs can be selected with the normal `network.kind` override.
Different local evidence histories can now drive different posterior trajectories.

The separate `shared-evidence-factionalization-*.yaml` examples are cross-checks
of the paper's common-evidence setting. They use `case: shared-factionalization1`
through `shared-factionalization3`. The original names `factionalization1` through
`factionalization3` remain aliases for these shared-evidence controls.
In these controls every agent receives the same likelihood evidence on the *persistent* node `H2`,
exactly once per round, independently of social graph degree. The supplied examples
use 20 updates with likelihoods `[0.35, 0.65]` for `[H2=0, H2=1]`. The captions call
this a "likelihood ratio of 0.65"; the accompanying earlier code supplies the two
likelihoods 0.65 and 0.35. The explicit ratio here is therefore `0.65 / 0.35`.
Root priors use an unscrambled Halton sequence with bases 2, 3 and 5, starting at
index 1 to avoid the all-zero boundary point. These are reproducible instances of
the published models, not a claim to reproduce an undocumented random seed.

`population_diagnostics()` reports the mean pairwise Jensen-Shannon divergence
between joint distributions and between their products of marginals, in nats.
It also reports `joint_kl` and `marginal_product_kl`: the mean directed KL over
all ordered pairs of distinct agents, equivalently the mean of
`(KL(P||Q)+KL(Q||P))/2` over unordered pairs. KL is unbounded and is infinite when
positive probability under one distribution has zero probability under the other.
No arbitrary epsilon is added to make KL finite. For highly concentrated posteriors,
use the retained log probabilities to distinguish real support differences from
floating-point underflow.
It also reports population variance and mean absolute off-diagonal covariance.
Jensen-Shannon divergence is bounded by `log(2)`, regardless of the number of
variables. The covariance is `abs(mean((x-mean(x))*(y-mean(y))))`.
These use the standard mathematical definitions; apparent sign and typesetting
errors in the paper are not reproduced. No rule forces joint divergence to decline
after every realized observation. Directed KL obeys an expectation identity under
the first agent's predictive distribution. The corresponding JS chain identity
uses posterior mixture weights, not an equal-weight JS of posteriors. Equal-weight
JS can increase even in expectation under the shared-CPT model. Thus the KL
expectation result does not by itself establish the same claim for ordinary JS.

## Bayesian source reliability

`BayesianReliabilityOp` keeps binary `H` and a persistent binary `R_j` for each
incoming neighbour. `R_j=1` means that neighbour's reports follow the informative
trial distribution. Its reporting model is

\[
K_{j,t}\mid H,R_j \sim \operatorname{Binomial}(n_{j,t},p_{H,R_j}).
\]

For reliable sources, `p(H=1)=0.5+epsilon` and `p(H=0)=0.5-epsilon`. For unreliable
sources, `unreliable_mode: uninformative` uses a constant probability (default 0.5).
`unreliable_mode: opposite` reverses the informative probabilities. This permits
Bayesian reversal of testimony when a source is believed systematically misleading.

After each report the recipient updates `P(H,R_j | reports)`. A report agreeing
with its expectations can support the reliable-source hypothesis; disagreement can
undermine it. Trust is not mechanically decreased by a belief-distance formula.
Further evidence about `H` can also change beliefs about previously heard sources.

Initially `H` and the `R_j` are independent. The posterior factorizes as
`P(H) * product_j P(R_j | H)`, because each report depends only on `H` and its own
source. The Python backend stores these conditional factors; the R backend queries
`H` and each `(H,R_j)` pair using the same bnlearn model. Neither enumerates the full
`2**(degree+1)` joint. This is exact under the stated conditional independencies,
including the dependence between source reliabilities induced by uncertainty in `H`.

Agents treat their own private trials as reliable. By default all agents actually
report reliable data, as in O'Connor and Weatherall's original experiment. Set
`actual_reliability` or `unreliablenodes` to introduce persistent unreliable sources.
Unreliable agents still learn from their own accurate private trials but send fresh
reports from the configured unreliable distribution. This models corrupted
reporting rather than impaired private observation. A report is generated once per
source and shared unchanged with all recipients.

This is a new Bayesian extension of the O'Connor–Weatherall evidence-sharing model,
not a reproduction of their distance heuristic. Inference is Bayesian **conditional
on the report likelihoods and the experiment schedule**. The model does not infer
others' hidden priors, reason recursively about their strategies, or treat their
decision to experiment as additional evidence. It does not repeatedly condition on
the same announced credence as if it were independent testimony. This explicit
scope is important when interpreting "fully Bayesian" social learning.

## Configuration

All example YAML files are standalone and accept PolyGraphs' normal overrides:

```bash
polygraphs -f examples/multibelief/meta-cycle.yaml local-overrides.yaml
```

For instance, `local-overrides.yaml` can contain:

```yaml
multibelief.backend: numpy
epsilon: 0.1
simulation.steps: 500
```

To define a network, select `case: custom` for fresh bandit observations or
`case: shared` for common likelihood evidence. Provide `model.nodes`, each containing
`name`, optional `states` (default `["0", "1"]`), `parents`, and `cpt`.
CPT axes are the listed parents followed by the child. Every final-axis row must
sum to one. Cycles, invalid probabilities and undefined parents are rejected.
`custom-belief-network.yaml` is a complete example with a third hypothesis linked
to `H`.

For a custom bandit, `observation.parents` and `observation.probability` define a
fresh Bernoulli observation conditional on the listed persistent variables. For
shared evidence, specify a list such as

```yaml
multibelief.shared_evidence:
  - variables: [H2]
    likelihood: [0.35, 0.65]
```

Several factors can occur in each batch. `evidence_schedule` can instead supply one
list of factors per round; it must cover the configured horizon. Likelihoods are
relative weights, not desired posterior probabilities.

`prior_mode` is `uniform`, `halton` or `fixed`. Automatic priors require binary
roots. `fixed` retains the supplied CPTs, including categorical priors. For the
focal root in uniform mode, the normal PolyGraphs `init` settings apply. `priors`
can override each binary root with a probability or a list of one probability per
agent. `agent_models` can override individual agents' structures and CPTs using
string agent IDs; they must share ordered node names and state labels. Use `fixed`
to retain the priors in those individual models.

## Stopping and outputs

Bandit examples stop when no agent chooses B or all agents choose B and have focal
credence greater than `lowerupper` (normally 0.99), or at `simulation.steps`.
The 0.99 rule is a conventional threshold, not a proof of permanent convergence.
Set `stop_on_consensus: false` for a fixed horizon. Shared-evidence and propaganda
runs always require a finite horizon and are not stopped by a scalar belief
threshold. The distance-based polarization stopping rule is never applied to these
new operations. Mixed beliefs at a finite horizon do not establish stable
polarization.

The normal `.bin`, `.hd5`, `configuration.json` and `data.csv` outputs remain.
Each simulation additionally writes `<prefix>.beliefs.json`, specifying node order,
state labels, initial networks and backend versions. HDF5 snapshots contain:

- `beliefs`: the focal marginal, for existing analysis tools;
- `beliefs_by_node/<name>`: every state's marginal for every agent;
- `belief_joint`: flattened joints for `MultiBeliefOp`, in C order;
- `expected_payoff`: the joint-derived bandit expectation;
- `reliability_belief`: the recipient's belief about each directed edge's source;
- `private_payoffs`: private evidence in the learned-reliability model.

Node names in HDF5 paths are URL-encoded. The `.bin` file gives the graph and
initial beliefs; snapshots record subsequent rounds. Full joints in the reliability
model are deliberately not materialized. Its edge and hypothesis marginals are
saved instead.

Print a CSV of the saved diagnostics, including the initial state, with:

```bash
python examples/multibelief/analyse.py path/to/result-directory
```

Use `--prefix` when analysing a repeat other than simulation `1`.

## Validation

```bash
python -m pytest tests
```

Tests cover analytical binomial posteriors, incremental-versus-replay equivalence,
separate observed data nodes, induced dependence, impossible evidence, long
histories, categorical CPT ordering, source reliability, graph communication,
stopping, snapshots and existing operators. Tests marked `bnlearn` compare against
the real R backend and skip if `Rscript` is unavailable. A suite with skipped R
tests has not validated that backend on that machine. Tests also run short
configuration checks; they do not establish population-level scientific results.

For the supplied Halton populations after 20 fixed updates, the deterministic
checks give the following mean pairwise divergences over **all** belief variables:

| Case | Joint JS, initial -> final | Marginal-product JS, initial -> final |
|---|---|---|
| 1 | 0.091579 -> 0.049365 | 0.127267 -> 0.049366 |
| 2 | 0.181880 -> 0.162877 | 0.191217 -> 0.197496 |
| 3 | 0.252992 -> 0.253919 | 0.263038 -> 0.296345 |

Case 3's joint divergence increases slightly. Its final posterior was independently
checked with bnlearn/gRain and its divergence with SciPy's Jensen-Shannon distance
squared. This particular instance therefore does not satisfy the section 3.3
criterion of decreasing joint JS, although its belief trajectories and increasing
cross-belief covariance exhibit faction structure. The implementation reports this
distinction rather than assuming that every figure must meet every proposed measure.

## Sources

- Freeborn, O'Connor and Weatherall, *Multiple Beliefs in Epistemic Networks*, draft,
  Models 1–2. Its trial count, propaganda strength and tentative findings remain
  distinguished from the explicit example choices above.
- Freeborn (2024), [Rational factionalization for agents with probabilistically
  related beliefs](https://doi.org/10.1007/s11229-024-04491-5), section 4.
- O'Connor and Weatherall (2018), [Scientific Polarization](https://arxiv.org/abs/1712.04561),
  especially section 3.1 and footnote 23.
- bnlearn, [custom parameter specification](https://www.bnlearn.com/documentation/man/bn.fit.html)
  and [exact inference with gRain](https://www.bnlearn.com/examples/pkg-gRain/).
