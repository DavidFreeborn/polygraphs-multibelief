"""Write the precise protocol alongside machine-readable execution metadata."""

import html
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    settings = json.loads((HERE / "data/execution.json").read_text())
    validation = json.loads((HERE / "data/validation.json").read_text())
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (HERE / "requirements-exact.txt").write_text(freeze, encoding="utf-8")
    root = HERE.parents[1]
    files = (
        list((root / "polygraphs").rglob("*.py"))
        + list((root / "polygraphs/beliefs").glob("*.R"))
        + list((root / "tests").glob("*.py"))
        + list(HERE.glob("*.py"))
    )
    provenance = {
        str(p.relative_to(root))
        .replace("\\", "/"): hashlib.sha256(p.read_bytes())
        .hexdigest()
        for p in files
    }
    (HERE / "data/source-sha256.json").write_text(json.dumps(provenance, indent=2))
    text = (
        """MULTIBELIEF NETWORK STUDY: COMPLETE EXPERIMENT SETTINGS
17 November 2026

1. Design
72 conditions, 20 repetitions per condition, 500 rounds, 60 agents.
48 basic conditions: 4 belief models x 4 graphs x 3 information treatments.
24 trust conditions: 4 graphs x 3 treatments x 2 initial reliability priors.
No population size, horizon, epsilon or repetition reduction was needed.
Round zero is recorded. Every run continues to round 500, even after apparent
consensus. Agents can cease experimenting endogenously. 500 is a fixed horizon,
not evidence of asymptotic convergence. No extra result-selected runs were added.

2. Trial and decision semantics
A is known to have Bernoulli success probability 0.5. Genuine B has probability
0.55. Epsilon is 0.05. A myopic agent chooses B iff its posterior expected B
payoff is strictly greater than 0.5; ties choose A. Every B chooser performs
10 trials per round. A choosers generate zero B trials. Outcomes are synchronous:
actions use the previous round's posterior, then new evidence is shared and all
agents update. No action costs, forced exploration or repeated forwarding.
Each recipient uses its own evidence once and each immediate neighbour's fresh
batch once. Self loops and duplicate edges do not duplicate evidence.

3. Graphs
Undirected cycle of 60 labelled vertices (0,...,59), complete graph, and G(60,p)
Erdos-Renyi graphs for p=0.1 and p=0.5. Graph edges are stored in both directions
in DGL; no self loops are stored in experiment graphs. Own evidence is added
separately. ER graphs are sampled independently for each repeat and probability,
without connectivity rejection. Graph and unreliable identities are paired across
models/treatments at a repeat seed. Expected degrees: 2,59,5.9,29.5 respectively.
Realized degree means and connected-component counts are in metrics.csv.

4. Belief-network models
All variables are binary, with axis state order [0,1]. D is fresh for every trial.
Persistent-state arrays enumerate nodes in C order (last index varies fastest).

Baseline: persistent H; H -> D. P(D=1|H=0,1)=[0.45,0.55].
Converging candidate: independent root priors H,S; H -> D <- S.
P(D=1|H,S)=[[0.45,0.50],[0.50,0.55]].
Polarising candidate: independent root priors H,S; H -> D <- S.
P(D=1|H,S)=[[0.55,0.45],[0.45,0.55]].
These three-node CPTs are experimental choices, not the draft's strong/weak table.
For the converging candidate genuine outcomes correspond to H=S=1.
For the polarising candidate they correspond equally to 00 and 11: no experimental
history distinguishes those states. No arbitrary root truth is used for scoring.

Factionalising: ordered persistent nodes H1,H4,H5,H3,H2. Roots H1,H4,H5.
Parents of H3: [H4,H5]. Parents of H2: [H1,H3]. In both cases P(child=1)
=[[0.9,0.1],[0.1,0.9]]. D depends only on H2, with probabilities [0.45,0.55].
Genuine data fix the observation process at the H2=1 likelihood; no ground-truth
assignment of the other hypotheses is needed or assumed for accuracy scores.

Initial root probabilities are independent Uniform(0,1) draws, independently
across agents and roots. Root probabilities are not all 0.5 and not the earlier
Halton population. The same first root column initializes baseline H, both
three-node H variables, trust H and factionalising H1. Second and third columns
initialize S / H4 / H5 as applicable. These are paired root priors; induced H2
priors in the five-hypothesis model are not forced to match the baseline.

5. Source treatments
All reliable: zero altered sources.
Misinformation: exactly six sources have observed B successes sampled at 0.50.
Disinformation: exactly six sources have observed B successes sampled at 0.45.
The six identities are sampled uniformly without replacement for each repeat.
They are fixed throughout that run, and paired across treatments and graphs.
In basic belief models this matches native unreliable-observation semantics:
altered observations are used both by the unreliable source itself and by its
neighbours. All recipients apply the honest likelihood without source correction.
There are no externally imposed belief adjustments or propaganda likelihoods.
Unreliable sources still use the ordinary expected-payoff decision rule and can
stop experimenting. They are not permanently broadcasting zealots.

6. Trust model
Each recipient maintains H and a persistent R_j for each immediate neighbour j.
R_j=1 means reliable; R_j=0 means the treatment-specific alternative.
Reliable source probabilities given H=0,1: [0.45,0.55].
Uninformative alternative: [0.50,0.50]. Reversed alternative: [0.55,0.45].
All-reliable and misinformation conditions use the uninformative alternative;
disinformation uses the reversed alternative. This assumption is known to agents.
Initial H and R_j are independent, and R_j priors are independent across sources.
Initial trust is 0.50 or the true reliable fraction: 1.00 in all-reliable runs,
0.90 otherwise. Knowing the marginal proportion does NOT encode knowledge that
exactly six sources are unreliable. Prior 1 is dogmatic and is a benchmark.

Every trust-model agent receives genuine private outcomes with success probability
0.55. Its outgoing reports are replaced at probability 0.50 or 0.45 if its source
type is unreliable. The recipient uses its own genuine private data and incoming
reports, never its own corrupted outgoing report. This preserves the established
BayesianReliabilityOp mechanism. Consequently contrasts against basic models
change both source inference and unreliable agents' own observation mechanism.
They are not interpreted as a pure causal effect of adding trust. Ordinary-agent
summaries (excluding the six unreliable sources) are provided separately.

Updates use the exact joint factorisation P(H) product_j P(R_j|H). For cumulative
source successes k_j and trials n_j, each (H,R_j) factor is its prior weight times
p(H,R_j)^k_j [1-p(H,R_j)]^(n_j-k_j). Summing R_j gives the source's likelihood for
H. The product of these source likelihoods and the private-data likelihood updates
H; normalization within H gives P(R_j|H). Marginal trust sums over H.
This reconstructs from original priors and cumulative counts and is exact for
the specified model. Trial counts and the experimental schedule are conditioned
on. Agents do not model neighbours' hidden policies or infer from their silence.
With reversed likelihoods and prior trust .50, report likelihoods for H cancel.
H then equals the private-evidence-only posterior. Initially A-preferring agents
never collect private B evidence and retain their initial H probability. Reliability
beliefs can nevertheless change, and privately experimenting agents can learn.

7. Randomness and pairing
Repetition seeds: integers 1729 through 1748 inclusive.
Graph: networkx.erdos_renyi_graph(N,p,seed=seed+10000).
Root draws: numpy.default_rng(seed+20000).uniform(size=(60,3)).
Bad IDs: sorted(default_rng(seed+30000).choice(60,6,replace=False)).
Trials: default_rng(seed+40000).random((2,60,10)) each round.
The first array generates genuine trials at threshold .55. The second generates
bad-source trials at .50 or .45. All arrays are consumed every round even for
inactive agents. Inactive agents' success/trial counts are then zeroed. This pairs
underlying uniforms without pretending endogenous action histories stay equal.
NumPy global and torch initialization seeds are also set to seed; inference
priors are explicitly replaced by the specified paired draws where necessary.

8. Metrics
Recorded at every round 0..20, then every fifth round through 500. Raw success and
trial counts are retained for ALL rounds. Posterior node marginals are retained
at measurement rounds for every run. Full joint log posteriors at measurement
rounds and edge trust histories are retained for prespecified seed 1729; every
run retains initial/final log joint states and final trust factors. These plus
all raw counts suffice to reconstruct unsaved states without random generation.

Population node mean = sum_i P_i(X=1)/N. Population node SD uses divisor N.
D_next = posterior expected probability of a fresh B success.
Joint KL = sum_{i!=j} KL(P_i||P_j)/(N(N-1)).
Joint JS = mean_{i<j} [KL(P_i||M_ij)+KL(P_j||M_ij)]/2, M_ij=(P_i+P_j)/2.
Marginal-product metrics replace each joint with the product of its node marginals.
All logs are natural, hence nats. JS is divergence, not square-root distance.
KL uses finite log posterior values; no support smoothing or epsilon floors.
JS mixtures are calculated with logaddexp to avoid division-by-two underflow
of positive subnormal probabilities. Raw posteriors and actions were unchanged
by this numerical-analysis correction; affected measures were recomputed from
the saved evidence. Independent JS checks use the entropy identity instead.
Floating-point roundoff at zero is clipped only for the algebraic nonnegative KL
identity. Total correlation = mean_i KL(P_i || product of own marginals).
Joint distributions exclude D. The trust model's common-state divergence data
refer only to H, since different agents have different neighbour-variable sets;
they are not described as divergence of the complete trust networks.

Good-B belief = posterior probability that the state-specific B success rate is
strictly above .5: H=1 (baseline/trust), H=S=1 (converging), H=S (polarising), H2=1
(five hypotheses). Correct fraction counts probabilities >.95; all_correct means
all 60 exceed .95. Brier score for this true event is mean (1-belief)^2.
No accuracy claims are made for unidentifiable background-node assignments.
Trust scores average directed edges, not agents equally; high-degree recipients
therefore contribute more. Actual source labels group reliable/unreliable curves.
Source Brier score is squared error against the binary actual source label.
An exposed edge is one whose source has supplied at least one B trial.
Initial-A and initial-B cohorts are fixed at round zero separately for each run.
The source-exposure supplement additionally groups directed edges by sources
with no reports or at least 100 cumulative trials. Fractions use all edges of
that actual type as denominator. Trust means average available run-level group
means; empty groups are omitted. This conditions on endogenous experimentation,
not a causal comparison or an extra experimental treatment.

Main basic-model figures use all agents; supplements repeat all means, SDs and
distribution metrics on ordinary agents (54 affected / 60 control). Activity in
those supplements remains the whole-population fraction and is labelled as such.
Lines average per-run statistics; bands are the 25th and 75th percentiles across
20 runs (pandas default linear interpolation), NOT confidence intervals.
No averaging of distributions before calculating within-population divergence.

9. Replication
Install this fork from source with its dependency constraints. See requirements-
exact.txt for the environment used; its editable local path should be replaced
by the path of the new checkout. Windows Python 3.10 was used. DGLBACKEND=pytorch,
OMP_NUM_THREADS=1 and OPENBLAS_NUM_THREADS=1 were set. torch.set_num_threads(1).

From the repository root:
python -m pytest tests
python reports/2026-11-17-network-study/run_experiments.py --pilot
python reports/2026-11-17-network-study/run_experiments.py --rounds 500 --repeats 20
python reports/2026-11-17-network-study/validate_results.py
python reports/2026-11-17-network-study/source_exposure.py
python reports/2026-11-17-network-study/write_settings.py
python reports/2026-11-17-network-study/make_report.py

R cross-checks require Rscript plus bnlearn and gRain, with POLYGRAPHS_RSCRIPT set
if Rscript is not on PATH. Main simulations use the exact batched Python backend.
The original general and R backends remain available. Tests with R skipped do
not validate R on that installation. SHA256 manifests identify saved raw files.
No cloud compute, GPU, paid services or GitHub workflow runs were used.

10. Scope of conclusions
Native Bala-Goyal matched-path tests use 12 and 60 agents on cycle, complete
and random graphs for 30 rounds, comparing both actions and posteriors. The
comparison uses float64 arithmetic to test mathematical equivalence; native
default float32 arithmetic can saturate probabilities earlier. Trial outcomes
are identical in these tests, rather than merely drawn from the same distribution.
The Bayesian-network observation D is fresh; this is not repeated conditioning
on a single persistent data value. Additional tests compare joint and trust
updates with the reference operations and exercise subnormal-probability JS.

Twenty runs per setting and 500 rounds are a bounded descriptive study, not a
high-power estimate of rare failures or proof of long-run convergence. Density
changes evidence exposure. Treatments can change experimentation endogenously.
Networked results are distinct from the earlier common-evidence controls, which
remain in reports/2026-11-17. Candidate model labels are not assumed outcomes.

EXECUTION METADATA
"""
        + json.dumps(settings, indent=2)
        + "\n\nINDEPENDENT VALIDATION\n"
        + json.dumps(validation, indent=2)
        + "\n\nR BACKEND CHECKS OF SAVED POSTERIORS\n"
        + (HERE / "data/r-validation.json").read_text()
        + "\n\nRECORDED TEST RESULTS FOR THE ORIGINAL EXECUTION\n"
        + (HERE / "data/test-results.json").read_text()
    )
    (HERE / "experiment-settings.txt").write_text(text, encoding="utf-8")
    (HERE / "experiment-settings.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Experiment settings and replication</title><style>body{max-width:1000px;margin:40px auto;padding:20px}pre{white-space:pre-wrap;font:15px/1.55 monospace}</style><body><pre>'
        + html.escape(text)
        + "</pre></body></html>",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
