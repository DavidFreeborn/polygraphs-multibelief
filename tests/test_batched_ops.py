"""Independent checks of batched inference and native evidence semantics."""

import dgl
import numpy as np
import pytest
import torch
from scipy.special import expit, logit

from polygraphs import hyperparameters as hp
from polygraphs.ops import (
    BatchedMultiBeliefOp,
    BatchedBayesianReliabilityOp,
    MultiBeliefOp,
    BayesianReliabilityOp,
    BalaGoyalOp,
)
from polygraphs.beliefs.cases import binary_node, factionalization


def params():
    p = hp.PolyGraphHyperParameters()
    p.epsilon = 0.05
    p.trials = 10
    p.multibelief.backend = "numpy"
    p.multibelief.stop_on_consensus = False
    p.simulation.steps = 20
    p.init.kind = "constant"
    p.init.constant.value = 0.6
    return p


@pytest.mark.parametrize(
    "case", ["baseline", "meta", "custom", "networked-factionalization3"]
)
def test_batched_joint_matches_reference(case):
    p = params()
    p.multibelief.case = case
    if case == "custom":
        p.multibelief.model = {"nodes": [binary_node("H"), binary_node("S")]}
        p.multibelief.observation = {
            "parents": ["H", "S"],
            "probability": [[0.55, 0.45], [0.45, 0.55]],
        }
    edges = ([0, 0, 1, 2, 2, 3], [0, 1, 2, 1, 1, 0])
    g, h = [dgl.graph(edges, num_nodes=5) for _ in range(2)]
    np.random.seed(42)
    slow = MultiBeliefOp(g, p)
    np.random.seed(42)
    fast = BatchedMultiBeliefOp(h, p)
    rng = np.random.default_rng(10)
    for _ in range(20):
        payoffs = torch.tensor(
            np.stack((rng.binomial(10, 0.55, 5), np.full(5, 10)), 1),
            dtype=torch.float64,
        )
        slow.experiment = fast.experiment = lambda graph: graph.ndata.__setitem__(
            "payoffs", payoffs
        )
        slow(g)
        fast(h)
        np.testing.assert_allclose(
            fast.log_joint, [s.log_joint for s in slow.states], atol=2e-12
        )
        np.testing.assert_allclose(
            g.ndata["expected_payoff"], h.ndata["expected_payoff"], atol=2e-14
        )


@pytest.mark.parametrize("mode", ["uninformative", "opposite"])
@pytest.mark.parametrize("prior", [0.5, 0.9, 1.0])
def test_batched_trust_matches_reference(mode, prior):
    p = params()
    p.bayesian_reliability.prior = prior
    p.bayesian_reliability.unreliable_mode = mode
    edges = ([0, 0, 1, 2, 2, 3], [0, 1, 2, 1, 1, 0])
    g, h = [dgl.graph(edges, num_nodes=5) for _ in range(2)]
    slow, fast = BayesianReliabilityOp(g, p), BatchedBayesianReliabilityOp(h, p)
    rng = np.random.default_rng(11)
    for _ in range(20):
        reports = torch.tensor(
            np.stack((rng.binomial(10, 0.55, 5), np.full(5, 10)), 1),
            dtype=torch.float64,
        )
        private = torch.tensor(
            np.stack((rng.binomial(10, 0.55, 5), np.full(5, 10)), 1),
            dtype=torch.float64,
        )

        def experiment(graph):
            graph.ndata["payoffs"] = reports
            graph.ndata["private_payoffs"] = private

        slow.experiment = fast.experiment = experiment
        slow(g)
        fast(h)
        np.testing.assert_allclose(g.ndata["beliefs"], h.ndata["beliefs"], atol=1e-12)
        np.testing.assert_allclose(
            g.edata["reliability_belief"], h.edata["reliability_belief"], atol=1e-12
        )


@pytest.mark.parametrize("topology", ["cycle", "complete", "er"])
@pytest.mark.parametrize("n", [12, 60])
def test_native_bala_goyal_matched_trajectory(topology, n):
    rng = np.random.default_rng(123)
    adj = np.eye(n, dtype=bool)
    for i in range(n):
        for j in range(i):
            edge = (
                topology == "complete"
                or (topology == "cycle" and (i - j in (1, n - 1)))
                or (topology == "er" and rng.random() < 0.25)
            )
            adj[i, j] = adj[j, i] = edge
    g, h = [dgl.graph(np.where(adj), num_nodes=n) for _ in range(2)]
    p = params()
    # Float64 makes the mathematical comparison independent of native float32 rounding.
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        native = BalaGoyalOp(g, p)
        fast = BatchedMultiBeliefOp(h, p)
        priors = rng.uniform(0.1, 0.9, n)
        g.ndata["beliefs"] = torch.tensor(priors)
        fast.log_joint = np.log(np.stack((1 - priors, priors), 1))
        fast._publish(h)
        for _ in range(30):
            np.testing.assert_array_equal(
                g.ndata["beliefs"].numpy() > 0.5, fast.active(h).numpy()
            )
            counts = rng.binomial(10, 0.55, n)
            native.sample = lambda: torch.tensor(counts, dtype=torch.float64)

            def experiment(graph):
                active = fast.active(graph).numpy()
                graph.ndata["payoffs"] = torch.tensor(
                    np.stack((counts * active, 10 * active), 1), dtype=torch.float64
                )

            fast.experiment = experiment
            native(g)
            fast(h)
            np.testing.assert_allclose(
                g.ndata["beliefs"], h.ndata["beliefs"], atol=3e-12
            )
    finally:
        torch.set_default_dtype(old)


def test_symmetric_reports_leave_h_equal_to_private_only():
    p = params()
    p.bayesian_reliability.unreliable_mode = "opposite"
    g = dgl.graph(([0, 1, 2, 0, 1, 2], [1, 2, 0, 2, 0, 1]), num_nodes=3)
    m = BatchedBayesianReliabilityOp(g, p)
    initial = g.ndata["beliefs"].numpy().copy()
    for _ in range(50):
        m(g)
    k, n = m.private_counts.T
    expected = expit(logit(initial) + (2 * k - n) * np.log(0.55 / 0.45))
    np.testing.assert_allclose(g.ndata["beliefs"], expected, atol=3e-12)


@pytest.mark.parametrize("mode", ["uninformative", "opposite"])
def test_unreliable_observations_match_reference(mode):
    p = params()
    p.multibelief.source_mode = mode
    p.unreliablenodes = [1]
    g, h = [dgl.graph(([0, 1, 2], [1, 2, 0]), num_nodes=3) for _ in range(2)]
    slow, fast = MultiBeliefOp(g, p), BatchedMultiBeliefOp(h, p)
    for seed in range(20):
        np.random.seed(seed)
        slow(g)
        np.random.seed(seed)
        fast(h)
        np.testing.assert_array_equal(g.ndata["payoffs"], h.ndata["payoffs"])
        np.testing.assert_allclose(g.ndata["beliefs"], h.ndata["beliefs"], atol=1e-12)


def test_js_is_finite_for_subnormal_probabilities():
    from polygraphs.beliefs.metrics import jensen_shannon

    tiny = np.nextafter(0.0, 1.0)
    value = jensen_shannon([tiny, 1.0], [0.0, 1.0])
    assert np.isfinite(value)
    assert 0 <= value < 1e-300
    assert jensen_shannon([0.0, 1.0], [1.0, 0.0]) == pytest.approx(np.log(2.0))
