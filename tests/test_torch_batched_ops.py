"""Controlled evidence checks against independent exact implementations."""

import json
from pathlib import Path

import dgl
import h5py
import numpy as np
import pytest
import torch

import polygraphs
from polygraphs.beliefs.cases import binary_node
from polygraphs.beliefs.network import ImpossibleEvidence
from polygraphs.ops import (
    MultiBeliefOp, BayesianReliabilityOp, BatchedMultiBeliefOp,
    BatchedBayesianReliabilityOp, TorchMultiBeliefOp, TorchBayesianReliabilityOp,
    BalaGoyalOp, getbyname,
    ReportedMultiBeliefOp,
)
from tests.test_batched_ops import params


def test_examples_and_replay_match_current_study_specifications():
    from scripts.validate_torch_cpp import specifications

    root = Path(__file__).resolve().parents[1]
    published = json.loads((root / "reports/current/model-specifications.json").read_text())
    for index, name in enumerate(("baseline", "converging", "polarizing", "factionalizing")):
        nodes, observation = specifications(index)
        p = polygraphs.hyperparameters.PolyGraphHyperParameters.load(
            [str(root / f"examples/multibelief/torch-{name}.yaml")]
        )
        p.simulation.steps = 2
        g = graph()
        model = TorchMultiBeliefOp(g, p)
        assert list(model.network.names) == published[index]["latent_nodes"]
        assert [node["name"] for node in nodes] == published[index]["latent_nodes"]
        assert model.belief_metadata()["observation_parents"] == observation["parents"]
        np.testing.assert_array_equal(model.metadata["observation_probability"], observation["probability"])
    trust = polygraphs.hyperparameters.PolyGraphHyperParameters.load(
        [str(root / "examples/multibelief/torch-trust.yaml")]
    )
    assert trust.bayesian_reliability.prior == published[4]["initial_source_reliability"]


def study_params(case):
    p = params()
    p.multibelief.case = "custom"
    p.multibelief.prior_mode = "fixed"
    nodes = [binary_node("H")]
    if case in ("converging", "polarizing"):
        nodes += [binary_node("S")]
    elif case == "factionalizing":
        nodes = [binary_node("X"), binary_node("Y", ["H", "S"], [[.9, .1], [.1, .9]]),
                 binary_node("H"), binary_node("S")]
    elif case == "32-state":
        p.multibelief.case = "networked-factionalization3"
        return p
    p.multibelief.model = {"nodes": nodes}
    p.multibelief.priors = {"H": .31}
    if case in ("converging", "polarizing"):
        p.multibelief.priors["S"] = .73
    elif case == "factionalizing":
        p.multibelief.priors = {"X": .31, "H": .24, "S": .81}
    scope = (["H"] if case == "baseline" else
             ["X", "Y"] if case == "factionalizing" else ["H", "S"])
    probability = ([.45, .55] if case == "baseline" else
                   [[.45, .45], [.45, .55]] if case == "converging" else
                   [[.55, .45], [.45, .55]])
    p.multibelief.observation = {"parents": scope, "probability": probability}
    p.multibelief.true_probability = .55
    return p


def graph(n=5):
    # Parallel edges, self-loop, asymmetric edges and an isolated agent.
    return dgl.graph(([0, 0, 1, 2, 2, 3], [0, 1, 2, 1, 1, 0]), num_nodes=n)


@pytest.mark.parametrize("case", ["baseline", "converging", "polarizing", "factionalizing", "32-state"])
@pytest.mark.parametrize("decision", ["hypothesis", "expected_payoff"])
def test_joint_sequence_matches_both_references(case, decision):
    p = study_params(case)
    p.multibelief.decision = decision
    np.random.seed(812)
    gs = [graph() for _ in range(3)]
    models = [op(g, p) for op, g in zip(
        (MultiBeliefOp, BatchedMultiBeliefOp, TorchMultiBeliefOp), gs)]
    rng = np.random.default_rng(77)
    for step in range(40):
        trials = rng.integers(0, 11, 5)
        payoff = torch.tensor(np.stack((rng.binomial(trials, .55), trials), 1), dtype=torch.float64)
        for m, g in zip(models, gs):
            m.experiment = lambda g: g.ndata.__setitem__("payoffs", payoff)
            m(g)
        for key in gs[0].ndata:
            if key.startswith("belief") or key == "expected_payoff":
                for g in gs[1:]:
                    torch.testing.assert_close(g.ndata[key], gs[0].ndata[key], atol=2e-12, rtol=2e-12)
        torch.testing.assert_close(models[2].log_joint, torch.tensor(models[1].log_joint), atol=1e-10, rtol=1e-12)
        np.testing.assert_array_equal(models[2].received, models[1].received)
        np.testing.assert_array_equal(models[2].cumulative_received, models[1].cumulative_received)
        torch.testing.assert_close(models[0].active(gs[0]), models[2].active(gs[2]))


@pytest.mark.parametrize("case", ["baseline", "converging", "polarizing", "factionalizing", "32-state"])
@pytest.mark.parametrize("mode", ["uninformative", "opposite"])
def test_private_and_report_channels_match_reported_reference(case, mode):
    p = study_params(case)
    p.multibelief.backend = "numpy"
    p.multibelief.source_mode = mode
    p.unreliablenodes = [1]
    g, h = graph(), graph()
    slow, fast = ReportedMultiBeliefOp(g, p), TorchMultiBeliefOp(h, p)
    rng = np.random.default_rng(88)
    for _ in range(40):
        trials = rng.integers(0, 11, 5)
        private, reports = [torch.tensor(np.stack((rng.binomial(trials, probability), trials), 1),
                                        dtype=torch.float64) for probability in (.55, .45)]
        def evidence(g):
            g.ndata["payoffs"], g.ndata["private_payoffs"] = reports, private
        slow.experiment = fast.experiment = evidence
        slow(g)
        fast(h)
        torch.testing.assert_close(g.ndata["belief_joint"], h.ndata["belief_joint"], atol=3e-12, rtol=3e-12)
        torch.testing.assert_close(g.ndata["expected_payoff"], h.ndata["expected_payoff"], atol=3e-12, rtol=3e-12)
        np.testing.assert_array_equal(slow.cumulative_received, fast.cumulative_received)


@pytest.mark.parametrize("mode", ["uninformative", "opposite"])
@pytest.mark.parametrize("prior", [0., .5, .9, 1.])
def test_trust_sequence_matches_both_references(mode, prior):
    p = params()
    p.bayesian_reliability.prior = prior
    p.bayesian_reliability.source_priors = {"2": .3}
    p.bayesian_reliability.unreliable_mode = mode
    gs = [graph() for _ in range(3)]
    models = [op(g, p) for op, g in zip(
        (BayesianReliabilityOp, BatchedBayesianReliabilityOp, TorchBayesianReliabilityOp), gs)]
    rng = np.random.default_rng(63)
    for _ in range(40):
        trials = rng.integers(0, 11, 5)
        reports, private = [torch.tensor(np.stack((rng.binomial(trials, .55), trials), 1),
                                        dtype=torch.float64) for _ in range(2)]
        def evidence(g):
            g.ndata["payoffs"], g.ndata["private_payoffs"] = reports, private
        for m, g in zip(models, gs):
            m.experiment = evidence
            m(g)
        for g in gs[1:]:
            torch.testing.assert_close(g.ndata["belief:H"], gs[0].ndata["belief:H"], atol=3e-12, rtol=3e-12)
            torch.testing.assert_close(g.edata["reliability_belief"], gs[0].edata["reliability_belief"], atol=3e-12, rtol=3e-12)
        np.testing.assert_array_equal(models[2].cumulative_received, models[1].cumulative_received)
        for edge, (source, recipient) in enumerate(zip(models[2].sources, models[2].recipients)):
            pair = (models[2].log_h[recipient, :, None] + models[2].log_r_given_h[edge]).exp()
            torch.testing.assert_close(pair, torch.tensor(models[0].states[int(recipient)].pair(int(source))), atol=3e-12, rtol=3e-12)


@pytest.mark.parametrize("prior", [0., .5, 1.])
@pytest.mark.parametrize("probability", [0., 1.])
def test_deterministic_source_likelihoods(prior, probability):
    p = params()
    p.bayesian_reliability.prior = prior
    p.bayesian_reliability.unreliable_probability = probability
    gs = [dgl.graph(([0], [1]), num_nodes=2) for _ in range(2)]
    slow, fast = [op(g, p) for op, g in zip((BayesianReliabilityOp, TorchBayesianReliabilityOp), gs)]
    for successes in [int(10 * probability), 5, 0]:
        report = torch.tensor([[successes, 10], [0, 0]], dtype=torch.float64)
        def evidence(g):
            g.ndata["payoffs"] = report
            g.ndata["private_payoffs"] = torch.zeros((2, 2), dtype=torch.float64)
        slow.experiment = fast.experiment = evidence
        if prior == 0 and successes != 10 * probability:
            for m, g in zip((slow, fast), gs):
                with pytest.raises(ImpossibleEvidence):
                    m(g)
            break
        for m, g in zip((slow, fast), gs):
            m(g)
        torch.testing.assert_close(gs[0].ndata["beliefs"], gs[1].ndata["beliefs"])
        torch.testing.assert_close(gs[0].edata["reliability_belief"], gs[1].edata["reliability_belief"])


@pytest.mark.parametrize("operator", [TorchMultiBeliefOp, TorchBayesianReliabilityOp])
def test_sampling_is_torch_seeded_and_has_no_numpy_roundtrip(operator, monkeypatch):
    p = params()
    p.unreliablenodes = [1]
    p.multibelief.source_mode = "opposite"
    gs = [graph() for _ in range(2)]
    # Construction uses the same NumPy-generated reference priors/source identities.
    np.random.seed(40)
    a = operator(gs[0], p)
    np.random.seed(40)
    b = operator(gs[1], p)
    def forbidden(*args, **kwargs):
        raise AssertionError("NumPy called in tensor simulation loop")
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "numpy", forbidden)
        patch.setattr(np.random, "binomial", forbidden)
        for seed in range(12):
            torch.manual_seed(seed)
            a(gs[0])
            torch.manual_seed(seed)
            b(gs[1])
            torch.testing.assert_close(gs[0].ndata["payoffs"], gs[1].ndata["payoffs"], rtol=0, atol=0)
            torch.testing.assert_close(gs[0].ndata["beliefs"], gs[1].ndata["beliefs"], rtol=0, atol=0)
            if operator == TorchBayesianReliabilityOp:
                honest = a.actual_reliability
                torch.testing.assert_close(gs[0].ndata["payoffs"][honest], gs[0].ndata["private_payoffs"][honest])


@pytest.mark.parametrize("operator", [TorchMultiBeliefOp, TorchBayesianReliabilityOp])
def test_empty_edges_and_dogmatic_priors(operator):
    p = params()
    p.multibelief.case = "baseline"
    p.init.constant.value = 1.
    g = dgl.graph(([], []), num_nodes=1)
    m = operator(g, p)
    for _ in range(3):
        m(g)
    assert g.ndata["beliefs"].item() == 1.
    assert m.termination(g, .99, .01) == (False, False, False)
    p.multibelief.stop_on_consensus = True
    assert m.termination(g, .99, .01) == (False, True, False)


@pytest.mark.parametrize("topology", ["cycle", "complete", "random"])
def test_native_baseline_controlled_decisions_and_trajectory(topology):
    n = 60
    rng = np.random.default_rng(123)
    adjacency = np.eye(n, dtype=bool)
    for i in range(n):
        for j in range(i):
            edge = topology == "complete" or (topology == "cycle" and i-j in (1, n-1)) or (topology == "random" and rng.random() < .2)
            adjacency[i, j] = adjacency[j, i] = edge
    p = params()
    p.multibelief.case = "baseline"
    gs = [dgl.graph(np.where(adjacency), num_nodes=n) for _ in range(2)]
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        slow, fast = BalaGoyalOp(gs[0], p), TorchMultiBeliefOp(gs[1], p)
        priors = torch.tensor(rng.uniform(.1, .9, n))
        gs[0].ndata["beliefs"] = priors
        fast.log_joint.copy_(torch.stack((1-priors, priors), 1).log())
        fast._publish(gs[1])
        for _ in range(60):
            torch.testing.assert_close(gs[0].ndata["beliefs"] > .5, fast.active(gs[1]))
            counts = torch.tensor(rng.binomial(10, .55, n))
            slow.sample = lambda: counts
            def evidence(g):
                active = fast.active(g)
                g.ndata["payoffs"] = torch.stack((counts * active, 10 * active), 1).to(torch.float64)
            fast.experiment = evidence
            slow(gs[0])
            fast(gs[1])
            torch.testing.assert_close(gs[0].ndata["payoffs"], gs[1].ndata["payoffs"])
            torch.testing.assert_close(gs[0].ndata["beliefs"], gs[1].ndata["beliefs"], atol=4e-12, rtol=4e-12)
    finally:
        torch.set_default_dtype(old_dtype)


@pytest.mark.parametrize("operator", [TorchMultiBeliefOp, TorchBayesianReliabilityOp])
def test_full_simulation_metadata_and_snapshots(operator, tmp_path):
    p = params()
    p.multibelief.case = "baseline"
    p.op = operator.__name__
    p.network.kind, p.network.size = "cycle", 5
    p.simulation.results = str(tmp_path / "results")
    p.simulation.steps, p.simulation.repeats = 3, 1
    p.logging.enabled = False
    p.snapshots.enabled, p.snapshots.messages = True, True
    polygraphs.simulate(p)
    assert getbyname(operator.__name__) is operator
    metadata = json.loads(next(tmp_path.rglob("*.beliefs.json")).read_text())
    assert metadata["backend"] == "torch"
    assert metadata["dtype"] == "float64"
    with h5py.File(next(tmp_path.rglob("*.hd5"))) as f:
        assert set(f["beliefs"]) == {"1", "2", "3"}
        assert f["beliefs/3"].dtype == np.float64
        assert "payoffs" in f
        assert "belief_joint" in f if operator == TorchMultiBeliefOp else "reliability_belief" in f


def test_validation_preserves_original_params():
    p = params()
    p.multibelief.case = "shared"
    with pytest.raises(ValueError):
        TorchMultiBeliefOp(graph(), p)
    p = params()
    p.device = "cuda"
    with pytest.raises(ValueError, match="device"):
        TorchMultiBeliefOp(graph(), p)
    p = params()
    p.multibelief.backend = "bnlearn"
    TorchMultiBeliefOp(graph(), p)
    assert p.multibelief.backend == "bnlearn"


def test_calibrated_joint_is_restored_and_recorded():
    p = study_params("polarizing")
    g = graph()
    m = TorchMultiBeliefOp(g, p)
    initial = torch.arange(1, 21, dtype=torch.float64).reshape(5, 4)
    initial /= initial.sum(1, keepdim=True)
    m.set_initial_log_joint(g, initial.log())
    torch.testing.assert_close(g.ndata["belief_joint"], initial)
    np.testing.assert_allclose(m.belief_metadata()["initial_joint"], initial)
    with pytest.raises(ValueError, match="shape"):
        m.set_initial_log_joint(g, torch.zeros(5, 2))
    m.experiment = lambda g: g.ndata.__setitem__("payoffs", torch.ones((5, 2), dtype=torch.float64))
    m(g)
    with pytest.raises(ValueError, match="before"):
        m.set_initial_log_joint(g, initial.log())


def test_nonbinary_states_and_heterogeneous_agent_cpts():
    p = params()
    p.multibelief.case, p.multibelief.prior_mode = "custom", "fixed"
    nodes = [{"name": "H", "states": ["low", "medium", "high"], "cpt": [.2, .3, .5]},
             binary_node("S", ["H"], [.2, .7, .8])]
    p.multibelief.model = {"nodes": nodes}
    p.multibelief.agent_models = {"2": {"nodes": [nodes[0], binary_node("S", ["H"], [.8, .3, .2])]}}
    p.multibelief.observation = {"parents": ["S", "H"], "probability": [[.2, .4, .6], [.4, .6, .8]]}
    g, h = graph(), graph()
    slow, fast = MultiBeliefOp(g, p), TorchMultiBeliefOp(h, p)
    payoff = torch.tensor([[2, 5], [3, 5], [1, 5], [0, 0], [4, 5]], dtype=torch.float64)
    slow.experiment = fast.experiment = lambda g: g.ndata.__setitem__("payoffs", payoff)
    for _ in range(20):
        slow(g)
        fast(h)
        torch.testing.assert_close(g.ndata["belief_joint"], h.ndata["belief_joint"], atol=2e-12, rtol=2e-12)
        torch.testing.assert_close(g.ndata["belief:H"], h.ndata["belief:H"], atol=2e-12, rtol=2e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware unavailable")
@pytest.mark.parametrize("operator", [TorchMultiBeliefOp, TorchBayesianReliabilityOp])
def test_cuda_matches_cpu(operator):
    p = params()
    p.multibelief.case = "baseline"
    g = graph()
    np.random.seed(19)
    cpu = operator(g, p)
    p.device = "cuda:0"
    try:
        h = graph().to("cuda:0")
    except dgl.DGLError:
        pytest.skip("Installed DGL wheel has no CUDA support")
    np.random.seed(19)
    gpu = operator(h, p)
    rng = np.random.default_rng(20)
    for _ in range(30):
        payoff = torch.tensor(np.stack((rng.binomial(10, .55, 5), np.full(5, 10)), 1), dtype=torch.float64)
        def evidence(g):
            g.ndata["payoffs"] = payoff.to(g.device)
            g.ndata["private_payoffs"] = payoff.to(g.device)
        cpu.experiment = gpu.experiment = evidence
        cpu(g)
        gpu(h)
        for key in ("beliefs", "belief:H"):
            torch.testing.assert_close(g.ndata[key], h.ndata[key].cpu(), atol=2e-11, rtol=2e-11)
        if operator == TorchBayesianReliabilityOp:
            torch.testing.assert_close(g.edata["reliability_belief"], h.edata["reliability_belief"].cpu(), atol=2e-11, rtol=2e-11)
