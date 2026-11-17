"""Tests through real PolyGraphs operations, graph storage and configuration."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import dgl
import h5py
import numpy as np
import pytest
import torch

import polygraphs as pg
from polygraphs import hyperparameters as hp
from polygraphs import ops
from polygraphs.beliefs import BeliefState, binomial_log_likelihood
from polygraphs.beliefs.cases import bandit, bandit_observation
from polygraphs.ops.multibelief import MultiBeliefOp, BayesianReliabilityOp


def parameters(case="baseline", count=3):
    params = hp.PolyGraphHyperParameters()
    params.op = "MultiBeliefOp"
    params.epsilon = 0.1
    params.trials = 10
    params.init.kind = "constant"
    params.init.constant.value = 0.6
    params.multibelief.case = case
    params.multibelief.backend = "numpy"
    params.multibelief.stop_on_consensus = False
    params.network.kind = "complete"
    params.network.size = count
    params.simulation.steps = 2
    params.simulation.results = None
    params.logging.enabled = False
    return params


def fixed_payoffs(model, payoffs):
    def experiment(graph):
        graph.ndata["payoffs"] = torch.tensor(payoffs, dtype=torch.float64)

    model.experiment = experiment


def test_operator_registration():
    assert ops.getbyname("MultiBeliefOp") is MultiBeliefOp
    assert ops.getbyname("BayesianReliabilityOp") is BayesianReliabilityOp


def test_directed_edges_self_loops_parallel_edges_and_isolated_nodes():
    graph = dgl.graph(([0, 0, 0, 1, 1], [0, 1, 1, 1, 1]), num_nodes=3)
    model = MultiBeliefOp(graph, parameters())
    fixed_payoffs(model, [[8, 10], [4, 10], [7, 10]])
    model(graph)
    scope, probability = bandit_observation(0.1)
    for index, successes, trials in [(0, 8, 10), (1, 12, 20), (2, 7, 10)]:
        reference = BeliefState(bandit().with_priors({"H": float(torch.tensor(0.6))}))
        reference.update(
            [(scope, binomial_log_likelihood(probability, successes, trials))]
        )
        assert graph.ndata["beliefs"][index] == pytest.approx(
            reference.marginal("H")[1]
        )


def test_reports_are_fresh_and_not_forwarded_again():
    graph = dgl.graph(([0, 1], [1, 2]), num_nodes=3)
    model = MultiBeliefOp(graph, parameters())
    fixed_payoffs(model, [[8, 10], [0, 0], [0, 0]])
    model(graph)
    first = graph.ndata["beliefs"].clone()
    assert first[0] == first[1]
    assert first[2] == pytest.approx(0.6)
    fixed_payoffs(model, [[0, 0], [0, 0], [0, 0]])
    model(graph)
    torch.testing.assert_close(first, graph.ndata["beliefs"])


def test_own_data_is_counted_once_with_or_without_self_loops():
    params = parameters()
    graphs = [
        dgl.graph(([], []), num_nodes=3),
        dgl.graph(([0, 1, 2], [0, 1, 2]), num_nodes=3),
    ]
    models = [MultiBeliefOp(graph, params) for graph in graphs]
    for graph, model in zip(graphs, models):
        fixed_payoffs(model, [[8, 10], [4, 10], [7, 10]])
        model(graph)
    torch.testing.assert_close(graphs[0].ndata["beliefs"], graphs[1].ndata["beliefs"])


def test_expected_payoff_uses_joint_not_independent_marginals():
    params = parameters("meta", count=1)
    params.multibelief.doubtful_epsilon = 0.0
    params.multibelief.priors = {"H": 0.9, "S": 0.9}
    graph = dgl.graph(([], []), num_nodes=1)
    model = MultiBeliefOp(graph, params)
    fixed_payoffs(model, [[1, 10]])
    model(graph)
    state = model.states[0]
    correct = state.expectation(model.observation_scope, model.observation_probability)
    naive = 0.5 + 0.1 * (2 * state.marginal("H")[1] - 1) * state.marginal("S")[1]
    assert graph.ndata["expected_payoff"][0] == pytest.approx(correct)
    assert abs(correct - naive) > 0.001


def test_factionalization_common_evidence_does_not_depend_on_graph_degree():
    params = parameters("factionalization2")
    params.multibelief.prior_mode = "halton"
    graphs = [dgl.graph(([], []), num_nodes=3), pg.graphs.complete_(3)]
    models = [MultiBeliefOp(graph, params) for graph in graphs]
    for graph, model in zip(graphs, models):
        pg.simulate_(graph, model, steps=2)
    torch.testing.assert_close(
        graphs[0].ndata["belief_joint"], graphs[1].ndata["belief_joint"]
    )


def test_shared_evidence_runs_full_horizon_even_when_target_is_certain():
    params = parameters("factionalization1")
    params.multibelief.shared_evidence = [{"variables": ["H2"], "likelihood": [0, 1]}]
    graph = pg.graphs.complete_(3)
    model = MultiBeliefOp(graph, params)
    result = pg.simulate_(graph, model, steps=4)
    assert result[0] == 4
    assert result[2] == "?"
    assert not result[4]


def test_propaganda_updates_connected_beliefs_and_prevents_false_absorption():
    params = parameters("meta", count=1)
    params.multibelief.priors = {"H": 0.7, "S": 0.8}
    params.multibelief.propaganda.probability = 1
    params.multibelief.stop_on_consensus = True
    graph = dgl.graph(([], []), num_nodes=1)
    model = MultiBeliefOp(graph, params)
    fixed_payoffs(model, [[7, 10]])
    model(graph)
    previous = graph.ndata["beliefs"].clone()
    fixed_payoffs(model, [[0, 0]])
    model(graph)
    assert not torch.equal(previous, graph.ndata["beliefs"])
    assert model.termination(graph, 0.99, 0.5)[1] is False


def test_reliability_edges_store_recipient_belief_in_source():
    params = parameters(count=3)
    graph = dgl.graph(([0, 1], [1, 0]), num_nodes=3)
    model = BayesianReliabilityOp(graph, params)

    def experiment(graph):
        graph.ndata["payoffs"] = torch.tensor(
            [[9, 10], [1, 10], [0, 0]], dtype=torch.float64
        )
        graph.ndata["private_payoffs"] = graph.ndata["payoffs"].clone()

    model.experiment = experiment
    model(graph)
    assert graph.edata["reliability_belief"][0] == pytest.approx(
        model.states[1].reliability(0)
    )
    assert graph.edata["reliability_belief"][1] == pytest.approx(
        model.states[0].reliability(1)
    )
    assert graph.ndata["beliefs"][2] == pytest.approx(0.6)


def test_known_reliable_source_matches_baseline_with_same_observations():
    params = parameters(count=2)
    params.bayesian_reliability.prior = 1
    first, second = pg.graphs.complete_(2), pg.graphs.complete_(2)
    baseline = MultiBeliefOp(first, params)
    reliability = BayesianReliabilityOp(second, params)
    fixed_payoffs(baseline, [[8, 10], [6, 10]])

    def experiment(graph):
        graph.ndata["payoffs"] = torch.tensor([[8, 10], [6, 10]], dtype=torch.float64)
        graph.ndata["private_payoffs"] = graph.ndata["payoffs"].clone()

    reliability.experiment = experiment
    baseline(first)
    reliability(second)
    torch.testing.assert_close(first.ndata["beliefs"], second.ndata["beliefs"])


def test_real_simulation_stores_all_beliefs_and_metadata(tmp_path):
    params = parameters("meta")
    params.simulation.results = str(tmp_path / "result")
    params.snapshots.enabled = True
    pg.random(123)
    result = pg.simulate(params)
    assert result.frame.iloc[0]["steps"] == 2
    output = Path(params.simulation.results)
    metadata = json.loads((output / "1.beliefs.json").read_text())
    assert metadata["nodes"] == ["H", "S"]
    graph = dgl.load_graphs(str(output / "1.bin"))[0][0]
    assert "belief:S" in graph.ndata
    with h5py.File(output / "1.hd5") as stream:
        assert stream["beliefs_by_node/S/2"].shape == (3, 2)
        assert stream["belief_joint/2"].shape == (3, 4)
        np.testing.assert_allclose(stream["belief_joint/2"][:].sum(axis=1), 1)
        assert stream["expected_payoff/2"].shape == (3,)
    analysis = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parents[1] / "examples/multibelief/analyse.py"),
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert analysis.returncode == 0, analysis.stderr
    assert "joint_js" in analysis.stdout
    assert len(analysis.stdout.strip().splitlines()) == 4


def test_source_reliability_saved_in_snapshots(tmp_path):
    params = parameters(count=2)
    params.op = "BayesianReliabilityOp"
    params.simulation.results = str(tmp_path / "result")
    params.snapshots.enabled = True
    pg.simulate(params)
    with h5py.File(Path(params.simulation.results) / "1.hd5") as stream:
        assert "reliability_belief/2" in stream
        assert "private_payoffs/2" in stream


def test_seed_reproduces_initialization_and_updates():
    outcomes = []
    for _ in range(2):
        pg.random(23)
        graph = pg.graphs.complete_(3)
        model = MultiBeliefOp(graph, parameters("meta"))
        model(graph)
        outcomes.append(graph.ndata["belief_joint"].clone())
    torch.testing.assert_close(*outcomes, rtol=0, atol=0)


def test_existing_bala_goyal_operation_still_runs():
    params = parameters()
    params.op = "BalaGoyalOp"
    result = pg.simulate(params)
    assert result.frame.iloc[0]["undefined"] == False


def test_direct_runner_uses_the_configured_convention_for_thresholds():
    params = parameters()
    params.multibelief.stop_on_consensus = True
    graph = dgl.graph(([], []), num_nodes=3)
    model = MultiBeliefOp(graph, params)
    fixed_payoffs(model, [[6, 10], [6, 10], [6, 10]])
    result = pg.simulate_(graph, model, steps=2)
    assert result[0] == 2
    assert not result[4]


def test_fixed_horizon_is_required_for_shared_and_propaganda():
    params = parameters("factionalization1")
    params.simulation.steps = 0
    with pytest.raises(ValueError, match="simulation.steps"):
        MultiBeliefOp(pg.graphs.complete_(3), params)


def test_agent_specific_cpts_are_respected():
    params = parameters("shared", count=2)
    params.multibelief.model = {
        "nodes": [
            {"name": "H", "cpt": [0.5, 0.5]},
            {"name": "D", "parents": ["H"], "cpt": [[0.9, 0.1], [0.1, 0.9]]},
        ]
    }
    params.multibelief.prior_mode = "fixed"
    params.multibelief.agent_models = {
        "1": {
            "nodes": [
                {"name": "H", "cpt": [0.5, 0.5]},
                {"name": "D", "parents": ["H"], "cpt": [[0.1, 0.9], [0.9, 0.1]]},
            ]
        }
    }
    params.multibelief.shared_evidence = [{"variables": ["D"], "likelihood": [0, 1]}]
    graph = pg.graphs.complete_(2)
    model = MultiBeliefOp(graph, params)
    model(graph)
    np.testing.assert_allclose(graph.ndata["beliefs"].numpy(), [0.9, 0.1])


@pytest.mark.parametrize("operator", [MultiBeliefOp, BayesianReliabilityOp])
@pytest.mark.parametrize("steps", [-1, 0.5, True])
def test_invalid_horizons_are_rejected(operator, steps, tmp_path):
    params = parameters()
    configuration = tmp_path / "invalid.json"
    configuration.write_text(json.dumps({"simulation": {"steps": steps}}))
    hp.PolyGraphHyperParameters.fromJSON(str(configuration), params)
    with pytest.raises(ValueError, match="simulation.steps"):
        operator(pg.graphs.complete_(3), params)


def test_reliability_rejects_unsupported_settings():
    params = parameters("meta")
    with pytest.raises(ValueError, match="unsupported"):
        BayesianReliabilityOp(pg.graphs.complete_(3), params)


def test_runner_closes_resources_if_initial_storage_fails(monkeypatch, tmp_path):
    closed = []

    class ResourceOp(MultiBeliefOp):
        def close(self):
            closed.append(True)
            super().close()

    params = parameters()
    params.op = None
    params.simulation.results = str(tmp_path / "result")

    def failed_store(*args):
        raise OSError("storage failure")

    monkeypatch.setattr(pg, "_storegraph", failed_store)
    with pytest.raises(OSError, match="storage failure"):
        pg.simulate(params, op=ResourceOp)
    assert closed == [True]


EXAMPLES = sorted(
    (Path(__file__).parents[1] / "examples" / "multibelief").glob("*.yaml")
)


@pytest.mark.parametrize("filename", EXAMPLES, ids=lambda path: path.stem)
@pytest.mark.parametrize(
    "backend", ["numpy", pytest.param("bnlearn", marks=pytest.mark.bnlearn)]
)
def test_each_standalone_example_runs_two_steps(filename, backend, tmp_path):
    if backend == "bnlearn" and not (
        os.environ.get("POLYGRAPHS_RSCRIPT") or shutil.which("Rscript")
    ):
        pytest.skip("Rscript is not installed")
    params = hp.PolyGraphHyperParameters.load([str(filename)])
    params.multibelief.backend = backend
    params.multibelief.stop_on_consensus = False
    params.simulation.steps = 2
    params.simulation.results = str(tmp_path / "result")
    params.logging.enabled = False
    pg.random(params.seed)
    if params.op.startswith("Batched") and backend == "bnlearn":
        with pytest.raises(ValueError, match="require the numpy backend"):
            pg.simulate(params)
        return
    result = pg.simulate(params)
    assert result.frame.iloc[0]["steps"] == 2
    assert not result.frame.iloc[0]["undefined"]
    with h5py.File(Path(params.simulation.results) / "1.hd5") as stream:
        assert np.isfinite(stream["beliefs/2"][:]).all()


def test_installed_cli_loads_example_and_overlay(tmp_path):
    overlay = tmp_path / "test.yaml"
    overlay.write_text(
        "multibelief.backend: numpy\nmultibelief.stop_on_consensus: false\n"
        "simulation.steps: 2\nsimulation.results: null\n"
        "snapshots.enabled: false\nlogging.enabled: false\n"
    )
    result = subprocess.run(
        [sys.executable, "-m", "polygraphs.run", "-f", str(EXAMPLES[0]), str(overlay)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "Bye." in result.stdout


@pytest.mark.parametrize("case", [1, 2, 3])
def test_twenty_factionalization_updates_match_closed_form(case):
    params = hp.PolyGraphHyperParameters.load(
        [
            str(
                Path(__file__).parents[1]
                / "examples/multibelief"
                / f"shared-evidence-factionalization-{case}.yaml"
            )
        ]
    )
    params.multibelief.backend = "numpy"
    params.simulation.results = None
    graph = pg.graphs.create(params.network)
    model = MultiBeliefOp(graph, params)
    initial = [state.joint.copy() for state in model.states]
    for _ in range(20):
        model(graph)
    for state, prior in zip(model.states, initial):
        axis = state.network.names.index("H2")
        shape = [1] * prior.ndim
        shape[axis] = 2
        expected = prior * np.array([0.35, 0.65]).reshape(shape) ** 20
        expected /= expected.sum()
        np.testing.assert_allclose(state.joint, expected, atol=1e-13)


@pytest.mark.parametrize("case", [1, 2, 3])
def test_networked_factionalization_uses_only_own_and_incoming_evidence(case):
    params = parameters(f"networked-factionalization{case}")
    params.multibelief.prior_mode = "halton"
    graph = dgl.graph(([0], [1]), num_nodes=3)
    model = MultiBeliefOp(graph, params)
    before = [state.joint.copy() for state in model.states]
    assert not model.shared
    assert model.true_probability == 0.65
    fixed_payoffs(model, [[7, 10], [0, 0], [0, 0]])
    model(graph)
    for index in (0, 1):
        shape = [1] * before[index].ndim
        shape[model.network.names.index("H2")] = 2
        weights = (np.array([0.35, 0.65]) ** 7 * np.array([0.65, 0.35]) ** 3).reshape(
            shape
        )
        expected = before[index] * weights
        expected /= expected.sum()
        np.testing.assert_allclose(model.states[index].joint, expected, atol=1e-13)
    np.testing.assert_array_equal(model.states[2].joint, before[2])
