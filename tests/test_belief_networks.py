"""Analytical and independent-engine checks for Bayesian updates."""

from copy import deepcopy
import os
import shutil

import numpy as np
import pytest
from scipy.special import expit, logit

from polygraphs.beliefs import (
    BayesianNetwork,
    BeliefState,
    ImpossibleEvidence,
    binomial_log_likelihood,
)
from polygraphs.beliefs.bnlearn import BnlearnBackend
from polygraphs.beliefs.cases import (
    bandit,
    bandit_observation,
    binary_node,
    factionalization,
)
from polygraphs.beliefs.metrics import jensen_shannon, population_diagnostics
from polygraphs.beliefs.reliability import ReliabilityState


@pytest.fixture(scope="module")
def r_backend():
    if not (os.environ.get("POLYGRAPHS_RSCRIPT") or shutil.which("Rscript")):
        pytest.skip("Rscript is not installed")
    with BnlearnBackend() as backend:
        yield backend


def test_baseline_matches_analytical_binomial_posterior():
    state = BeliefState(bandit().with_priors({"H": 0.37}))
    scope, probability = bandit_observation(0.1)
    state.update([(scope, binomial_log_likelihood(probability, 67, 100))])
    expected = expit(logit(0.37) + 34 * np.log(0.6 / 0.4))
    assert state.marginal("H")[1] == pytest.approx(expected, abs=1e-14)


def test_sequential_batch_and_reordered_updates_agree():
    network = bandit(True).with_priors({"H": 0.31, "S": 0.73})
    scope, probability = bandit_observation(0.2, 0.04)
    sequence = [(1, 1), (0, 1), (6, 9), (2, 7)]
    sequential, batch, reordered = [BeliefState(network) for _ in range(3)]
    for successes, trials in sequence:
        sequential.update(
            [(scope, binomial_log_likelihood(probability, successes, trials))]
        )
    batch.update([(scope, binomial_log_likelihood(probability, 9, 18))])
    for successes, trials in reversed(sequence):
        reordered.update(
            [(scope, binomial_log_likelihood(probability, successes, trials))]
        )
    np.testing.assert_allclose(sequential.joint, batch.joint, atol=1e-14)
    np.testing.assert_allclose(sequential.joint, reordered.joint, atol=1e-14)
    np.testing.assert_allclose(sequential.joint, sequential.replay(), atol=1e-14)


def test_collapsed_likelihood_equals_distinct_observed_data_nodes():
    network = bandit(True).with_priors({"H": 0.3, "S": 0.8})
    scope, probability = bandit_observation(0.2, 0.05)
    collapsed = BeliefState(network)
    collapsed.update([(scope, binomial_log_likelihood(probability, 2, 3))])
    expanded = BeliefState(
        BayesianNetwork(
            network.nodes + [binary_node(f"D{i}", scope, probability) for i in range(3)]
        )
    )
    expanded.update(
        [
            ((f"D{i}",), np.array([0, -np.inf] if value == 0 else [-np.inf, 0]))
            for i, value in enumerate([1, 0, 1])
        ]
    )
    np.testing.assert_allclose(collapsed.joint, expanded.marginal(scope), atol=1e-14)


def test_marginal_only_reinitialization_is_not_bayesian_iteration():
    network = bandit(True).with_priors({"H": 0.4, "S": 0.7})
    scope, probability = bandit_observation(0.3, 0.0)
    correct = BeliefState(network)
    correct.update([(scope, binomial_log_likelihood(probability, 3, 3))])
    naive = BeliefState(
        network.with_priors({name: correct.marginal(name)[1] for name in network.names})
    )
    assert not np.allclose(correct.joint, naive.joint)
    evidence = [(scope, binomial_log_likelihood(probability, 0, 4))]
    correct.update(evidence)
    naive.update(evidence)
    assert abs(correct.marginal("H")[1] - naive.marginal("H")[1]) > 0.01


def test_long_evidence_histories_preserve_log_information_and_allow_reversal():
    state = BeliefState(bandit().with_priors({"H": 0.3}))
    scope, probability = bandit_observation(0.1)
    state.update([(scope, binomial_log_likelihood(probability, 10000, 10000))])
    assert state.marginal("H")[1] == 1.0
    state.update([(scope, binomial_log_likelihood(probability, 0, 10000))])
    assert state.marginal("H")[1] == pytest.approx(0.3, abs=1e-11)
    assert len(state.factors) == 1


def test_impossible_evidence_is_atomic():
    state = BeliefState(bandit().with_priors({"H": 0.0}))
    initial = state.joint.copy()
    with pytest.raises(ImpossibleEvidence):
        state.update([(("H",), np.array([-np.inf, 0]))])
    np.testing.assert_array_equal(state.joint, initial)
    assert state.factors == {}


def test_likelihood_scale_and_no_data_do_not_change_answer():
    a, b = BeliefState(bandit()), BeliefState(bandit())
    a.update([(("H",), np.log([0.2, 0.6]))])
    b.update([(("H",), np.log([2, 6]))])
    np.testing.assert_allclose(a.joint, b.joint)
    b.update([(("H",), binomial_log_likelihood([0, 1], 0, 0))])
    np.testing.assert_allclose(a.joint, b.joint)


@pytest.mark.parametrize(
    "successes,trials", [(1.2, 3), (-1, 2), (3, 2), (0, -1), (np.nan, 3)]
)
def test_invalid_counts_are_rejected(successes, trials):
    with pytest.raises(ValueError):
        binomial_log_likelihood([0.3, 0.7], successes, trials)


def test_network_validation_and_no_mutation_of_specification():
    nodes = [binary_node("A"), binary_node("B", ["A"], [0.2, 0.8])]
    original = deepcopy(nodes)
    network = BayesianNetwork(nodes)
    network.with_priors({"A": 0.9})
    assert nodes == original
    for invalid in [
        nodes + [binary_node("A")],
        [binary_node("A", ["B"], [0.1, 0.9]), binary_node("B", ["A"], [0.2, 0.8])],
        [binary_node("A", ["missing"], [0.1, 0.9])],
        [{"name": "A", "cpt": [0.1, 0.1]}],
    ]:
        with pytest.raises(ValueError):
            BayesianNetwork(invalid)


def test_categorical_nodes_and_reversed_query_order():
    network = BayesianNetwork(
        [
            {"name": "A", "states": ["a", "b", "c"], "cpt": [0.2, 0.3, 0.5]},
            binary_node("B", ["A"], [0.1, 0.4, 0.8]),
        ]
    )
    state = BeliefState(network)
    state.update([(("B",), np.array([-np.inf, 0]))])
    np.testing.assert_allclose(state.marginal("A"), np.array([0.02, 0.12, 0.4]) / 0.54)
    np.testing.assert_allclose(state.marginal(("B", "A")), state.joint.T)


def test_reliability_factorization_equals_full_joint_and_preserves_cross_source_effects():
    specialized = ReliabilityState(0.45, {2: 0.7, 4: 0.4})
    generic = BeliefState(specialized.network)
    probability = np.array([[0.5, 0.3], [0.5, 0.7]])
    for evidence in [
        [(("H", "R2"), binomial_log_likelihood(probability, 7, 10))],
        [(("H", "R4"), binomial_log_likelihood(probability, 1, 10))],
        [(("H",), binomial_log_likelihood([0.3, 0.7], 6, 10))],
    ]:
        specialized.update(evidence)
        generic.update(evidence)
        assert specialized.belief == pytest.approx(generic.marginal("H")[1], abs=1e-14)
        for source in [2, 4]:
            np.testing.assert_allclose(
                specialized.pair(source),
                generic.marginal(("H", f"R{source}")),
                atol=1e-14,
            )
    before = specialized.reliability(2)
    specialized.update([(("H",), np.log([0.01, 0.99]))])
    assert abs(before - specialized.reliability(2)) > 0.01


def test_agreement_and_disagreement_change_reliability_by_bayes_rule():
    agreed, disagreed = ReliabilityState(0.9, {1: 0.5}), ReliabilityState(0.9, {1: 0.5})
    probability = np.array([[0.5, 0.2], [0.5, 0.8]])
    agreed.update([(("H", "R1"), binomial_log_likelihood(probability, 1, 1))])
    disagreed.update([(("H", "R1"), binomial_log_likelihood(probability, 0, 1))])
    assert agreed.reliability(1) > 0.5
    assert disagreed.reliability(1) < 0.5
    assert agreed.reliability(1) == pytest.approx(0.74 / (0.74 + 0.5))


def test_known_unreliable_reports_are_uninformative_or_reversed_as_specified():
    uninformative = ReliabilityState(0.6, {1: 0.0})
    opposite = ReliabilityState(0.6, {1: 0.0})
    uninformative.update([(("H", "R1"), np.log([[0.5, 0.2], [0.5, 0.8]]))])
    opposite.update([(("H", "R1"), np.log([[0.8, 0.2], [0.2, 0.8]]))])
    assert uninformative.belief == pytest.approx(0.6)
    assert opposite.belief < 0.6


def test_reliability_model_scales_without_enumerating_joint():
    state = ReliabilityState(0.5, {source: 0.6 for source in range(100)})
    state.update(
        [
            (("H", f"R{source}"), np.log([[0.5, 0.45], [0.5, 0.55]]))
            for source in range(100)
        ]
    )
    assert state.belief > 0.9
    assert len(state.log_r_given_h) == 100


def test_js_uses_standard_nonnegative_definition_and_bound():
    assert jensen_shannon([1, 0], [0, 1]) == pytest.approx(np.log(2))
    assert jensen_shannon([0.2, 0.8], [0.2, 0.8]) == 0
    states = [
        BeliefState(factionalization(2).with_priors({"H1": h, "H3": s}))
        for h, s in [(0.2, 0.3), (0.8, 0.7)]
    ]
    result = population_diagnostics(states)
    assert 0 <= result["joint_js"] <= np.log(2)
    from scipy.stats import entropy
    from polygraphs.beliefs.metrics import kullback_leibler

    first, second = states[0].joint.ravel(), states[1].joint.ravel()
    assert kullback_leibler(first, second) == pytest.approx(entropy(first, second))
    expected = (entropy(first, second) + entropy(second, first)) / 2
    assert result["joint_kl"] == pytest.approx(expected)
    assert np.isinf(kullback_leibler([1, 0], [0, 1]))
    assert 0 <= result["marginal_product_js"] <= np.log(2)


@pytest.mark.bnlearn
@pytest.mark.parametrize("case", [1, 2, 3])
def test_bnlearn_matches_python_for_factionalization(case, r_backend):
    network = factionalization(case)
    priors = {name: value for name, value in zip(network.roots, [0.23, 0.71, 0.46])}
    network = network.with_priors(priors)
    reference, actual = BeliefState(network), BeliefState(network, r_backend)
    for weights in [[0.35, 0.65], [0.8, 0.2], [0.1, 0.9]]:
        evidence = [(("H2",), np.log(weights))]
        reference.update(evidence)
        actual.update(evidence)
        np.testing.assert_allclose(actual.joint, reference.joint, atol=2e-12)


@pytest.mark.bnlearn
def test_bnlearn_matches_python_for_repeated_trials_and_propaganda(r_backend):
    network = bandit(True).with_priors({"H": 0.23, "S": 0.71})
    reference, actual = BeliefState(network), BeliefState(network, r_backend)
    scope, probability = bandit_observation(0.2, 0.01)
    for successes in [7, 2, 9, 5]:
        evidence = [
            (scope, binomial_log_likelihood(probability, successes, 10)),
            (("S",), np.log([1, 0.5])),
        ]
        reference.update(evidence)
        actual.update(evidence)
        np.testing.assert_allclose(actual.joint, reference.joint, atol=2e-12)


@pytest.mark.bnlearn
def test_bnlearn_reliability_matches_full_joint(r_backend):
    state = ReliabilityState(0.3, {1: 0.7, 2: 0.6, 3: 0.4}, r_backend)
    reference = BeliefState(state.network)
    evidence = [
        (("H", "R1"), np.log([[0.5, 0.3], [0.5, 0.7]])),
        (("H", "R2"), np.log([[0.5, 0.7], [0.5, 0.3]])),
    ]
    state.update(evidence)
    reference.update(evidence)
    assert state.belief == pytest.approx(reference.marginal("H")[1], abs=2e-12)
    for source in state.sources:
        np.testing.assert_allclose(
            state.pair(source), reference.marginal(("H", f"R{source}")), atol=2e-12
        )


@pytest.mark.bnlearn
def test_bnlearn_categorical_asymmetric_cpts_and_parent_order(r_backend):
    network = BayesianNetwork(
        [
            {"name": "A", "states": ["a", "b", "c"], "cpt": [0.1, 0.3, 0.6]},
            binary_node("B", probability=0.7),
            binary_node("C", ["B", "A"], [[0.1, 0.2, 0.3], [0.4, 0.8, 0.9]]),
        ]
    )
    reference, actual = BeliefState(network), BeliefState(network, r_backend)
    evidence = [(("C",), np.log([0.25, 0.75]))]
    reference.update(evidence)
    actual.update(evidence)
    np.testing.assert_allclose(actual.joint, reference.joint, atol=2e-12)
