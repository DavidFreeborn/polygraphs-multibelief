"""Belief networks for the multibelief and factionalization examples."""

import numpy as np

from .network import BayesianNetwork


def binary_node(name, parents=(), probability=0.5):
    probability = np.asarray(probability, dtype=float)
    return {
        "name": name,
        "parents": list(parents),
        "cpt": np.stack((1 - probability, probability), axis=-1).tolist(),
    }


def bandit(meta=False):
    nodes = [binary_node("H")]
    if meta:
        nodes.append(binary_node("S"))
    return BayesianNetwork(nodes)


def bandit_observation(epsilon, doubtful_epsilon=None):
    if not 0 < epsilon < 0.5:
        raise ValueError("epsilon must lie strictly between 0 and 0.5")
    if doubtful_epsilon is None:
        return ("H",), np.array([0.5 - epsilon, 0.5 + epsilon])
    if not 0 <= doubtful_epsilon <= epsilon:
        raise ValueError("doubtful_epsilon must lie between 0 and epsilon")
    return ("H", "S"), np.array(
        [
            [0.5 - doubtful_epsilon, 0.5 - epsilon],
            [0.5 + doubtful_epsilon, 0.5 + epsilon],
        ]
    )


def factionalization(case):
    """Networks from Freeborn (2024), section 4, figures 2, 4 and 6."""
    if case == 1:
        nodes = [binary_node("H1"), binary_node("H2", ["H1"], [0.1, 0.9])]
    elif case == 2:
        nodes = [
            binary_node("H1"),
            binary_node("H3"),
            binary_node("H2", ["H1", "H3"], [[0.9, 0.1], [0.1, 0.9]]),
        ]
    elif case == 3:
        nodes = [
            binary_node("H1"),
            binary_node("H4"),
            binary_node("H5"),
            binary_node("H3", ["H4", "H5"], [[0.9, 0.1], [0.1, 0.9]]),
            binary_node("H2", ["H1", "H3"], [[0.9, 0.1], [0.1, 0.9]]),
        ]
    else:
        raise ValueError("Factionalization case must be 1, 2 or 3")
    return BayesianNetwork(nodes)
