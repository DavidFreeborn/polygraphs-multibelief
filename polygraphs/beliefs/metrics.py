"""Population diagnostics for joint and marginal belief distributions."""

from itertools import combinations

import numpy as np
from scipy.special import rel_entr


def jensen_shannon(first, second):
    """Jensen-Shannon divergence in nats, bounded by log(2)."""
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if (
        first.shape != second.shape
        or not np.isfinite(first).all()
        or not np.isfinite(second).all()
        or np.any(first < 0)
        or np.any(second < 0)
        or not np.isclose(first.sum(), 1)
        or not np.isclose(second.sum(), 1)
    ):
        raise ValueError("Expected probability distributions on the same support")
    # A positive subnormal probability can become zero when divided by two.
    # Forming the mixture in log space avoids a spurious infinite divergence.
    with np.errstate(divide="ignore"):
        log_first, log_second = np.log(first), np.log(second)
        log_mixture = np.logaddexp(log_first, log_second) - np.log(2.)
    a, b = first > 0, second > 0
    value = (np.sum(first[a] * (log_first[a] - log_mixture[a]))
             + np.sum(second[b] * (log_second[b] - log_mixture[b]))) / 2
    return max(0., float(value))


def kullback_leibler(first, second):
    """Directed KL(P || Q), in nats; infinite for unmatched positive support."""
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if (
        first.shape != second.shape
        or not np.isfinite(first).all()
        or not np.isfinite(second).all()
        or np.any(first < 0)
        or np.any(second < 0)
        or not np.isclose(first.sum(), 1)
        or not np.isclose(second.sum(), 1)
    ):
        raise ValueError("Expected probability distributions on the same support")
    return float(rel_entr(first, second).sum())


def population_diagnostics(states, variables=None):
    """Mean pairwise JS for joints and products of marginals, plus covariance."""
    if len(states) < 2:
        raise ValueError("Population diagnostics require at least two agents")
    variables = tuple(variables or states[0].network.names)
    if any(
        state.network.shape(variables) != states[0].network.shape(variables)
        for state in states
    ):
        raise ValueError(
            "Agents must use the same state spaces for diagnostic variables"
        )
    return distribution_diagnostics(
        np.array([state.marginal(variables) for state in states])
    )


def distribution_diagnostics(joints):
    """Diagnostics for an array with axes (agent, variable_1, ..., variable_k)."""
    joints = np.asarray(joints, dtype=float)
    if (
        joints.ndim < 2
        or len(joints) < 2
        or not np.isfinite(joints).all()
        or np.any(joints < 0)
        or any(card < 2 for card in joints.shape[1:])
        or not np.allclose(joints.reshape(len(joints), -1).sum(axis=1), 1)
    ):
        raise ValueError("Expected at least two normalized joint distributions")
    products, beliefs = [], []
    for joint in joints:
        marginals = [
            joint.sum(axis=tuple(axis for axis in range(joint.ndim) if axis != index))
            for index in range(joint.ndim)
        ]
        independent = marginals[0]
        for marginal in marginals[1:]:
            independent = np.multiply.outer(independent, marginal)
        products.append(independent)
        beliefs.append([marginal[1] for marginal in marginals])
    pairs = list(combinations(range(len(joints)), 2))
    beliefs = np.array(beliefs)
    covariance = np.atleast_2d(np.cov(beliefs, rowvar=False, bias=True))
    off_diagonal = covariance[np.triu_indices(joints.ndim - 1, k=1)]
    return {
        "joint_js": float(
            np.mean([jensen_shannon(joints[i], joints[j]) for i, j in pairs])
        ),
        "marginal_product_js": float(
            np.mean([jensen_shannon(products[i], products[j]) for i, j in pairs])
        ),
        "joint_kl": float(
            np.mean(
                [
                    (
                        kullback_leibler(joints[i], joints[j])
                        + kullback_leibler(joints[j], joints[i])
                    )
                    / 2
                    for i, j in pairs
                ]
            )
        ),
        "marginal_product_kl": float(
            np.mean(
                [
                    (
                        kullback_leibler(products[i], products[j])
                        + kullback_leibler(products[j], products[i])
                    )
                    / 2
                    for i, j in pairs
                ]
            )
        ),
        "mean_variance": float(np.var(beliefs, axis=0).mean()),
        "mean_absolute_covariance": (
            float(np.abs(off_diagonal).mean()) if off_diagonal.size else 0.0
        ),
    }
