"""Small discrete networks with explicit likelihood evidence."""

from copy import deepcopy
from itertools import product
import math

import numpy as np
from scipy.special import logsumexp, xlogy


class ImpossibleEvidence(ValueError):
    """The evidence has zero probability under the agent's model."""


def binomial_log_likelihood(probability, successes, trials):
    """Log likelihood up to the state-independent binomial coefficient."""
    if (
        isinstance(trials, bool)
        or isinstance(successes, bool)
        or not np.isfinite([successes, trials]).all()
        or int(trials) != trials
        or int(successes) != successes
        or not 0 <= successes <= trials
    ):
        raise ValueError("Counts must be integers satisfying 0 <= successes <= trials")
    probability = np.asarray(probability, dtype=float)
    if not np.isfinite(probability).all() or np.any(
        (probability < 0) | (probability > 1)
    ):
        raise ValueError("Probabilities must be finite and in [0, 1]")
    return xlogy(successes, probability) + xlogy(trials - successes, 1 - probability)


def normalize(log_weights):
    total = logsumexp(log_weights)
    if not np.isfinite(total):
        raise ImpossibleEvidence(
            "Evidence has zero probability under the current model"
        )
    return log_weights - total


class BayesianNetwork:
    """A finite DAG. CPT axes are parents in order, followed by the child."""

    def __init__(self, nodes):
        self.nodes = deepcopy(list(nodes))
        self.names = tuple(node["name"] for node in self.nodes)
        if not self.names or len(set(self.names)) != len(self.names):
            raise ValueError("Network requires distinct node names")
        if any(
            not isinstance(name, str) or not name or name.startswith("__evidence_")
            for name in self.names
        ):
            raise ValueError(
                "Node names must be nonempty strings without the __evidence_ prefix"
            )
        self.cards = {}
        for node in self.nodes:
            unknown = set(node) - {"name", "parents", "states", "cpt"}
            if unknown:
                raise ValueError(f"Unknown node fields: {sorted(unknown)}")
            node.setdefault("parents", [])
            node.setdefault("states", ["0", "1"])
            states = node["states"]
            if (
                len(states) < 2
                or len(set(states)) != len(states)
                or any(not isinstance(state, str) for state in states)
            ):
                raise ValueError(
                    "States must be distinct strings; at least two per node"
                )
            self.cards[node["name"]] = len(states)
        for node in self.nodes:
            parents = node["parents"]
            if (
                len(set(parents)) != len(parents)
                or node["name"] in parents
                or not set(parents) <= set(self.names)
            ):
                raise ValueError(f"Invalid parents for {node['name']}")
            cpt = np.asarray(node["cpt"], dtype=float)
            shape = self.shape(tuple(parents) + (node["name"],))
            if (
                cpt.shape != shape
                or not np.isfinite(cpt).all()
                or np.any(cpt < 0)
                or np.any(cpt > 1)
                or not np.allclose(cpt.sum(axis=-1), 1, rtol=0, atol=1e-12)
            ):
                raise ValueError(
                    f"Invalid CPT for {node['name']}; expected shape {shape}"
                )
            node["cpt"] = cpt.tolist()
        remaining = {node["name"]: set(node["parents"]) for node in self.nodes}
        while remaining:
            roots = {name for name, parents in remaining.items() if not parents}
            if not roots:
                raise ValueError("Belief network must be acyclic")
            remaining = {
                name: parents - roots
                for name, parents in remaining.items()
                if name not in roots
            }

    @property
    def roots(self):
        return tuple(node["name"] for node in self.nodes if not node["parents"])

    def shape(self, scope):
        if len(set(scope)) != len(scope) or not set(scope) <= set(self.names):
            raise ValueError(f"Invalid variable scope: {scope}")
        return tuple(self.cards[name] for name in scope)

    def with_priors(self, priors):
        if not set(priors) <= set(self.roots):
            raise ValueError("Only root priors can be replaced")
        nodes = deepcopy(self.nodes)
        for node in nodes:
            if node["name"] in priors:
                prior = priors[node["name"]]
                node["cpt"] = [1 - prior, prior] if np.isscalar(prior) else list(prior)
        return BayesianNetwork(nodes)

    def assignments(self, max_states=65536):
        shape = self.shape(self.names)
        if math.prod(shape) > max_states:
            raise ValueError(f"Joint distribution exceeds max_states={max_states}")
        return np.array(list(product(*(range(card) for card in shape))), dtype=int)

    def log_joint(self, assignments):
        result = np.zeros(len(assignments))
        for node in self.nodes:
            scope = tuple(node["parents"]) + (node["name"],)
            columns = tuple(assignments[:, self.names.index(name)] for name in scope)
            with np.errstate(divide="ignore"):
                result += np.log(np.asarray(node["cpt"])[columns])
        return normalize(result)


class BeliefState:
    """Exact joint posterior, retaining dependencies induced by past evidence."""

    def __init__(self, network, backend=None, max_states=65536):
        self.network = network
        self.backend = backend
        self.assignments = network.assignments(max_states)
        self.initial_log_joint = network.log_joint(self.assignments)
        self.log_joint = self.initial_log_joint.copy()
        self.factors = {}
        if backend is not None:
            self._infer(self.factors)

    @property
    def joint(self):
        return np.exp(self.log_joint).reshape(self.network.shape(self.network.names))

    def _expanded(self, scope, values):
        columns = tuple(
            self.assignments[:, self.network.names.index(name)] for name in scope
        )
        return values[columns]

    def _infer(self, factors):
        if self.backend is None:
            result = self.initial_log_joint.copy()
            for scope, values in factors.items():
                result += self._expanded(scope, values)
            self.log_joint = normalize(result)
        else:
            # Check support in log space before calling an external inference engine.
            support = self.initial_log_joint.copy()
            for scope, values in factors.items():
                support += self._expanded(scope, values)
            normalize(support)
            posterior = self.backend.infer(self.network, factors, [self.network.names])[
                0
            ]
            with np.errstate(divide="ignore"):
                self.log_joint = normalize(np.log(posterior.ravel()))

    def update(self, evidence):
        """Apply (scope, log likelihood) factors as one atomic observation batch."""
        factors = {scope: values.copy() for scope, values in self.factors.items()}
        incremental = self.log_joint.copy()
        for scope, values in evidence:
            scope = tuple(scope)
            values = np.asarray(values, dtype=float)
            if (
                not scope
                or values.shape != self.network.shape(scope)
                or np.isnan(values).any()
                or np.isposinf(values).any()
            ):
                raise ValueError(f"Invalid log likelihood for {scope}")
            factors[scope] = factors.get(scope, np.zeros_like(values)) + values
            maximum = np.max(factors[scope])
            if not np.isfinite(maximum):
                raise ImpossibleEvidence("Mutually inconsistent likelihood evidence")
            factors[scope] -= maximum
            if self.backend is None:
                incremental += self._expanded(scope, values)
        if self.backend is None:
            self.log_joint = normalize(incremental)
        else:
            self._infer(factors)
        self.factors = factors

    def replay(self):
        """Reconstruct the posterior from the original prior and accumulated factors."""
        result = self.initial_log_joint.copy()
        for scope, values in self.factors.items():
            result += self._expanded(scope, values)
        return np.exp(normalize(result)).reshape(self.network.shape(self.network.names))

    def marginal(self, scope):
        scope = (scope,) if isinstance(scope, str) else tuple(scope)
        shape = self.network.shape(scope)
        indices = tuple(
            self.assignments[:, self.network.names.index(name)] for name in scope
        )
        flat = np.ravel_multi_index(indices, shape)
        return np.bincount(
            flat, weights=np.exp(self.log_joint), minlength=math.prod(shape)
        ).reshape(shape)

    def expectation(self, scope, values):
        values = np.asarray(values, dtype=float)
        if (
            values.shape != self.network.shape(tuple(scope))
            or not np.isfinite(values).all()
        ):
            raise ValueError("Invalid values for expectation")
        return float(np.sum(self.marginal(scope) * values))
