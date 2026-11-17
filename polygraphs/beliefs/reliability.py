"""Exact source-reliability learning with a shared binary hypothesis."""

import numpy as np
from scipy.special import logsumexp

from .cases import binary_node
from .network import BayesianNetwork, ImpossibleEvidence, normalize


class ReliabilityState:
    """P(H, R_1, ..., R_k) = P(H) product_j P(R_j | H), in linear space."""

    def __init__(self, prior, source_priors, backend=None):
        self.sources = tuple(source_priors)
        self.network = BayesianNetwork(
            [binary_node("H", probability=prior)]
            + [
                binary_node(f"R{source}", probability=source_priors[source])
                for source in self.sources
            ]
        )
        self.backend = backend
        self.factors = {}
        with np.errstate(divide="ignore"):
            self._prior_h = np.log([1 - prior, prior])
            self._prior_r = {
                source: np.log([1 - source_priors[source], source_priors[source]])
                for source in self.sources
            }
        self.log_h = self._prior_h.copy()
        self.log_r_given_h = {
            source: np.tile(value, (2, 1)) for source, value in self._prior_r.items()
        }
        self._posterior_pairs = None
        if backend is not None:
            self.update([])

    @property
    def belief(self):
        return float(np.exp(self.log_h[1]))

    def reliability(self, source):
        return float(self.pair(source)[:, 1].sum())

    def pair(self, source):
        if self._posterior_pairs is not None:
            return self._posterior_pairs[source].copy()
        return np.exp(self.log_h[:, None] + self.log_r_given_h[source])

    def update(self, evidence):
        factors = {scope: values.copy() for scope, values in self.factors.items()}
        allowed = {("H",)} | {("H", f"R{source}") for source in self.sources}
        for scope, values in evidence:
            scope = tuple(scope)
            values = np.asarray(values, dtype=float)
            if (
                scope not in allowed
                or values.shape != self.network.shape(scope)
                or np.isnan(values).any()
                or np.isposinf(values).any()
            ):
                raise ValueError(f"Invalid reliability likelihood for {scope}")
            factors[scope] = factors.get(scope, np.zeros_like(values)) + values
            maximum = np.max(factors[scope])
            if not np.isfinite(maximum):
                raise ImpossibleEvidence("Mutually inconsistent source evidence")
            factors[scope] -= maximum
        log_h = self._prior_h + factors.get(("H",), np.zeros(2))
        conditionals = {}
        for source in self.sources:
            log_r = self._prior_r[source][None, :] + factors.get(
                ("H", f"R{source}"), np.zeros((2, 2))
            )
            normalizer = logsumexp(log_r, axis=1)
            log_h += normalizer
            # A conditional on an impossible H is immaterial to the joint.
            conditionals[source] = np.full((2, 2), -np.log(2.0))
            valid = np.isfinite(normalizer)
            conditionals[source][valid] = log_r[valid] - normalizer[valid, None]
        log_h = normalize(log_h)
        pairs = None
        if self.backend is not None:
            queries = [("H",)] + [("H", f"R{source}") for source in self.sources]
            posteriors = self.backend.infer(self.network, factors, queries)
            with np.errstate(divide="ignore"):
                log_h = normalize(np.log(posteriors[0]))
            pairs = dict(zip(self.sources, posteriors[1:]))
        self.log_h = log_h
        self.log_r_given_h = conditionals
        self._posterior_pairs = pairs
        self.factors = factors
