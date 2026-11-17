"""Vectorized exact inference for fixed-horizon network experiments.

The reference operations validate configuration and construct priors. These
operations retain posterior arrays instead of individual BeliefState objects.
"""

import numpy as np
import torch
from scipy.special import logsumexp

from .common import NoOp
from .multibelief import MultiBeliefOp, BayesianReliabilityOp


class BatchedMultiBeliefOp(NoOp):
    """Exact finite-joint updates with native-style unreliable observations."""

    def __init__(self, graph, params):
        torch.nn.Module.__init__(self)
        if params.multibelief.backend != "numpy":
            raise ValueError("Batched operations require the numpy backend")
        reference = MultiBeliefOp(graph, params)
        if reference.shared or reference.propaganda_probability:
            raise ValueError("Batched operation requires fresh network evidence")
        self.params = params
        self.settings = params.multibelief
        self.network = reference.network
        self.target = reference.target
        self.target_state = reference.target_state
        self.neighbours = reference.neighbours
        self.metadata = reference.belief_metadata()
        self.assignments = reference.states[0].assignments
        self.initial_log_joint = np.array([s.log_joint for s in reference.states])
        self.log_joint = self.initial_log_joint.copy()
        self.observation_probability = reference.states[0]._expanded(
            reference.observation_scope, reference.observation_probability
        )
        if np.any(
            (self.observation_probability <= 0) | (self.observation_probability >= 1)
        ):
            raise ValueError(
                "Batched observation probabilities must be strictly inside (0, 1)"
            )
        self.true_probability = reference.true_probability
        self.adjacency = np.eye(graph.num_nodes())
        for recipient, sources in enumerate(self.neighbours):
            self.adjacency[recipient, list(sources)] = 1
        self.source_probability = np.full(graph.num_nodes(), self.true_probability)
        mode = self.settings.source_mode
        if mode not in ("honest", "uninformative", "opposite"):
            raise ValueError("source_mode must be honest, uninformative or opposite")
        for source in params.unreliablenodes:
            if not isinstance(source, int) or not 0 <= source < graph.num_nodes():
                raise ValueError("Invalid unreliable source")
            if mode != "honest":
                self.source_probability[source] = (
                    0.5 if mode == "uninformative" else 1 - self.true_probability
                )
        self.received = np.zeros(graph.num_nodes())
        self.cumulative_received = np.zeros(graph.num_nodes())
        self._publish(graph)

    def _publish(self, graph):
        joint = np.exp(self.log_joint)
        graph.ndata["belief_joint"] = torch.from_numpy(joint)
        for k, name in enumerate(self.network.names):
            marginal = np.stack(
                [
                    joint[:, self.assignments[:, k] == s].sum(1)
                    for s in range(self.network.cards[name])
                ],
                axis=1,
            )
            graph.ndata[f"belief:{name}"] = torch.from_numpy(marginal)
        graph.ndata["beliefs"] = graph.ndata[f"belief:{self.target}"][
            :, self.target_state
        ]
        graph.ndata["expected_payoff"] = torch.from_numpy(
            joint @ self.observation_probability
        )

    def active(self, graph):
        if self.settings.decision == "hypothesis":
            return graph.ndata["beliefs"] > 0.5
        return graph.ndata["expected_payoff"] > self.settings.safe_payoff

    def experiment(self, graph):
        trials = np.where(self.active(graph).numpy(), self.params.trials, 0)
        successes = np.random.binomial(trials, self.source_probability)
        graph.ndata["payoffs"] = torch.from_numpy(
            np.stack((successes, trials), axis=1).astype(float)
        )

    def forward(self, graph, *args, **kwargs):
        self.experiment(graph)
        counts = self.adjacency @ graph.ndata["payoffs"].numpy()
        self.received = counts[:, 1]
        self.cumulative_received += self.received
        p = self.observation_probability
        self.log_joint += counts[:, :1] * np.log(p) + (
            counts[:, 1:] - counts[:, :1]
        ) * np.log1p(-p)
        self.log_joint -= logsumexp(self.log_joint, axis=1, keepdims=True)
        self._publish(graph)
        return graph.ndata["beliefs"]

    def termination(self, graph, lowerupper, upperlower):
        active = self.active(graph)
        converged = self.settings.stop_on_consensus and bool(
            not active.any()
            or (active.all() and torch.all(graph.ndata["beliefs"] > lowerupper))
        )
        return (
            bool(np.isnan(self.log_joint).any() or np.isposinf(self.log_joint).any()),
            converged,
            False,
        )

    def consensus(self, graph):
        active = self.active(graph)
        return "B" if active.all() else ("A" if not active.any() else "?")

    def belief_metadata(self):
        return {
            **self.metadata,
            "implementation": type(self).__name__,
            "source_mode": self.settings.source_mode,
            "source_probability": self.source_probability.tolist(),
        }

    def close(self):
        pass


class BatchedBayesianReliabilityOp(NoOp):
    """Exact edge-factor inference, linear in the number of social edges."""

    def __init__(self, graph, params):
        torch.nn.Module.__init__(self)
        if params.multibelief.backend != "numpy":
            raise ValueError("Batched operations require the numpy backend")
        reference = BayesianReliabilityOp(graph, params)
        self.params = params
        self.settings = params.bayesian_reliability
        self.metadata = reference.belief_metadata()
        self.actual_reliability = reference.actual_reliability
        self.reliable_probability = reference.reliable_probability
        self.unreliable_probability = reference.unreliable_probability
        self.report_probability = reference.report_probability
        self.neighbours = reference.neighbours
        edges = [(s, r) for r, sources in enumerate(self.neighbours) for s in sources]
        self.sources = np.array([s for s, r in edges], dtype=int)
        self.recipients = np.array([r for s, r in edges], dtype=int)
        self.prior_h = np.array([s.log_h for s in reference.states])
        self.log_h = self.prior_h.copy()
        self.prior_r = np.array(
            [reference.states[r]._prior_r[s] for s, r in edges]
        ).reshape(-1, 2)
        self.report_counts = np.zeros((graph.num_nodes(), 2))
        self.private_counts = np.zeros_like(self.report_counts)
        self.received = np.zeros(graph.num_nodes())
        self.cumulative_received = np.zeros(graph.num_nodes())
        lookup = {edge: k for k, edge in enumerate(edges)}
        src, dst = graph.edges()
        self.edge_indices = np.array(
            [lookup.get((s, r), -1) for s, r in zip(src.tolist(), dst.tolist())]
        )
        self.log_r_given_h = np.repeat(self.prior_r[:, None, :], 2, axis=1)
        self._publish(graph)

    def active(self, graph):
        return graph.ndata["beliefs"] > 0.5

    def experiment(self, graph):
        trials = np.where(self.active(graph).numpy(), self.params.trials, 0)
        private = np.random.binomial(trials, self.reliable_probability[1])
        reports = private.copy()
        bad = ~self.actual_reliability
        reports[bad] = np.random.binomial(trials[bad], self.unreliable_probability[1])
        graph.ndata["payoffs"] = torch.from_numpy(
            np.stack((reports, trials), axis=1).astype(float)
        )
        graph.ndata["private_payoffs"] = torch.from_numpy(
            np.stack((private, trials), axis=1).astype(float)
        )

    def forward(self, graph, *args, **kwargs):
        self.experiment(graph)
        reports = graph.ndata["payoffs"].numpy()
        private = graph.ndata["private_payoffs"].numpy()
        self.report_counts += reports
        self.private_counts += private
        self.received = private[:, 1].copy()
        np.add.at(self.received, self.recipients, reports[self.sources, 1])
        self.cumulative_received += self.received
        counts = self.report_counts[self.sources]
        p = self.report_probability
        log_r = (
            self.prior_r[:, None, :]
            + counts[:, 0, None, None] * np.log(p)
            + (counts[:, 1] - counts[:, 0])[:, None, None] * np.log1p(-p)
        )
        norm = logsumexp(log_r, axis=2)
        self.log_r_given_h = log_r - norm[:, :, None]
        p = self.reliable_probability
        self.log_h = (
            self.prior_h
            + self.private_counts[:, :1] * np.log(p)
            + (self.private_counts[:, 1:] - self.private_counts[:, :1]) * np.log1p(-p)
        )
        np.add.at(self.log_h, self.recipients, norm)
        self.log_h -= logsumexp(self.log_h, axis=1, keepdims=True)
        self._publish(graph)
        return graph.ndata["beliefs"]

    def _publish(self, graph):
        graph.ndata["belief:H"] = torch.from_numpy(np.exp(self.log_h))
        graph.ndata["beliefs"] = graph.ndata["belief:H"][:, 1]
        self.trust = np.exp(self.log_h[self.recipients, :, None] + self.log_r_given_h)[
            :, :, 1
        ].sum(1)
        values = np.ones(len(self.edge_indices))
        mask = self.edge_indices >= 0
        values[mask] = self.trust[self.edge_indices[mask]]
        graph.edata["reliability_belief"] = torch.from_numpy(values)
        graph.ndata["actual_reliability"] = torch.from_numpy(self.actual_reliability)

    def termination(self, graph, lowerupper, upperlower):
        beliefs = graph.ndata["beliefs"]
        return (
            bool(np.isnan(self.log_h).any() or np.isposinf(self.log_h).any()),
            self.params.multibelief.stop_on_consensus
            and bool(torch.all(beliefs <= 0.5) or torch.all(beliefs > lowerupper)),
            False,
        )

    def consensus(self, graph):
        active = self.active(graph)
        return "B" if active.all() else ("A" if not active.any() else "?")

    def belief_metadata(self):
        return {**self.metadata, "implementation": type(self).__name__}

    def close(self):
        pass
