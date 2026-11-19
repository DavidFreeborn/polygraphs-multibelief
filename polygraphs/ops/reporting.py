"""Finite belief networks with genuine private data and altered outgoing reports."""

import numpy as np
from scipy.special import logsumexp
import torch

from .batched import BatchedMultiBeliefOp


class ReportedMultiBeliefOp(BatchedMultiBeliefOp):
    """Keep private observations distinct from each source's public report."""

    def experiment(self, graph):
        trials = np.where(self.active(graph).numpy(), self.params.trials, 0)
        private = np.random.binomial(trials, self.true_probability)
        reports = private.copy()
        if self.settings.source_mode != "honest":
            indices = self.params.unreliablenodes
            reports[indices] = np.random.binomial(
                trials[indices], self.source_probability[indices]
            )
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
        counts = self.adjacency @ reports + private - reports
        self.received = counts[:, 1]
        self.cumulative_received += self.received
        probability = self.observation_probability
        self.log_joint += counts[:, :1] * np.log(probability) + (
            counts[:, 1:] - counts[:, :1]
        ) * np.log1p(-probability)
        self.log_joint -= logsumexp(self.log_joint, axis=1, keepdims=True)
        self._publish(graph)
        return graph.ndata["beliefs"]

    def belief_metadata(self):
        return {
            **super().belief_metadata(),
            "source_mechanism": "genuine private data; potentially altered reports",
        }
