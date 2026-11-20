"""Device-resident exact Bayesian operators for fresh network evidence.

Configuration and priors are compiled once with the independent NumPy backend.
Every subsequent experiment, evidence reduction, posterior and marginal stays
in float64 PyTorch tensors on the graph's device. No R process is started.
"""

from copy import deepcopy

import torch

from .common import NoOp
from .multibelief import MultiBeliefOp, BayesianReliabilityOp
from ..beliefs.network import ImpossibleEvidence


def _compile(operator, graph, params):
    device = torch.device(params.device)
    if device.type != graph.device.type or (
        device.index is not None and device.index != graph.device.index
    ):
        raise ValueError("params.device must match the graph device")
    if graph.device.type not in ("cpu", "cuda"):
        raise ValueError("Torch Bayesian operations support CPU and CUDA float64")
    settings = deepcopy(params)
    settings.device = "cpu"
    # Use the existing exact implementation only to validate and compile priors.
    settings.multibelief.backend = "numpy"
    return operator(graph.to("cpu").clone(), settings)


def _tensor(value, device, dtype=torch.float64):
    return torch.as_tensor(value, dtype=dtype, device=device).clone()


def _likelihood(successes, trials, log_p, log_q):
    """Handle zero counts and deterministic likelihoods without 0 * -inf."""
    failures = trials - successes
    return torch.where(successes == 0, 0.0, successes * log_p) + torch.where(
        failures == 0, 0.0, failures * log_q
    )


def _normalize(log_joint):
    normalizer = torch.logsumexp(log_joint, dim=1, keepdim=True)
    if not bool(torch.isfinite(normalizer).all()):
        raise ImpossibleEvidence("Evidence has zero probability under an agent's model")
    return log_joint - normalizer


class _TorchBayesianOp(NoOp):
    def _buffer(self, name, value, dtype=torch.float64):
        self.register_buffer(name, _tensor(value, self._device, dtype))
        return getattr(self, name)

    def consensus(self, graph):
        active = self.active(graph)
        return "B" if bool(active.all()) else ("A" if not bool(active.any()) else "?")

    def close(self):
        pass

    def belief_metadata(self):
        return {
            **self.metadata,
            "implementation": type(self).__name__,
            "backend": "torch",
            "initialization_backend": "numpy",
            "backend_versions": {"torch": str(torch.__version__)},
            "device": str(self._device),
            "dtype": "float64",
        }


class TorchMultiBeliefOp(_TorchBayesianOp):
    """Finite-joint inference with one tensor row per agent.

    Supports baseline, meta, custom and networked factionalization cases, with
    honest, uninformative or opposite sources. Shared evidence and propaganda
    retain their reference operators. Social edges are deduplicated; one's own
    fresh evidence is used once, regardless of self-loops or parallel edges.
    """

    def __init__(self, graph, params):
        torch.nn.Module.__init__(self)
        reference = _compile(MultiBeliefOp, graph, params)
        if reference.shared or reference.propaganda_probability:
            raise ValueError("TorchMultiBeliefOp requires fresh network evidence without propaganda")
        self._device = graph.device
        self.params = params
        self.settings = params.multibelief
        self.network = reference.network
        self.target = reference.target
        self.target_state = reference.target_state
        self.neighbours = reference.neighbours
        self.metadata = reference.belief_metadata()
        self.true_probability = reference.true_probability
        self._buffer("assignments", reference.states[0].assignments, torch.long)
        initial = torch.stack([_tensor(s.log_joint, self._device) for s in reference.states])
        self._buffer("initial_log_joint", initial)
        self._buffer("log_joint", initial)
        probability = reference.states[0]._expanded(
            reference.observation_scope, reference.observation_probability
        )
        self._buffer("observation_probability", probability)
        if bool(((self.observation_probability <= 0) | (self.observation_probability >= 1)).any()):
            raise ValueError("TorchMultiBeliefOp observation probabilities must be inside (0, 1)")
        self._buffer("log_p", probability).log_()
        self._buffer("log_q", 1 - probability).log_()
        source_probability = torch.full((graph.num_nodes(),), self.true_probability,
                                        dtype=torch.float64, device=self._device)
        if self.settings.source_mode != "honest":
            source_probability[params.unreliablenodes] = (
                .5 if self.settings.source_mode == "uninformative" else 1 - self.true_probability
            )
        self._buffer("source_probability", source_probability)
        bad_sources = torch.zeros(graph.num_nodes(), dtype=torch.bool, device=self._device)
        if self.settings.source_mode != "honest":
            bad_sources[params.unreliablenodes] = True
        self._buffer("bad_sources", bad_sources, torch.bool)
        edges = [(s, r) for r, sources in enumerate(self.neighbours) for s in sources]
        self._buffer("sources", [s for s, _ in edges], torch.long)
        self._buffer("recipients", [r for _, r in edges], torch.long)
        self._buffer("received", torch.zeros(graph.num_nodes()))
        self._buffer("cumulative_received", torch.zeros(graph.num_nodes()))
        # A single matrix multiplication publishes all marginal probabilities.
        masks = []
        self.marginal_slices = {}
        offset = 0
        for k, name in enumerate(self.network.names):
            card = self.network.cards[name]
            masks.extend([(self.assignments[:, k] == s) for s in range(card)])
            self.marginal_slices[name] = slice(offset, offset + card)
            offset += card
        self.register_buffer("marginal_mask", torch.stack(masks, dim=1).to(torch.float64))
        self._publish(graph)

    def _publish(self, graph):
        joint = self.log_joint.exp()
        graph.ndata["belief_joint"] = joint
        marginals = joint @ self.marginal_mask
        for name, columns in self.marginal_slices.items():
            graph.ndata[f"belief:{name}"] = marginals[:, columns]
        graph.ndata["beliefs"] = graph.ndata[f"belief:{self.target}"][:, self.target_state]
        graph.ndata["expected_payoff"] = joint @ self.observation_probability

    @torch.no_grad()
    def set_initial_log_joint(self, graph, log_joint):
        """Restore a calibrated joint in this operator's published C state order.

        This is an initialization operation, before any evidence is processed.
        It permits the matched-P(G) study's dependent priors without pretending
        that independent root probabilities can represent those dependencies.
        """
        if bool(self.cumulative_received.any()):
            raise ValueError("Initial joint must be set before processing evidence")
        values = _tensor(log_joint, self._device)
        if values.shape != self.log_joint.shape:
            raise ValueError("Initial log joint must have shape (agents, joint states)")
        values = _normalize(values)
        self.initial_log_joint.copy_(values)
        self.log_joint.copy_(values)
        self._publish(graph)

    def belief_metadata(self):
        return {
            **super().belief_metadata(),
            "source_mechanism": "genuine_private_corrupted_outgoing_reports",
            "source_probability": self.source_probability.tolist(),
            "initial_joint": self.initial_log_joint.exp().tolist(),
        }

    def active(self, graph):
        if self.settings.decision == "hypothesis":
            return graph.ndata["beliefs"] > 0.5
        return graph.ndata["expected_payoff"] > self.settings.safe_payoff

    @torch.no_grad()
    def experiment(self, graph):
        trials = self.active(graph).to(torch.float64) * self.params.trials
        private = torch.binomial(trials, torch.full_like(trials, self.true_probability))
        reports = private.clone()
        bad = self.bad_sources
        reports[bad] = torch.binomial(trials[bad], self.source_probability[bad])
        graph.ndata["payoffs"] = torch.stack((reports, trials), dim=1)
        graph.ndata["private_payoffs"] = torch.stack((private, trials), dim=1)

    @torch.no_grad()
    def forward(self, graph, *args, **kwargs):
        self.experiment(graph)
        payoffs = graph.ndata["payoffs"]
        counts = graph.ndata.get("private_payoffs", payoffs).clone()
        counts.index_add_(0, self.recipients, payoffs[self.sources])
        self.received.copy_(counts[:, 1])
        updated = self.log_joint + _likelihood(
            counts[:, :1], counts[:, 1:], self.log_p, self.log_q
        )
        self.log_joint.copy_(_normalize(updated))
        self.cumulative_received.add_(self.received)
        self._publish(graph)
        return graph.ndata["beliefs"]

    def termination(self, graph, lowerupper, upperlower):
        active = self.active(graph)
        converged = self.settings.stop_on_consensus and bool(
            not active.any()
            or (active.all() and torch.all(graph.ndata["beliefs"] > lowerupper))
        )
        invalid = bool(torch.isnan(self.log_joint).any() or torch.isposinf(self.log_joint).any())
        return invalid, converged, False


class TorchBayesianReliabilityOp(_TorchBayesianOp):
    """Exact P(H) product_s P(R_s | H), using four values per social edge.

    Cumulative sufficient statistics prevent either forgetting reliability
    dependence or counting old reports twice. Private observations always come
    from the reliable experiment, including for agents that send false reports.
    """

    def __init__(self, graph, params):
        torch.nn.Module.__init__(self)
        reference = _compile(BayesianReliabilityOp, graph, params)
        self._device = graph.device
        self.params = params
        self.settings = params.bayesian_reliability
        self.neighbours = reference.neighbours
        self.metadata = reference.belief_metadata()
        for name in ("reliable_probability", "unreliable_probability", "report_probability"):
            self._buffer(name, getattr(reference, name))
        edges = [(s, r) for r, sources in enumerate(self.neighbours) for s in sources]
        self._buffer("sources", [s for s, _ in edges], torch.long)
        self._buffer("recipients", [r for _, r in edges], torch.long)
        prior_h = torch.stack([_tensor(s.log_h, self._device) for s in reference.states])
        prior_r = torch.stack([_tensor(reference.states[r]._prior_r[s], self._device)
                               for s, r in edges]) if edges else torch.empty((0, 2), device=self._device)
        self._buffer("prior_h", prior_h)
        self._buffer("log_h", prior_h)
        self._buffer("prior_r", prior_r)
        self._buffer("log_r_given_h", prior_r[:, None, :].expand(-1, 2, -1))
        for name in ("report_counts", "private_counts"):
            self._buffer(name, torch.zeros((graph.num_nodes(), 2)))
        for name in ("received", "cumulative_received"):
            self._buffer(name, torch.zeros(graph.num_nodes()))
        self._buffer("actual_reliability", reference.actual_reliability, torch.bool)
        lookup = {edge: k for k, edge in enumerate(edges)}
        src, dst = graph.edges()
        self._buffer("edge_indices", [lookup.get((s, r), -1)
                                      for s, r in zip(src.tolist(), dst.tolist())], torch.long)
        self._buffer("log_report_p", reference.report_probability).log_()
        self._buffer("log_report_q", 1 - reference.report_probability).log_()
        self._buffer("log_private_p", reference.reliable_probability).log_()
        self._buffer("log_private_q", 1 - reference.reliable_probability).log_()
        self.register_buffer("trust", torch.empty(len(self.sources), dtype=torch.float64, device=self._device))
        self._publish(graph)

    def active(self, graph):
        return graph.ndata["beliefs"] > 0.5

    @torch.no_grad()
    def experiment(self, graph):
        trials = self.active(graph).to(torch.float64) * self.params.trials
        private = torch.binomial(trials, self.reliable_probability[1].expand_as(trials))
        # Honest reports are the private evidence itself, not another draw.
        reports = private.clone()
        bad = ~self.actual_reliability
        reports[bad] = torch.binomial(
            trials[bad], self.unreliable_probability[1].expand_as(trials[bad])
        )
        graph.ndata["payoffs"] = torch.stack((reports, trials), dim=1)
        graph.ndata["private_payoffs"] = torch.stack((private, trials), dim=1)

    @torch.no_grad()
    def forward(self, graph, *args, **kwargs):
        self.experiment(graph)
        reports = graph.ndata["payoffs"]
        private = graph.ndata["private_payoffs"]
        report_counts = self.report_counts + reports
        private_counts = self.private_counts + private
        counts = report_counts[self.sources]
        log_r = self.prior_r[:, None, :] + _likelihood(
            counts[:, 0, None, None], counts[:, 1, None, None],
            self.log_report_p, self.log_report_q,
        )
        norm = torch.logsumexp(log_r, dim=2)
        # A conditional given an impossible H is arbitrary. Use a finite value
        # so its zero posterior mass cannot produce NaN in trust marginals.
        conditional = torch.where(
            torch.isfinite(norm[:, :, None]),
            log_r - norm[:, :, None],
            torch.full_like(log_r, -0.6931471805599453),
        )
        log_h = self.prior_h + _likelihood(
            private_counts[:, :1], private_counts[:, 1:],
            self.log_private_p, self.log_private_q,
        )
        log_h.index_add_(0, self.recipients, norm)
        updated = _normalize(log_h)
        self.log_h.copy_(updated)
        self.log_r_given_h.copy_(conditional)
        self.report_counts.copy_(report_counts)
        self.private_counts.copy_(private_counts)
        self.received.copy_(private[:, 1])
        self.received.index_add_(0, self.recipients, reports[self.sources, 1])
        self.cumulative_received.add_(self.received)
        self._publish(graph)
        return graph.ndata["beliefs"]

    def _publish(self, graph):
        graph.ndata["belief:H"] = self.log_h.exp()
        graph.ndata["beliefs"] = graph.ndata["belief:H"][:, 1]
        self.trust.copy_((self.log_h[self.recipients] + self.log_r_given_h[:, :, 1]).exp().sum(1))
        values = torch.ones(len(self.edge_indices), dtype=torch.float64, device=self._device)
        mask = self.edge_indices >= 0
        values[mask] = self.trust[self.edge_indices[mask]]
        graph.edata["reliability_belief"] = values
        graph.ndata["actual_reliability"] = self.actual_reliability

    def termination(self, graph, lowerupper, upperlower):
        beliefs = graph.ndata["beliefs"]
        invalid = bool(torch.isnan(self.log_h).any() or torch.isposinf(self.log_h).any())
        converged = self.params.multibelief.stop_on_consensus and bool(
            torch.all(beliefs <= 0.5) or torch.all(beliefs > lowerupper)
        )
        return invalid, converged, False
