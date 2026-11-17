"""PolyGraphs operations with discrete Bayesian belief networks."""

import numpy as np
import torch
from scipy.stats import qmc

from .common import NoOp
from ..beliefs import BayesianNetwork, BeliefState, binomial_log_likelihood
from ..beliefs.bnlearn import BnlearnBackend
from ..beliefs.cases import bandit, bandit_observation, factionalization
from ..beliefs.reliability import ReliabilityState


def _probability(value, label):
    if not np.isscalar(value) or not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{label} must lie in [0, 1]")
    return float(value)


def _backend(settings):
    if settings.backend == "numpy":
        return None
    if settings.backend == "bnlearn":
        return BnlearnBackend(settings.rscript, settings.timeout)
    raise ValueError("multibelief.backend must be bnlearn or numpy")


def _incoming(graph):
    sources, destinations = graph.edges()
    neighbours = [set() for _ in range(graph.num_nodes())]
    for source, destination in zip(sources.tolist(), destinations.tolist()):
        if source != destination:
            neighbours[destination].add(source)
    return [tuple(sorted(sources)) for sources in neighbours]


def _validate_runtime(graph, params):
    if params.device != "cpu" or graph.device.type != "cpu":
        raise ValueError("Bayesian network operations currently require device: cpu")
    if graph.num_nodes() < 1:
        raise ValueError("At least one agent is required")
    if (
        isinstance(params.trials, bool)
        or not isinstance(params.trials, int)
        or params.trials < 1
    ):
        raise ValueError("trials must be a positive integer")
    steps = params.simulation.steps
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("simulation.steps must be a nonnegative integer")
    if not isinstance(params.multibelief.stop_on_consensus, bool):
        raise ValueError("stop_on_consensus must be boolean")


class MultiBeliefOp(NoOp):
    """Share fresh observations and update every belief in each agent's network."""

    def __init__(self, graph, params):
        _validate_runtime(graph, params)
        super().__init__(graph, params)
        self.settings = params.multibelief
        self.params = params
        self.neighbours = _incoming(graph)
        self.round = 0
        self.backend = None
        self.shared = (
            self.settings.case.startswith("factionalization")
            or self.settings.case == "shared"
        )
        self._configure_network()
        self._configure_observation()
        self._configure_evidence()
        if (
            self.shared
            or not self.settings.stop_on_consensus
            or self.propaganda_probability > 0
        ):
            if params.simulation.steps <= 0:
                raise ValueError(
                    "Shared evidence, propaganda and fixed-horizon runs require simulation.steps > 0"
                )
        self.target = self.settings.target
        self.target_state = self.settings.target_state
        if self.settings.case.startswith("factionalization") and self.target == "H":
            self.target = "H2"
        if (
            self.target not in self.network.names
            or not isinstance(self.target_state, int)
            or not 0 <= self.target_state < self.network.cards[self.target]
        ):
            raise ValueError("Invalid target node or target_state")
        self.states = []
        if not set(self.settings.agent_models) <= {
            str(index) for index in range(graph.num_nodes())
        }:
            raise ValueError("agent_models contains an invalid agent ID")
        priors = self._initial_priors(graph)
        self.backend = _backend(self.settings)
        try:
            for index in range(graph.num_nodes()):
                override = self.settings.agent_models.get(str(index))
                network = (
                    BayesianNetwork(override["nodes"]) if override else self.network
                )
                if network.names != self.network.names or any(
                    a["states"] != b["states"]
                    for a, b in zip(network.nodes, self.network.nodes)
                ):
                    raise ValueError(
                        "Agent models must share the ordered node names and state labels"
                    )
                applicable = {
                    name: value
                    for name, value in priors[index].items()
                    if name in network.roots
                }
                network = network.with_priors(applicable)
                self.states.append(
                    BeliefState(network, self.backend, self.settings.max_states)
                )
            self._publish(graph)
        except Exception:
            self.close()
            raise

    def _configure_network(self):
        case = self.settings.case
        if case in ("baseline", "meta"):
            self.network = bandit(meta=case == "meta")
        elif case in ("factionalization1", "factionalization2", "factionalization3"):
            self.network = factionalization(int(case[-1]))
        elif case in ("custom", "shared") and "nodes" in self.settings.model:
            self.network = BayesianNetwork(self.settings.model["nodes"])
        else:
            raise ValueError(f"Unknown or incomplete multibelief case: {case}")

    def _configure_observation(self):
        self.observation_scope, self.observation_probability = (), None
        if self.shared:
            return
        if self.settings.case in ("baseline", "meta"):
            doubtful = None
            if self.settings.case == "meta":
                doubtful = self.settings.doubtful_epsilon
                if doubtful is None:
                    doubtful = self.params.epsilon / 2
            scope, probability = bandit_observation(self.params.epsilon, doubtful)
        else:
            observation = self.settings.observation
            scope = tuple(observation.get("parents", []))
            probability = np.asarray(observation.get("probability"), dtype=float)
        if (
            not scope
            or probability.shape != self.network.shape(scope)
            or not np.isfinite(probability).all()
            or np.any(probability < 0)
            or np.any(probability > 1)
        ):
            raise ValueError(
                "Observation needs valid parents and Bernoulli probability table"
            )
        self.observation_scope, self.observation_probability = scope, probability
        truth = self.settings.true_probability
        self.true_probability = _probability(
            0.5 + self.params.epsilon if truth is None else truth, "true_probability"
        )
        _probability(self.settings.safe_payoff, "safe_payoff")
        if self.settings.decision not in ("expected_payoff", "hypothesis"):
            raise ValueError("decision must be expected_payoff or hypothesis")

    def _parse_evidence(self, specifications):
        evidence = []
        for specification in specifications:
            scope = tuple(specification["variables"])
            likelihood = np.asarray(specification["likelihood"], dtype=float)
            if (
                not scope
                or likelihood.shape != self.network.shape(scope)
                or not np.isfinite(likelihood).all()
                or np.any(likelihood < 0)
                or not np.any(likelihood > 0)
            ):
                raise ValueError(f"Invalid likelihood evidence for {scope}")
            with np.errstate(divide="ignore"):
                evidence.append((scope, np.log(likelihood)))
        return evidence

    def _configure_evidence(self):
        specifications = self.settings.shared_evidence
        if not self.shared and (specifications or self.settings.evidence_schedule):
            raise ValueError(
                "shared_evidence and evidence_schedule require a shared-evidence case"
            )
        if not specifications and self.settings.case.startswith("factionalization"):
            specifications = [{"variables": ["H2"], "likelihood": [0.35, 0.65]}]
        self.shared_evidence = self._parse_evidence(specifications)
        self.evidence_schedule = [
            self._parse_evidence(batch) for batch in self.settings.evidence_schedule
        ]
        if self.shared and not self.shared_evidence and not self.evidence_schedule:
            raise ValueError("Shared-evidence mode requires evidence")
        if (
            self.evidence_schedule
            and len(self.evidence_schedule) < self.params.simulation.steps
        ):
            raise ValueError("evidence_schedule must cover every simulation step")
        propaganda = self.settings.propaganda
        self.propaganda_probability = _probability(
            propaganda.probability, "propaganda probability"
        )
        self.propaganda_evidence = []
        if self.propaganda_probability:
            self.propaganda_evidence = self._parse_evidence(
                [{"variables": [propaganda.node], "likelihood": propaganda.likelihood}]
            )

    def _initial_priors(self, graph):
        roots = self.network.roots
        count = graph.num_nodes()
        mode = self.settings.prior_mode
        if mode not in ("uniform", "halton", "fixed"):
            raise ValueError("prior_mode must be uniform, halton or fixed")
        if mode != "fixed" and any(self.network.cards[name] != 2 for name in roots):
            raise ValueError(
                "Automatic root priors currently require binary roots; use prior_mode: fixed"
            )
        if mode == "halton":
            sampler = qmc.Halton(len(roots), scramble=False)
            sampler.fast_forward(1)
            values = sampler.random(count)
        elif mode == "uniform":
            values = np.random.uniform(size=(count, len(roots)))
        priors = [{} for _ in range(count)]
        for column, name in enumerate(roots):
            if mode != "fixed":
                for index in range(count):
                    priors[index][name] = float(values[index, column])
            if mode == "uniform" and name == self.target:
                for index, value in enumerate(graph.ndata["beliefs"].tolist()):
                    priors[index][name] = value
        if not set(self.settings.priors) <= set(roots):
            raise ValueError("Configured priors must name root nodes")
        for name, value in self.settings.priors.items():
            if np.isscalar(value):
                value = [value] * count
            if len(value) != count:
                raise ValueError(f"Prior list for {name} must have one entry per agent")
            for index, entry in enumerate(value):
                priors[index][name] = _probability(entry, f"prior for {name}")
        return priors

    def _publish(self, graph):
        graph.ndata["beliefs"] = torch.tensor(
            [state.marginal(self.target)[self.target_state] for state in self.states],
            dtype=torch.float64,
        )
        for name in self.network.names:
            graph.ndata[f"belief:{name}"] = torch.tensor(
                np.array([state.marginal(name) for state in self.states]),
                dtype=torch.float64,
            )
        graph.ndata["belief_joint"] = torch.tensor(
            np.array([state.joint.ravel() for state in self.states]),
            dtype=torch.float64,
        )
        if not self.shared:
            graph.ndata["expected_payoff"] = torch.tensor(
                [
                    state.expectation(
                        self.observation_scope, self.observation_probability
                    )
                    for state in self.states
                ],
                dtype=torch.float64,
            )

    def active(self, graph):
        if self.settings.decision == "hypothesis":
            return graph.ndata["beliefs"] > 0.5
        return graph.ndata["expected_payoff"] > self.settings.safe_payoff

    def experiment(self, graph):
        count = graph.num_nodes()
        trials = np.where(self.active(graph).numpy(), self.params.trials, 0)
        successes = np.random.binomial(trials, self.true_probability)
        graph.ndata["payoffs"] = torch.tensor(
            np.stack((successes, trials), axis=1), dtype=torch.float64
        )

    def forward(self, graph, *args, **kwargs):
        self.round += 1
        batches = [[] for _ in self.states]
        if self.shared:
            evidence = (
                self.evidence_schedule[self.round - 1]
                if self.evidence_schedule
                else self.shared_evidence
            )
            batches = [list(evidence) for _ in self.states]
            graph.ndata["payoffs"] = torch.zeros(
                (len(self.states), 2), dtype=torch.float64
            )
        else:
            self.experiment(graph)
            payoffs = graph.ndata["payoffs"].numpy()
            for recipient, neighbours in enumerate(self.neighbours):
                successes, trials = payoffs[[recipient, *neighbours]].sum(axis=0)
                if trials:
                    batches[recipient].append(
                        (
                            self.observation_scope,
                            binomial_log_likelihood(
                                self.observation_probability, successes, trials
                            ),
                        )
                    )
        for index, state in enumerate(self.states):
            if (
                self.propaganda_probability
                and np.random.random() < self.propaganda_probability
            ):
                batches[index].extend(self.propaganda_evidence)
            if batches[index]:
                state.update(batches[index])
        self._publish(graph)
        return graph.ndata["beliefs"]

    def termination(self, graph, lowerupper, upperlower):
        invalid = any(
            not torch.isfinite(value).all().item()
            for key, value in graph.ndata.items()
            if key.startswith("belief")
        )
        converged = False
        if (
            not self.shared
            and self.settings.stop_on_consensus
            and not self.propaganda_probability
        ):
            active = self.active(graph)
            converged = bool(
                not active.any()
                or (active.all() and torch.all(graph.ndata["beliefs"] > lowerupper))
            )
        return invalid, converged, False

    def consensus(self, graph):
        if self.shared:
            return "?"
        active = self.active(graph)
        return "B" if active.all() else ("A" if not active.any() else "?")

    def close(self):
        if self.backend is not None:
            self.backend.close()

    def belief_metadata(self):
        return {
            "nodes": list(self.network.names),
            "states": {node["name"]: node["states"] for node in self.network.nodes},
            "joint_order": "C",
            "target": self.target,
            "target_state": self.target_state,
            "agent_networks": [state.network.nodes for state in self.states],
            "backend": self.settings.backend,
            "backend_versions": self.backend.versions if self.backend else {},
            "observation_parents": list(self.observation_scope),
            "observation_probability": (
                None if self.shared else self.observation_probability.tolist()
            ),
        }


class BayesianReliabilityOp(NoOp):
    """Joint Bayesian learning of H and each neighbour's persistent reliability."""

    def __init__(self, graph, params):
        _validate_runtime(graph, params)
        settings = params.multibelief
        if (
            settings.case != "baseline"
            or settings.model
            or settings.agent_models
            or settings.priors
            or settings.shared_evidence
            or settings.evidence_schedule
            or settings.propaganda.probability
            or settings.target != "H"
            or settings.target_state != 1
            or settings.prior_mode != "uniform"
            or settings.true_probability is not None
            or settings.safe_payoff != 0.5
            or settings.doubtful_epsilon is not None
        ):
            raise ValueError(
                "BayesianReliabilityOp uses init and bayesian_reliability settings; "
                "custom multibelief models/evidence/priors are unsupported"
            )
        if not set(params.bayesian_reliability.source_priors) <= {
            str(index) for index in range(graph.num_nodes())
        }:
            raise ValueError("source_priors contains an invalid source ID")
        super().__init__(graph, params)
        self.params = params
        self.settings = params.bayesian_reliability
        self.neighbours = _incoming(graph)
        _, self.reliable_probability = bandit_observation(params.epsilon)
        if self.settings.unreliable_mode == "uninformative":
            probability = _probability(
                self.settings.unreliable_probability, "unreliable_probability"
            )
            self.unreliable_probability = np.full(2, probability)
        elif self.settings.unreliable_mode == "opposite":
            self.unreliable_probability = self.reliable_probability[::-1].copy()
        else:
            raise ValueError("unreliable_mode must be uninformative or opposite")
        self.report_probability = np.stack(
            (self.unreliable_probability, self.reliable_probability), axis=1
        )
        prior = _probability(self.settings.prior, "reliability prior")
        actual = _probability(self.settings.actual_reliability, "actual_reliability")
        self.actual_reliability = np.random.binomial(
            1, actual, graph.num_nodes()
        ).astype(bool)
        for source in params.unreliablenodes:
            if not isinstance(source, int) or not 0 <= source < graph.num_nodes():
                raise ValueError("unreliablenodes contains an invalid node")
            self.actual_reliability[source] = False
        self.states = []
        self.backend = _backend(params.multibelief)
        try:
            for recipient, sources in enumerate(self.neighbours):
                priors = {
                    source: _probability(
                        self.settings.source_priors.get(str(source), prior),
                        "source prior",
                    )
                    for source in sources
                }
                self.states.append(
                    ReliabilityState(
                        float(graph.ndata["beliefs"][recipient]), priors, self.backend
                    )
                )
            if (
                not params.multibelief.stop_on_consensus
                and params.simulation.steps <= 0
            ):
                raise ValueError("Fixed-horizon runs require simulation.steps > 0")
            self._publish(graph)
        except Exception:
            self.close()
            raise

    def _publish(self, graph):
        graph.ndata["beliefs"] = torch.tensor(
            [state.belief for state in self.states], dtype=torch.float64
        )
        graph.ndata["belief:H"] = torch.stack(
            (1 - graph.ndata["beliefs"], graph.ndata["beliefs"]), dim=1
        )
        sources, destinations = graph.edges()
        graph.edata["reliability_belief"] = torch.tensor(
            [
                (
                    1.0
                    if source == destination
                    else self.states[destination].reliability(source)
                )
                for source, destination in zip(sources.tolist(), destinations.tolist())
            ],
            dtype=torch.float64,
        )
        graph.ndata["actual_reliability"] = torch.tensor(
            self.actual_reliability, dtype=torch.bool
        )

    def experiment(self, graph):
        trials = np.where(graph.ndata["beliefs"].numpy() > 0.5, self.params.trials, 0)
        private = np.random.binomial(trials, self.reliable_probability[1])
        reports = private.copy()
        unreliable = ~self.actual_reliability
        reports[unreliable] = np.random.binomial(
            trials[unreliable], self.unreliable_probability[1]
        )
        graph.ndata["payoffs"] = torch.tensor(
            np.stack((reports, trials), axis=1), dtype=torch.float64
        )
        graph.ndata["private_payoffs"] = torch.tensor(
            np.stack((private, trials), axis=1), dtype=torch.float64
        )

    def forward(self, graph, *args, **kwargs):
        self.experiment(graph)
        reports = graph.ndata["payoffs"].numpy()
        private = graph.ndata["private_payoffs"].numpy()
        for recipient, state in enumerate(self.states):
            evidence = []
            successes, trials = private[recipient]
            if trials:
                evidence.append(
                    (
                        ("H",),
                        binomial_log_likelihood(
                            self.reliable_probability, successes, trials
                        ),
                    )
                )
            for source in self.neighbours[recipient]:
                successes, trials = reports[source]
                if trials:
                    evidence.append(
                        (
                            ("H", f"R{source}"),
                            binomial_log_likelihood(
                                self.report_probability, successes, trials
                            ),
                        )
                    )
            if evidence:
                state.update(evidence)
        self._publish(graph)
        return graph.ndata["beliefs"]

    def termination(self, graph, lowerupper, upperlower):
        beliefs = graph.ndata["beliefs"]
        invalid = not bool(torch.isfinite(beliefs).all())
        converged = self.params.multibelief.stop_on_consensus and bool(
            torch.all(beliefs <= 0.5) or torch.all(beliefs > lowerupper)
        )
        return invalid, converged, False

    def consensus(self, graph):
        beliefs = graph.ndata["beliefs"]
        return (
            "B"
            if torch.all(beliefs > 0.5)
            else ("A" if torch.all(beliefs <= 0.5) else "?")
        )

    def close(self):
        if self.backend is not None:
            self.backend.close()

    def belief_metadata(self):
        return {
            "nodes": ["H"],
            "states": {"H": ["0", "1"]},
            "target": "H",
            "target_state": 1,
            "reliability_direction": "source -> recipient; recipient's belief in source",
            "backend": self.params.multibelief.backend,
            "backend_versions": self.backend.versions if self.backend else {},
            "agent_networks": [state.network.nodes for state in self.states],
            "report_probability": self.report_probability.tolist(),
            "report_axes": ["H", "reliable"],
        }
