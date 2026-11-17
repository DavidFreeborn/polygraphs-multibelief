"""Small, fixed-seed experiments and exact checks for multibelief networks."""

from itertools import permutations
import json
from pathlib import Path
import time

import dgl
import numpy as np
import pandas as pd
from scipy.special import rel_entr, xlogy, logsumexp
from scipy.spatial.distance import jensenshannon
from scipy.stats import binom, qmc
import torch

import polygraphs as pg
from polygraphs import hyperparameters as hp
from polygraphs.beliefs import BeliefState
from polygraphs.beliefs.bnlearn import BnlearnBackend
from polygraphs.beliefs.cases import factionalization
from polygraphs.ops.multibelief import MultiBeliefOp, BayesianReliabilityOp

HERE = Path(__file__).resolve().parent
PLAN = json.loads((HERE / "workload.json").read_text())
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)
START = time.perf_counter()


def budget():
    if time.perf_counter() - START > 600:
        raise RuntimeError("Approved ten-minute computation budget reached")


def js_pairs(joints):
    flat = joints.reshape(len(joints), -1)
    i, j = np.triu_indices(len(flat), 1)
    p, q = flat[i], flat[j]
    m = (p + q) / 2
    return (rel_entr(p, m).sum(-1) + rel_entr(q, m).sum(-1)) / 2


def probability_roundoff(values):
    values = np.asarray(values)
    if (
        not np.isfinite(values).all()
        or np.any(values < -1e-12)
        or np.any(values > 1 + 1e-12)
    ):
        raise ValueError("Invalid probability outside floating-point tolerance")
    return np.clip(values, 0, 1)


def marginal_probabilities(joints):
    return np.stack(
        [
            joints.sum(axis=tuple(j + 1 for j in range(joints.ndim - 1) if j != k))[
                :, 1
            ]
            for k in range(joints.ndim - 1)
        ],
        axis=1,
    )


def mean_ordered_kl(log_probabilities):
    """Mean over ordered agent pairs, retaining tiny probabilities in log space."""
    logs = np.asarray(log_probabilities).reshape(len(log_probabilities), -1)
    if not np.isfinite(logs).all():
        raise ValueError("This study's log-KL calculation requires full prior support")
    p = np.exp(logs)
    n = len(logs)
    value = (
        n
        / (n - 1)
        * (np.mean(np.sum(p * logs, axis=1)) - np.sum(p.mean(0) * logs.mean(0)))
    )
    assert value > -1e-11
    return max(0.0, float(value))


def metrics(joints, log_joints=None):
    beliefs = marginal_probabilities(joints)
    product = np.ones((len(joints),) + (1,) * (joints.ndim - 1))
    for k in range(joints.ndim - 1):
        marginal = joints.sum(
            axis=tuple(j + 1 for j in range(joints.ndim - 1) if j != k)
        )
        shape = [len(joints)] + [1] * (joints.ndim - 1)
        shape[k + 1] = joints.shape[k + 1]
        product = product * marginal.reshape(shape)
    cov = np.atleast_2d(np.cov(beliefs, rowvar=False, bias=True))
    off = cov[np.triu_indices(len(cov), 1)]
    logs = np.log(joints) if log_joints is None else np.asarray(log_joints)
    marginal_logs = [
        logsumexp(logs, axis=tuple(j + 1 for j in range(joints.ndim - 1) if j != k))
        for k in range(joints.ndim - 1)
    ]
    return {
        "joint_js": float(js_pairs(joints).mean()),
        "marginal_product_js": float(js_pairs(product).mean()),
        "joint_kl": mean_ordered_kl(logs),
        "marginal_product_kl": sum(mean_ordered_kl(logs) for logs in marginal_logs),
        "mean_sd": float(beliefs.std(axis=0).mean()),
        "mean_variance": float(beliefs.var(axis=0).mean()),
        "mean_absolute_covariance": float(np.abs(off).mean()) if len(off) else 0.0,
        "mean_total_correlation": float(
            rel_entr(joints, product).reshape(len(joints), -1).sum(1).mean()
        ),
        "mean_joint_entropy": float(
            -xlogy(joints, joints).reshape(len(joints), -1).sum(1).mean()
        ),
    }


def initial_population(case, roots):
    network = factionalization(case)
    return np.array(
        [
            BeliefState(network.with_priors(dict(zip(network.roots, row)))).joint
            for row in roots
        ]
    )


def condition(initial, weights):
    shape = (1,) * (initial.ndim - 1) + (2,)
    result = initial * np.asarray(weights).reshape(shape)
    return result / result.reshape(len(result), -1).sum(1).reshape(
        (-1,) + (1,) * (initial.ndim - 1)
    )


def faction_study():
    rows, belief_rows = [], []
    populations = {}
    for case, count in zip([1, 2, 3], [15, 40, 60]):
        network = factionalization(case)
        sampler = qmc.Halton(len(network.roots), scramble=False)
        sampler.fast_forward(1)
        roots = sampler.random(count)
        initial = initial_population(case, roots)
        history = np.array(
            [condition(initial, np.array([0.35, 0.65]) ** t) for t in range(21)]
        )
        sequential = BeliefState(
            network.with_priors(dict(zip(network.roots, roots[0])))
        )
        for t, joint in enumerate(history):
            if t:
                sequential.update([(("H2",), np.log([0.35, 0.65]))])
            np.testing.assert_allclose(sequential.joint, joint[0], atol=2e-14)
            rows.append({"case": case, "round": t, **metrics(joint)})
            beliefs = marginal_probabilities(joint)
            for index, name in enumerate(network.names):
                belief_rows.append(
                    {
                        "case": case,
                        "round": t,
                        "node": name,
                        "mean": beliefs[:, index].mean(),
                        "sd": beliefs[:, index].std(),
                    }
                )
        np.savez_compressed(
            DATA / f"faction-{case}.npz",
            roots=roots,
            joints=history,
            names=network.names,
        )
        populations[case] = initial
        print(f"Faction case {case} checked", flush=True)
    pd.DataFrame(rows).to_csv(DATA / "faction-metrics.csv", index=False)
    pd.DataFrame(belief_rows).to_csv(DATA / "faction-beliefs.csv", index=False)
    initial = populations[3]
    final = condition(initial, np.array([0.35, 0.65]) ** 20)
    p, q = np.triu_indices(60, 1)
    scipy_js = (
        jensenshannon(final.reshape(60, -1)[p], final.reshape(60, -1)[q], axis=1) ** 2
    )
    np.testing.assert_allclose(scipy_js, js_pairs(final), atol=3e-15)
    with BnlearnBackend() as backend:
        roots = np.load(DATA / "faction-3.npz")["roots"]
        for index in [0, 7, 19, 39, 59]:
            network = factionalization(3)
            state = BeliefState(
                network.with_priors(dict(zip(network.roots, roots[index]))), backend
            )
            state.update([(("H2",), 20 * np.log([0.35, 0.65]))])
            np.testing.assert_allclose(state.joint, final[index], atol=3e-12)
        versions = backend.versions
    sensitivity = []
    for kind in ["uniform", "scrambled_halton"]:
        for seed in range(100):
            budget()
            roots = (
                np.random.default_rng(1000 + seed).uniform(size=(60, 3))
                if kind == "uniform"
                else qmc.Halton(3, scramble=True, seed=1100 + seed).random(60)
            )
            before = initial_population(3, roots)
            after = condition(before, np.array([0.35, 0.65]) ** 20)
            a, b = js_pairs(before).mean(), js_pairs(after).mean()
            sensitivity.append(
                {
                    "kind": kind,
                    "seed": seed,
                    "initial_js": a,
                    "final_js": b,
                    "delta_js": b - a,
                }
            )
    pd.DataFrame(sensitivity).to_csv(DATA / "prior-sensitivity.csv", index=False)
    order_rows = []
    roots = np.load(DATA / "faction-3.npz")["roots"]
    for order in permutations(range(3)):
        before = initial_population(3, roots[:, order])
        after = condition(before, np.array([0.35, 0.65]) ** 20)
        order_rows.append(
            {
                "root_columns": str(order),
                "delta_js": js_pairs(after).mean() - js_pairs(before).mean(),
            }
        )
    pd.DataFrame(order_rows).to_csv(DATA / "halton-order.csv", index=False)
    size_rows = []
    for count in [15, 30, 60, 120]:
        sampler = qmc.Halton(3, scramble=False)
        sampler.fast_forward(1)
        before = initial_population(3, sampler.random(count))
        after = condition(before, np.array([0.35, 0.65]) ** 20)
        size_rows.append(
            {
                "agents": count,
                "delta_js": js_pairs(after).mean() - js_pairs(before).mean(),
            }
        )
    pd.DataFrame(size_rows).to_csv(DATA / "population-size.csv", index=False)
    audit = chain_audit(initial)
    audit["bnlearn_versions"] = versions
    audit["max_scipy_js_error"] = float(np.max(np.abs(scipy_js - js_pairs(final))))
    audit["pair_fraction_increasing"] = float(
        np.mean(js_pairs(final) > js_pairs(initial))
    )
    audit["initial_js"] = float(js_pairs(initial).mean())
    audit["final_js"] = float(js_pairs(final).mean())
    audit["relative_increase_percent"] = float(
        100 * (audit["final_js"] / audit["initial_js"] - 1)
    )
    (DATA / "faction-audit.json").write_text(json.dumps(audit, indent=2))
    print("Faction sensitivity and chain identities checked", flush=True)


def chain_audit(initial):
    """Exact expectations over all 21 counts, not a Monte Carlo approximation."""
    flat = initial.reshape(len(initial), -1)
    i, j = np.triu_indices(len(flat), 1)
    p, q = flat[i], flat[j]
    prior_js = js_pairs(initial)
    prior_kl = rel_entr(p, q).sum(1)
    h2 = np.tile([0, 1], flat.shape[1] // 2)
    results = {}
    for experiment in ["hard_H2", "20_signals"]:
        if experiment == "hard_H2":
            likelihoods = np.array([h2 == 0, h2 == 1], dtype=float)
        else:
            likelihoods = binom.pmf(
                np.arange(21)[:, None], 20, np.where(h2, 0.65, 0.35)[None, :]
            )
        predictive = flat @ likelihoods.T
        np.testing.assert_allclose(predictive.sum(1), 1, atol=1e-13)
        expected_equal_js = np.zeros(len(i))
        expected_weighted_js = np.zeros(len(i))
        expected_kl = np.zeros(len(i))
        outcome_rows = []
        for k, likelihood in enumerate(likelihoods):
            post = flat * likelihood / predictive[:, k, None]
            pp, qq = post[i], post[j]
            weight = predictive[i, k] / (predictive[i, k] + predictive[j, k])
            mixture_probability = (predictive[i, k] + predictive[j, k]) / 2
            mix_equal = (pp + qq) / 2
            js_equal = (
                rel_entr(pp, mix_equal).sum(1) + rel_entr(qq, mix_equal).sum(1)
            ) / 2
            mix_weighted = weight[:, None] * pp + (1 - weight[:, None]) * qq
            js_weighted = weight * rel_entr(pp, mix_weighted).sum(1) + (
                1 - weight
            ) * rel_entr(qq, mix_weighted).sum(1)
            expected_equal_js += mixture_probability * js_equal
            expected_weighted_js += mixture_probability * js_weighted
            expected_kl += predictive[i, k] * rel_entr(pp, qq).sum(1)
            outcome_rows.append(
                {
                    "experiment": experiment,
                    "outcome": k,
                    "population_mean_predictive_probability": predictive[:, k].mean(),
                    "joint_js": js_equal.mean(),
                    "delta_js": (js_equal - prior_js).mean(),
                }
            )
        pm = (predictive[i] + predictive[j]) / 2
        predictive_js = (
            rel_entr(predictive[i], pm).sum(1) + rel_entr(predictive[j], pm).sum(1)
        ) / 2
        predictive_kl = rel_entr(predictive[i], predictive[j]).sum(1)
        js_error = np.max(np.abs(prior_js - predictive_js - expected_weighted_js))
        kl_error = np.max(np.abs(prior_kl - predictive_kl - expected_kl))
        assert js_error < 5e-13 and kl_error < 5e-13
        results[experiment] = {
            "expected_equal_weight_js": float(expected_equal_js.mean()),
            "expected_equal_weight_delta": float((expected_equal_js - prior_js).mean()),
            "expected_weighted_js": float(expected_weighted_js.mean()),
            "predictive_js": float(predictive_js.mean()),
            "max_weighted_js_identity_error": float(js_error),
            "max_directed_kl_identity_error": float(kl_error),
            "pairs_expected_equal_js_increase": int(
                (expected_equal_js > prior_js + 1e-12).sum()
            ),
            "pairs": len(i),
        }
        pd.DataFrame(outcome_rows).to_csv(
            DATA / f"outcomes-{experiment}.csv", index=False
        )
    return results


def make_parameters(scenario, seed):
    p = hp.PolyGraphHyperParameters()
    p.op = (
        "BayesianReliabilityOp"
        if scenario.startswith("reliability")
        else "MultiBeliefOp"
    )
    p.device, p.seed, p.trials, p.epsilon = (
        "cpu",
        seed,
        PLAN["trials_per_active_agent_per_round"],
        PLAN["epsilon"],
    )
    p.init.kind = "uniform"
    p.multibelief.backend = "numpy"
    p.multibelief.stop_on_consensus = False
    p.network.kind = "complete" if "complete" in scenario else "cycle"
    p.network.size = PLAN["network_agents"]
    p.simulation.steps = PLAN["rounds"]
    p.simulation.results = None
    p.logging.enabled = False
    if scenario.startswith("meta") or scenario.startswith("hypothesis-propaganda"):
        p.multibelief.case = "meta"
    if "decision-control" in scenario:
        p.multibelief.decision = "hypothesis"
    if "propaganda" in scenario:
        p.multibelief.propaganda.probability = 0.1
        p.multibelief.propaganda.node = (
            "H" if scenario.startswith("hypothesis") else "S"
        )
    if "mixed" in scenario:
        p.unreliablenodes = PLAN["mixed_sources"]
    if "opposite" in scenario:
        p.bayesian_reliability.unreliable_mode = "opposite"
    return p


def social_study():
    rows, summary = [], []
    for scenario in PLAN["scenarios"]:
        started = time.perf_counter()
        for seed in PLAN["seeds"]:
            budget()
            pg.random(seed)
            params = make_parameters(scenario, seed)
            n = params.network.size
            if params.network.kind == "complete":
                sources, destinations = zip(
                    *[(i, j) for i in range(n) for j in range(n) if i != j]
                )
            else:
                sources, destinations = zip(
                    *[(i, j) for i in range(n) for j in ((i - 1) % n, (i + 1) % n)]
                )
            graph = dgl.graph((sources, destinations), num_nodes=n)
            cls = (
                BayesianReliabilityOp
                if scenario.startswith("reliability")
                else MultiBeliefOp
            )
            model = cls(graph, params)
            # Common H and S priors across matched model variants. Reset trial RNG after initialization.
            np.random.seed(seed + 10000)
            (
                beliefs,
                secondary,
                joint_history,
                trust_history,
                action_history,
                payoff_history,
            ) = ([], [], [], [], [], [])
            cumulative_trials = 0
            try:
                for t in range(params.simulation.steps + 1):
                    budget()
                    b = probability_roundoff(graph.ndata["beliefs"].numpy().copy())
                    active = (
                        model.active(graph).numpy()
                        if isinstance(model, MultiBeliefOp)
                        else b > 0.5
                    )
                    beliefs.append(b)
                    action_history.append(active)
                    trial_count = int(graph.ndata["payoffs"][:, 1].sum()) if t else 0
                    cumulative_trials += trial_count
                    payoff_history.append(
                        graph.ndata["payoffs"].numpy().copy() if t else np.zeros((n, 2))
                    )
                    row = {
                        "scenario": scenario,
                        "seed": seed,
                        "round": t,
                        "mean_h": b.mean(),
                        "sd_h": b.std(),
                        "h_js": js_pairs(np.stack((1 - b, b), axis=1)).mean(),
                        "h_brier": np.mean((1 - b) ** 2),
                        "h_entropy": np.mean(-xlogy(b, b) - xlogy(1 - b, 1 - b)),
                        "active_fraction": active.mean(),
                        "confident_correct": (b > 0.95).mean(),
                        "confident_wrong": (b < 0.05).mean(),
                        "cumulative_trials": cumulative_trials,
                    }
                    if isinstance(model, MultiBeliefOp):
                        joints = np.array([state.joint for state in model.states])
                        joint_history.append(joints)
                        logs = np.array(
                            [
                                state.log_joint.reshape(state.joint.shape)
                                for state in model.states
                            ]
                        )
                        row.update(metrics(joints, logs))
                        if "S" in model.network.names:
                            s = probability_roundoff(
                                graph.ndata["belief:S"][:, 1].numpy().copy()
                            )
                            secondary.append(s)
                            row.update(mean_s=s.mean(), sd_s=s.std())
                            product_mean = 0.5 + params.epsilon * (2 * b - 1) * (
                                0.5 + 0.5 * s
                            )
                            correct_mean = graph.ndata["expected_payoff"].numpy()
                            row["decision_disagreement"] = np.mean(
                                (b > 0.5) != (correct_mean > 0.5)
                            )
                            row["mean_dependence_payoff_error"] = np.mean(
                                np.abs(correct_mean - product_mean)
                            )
                    else:
                        trust = probability_roundoff(
                            graph.edata["reliability_belief"].numpy().copy()
                        )
                        trust_history.append(trust)
                        actual = model.actual_reliability[np.array(sources)]
                        row.update(
                            mean_trust=trust.mean(),
                            trust_brier=np.mean((trust - actual) ** 2),
                            trust_reliable=trust[actual].mean(),
                            trust_unreliable=(
                                trust[~actual].mean() if (~actual).any() else np.nan
                            ),
                        )
                    rows.append(row)
                    if t < params.simulation.steps:
                        model(graph)
                summary.append(
                    {
                        **rows[-1],
                        "all_correct": bool((b > 0.95).all()),
                        "no_testers": bool(not active.any()),
                        "some_wrong": bool((b < 0.05).any()),
                    }
                )
                payload = dict(
                    h=beliefs,
                    active=action_history,
                    payoffs=payoff_history,
                    sources=sources,
                    destinations=destinations,
                )
                if secondary:
                    payload["s"] = secondary
                if joint_history:
                    payload["joint"] = joint_history
                if trust_history:
                    payload["trust"] = trust_history
                    payload["actual_reliability"] = model.actual_reliability
                np.savez_compressed(DATA / f"{scenario}-{seed}.npz", **payload)
            finally:
                model.close()
        print(
            f"{scenario}: {len(PLAN['seeds'])} seeds, {time.perf_counter() - started:.1f}s",
            flush=True,
        )
    pd.DataFrame(rows).to_csv(DATA / "social-timeseries.csv", index=False)
    pd.DataFrame(summary).to_csv(DATA / "social-endpoints.csv", index=False)


if __name__ == "__main__":
    torch.set_num_threads(1)
    faction_study()
    social_study()
    (DATA / "execution.json").write_text(
        json.dumps(
            {
                "elapsed_seconds": time.perf_counter() - START,
                "numpy": np.__version__,
                "torch": torch.__version__,
                "dgl": dgl.__version__,
                "approved_workload": PLAN,
            },
            indent=2,
        )
    )
    print(f"Completed in {time.perf_counter() - START:.1f}s", flush=True)
