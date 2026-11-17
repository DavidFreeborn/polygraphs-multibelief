"""Fixed-seed, paired experiments on four 60-agent social networks."""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import dgl
import networkx as nx
import numpy as np
import pandas as pd
import scipy
from scipy.special import logsumexp, rel_entr
import torch

from polygraphs import hyperparameters as hp
from polygraphs.beliefs.cases import binary_node, factionalization
from polygraphs.ops import BatchedMultiBeliefOp, BatchedBayesianReliabilityOp

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
NETWORKS = ["cycle", "complete", "er01", "er05"]
MODELS = ["baseline", "converging", "polarising", "factionalising"]
TREATMENTS = ["honest", "misinformation", "disinformation"]
N, EPSILON, TRIALS = 60, 0.05, 10


def conditions():
    rows = [
        dict(model=m, network=g, treatment=t, trust_prior=None)
        for m in MODELS
        for g in NETWORKS
        for t in TREATMENTS
    ]
    rows += [
        dict(model="trust", network=g, treatment=t, trust_prior=p)
        for g in NETWORKS
        for t in TREATMENTS
        for p in [0.5, 1.0 if t == "honest" else 0.9]
    ]
    for row in rows:
        row["id"] = "-".join(str(v) for v in row.values() if v is not None)
    return rows


def graph_for(kind, seed):
    if kind == "cycle":
        graph = nx.cycle_graph(N)
    elif kind == "complete":
        graph = nx.complete_graph(N)
    else:
        graph = nx.erdos_renyi_graph(
            N, 0.1 if kind == "er01" else 0.5, seed=seed + 10000
        )
    adj = nx.to_numpy_array(graph, dtype=float)
    src, dst = np.where(adj)
    return dgl.graph((src, dst), num_nodes=N), graph


def configure(condition, seed, rounds):
    p = hp.PolyGraphHyperParameters()
    p.epsilon, p.trials = EPSILON, TRIALS
    p.network.size = N
    p.init.kind = "uniform"
    p.multibelief.backend = "numpy"
    p.multibelief.stop_on_consensus = False
    p.simulation.steps = rounds
    roots = np.random.default_rng(seed + 20000).uniform(size=(N, 3))
    bad = np.sort(np.random.default_rng(seed + 30000).choice(N, 6, replace=False))
    p.unreliablenodes = [] if condition["treatment"] == "honest" else bad.tolist()
    mode = {
        "honest": "honest",
        "misinformation": "uninformative",
        "disinformation": "opposite",
    }[condition["treatment"]]
    if condition["model"] == "trust":
        p.bayesian_reliability.prior = condition["trust_prior"]
        p.bayesian_reliability.unreliable_mode = (
            "opposite" if mode == "opposite" else "uninformative"
        )
    else:
        p.multibelief.source_mode = mode
        p.multibelief.prior_mode = "fixed"
        p.multibelief.true_probability = 0.55
        if condition["model"] == "baseline":
            p.multibelief.priors = {"H": roots[:, 0].tolist()}
        elif condition["model"] in ("converging", "polarising"):
            p.multibelief.case = "custom"
            p.multibelief.model = {"nodes": [binary_node("H"), binary_node("S")]}
            p.multibelief.priors = {
                "H": roots[:, 0].tolist(),
                "S": roots[:, 1].tolist(),
            }
            table = (
                [[0.45, 0.5], [0.5, 0.55]]
                if condition["model"] == "converging"
                else [[0.55, 0.45], [0.45, 0.55]]
            )
            p.multibelief.observation = {"parents": ["H", "S"], "probability": table}
        else:
            p.multibelief.case = "custom"
            p.multibelief.model = {"nodes": factionalization(3).nodes}
            p.multibelief.target = "H2"
            p.multibelief.priors = {
                name: roots[:, k].tolist() for k, name in enumerate(["H1", "H4", "H5"])
            }
            p.multibelief.observation = {"parents": ["H2"], "probability": [0.45, 0.55]}
    return p, roots


def make_model(c, seed, rounds):
    p, roots = configure(c, seed, rounds)
    np.random.seed(seed)
    torch.manual_seed(seed)
    graph, social = graph_for(c["network"], seed)
    cls = (
        BatchedBayesianReliabilityOp if c["model"] == "trust" else BatchedMultiBeliefOp
    )
    model = cls(graph, p)
    if c["model"] == "trust":
        model.prior_h = np.log(np.stack((1 - roots[:, 0], roots[:, 0]), 1))
        model.log_h = model.prior_h.copy()
        model._publish(graph)
    rng = np.random.default_rng(seed + 40000)

    def experiment(g):
        # All runs consume the same uniforms at a given seed and round,
        # even when action histories differ.
        uniforms = rng.random((2, N, TRIALS))
        active = model.active(g).numpy()
        trials = active.astype(int) * TRIALS
        genuine = (uniforms[0] < 0.55).sum(1) * active
        reports = genuine.copy()
        ids = p.unreliablenodes
        if ids:
            probability = 0.5 if c["treatment"] == "misinformation" else 0.45
            reports[ids] = (uniforms[1, ids] < probability).sum(1) * active[ids]
        g.ndata["payoffs"] = torch.from_numpy(
            np.stack((reports, trials), 1).astype(float)
        )
        if c["model"] == "trust":
            g.ndata["private_payoffs"] = torch.from_numpy(
                np.stack((genuine, trials), 1).astype(float)
            )

    model.experiment = experiment
    return model, graph, social, p


def ordered_kl(logs):
    n = len(logs)
    probs = np.exp(logs)
    return max(
        0.0,
        float(
            n
            / (n - 1)
            * (np.mean(np.sum(probs * logs, 1)) - np.sum(probs.mean(0) * logs.mean(0)))
        ),
    )


def pair_js(logs):
    i, j = np.triu_indices(len(logs), 1)
    p, q = logs[i], logs[j]
    mid = np.logaddexp(p, q) - np.log(2.0)
    return max(
        0.0,
        float(
            np.mean(
                (np.sum(np.exp(p) * (p - mid), 1) + np.sum(np.exp(q) * (q - mid), 1))
                / 2
            )
        ),
    )


def joint_metrics(logs, assignments):
    marginal_logs = [
        np.stack(
            [logsumexp(logs[:, assignments[:, k] == s], axis=1) for s in [0, 1]], 1
        )
        for k in range(assignments.shape[1])
    ]
    product_logs = sum(m[:, assignments[:, k]] for k, m in enumerate(marginal_logs))
    probs = np.exp(logs)
    return dict(
        joint_kl=ordered_kl(logs),
        product_kl=sum(ordered_kl(m) for m in marginal_logs),
        joint_js=pair_js(logs),
        product_js=pair_js(product_logs),
        dependence=float(np.mean(np.sum(probs * (logs - product_logs), 1))),
    )


def one_run(c, seed, rounds, save=True, deadline=None):
    start = time.perf_counter()
    model, g, social, p = make_model(c, seed, rounds)
    trust = c["model"] == "trust"
    good_agents = np.ones(N, dtype=bool)
    good_agents[p.unreliablenodes] = False
    sample_rounds = sorted(
        set(range(min(21, rounds + 1))) | set(range(0, rounds + 1, 5)) | {rounds}
    )
    rows, node_rows, snapshots, log_snapshots, trust_snapshots, reports, privates = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    initial = (model.log_h if trust else model.log_joint).copy()
    initial_a = ~model.active(g).numpy()
    names = ["H"] if trust else list(model.network.names)
    for t in range(rounds + 1):
        if deadline and time.perf_counter() > deadline:
            raise TimeoutError("Approved simulation wall-time cap reached")
        if t in sample_rounds:
            logs = model.log_h if trust else model.log_joint
            assignments = np.array([[0], [1]]) if trust else model.assignments
            beliefs = np.column_stack(
                [g.ndata[f"belief:{name}"][:, 1].numpy() for name in names]
            )
            active = model.active(g).numpy()
            if trust:
                prediction = 0.45 + 0.1 * beliefs[:, 0]
                good = beliefs[:, 0]
            else:
                prediction = g.ndata["expected_payoff"].numpy()
                good = np.exp(logs)[:, model.observation_probability > 0.5].sum(1)
            metrics = joint_metrics(logs, assignments)
            row = {
                **c,
                "seed": seed,
                "round": t,
                **metrics,
                "mean_good": good.mean(),
                "ordinary_good": good[good_agents].mean(),
                "correct_fraction": float(np.mean(good > 0.95)),
                "ordinary_correct_fraction": float(np.mean(good[good_agents] > 0.95)),
                "brier_good": np.mean((good - 1) ** 2),
                "active_fraction": active.mean(),
                "all_correct": bool(np.all(good > 0.95)),
                "no_testers": bool(not active.any()),
                "received_mean": model.cumulative_received.mean(),
                "components": nx.number_connected_components(social),
                "mean_degree": np.mean([d for _, d in social.degree()]),
            }
            for k, name in enumerate(names + ["D_next"]):
                b = beliefs[:, k] if k < len(names) else prediction
                node_rows.append(
                    {
                        **c,
                        "seed": seed,
                        "round": t,
                        "node": name,
                        "mean": b.mean(),
                        "sd": b.std(ddof=0),
                        "ordinary_mean": b[good_agents].mean(),
                        "ordinary_sd": b[good_agents].std(ddof=0),
                    }
                )
            if trust:
                actual = model.actual_reliability[model.sources]
                reliable = model.trust[actual]
                unreliable = model.trust[~actual]
                exposed = model.report_counts[model.sources, 1] > 0
                row.update(
                    trust_reliable=reliable.mean() if len(reliable) else np.nan,
                    trust_unreliable=unreliable.mean() if len(unreliable) else np.nan,
                    trust_brier=np.mean((model.trust - actual) ** 2),
                    reporting_edge_fraction=exposed.mean(),
                    initial_a_mean=(
                        beliefs[initial_a, 0].mean() if initial_a.any() else np.nan
                    ),
                    initial_b_mean=(
                        beliefs[~initial_a, 0].mean() if (~initial_a).any() else np.nan
                    ),
                )
                if seed == 1729:
                    trust_snapshots.append(model.trust.copy())
            # Retain all requested measures for the ordinary population too.
            row.update(
                {
                    "ordinary_" + key: value
                    for key, value in joint_metrics(
                        logs[good_agents], assignments
                    ).items()
                }
            )
            rows.append(row)
            snapshots.append(beliefs)
            if seed == 1729:
                log_snapshots.append(logs.copy())
        if t < rounds:
            model(g)
            reports.append(g.ndata["payoffs"].numpy().astype(np.int16))
            if trust:
                privates.append(g.ndata["private_payoffs"].numpy().astype(np.int16))
    if save:
        destination = DATA / c["id"]
        destination.mkdir(parents=True, exist_ok=True)
        payload = dict(
            rounds=sample_rounds,
            beliefs=snapshots,
            names=names,
            initial_log_joint=initial,
            final_log_joint=logs,
            reports=reports,
            edges=np.array(g.edges()),
            unreliable_ids=p.unreliablenodes,
            initial_a=initial_a,
        )
        if log_snapshots:
            payload["log_joint_snapshots"] = log_snapshots
        if trust:
            payload.update(
                private=privates,
                final_log_r_given_h=model.log_r_given_h,
                trust_sources=model.sources,
                trust_recipients=model.recipients,
                final_trust=model.trust,
            )
            if trust_snapshots:
                payload["trust_snapshots"] = trust_snapshots
        np.savez_compressed(destination / f"{seed}.npz", **payload)
    model.close()
    return rows, node_rows, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--rounds", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    torch.set_num_threads(1)
    DATA.mkdir(exist_ok=True)
    plan = conditions()
    start = time.perf_counter()
    if args.pilot:
        chosen = [
            next(
                c
                for c in plan
                if c["model"] == m
                and c["network"] == "complete"
                and c["treatment"] == "disinformation"
            )
            for m in MODELS + ["trust"]
        ]
        timings = []
        for c in chosen:
            _, _, seconds = one_run(c, 1729, 100, save=False)
            timings.append({**c, "seconds": seconds})
            print(c["id"], seconds, flush=True)
        (DATA / "pilot.json").write_text(json.dumps(timings, indent=2))
        return
    metadata = dict(
        agents=N,
        epsilon=EPSILON,
        trials=TRIALS,
        rounds=args.rounds,
        repeats=args.repeats,
        seeds=list(range(1729, 1729 + args.repeats)),
        conditions=plan,
        numpy=np.__version__,
        scipy=scipy.__version__,
        torch=torch.__version__,
        dgl=dgl.__version__,
        python=platform.python_version(),
        platform=platform.platform(),
        cap_seconds=5400,
    )
    (DATA / "settings.json").write_text(json.dumps(metadata, indent=2))
    rows, nodes, timings = [], [], []
    for i, c in enumerate(plan):
        condition_start = time.perf_counter()
        for seed in metadata["seeds"]:
            r, n, seconds = one_run(c, seed, args.rounds, deadline=start + 5400)
            rows.extend(r)
            nodes.extend(n)
            timings.append({"id": c["id"], "seed": seed, "seconds": seconds})
        # Checkpoint each condition; no costly reruns after interrupted reporting.
        pd.DataFrame([r for r in rows if r["id"] == c["id"]]).to_csv(
            DATA / c["id"] / "metrics.csv", index=False
        )
        pd.DataFrame([r for r in nodes if r["id"] == c["id"]]).to_csv(
            DATA / c["id"] / "nodes.csv", index=False
        )
        print(
            f"{i+1}/{len(plan)} {c['id']}: {time.perf_counter()-condition_start:.1f}s; elapsed {time.perf_counter()-start:.1f}s",
            flush=True,
        )
    pd.DataFrame(rows).to_csv(DATA / "metrics.csv", index=False)
    pd.DataFrame(nodes).to_csv(DATA / "nodes.csv", index=False)
    pd.DataFrame(timings).to_csv(DATA / "timings.csv", index=False)
    metadata["elapsed_seconds"] = time.perf_counter() - start
    (DATA / "execution.json").write_text(json.dumps(metadata, indent=2))
    print(
        f"Completed {len(timings)} runs in {metadata['elapsed_seconds']:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
