"""Approved networked five-node comparison with exact posterior cross-checks."""

import json
from pathlib import Path
import time

import dgl
import numpy as np
import pandas as pd
import torch

import polygraphs as pg
from polygraphs import hyperparameters as hp
from polygraphs.ops.multibelief import MultiBeliefOp
from run_study import metrics, marginal_probabilities, condition

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
PLAN = json.loads((HERE / "networked-faction-workload.json").read_text())


def main():
    torch.set_num_threads(1)
    start = time.perf_counter()
    rows = []
    for topology in PLAN["topologies"]:
        for seed in PLAN["seeds"]:
            pg.random(seed)
            p = hp.PolyGraphHyperParameters()
            p.op, p.device, p.seed, p.trials = "MultiBeliefOp", "cpu", seed, 10
            p.multibelief.backend = "numpy"
            p.multibelief.case = "networked-factionalization3"
            p.multibelief.prior_mode = "halton"
            p.multibelief.stop_on_consensus = False
            p.simulation.steps, p.simulation.results = 40, None
            p.logging.enabled = False
            n = PLAN["agents"]
            edges = (
                [(i, j) for i in range(n) for j in range(n) if i != j]
                if topology == "complete"
                else [(i, j) for i in range(n) for j in ((i - 1) % n, (i + 1) % n)]
            )
            source, destination = zip(*edges)
            graph = dgl.graph((source, destination), num_nodes=n)
            model = MultiBeliefOp(graph, p)
            initial = np.array([state.joint for state in model.states])
            control_initial = np.load(DATA / "faction-3.npz")["joints"][0]
            np.testing.assert_allclose(initial, control_initial, atol=1e-14)
            histories, log_histories, actions, payoffs, exposures = [], [], [], [], []
            cumulative = np.zeros((n, 2))
            try:
                for t in range(41):
                    if time.perf_counter() - start > 300:
                        raise RuntimeError("Approved five-minute compute cap reached")
                    joints = np.array([state.joint for state in model.states])
                    log_joints = np.array(
                        [
                            state.log_joint.reshape(state.joint.shape)
                            for state in model.states
                        ]
                    )
                    histories.append(joints)
                    log_histories.append(log_joints)
                    active = model.active(graph).numpy()
                    actions.append(active)
                    counts = (
                        graph.ndata["payoffs"].numpy().copy() if t else np.zeros((n, 2))
                    )
                    payoffs.append(counts)
                    if t:
                        for recipient, neighbours in enumerate(model.neighbours):
                            cumulative[recipient] += counts[
                                [recipient, *neighbours]
                            ].sum(0)
                    exposures.append(cumulative.copy())
                    # Accumulated received counts provide an independent closed-form update.
                    for index in [0, 19, 39, 59]:
                        k, trials = cumulative[index]
                        log_weights = k * np.log([0.35, 0.65]) + (trials - k) * np.log(
                            [0.65, 0.35]
                        )
                        reference = condition(
                            initial[index : index + 1],
                            np.exp(log_weights - log_weights.max()),
                        )[0]
                        np.testing.assert_allclose(joints[index], reference, atol=2e-12)
                    b = marginal_probabilities(joints)
                    row = {
                        "topology": topology,
                        "seed": seed,
                        "round": t,
                        "active_fraction": active.mean(),
                        "mean_received_trials": cumulative[:, 1].mean(),
                        **metrics(joints, log_joints),
                    }
                    for index, name in enumerate(model.network.names):
                        row[f"mean_{name}"] = b[:, index].mean()
                        row[f"sd_{name}"] = b[:, index].std()
                    rows.append(row)
                    if t < 40:
                        model(graph)
                np.savez_compressed(
                    DATA / f"networked-faction-{topology}-{seed}.npz",
                    joints=histories,
                    log_joints=log_histories,
                    active=actions,
                    payoffs=payoffs,
                    cumulative_received_counts=exposures,
                    names=model.network.names,
                )
            finally:
                model.close()
            print(f"{topology}, seed {seed}: complete", flush=True)
    pd.DataFrame(rows).to_csv(DATA / "networked-faction-timeseries.csv", index=False)
    (DATA / "networked-execution.json").write_text(
        json.dumps(
            {"elapsed_seconds": time.perf_counter() - start, "workload": PLAN}, indent=2
        )
    )


if __name__ == "__main__":
    main()
