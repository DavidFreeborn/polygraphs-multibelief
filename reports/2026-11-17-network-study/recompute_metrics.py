"""Recompute distribution metrics from saved evidence with log-space JS mixtures."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from run_experiments import configure, joint_metrics
from polygraphs.beliefs import BayesianNetwork
from polygraphs.beliefs.cases import bandit

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def main():
    settings = json.loads((DATA / "execution.json").read_text())
    all_frames = []
    for i, c in enumerate(settings["conditions"]):
        frame = pd.read_csv(DATA / c["id"] / "metrics.csv")
        for seed in settings["seeds"]:
            z = np.load(DATA / c["id"] / f"{seed}.npz")
            p, _ = configure(c, seed, settings["rounds"])
            ordinary = np.ones(60, dtype=bool)
            ordinary[z["unreliable_ids"].astype(int)] = False
            reports = np.concatenate((np.zeros((1, 60, 2)), z["reports"].cumsum(0)))
            if c["model"] == "trust":
                private = np.concatenate((np.zeros((1, 60, 2)), z["private"].cumsum(0)))
                a = np.array([[0], [1]])
                src, dst = z["trust_sources"], z["trust_recipients"]
                probability = (
                    np.array([[0.55, 0.45], [0.45, 0.55]])
                    if c["treatment"] == "disinformation"
                    else np.array([[0.5, 0.45], [0.5, 0.55]])
                )
            else:
                network = (
                    bandit()
                    if c["model"] == "baseline"
                    else BayesianNetwork(p.multibelief.model["nodes"])
                )
                a = network.assignments()
                scope, table = (
                    (("H",), np.array([0.45, 0.55]))
                    if c["model"] == "baseline"
                    else (
                        p.multibelief.observation["parents"],
                        np.array(p.multibelief.observation["probability"]),
                    )
                )
                probability = table[
                    tuple(a[:, network.names.index(name)] for name in scope)
                ]
                src, dst = z["edges"]
            for index in frame.index[frame.seed == seed]:
                t = int(frame.at[index, "round"])
                counts = reports[t]
                if c["model"] == "trust":
                    own = private[t]
                    logs = (
                        z["initial_log_joint"]
                        + own[:, :1] * np.log([0.45, 0.55])
                        + (own[:, 1:] - own[:, :1]) * np.log([0.55, 0.45])
                    )
                    prior = c["trust_prior"]
                    with np.errstate(divide="ignore"):
                        factors = (
                            np.log([1 - prior, prior])[None, None, :]
                            + counts[src, 0, None, None] * np.log(probability)
                            + (counts[src, 1] - counts[src, 0])[:, None, None]
                            * np.log1p(-probability)
                        )
                    np.add.at(logs, dst, logsumexp(factors, axis=2))
                else:
                    received = counts.copy()
                    np.add.at(received, dst, counts[src])
                    logs = (
                        z["initial_log_joint"]
                        + received[:, :1] * np.log(probability)
                        + (received[:, 1:] - received[:, :1]) * np.log1p(-probability)
                    )
                logs -= logsumexp(logs, axis=1, keepdims=True)
                values = joint_metrics(logs, a)
                values.update(
                    {
                        "ordinary_" + k: v
                        for k, v in joint_metrics(logs[ordinary], a).items()
                    }
                )
                for key, value in values.items():
                    frame.at[index, key] = value
        frame.to_csv(DATA / c["id"] / "metrics.csv", index=False)
        all_frames.append(frame)
        print(f"Recomputed {i+1}/72 {c['id']}", flush=True)
    pd.concat(all_frames, ignore_index=True).to_csv(DATA / "metrics.csv", index=False)


if __name__ == "__main__":
    main()
