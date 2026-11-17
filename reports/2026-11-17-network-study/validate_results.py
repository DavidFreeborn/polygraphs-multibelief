"""Reconstruct saved posteriors directly from priors and observation counts."""

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logsumexp, expit, logit, entr
from scipy.spatial.distance import jensenshannon

from run_experiments import configure, joint_metrics
from polygraphs.beliefs import BayesianNetwork
from polygraphs.beliefs.cases import bandit

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def main():
    settings = json.loads((DATA / "execution.json").read_text())
    metrics = pd.read_csv(DATA / "metrics.csv")
    last = metrics[metrics["round"] == settings["rounds"]].set_index(["id", "seed"])
    errors = dict(
        replay_log_posterior=0.0,
        trust_replay=0.0,
        joint_kl=0.0,
        joint_js=0.0,
        symmetric_private_only=0.0,
        symmetric_initial_a=0.0,
        polarising_odds=0.0,
    )
    runs, raw_bytes = 0, 0
    hashes = {}
    for c in settings["conditions"]:
        for seed in settings["seeds"]:
            path = DATA / c["id"] / f"{seed}.npz"
            z = np.load(path)
            logs = z["final_log_joint"]
            p, roots = configure(c, seed, settings["rounds"])
            reports = z["reports"].sum(0).astype(float)
            sources, recipients = z["edges"]
            assert len(z["reports"]) == settings["rounds"]
            assert len(z["unreliable_ids"]) == (0 if c["treatment"] == "honest" else 6)
            assert np.all(z["reports"][:, :, 0] <= z["reports"][:, :, 1])
            assert np.allclose(logsumexp(logs, axis=1), 0, atol=1e-11)
            if c["model"] != "trust":
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
                counts = reports.copy()
                np.add.at(counts, recipients, reports[sources])
                reconstructed = (
                    z["initial_log_joint"]
                    + counts[:, :1] * np.log(probability)
                    + (counts[:, 1:] - counts[:, :1]) * np.log1p(-probability)
                )
                reconstructed -= logsumexp(reconstructed, axis=1, keepdims=True)
                if c["model"] == "polarising":
                    errors["polarising_odds"] = max(
                        errors["polarising_odds"],
                        float(
                            np.max(
                                np.abs(
                                    (logs[:, 0] - logs[:, 3])
                                    - (
                                        z["initial_log_joint"][:, 0]
                                        - z["initial_log_joint"][:, 3]
                                    )
                                )
                            )
                        ),
                    )
            else:
                counts = z["private"].sum(0).astype(float)
                reconstructed = (
                    z["initial_log_joint"]
                    + counts[:, :1] * np.log([0.45, 0.55])
                    + (counts[:, 1:] - counts[:, :1]) * np.log([0.55, 0.45])
                )
                src, dst = z["trust_sources"], z["trust_recipients"]
                probability = (
                    np.array([[0.55, 0.45], [0.45, 0.55]])
                    if c["treatment"] == "disinformation"
                    else np.array([[0.5, 0.45], [0.5, 0.55]])
                )
                prior = c["trust_prior"]
                with np.errstate(divide="ignore"):
                    factor = (
                        np.log([1 - prior, prior])[None, None, :]
                        + reports[src, 0, None, None] * np.log(probability)
                        + (reports[src, 1] - reports[src, 0])[:, None, None]
                        * np.log1p(-probability)
                    )
                normalizers = logsumexp(factor, axis=2)
                np.add.at(reconstructed, dst, normalizers)
                reconstructed -= logsumexp(reconstructed, axis=1, keepdims=True)
                trust = np.exp(
                    reconstructed[dst, :, None] + factor - normalizers[:, :, None]
                )[:, :, 1].sum(1)
                errors["trust_replay"] = max(
                    errors["trust_replay"],
                    float(np.max(np.abs(trust - z["final_trust"]))),
                )
                if c["treatment"] == "disinformation" and prior == 0.5:
                    private_only = expit(
                        z["initial_log_joint"][:, 1]
                        - z["initial_log_joint"][:, 0]
                        + (2 * counts[:, 0] - counts[:, 1]) * np.log(0.55 / 0.45)
                    )
                    errors["symmetric_private_only"] = max(
                        errors["symmetric_private_only"],
                        float(np.max(np.abs(np.exp(logs[:, 1]) - private_only))),
                    )
                    initial = np.exp(z["initial_log_joint"][:, 1])
                    initial_a = z["initial_a"]
                    errors["symmetric_initial_a"] = max(
                        errors["symmetric_initial_a"],
                        float(
                            np.max(
                                np.abs(
                                    z["beliefs"][:, initial_a, 0] - initial[initial_a]
                                )
                            )
                        ),
                    )
            errors["replay_log_posterior"] = max(
                errors["replay_log_posterior"],
                float(np.max(np.abs(logs - reconstructed))),
            )
            if seed == settings["seeds"][0]:
                probs = np.exp(logs)
                # Explicit pair enumeration independent of the population identity.
                kl = np.mean(
                    [
                        np.sum(probs[i] * (logs[i] - logs[j]))
                        for i in range(60)
                        for j in range(60)
                        if i != j
                    ]
                )
                i, j = np.triu_indices(60, 1)
                js = np.mean(
                    entr((probs[i] + probs[j]) / 2).sum(1)
                    - (entr(probs[i]).sum(1) + entr(probs[j]).sum(1)) / 2
                )
                errors["joint_kl"] = max(
                    errors["joint_kl"], abs(kl - last.loc[(c["id"], seed), "joint_kl"])
                )
                errors["joint_js"] = max(
                    errors["joint_js"], abs(js - last.loc[(c["id"], seed), "joint_js"])
                )
            raw_bytes += path.stat().st_size
            hashes[str(path.relative_to(HERE)).replace("\\", "/")] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            runs += 1
    for key, value in errors.items():
        assert value < (1e-7 if key == "replay_log_posterior" else 1e-9), (key, value)
    baseline = metrics[metrics.model == "baseline"]
    assert np.allclose(baseline.joint_kl, baseline.product_kl, atol=1e-12)
    assert np.allclose(baseline.joint_js, baseline.product_js, atol=1e-12)
    assert np.all(
        np.isfinite(metrics[["joint_kl", "product_kl", "joint_js", "product_js"]])
    )
    assert len(last) == 72 * 20
    result = dict(
        runs_verified=runs,
        raw_bytes=raw_bytes,
        max_absolute_errors=errors,
        status="passed",
    )
    (DATA / "validation.json").write_text(json.dumps(result, indent=2))
    (DATA / "raw-sha256.json").write_text(json.dumps(hashes, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
