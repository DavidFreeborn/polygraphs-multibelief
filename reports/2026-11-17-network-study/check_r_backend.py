"""Check selected saved final posteriors with independent R bnlearn/gRain inference."""

import json
from pathlib import Path
import numpy as np
from polygraphs.beliefs import BayesianNetwork, BeliefState, binomial_log_likelihood
from polygraphs.beliefs.bnlearn import BnlearnBackend
from polygraphs.beliefs.cases import bandit
from run_experiments import conditions, configure, MODELS

HERE = Path(__file__).resolve().parent


def main():
    errors = []
    with BnlearnBackend() as backend:
        for c in conditions():
            if (
                c["model"] not in MODELS
                or c["network"] != "cycle"
                or c["treatment"] != "disinformation"
            ):
                continue
            z = np.load(HERE / "data" / c["id"] / "1729.npz")
            params, roots = configure(c, 1729, 500)
            net = (
                bandit()
                if c["model"] == "baseline"
                else BayesianNetwork(params.multibelief.model["nodes"])
            )
            scope, probability = (
                (("H",), np.array([0.45, 0.55]))
                if c["model"] == "baseline"
                else (
                    tuple(params.multibelief.observation["parents"]),
                    np.array(params.multibelief.observation["probability"]),
                )
            )
            counts = z["reports"].sum(0).astype(float)
            src, dst = z["edges"]
            received = counts.copy()
            np.add.at(received, dst, counts[src])
            for i in [0, 19, 59]:
                network = net.with_priors(
                    {name: params.multibelief.priors[name][i] for name in net.roots}
                )
                state = BeliefState(network, backend)
                k, n = received[i]
                state.update([(scope, binomial_log_likelihood(probability, k, n))])
                error = float(
                    np.max(
                        np.abs(state.joint.ravel() - np.exp(z["final_log_joint"][i]))
                    )
                )
                assert error < 2e-11, (c["id"], i, error)
                errors.append(
                    {"condition": c["id"], "agent": i, "max_absolute_error": error}
                )
        result = dict(
            versions=backend.versions,
            checks=errors,
            max_error=max(x["max_absolute_error"] for x in errors),
        )
    (HERE / "data/r-validation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
