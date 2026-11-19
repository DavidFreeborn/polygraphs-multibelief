"""Small reproducible CPU benchmark; construction, inference and sampling split.

All inference timings use identical fixed honest evidence. The C++ worker also
computes diagnostics and writes data; its full-run timing is reported separately
by validate_torch_cpp.py and is not a like-for-like inference benchmark.
"""

import argparse
import json
from pathlib import Path
import platform
import statistics
import time

import dgl
import numpy as np
import torch

from polygraphs import hyperparameters as hp
from polygraphs.ops import (
    MultiBeliefOp, BayesianReliabilityOp, BatchedMultiBeliefOp,
    BatchedBayesianReliabilityOp, TorchMultiBeliefOp, TorchBayesianReliabilityOp,
)


def config(case, size):
    p = hp.PolyGraphHyperParameters()
    p.epsilon, p.trials = .05, 10
    p.multibelief.backend = "numpy"
    p.multibelief.stop_on_consensus = False
    p.multibelief.case = "networked-factionalization3" if case == "32-state" else "baseline"
    p.simulation.steps = 500
    p.init.kind, p.init.constant.value = "constant", .6
    p.bayesian_reliability.prior = .9
    p.bayesian_reliability.actual_reliability = 1.
    p.bayesian_reliability.unreliable_mode = "opposite"
    return p


def measure(operator, case, size, rounds):
    p = config(case, size)
    sources = np.arange(size)
    destinations = (sources + 1) % size
    g = dgl.graph((np.r_[sources, destinations], np.r_[destinations, sources]), num_nodes=size)
    start = time.perf_counter()
    model = operator(g, p)
    construction = time.perf_counter() - start
    original_experiment = model.experiment
    evidence = torch.tensor(np.tile([6., 10.], (size, 1)), dtype=torch.float64)
    def experiment(graph):
        graph.ndata["payoffs"] = evidence
        graph.ndata["private_payoffs"] = evidence
    model.experiment = experiment
    for _ in range(5):
        model(g)
    timings = []
    for _ in range(3):
        start = time.perf_counter()
        for _ in range(rounds):
            model(g)
        timings.append((time.perf_counter() - start) / rounds)
    # Sampling alone, including publishing both evidence channels where supported.
    start = time.perf_counter()
    for _ in range(rounds):
        original_experiment(g)
    sampling = (time.perf_counter() - start) / rounds
    return {"operator": operator.__name__, "model": case, "agents": size,
            "construction_seconds": construction,
            "inference_seconds_per_round": statistics.median(timings),
            "sampling_seconds_per_round": sampling}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--sizes", type=int, nargs="+", default=[60, 1000])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    np.random.seed(13)
    torch.manual_seed(13)
    rows = []
    for size in args.sizes:
        for case in ("baseline", "32-state", "trust"):
            operators = ((BayesianReliabilityOp, BatchedBayesianReliabilityOp, TorchBayesianReliabilityOp)
                         if case == "trust" else
                         (MultiBeliefOp, BatchedMultiBeliefOp, TorchMultiBeliefOp))
            for operator in operators:
                # The per-agent reference is costly. Keep its measurement bounded.
                count = min(args.rounds, 10) if operator == operators[0] else args.rounds
                rows.append(measure(operator, case, size, count))
    result = {"platform": platform.platform(), "python": platform.python_version(),
              "torch": str(torch.__version__), "numpy": np.__version__, "dgl": dgl.__version__,
              "torch_threads": torch.get_num_threads(), "topology": "bidirectional cycle",
              "timing": "median of three batches; five warmup rounds; excludes construction",
              "cuda_available": torch.cuda.is_available(), "results": rows}
    output = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    print(output)


if __name__ == "__main__":
    main()
