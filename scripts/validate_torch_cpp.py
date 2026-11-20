"""Regenerate and replay the unmodified study worker through PyTorch/DGL.

Run from the repository root with PYTHONPATH=. and an archived worker.cpp.
The source is read and compiled, never edited. All regenerated data are temporary.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import dgl
import numpy as np
import torch

from polygraphs import hyperparameters as hp
from polygraphs.beliefs.cases import binary_node
from polygraphs.ops import TorchMultiBeliefOp, TorchBayesianReliabilityOp


def specifications(model):
    nodes = [binary_node("H")]
    if model:
        nodes.append(binary_node("S"))
    if model == 3:
        nodes = [binary_node("X"), binary_node("Y", ["H", "S"], [[.9, .1], [.1, .9]]),
                 binary_node("H"), binary_node("S")]
    if model == 0:
        observation = {"parents": ["H"], "probability": [.45, .55]}
    elif model == 1:
        observation = {"parents": ["H", "S"], "probability": [[.45, .45], [.45, .55]]}
    else:
        observation = {"parents": ["X", "Y"] if model == 3 else ["H", "S"],
                       "probability": [[.55, .45], [.45, .55]]}
    return nodes, observation


def replay(path, device="cpu"):
    run = json.loads(path.read_text())
    source, recipient = np.asarray(run["edges"]).T
    graph = dgl.graph((source, recipient), num_nodes=60).to(device)
    p = hp.PolyGraphHyperParameters()
    p.device, p.epsilon, p.trials = device, .05, 1
    p.multibelief.backend = "numpy"
    p.multibelief.stop_on_consensus = False
    p.simulation.steps = run["rounds"]
    p.unreliablenodes = run["bad"]
    trust = run["model"] == 4
    if trust:
        p.bayesian_reliability.prior = run["trust_prior"]
        p.bayesian_reliability.unreliable_mode = "opposite" if run["environment"] == 2 else "uninformative"
        model = TorchBayesianReliabilityOp(graph, p)
        initial = torch.tensor(run["initial"], dtype=torch.float64, device=device)
        model.prior_h.copy_(initial)
        model.log_h.copy_(initial)
        mapping = None
        good = torch.tensor([False, True], device=device)
    else:
        nodes, observation = specifications(run["model"])
        p.multibelief.case, p.multibelief.prior_mode = "custom", "fixed"
        p.multibelief.model = {"nodes": nodes}
        p.multibelief.observation = observation
        p.multibelief.true_probability = .55
        p.multibelief.source_mode = ["honest", "uninformative", "opposite"][run["environment"]]
        model = TorchMultiBeliefOp(graph, p)
        # C++ uses the first listed node as its fastest axis; PyTorch uses C order.
        weights = 2 ** torch.arange(len(nodes), device=device)
        mapping = (model.assignments * weights).sum(1)
        initial = torch.tensor(run["initial"], dtype=torch.float64, device=device)[:, mapping]
        model.set_initial_log_joint(graph, initial)
        good = model.observation_probability > .5
    model._publish(graph)
    raw = path.parent.parent / "raw" / (path.stem + ".gz")
    codes = np.frombuffer(gzip.decompress(raw.read_bytes()), dtype=np.uint8).reshape(-1, 60)
    assert len(codes) == run["rounds"]
    trajectory_error = 0.
    trust_error = 0.
    decisions = 0
    sampled_times = {int(t): k for k, t in enumerate(run["times"])}
    for step in range(len(codes) + 1):
        if step in sampled_times:
            row = run["series"][sampled_times[step]]
            joint = model.log_h.exp() if trust else model.log_joint.exp()
            pg = joint[:, good].sum(1)
            ph = graph.ndata["belief:H"][:, 1]
            actual = [pg.mean().item(), ph.std(unbiased=False).item(), ph.mean().item(), pg.std(unbiased=False).item()]
            expected = [row[0], row[1], row[11], row[12]]
            trajectory_error = max(trajectory_error, float(np.max(np.abs(np.asarray(actual) - expected))))
            if trust:
                bad = torch.zeros(60, dtype=torch.bool, device=device)
                bad[run["bad"]] = True
                unreliable = bad[model.sources]
                actual = [model.trust[~unreliable].mean().item(),
                          ((model.trust - (~unreliable).to(torch.float64)) ** 2).mean().item()]
                expected = [row[6], row[8]]
                if bool(unreliable.any()):
                    actual.append(model.trust[unreliable].mean().item())
                    expected.append(row[7])
                trust_error = max(trust_error, float(np.max(np.abs(np.asarray(actual) - expected))))
        if step == len(codes):
            break
        active = codes[step] != 255
        predicted = model.active(graph).cpu().numpy()
        if not np.array_equal(active, predicted):
            raise AssertionError((path.name, step, "experiment decisions differ"))
        decisions += len(active)
        private = np.stack((np.where(active, codes[step] & 15, 0), active), 1)
        reports = np.stack((np.where(active, codes[step] >> 4, 0), active), 1)
        def evidence(g):
            g.ndata["private_payoffs"] = torch.tensor(private, dtype=torch.float64, device=device)
            g.ndata["payoffs"] = torch.tensor(reports, dtype=torch.float64, device=device)
        model.experiment = evidence
        model(graph)
    actual = model.log_h if trust else model.log_joint[:, torch.argsort(mapping)]
    expected = torch.tensor(run["final"], dtype=torch.float64, device=device)
    log_error = (actual - expected).abs().max().item()
    posterior_error = (actual.exp() - expected.exp()).abs().max().item()
    assert log_error < 1e-8, (path.name, log_error)
    assert trajectory_error < 1e-10, (path.name, trajectory_error)
    assert trust_error < 1e-10, (path.name, trust_error)
    return {"log_posterior_error": log_error, "posterior_error": posterior_error,
            "trajectory_error": trajectory_error, "trust_error": trust_error,
            "decision_checks": decisions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path(__file__).resolve().parents[1] / "reports/current/worker.cpp")
    parser.add_argument("--rounds", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    source_bytes = args.source.read_bytes()
    checksum = hashlib.sha256(source_bytes).hexdigest()
    with tempfile.TemporaryDirectory(prefix="polygraphs-cpp-check-") as directory:
        root = Path(directory)
        subprocess.run(["clang++", "-O3", "-std=c++17", str(args.source), "-lz", "-o", str(root / "worker")], check=True)
        result = subprocess.run([str(root / "worker"), str(root / "data"), str(args.rounds), str(args.repeats), "all", "1729"], check=True, capture_output=True, text=True)
        checks = [replay(path, args.device) for path in sorted((root / "data/runs").glob("*.json"))]
    assert args.source.read_bytes() == source_bytes, "C++ source changed"
    assert len(checks) == 45 * args.repeats
    summary = {
        "source_sha256": checksum, "source_unchanged": True, "device": args.device,
        "runs": len(checks), "conditions": 45, "rounds": args.rounds,
        "repeats": args.repeats, "agents": 60,
        "decision_checks": sum(c["decision_checks"] for c in checks),
        **{"max_" + k: max(c[k] for c in checks) for k in checks[0] if k != "decision_checks"},
        "cpp_generation_log": result.stdout.strip(),
    }
    output = json.dumps(summary, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    print(output)


if __name__ == "__main__":
    main()
