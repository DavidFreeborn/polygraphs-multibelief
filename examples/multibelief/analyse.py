"""Print belief diagnostics from saved PolyGraphs snapshots, including time zero."""

import argparse
import csv
import json
from pathlib import Path
import sys

import dgl
import h5py
import numpy as np

from polygraphs.beliefs.metrics import distribution_diagnostics


def analyse(directory, prefix):
    directory = Path(directory)
    metadata = json.loads(
        (directory / f"{prefix}.beliefs.json").read_text(encoding="utf-8")
    )
    graph = dgl.load_graphs(str(directory / f"{prefix}.bin"))[0][0]
    rows = []

    def record(step, beliefs, joint=None, reliability=None):
        row = {"step": step, "mean_target": float(np.mean(beliefs))}
        if joint is not None:
            shape = [len(beliefs)] + [
                len(metadata["states"][name]) for name in metadata["nodes"]
            ]
            row.update(distribution_diagnostics(np.asarray(joint).reshape(shape)))
        if reliability is not None:
            sources, destinations = graph.edges()
            external = (sources != destinations).numpy()
            row["mean_reliability"] = (
                float(np.asarray(reliability)[external].mean())
                if external.any()
                else ""
            )
        rows.append(row)

    record(
        0,
        graph.ndata["beliefs"].numpy(),
        graph.ndata["belief_joint"].numpy() if "belief_joint" in graph.ndata else None,
        (
            graph.edata["reliability_belief"].numpy()
            if "reliability_belief" in graph.edata
            else None
        ),
    )
    with h5py.File(directory / f"{prefix}.hd5") as stream:
        for step in sorted(stream["beliefs"], key=int):
            record(
                int(step),
                stream[f"beliefs/{step}"][:],
                stream[f"belief_joint/{step}"][:] if "belief_joint" in stream else None,
                (
                    stream[f"reliability_belief/{step}"][:]
                    if "reliability_belief" in stream
                    else None
                ),
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--prefix", default="1", help="Simulation filename prefix")
    args = parser.parse_args()
    rows = analyse(args.directory, args.prefix)
    writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()
