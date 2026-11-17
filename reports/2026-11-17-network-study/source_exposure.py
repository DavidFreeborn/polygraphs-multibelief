"""Separate evidence-free source beliefs from source beliefs supported by reports."""

import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def main():
    settings = json.loads((HERE / "data/execution.json").read_text())
    rows = []
    for c in settings["conditions"]:
        if c["model"] != "trust":
            continue
        for seed in settings["seeds"]:
            z = np.load(HERE / "data" / c["id"] / f"{seed}.npz")
            src = z["trust_sources"]
            bad = np.isin(src, z["unreliable_ids"])
            trials = z["reports"][:, :, 1].sum(0)[src]
            for name, mask in [("reliable", ~bad), ("unreliable", bad)]:
                if not mask.any():
                    continue
                for category, eligible in [
                    ("all", np.ones(len(src), bool)),
                    ("no_reports", trials == 0),
                    ("at_least_100_trials", trials >= 100),
                ]:
                    selected = mask & eligible
                    rows.append(
                        {
                            **c,
                            "seed": seed,
                            "source_type": name,
                            "exposure": category,
                            "edge_count": int(selected.sum()),
                            "fraction_of_type": selected.sum() / mask.sum(),
                            "mean_trust": (
                                z["final_trust"][selected].mean()
                                if selected.any()
                                else np.nan
                            ),
                        }
                    )
    frame = pd.DataFrame(rows)
    frame.to_csv(HERE / "data/source-exposure.csv", index=False)
    summary = frame.groupby(
        ["network", "treatment", "trust_prior", "source_type", "exposure"]
    )[["fraction_of_type", "mean_trust"]].mean()
    summary.to_csv(HERE / "data/source-exposure-summary.csv")


if __name__ == "__main__":
    main()
