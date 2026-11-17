"""Validate saved study outputs without launching simulations."""

from hashlib import sha256
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.stats import entropy

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
plan = json.loads((HERE / "workload.json").read_text())
times = pd.read_csv(DATA / "social-timeseries.csv")
networked = pd.read_csv(DATA / "networked-faction-timeseries.csv")
assert len(times) == 10 * 12 * 81
assert len(networked) == 2 * 4 * 41
assert not np.isinf(times.select_dtypes("number").to_numpy()).any()
assert np.isfinite(networked.select_dtypes("number").to_numpy()).all()
for seed in plan["seeds"]:
    reference = np.load(DATA / f"baseline-cycle-{seed}.npz")["h"][0]
    for scenario in plan["scenarios"]:
        archive = np.load(DATA / f"{scenario}-{seed}.npz")
        np.testing.assert_allclose(archive["h"][0], reference, rtol=0, atol=2e-14)
        assert archive["h"].shape == (81, 12)
        assert np.isfinite(archive["h"]).all()
        if "joint" in archive:
            joints = archive["joint"]
            np.testing.assert_allclose(
                joints.reshape(81, 12, -1).sum(-1), 1, atol=2e-14
            )

max_kl_error = 0.0
for topology in ("cycle", "complete"):
    for seed in (1729, 1730, 1731, 1732):
        archive = np.load(DATA / f"networked-faction-{topology}-{seed}.npz")
        joints, logs = archive["joints"], archive["log_joints"]
        assert np.isfinite(logs).all()
        np.testing.assert_allclose(np.exp(logs), joints, atol=1e-15)
        np.testing.assert_allclose(joints.reshape(41, 60, -1).sum(-1), 1, atol=2e-13)
        for t in (0, 1, 5, 40):
            p = joints[t].reshape(60, -1)
            i, j = np.triu_indices(60, 1)
            independent = (
                (entropy(p[i], p[j], axis=1) + entropy(p[j], p[i], axis=1)) / 2
            ).mean()
            recorded = networked.loc[
                (networked.topology == topology)
                & (networked.seed == seed)
                & (networked["round"] == t),
                "joint_kl",
            ].item()
            max_kl_error = max(max_kl_error, abs(independent - recorded))
assert max_kl_error < 2e-11
html = (HERE / "results.html").read_text(encoding="utf-8")
for link in re.findall(r'href="([^"]+)"', html):
    if not link.startswith(("#", "http")):
        assert (HERE / link).exists(), link
assert html.count("data:image/png;base64,") == 9
validation = {
    "social_runs": 120,
    "networked_faction_runs": 8,
    "social_metric_rows": len(times),
    "networked_faction_metric_rows": len(networked),
    "matched_initial_H_priors": True,
    "normalized_saved_joints": True,
    "maximum_scipy_KL_error": max_kl_error,
    "html_local_links_valid": True,
}
(DATA / "validation.json").write_text(json.dumps(validation, indent=2))
lines = []
for path in sorted(HERE.rglob("*")):
    if path.is_file() and path.suffix in (".npz", ".png", ".pdf"):
        lines.append(
            f"{sha256(path.read_bytes()).hexdigest()}  {path.relative_to(HERE).as_posix()}"
        )
(HERE / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(validation, indent=2))
