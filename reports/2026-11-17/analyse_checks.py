"""Independent checks and compact tables from the saved study arrays."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit, rel_entr, xlogy

from run_study import js_pairs, condition, probability_roundoff, metrics

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
times = pd.read_csv(DATA / "social-timeseries.csv")
endpoints = pd.read_csv(DATA / "social-endpoints.csv")
max_roundoff = 0.0
for (scenario, seed), indexes in times.groupby(["scenario", "seed"]).groups.items():
    saved = np.load(DATA / f"{scenario}-{seed}.npz")
    raw = saved["h"]
    h = probability_roundoff(raw)
    max_roundoff = max(max_roundoff, float(np.max(np.abs(h - raw))))
    for index in indexes:
        t = int(times.at[index, "round"])
        values = h[t]
        times.at[index, "mean_h"] = values.mean()
        times.at[index, "sd_h"] = values.std()
        times.at[index, "h_js"] = js_pairs(
            np.stack((1 - values, values), axis=1)
        ).mean()
        times.at[index, "h_brier"] = np.mean((1 - values) ** 2)
        times.at[index, "h_entropy"] = np.mean(
            -xlogy(values, values) - xlogy(1 - values, 1 - values)
        )
        if "joint" in saved:
            diagnostic = metrics(saved["joint"][t])
            for metric in ["joint_kl", "marginal_product_kl"]:
                times.at[index, metric] = diagnostic[metric]
    endpoint_index = endpoints.index[
        (endpoints.scenario == scenario) & (endpoints.seed == seed)
    ][0]
    final_index = times.index[
        (times.scenario == scenario) & (times.seed == seed) & (times["round"] == 80)
    ][0]
    for metric in [
        "mean_h",
        "sd_h",
        "h_js",
        "h_brier",
        "h_entropy",
        "joint_kl",
        "marginal_product_kl",
    ]:
        endpoints.at[endpoint_index, metric] = times.at[final_index, metric]
assert not np.isinf(times.select_dtypes("number").to_numpy()).any()
assert times[["mean_h", "sd_h", "h_js", "h_brier", "h_entropy"]].notna().all().all()
times.to_csv(DATA / "social-timeseries.csv", index=False)
endpoints.to_csv(DATA / "social-endpoints.csv", index=False)
archive = np.load(DATA / "faction-3.npz")
faction_metrics = pd.read_csv(DATA / "faction-metrics.csv")
for case in (1, 2, 3):
    for t, joints in enumerate(np.load(DATA / f"faction-{case}.npz")["joints"]):
        for metric, value in metrics(joints).items():
            faction_metrics.loc[
                (faction_metrics.case == case) & (faction_metrics["round"] == t), metric
            ] = value
faction_metrics.to_csv(DATA / "faction-metrics.csv", index=False)
initial = archive["joints"][0]
flat = initial.reshape(60, -1)
i, j = np.triu_indices(60, 1)
prior_js = js_pairs(initial)
posteriors = [
    condition(initial, [1, 0]).reshape(60, -1),
    condition(initial, [0, 1]).reshape(60, -1),
]
predictive = initial.sum(axis=(1, 2, 3, 4))
expected = np.zeros(len(i))
branch_values = []
for d, post in enumerate(posteriors):
    a, b = post[i], post[j]
    mix = (a + b) / 2
    value = (rel_entr(a, mix).sum(1) + rel_entr(b, mix).sum(1)) / 2
    branch_values.append(value)
    expected += (predictive[i, d] + predictive[j, d]) / 2 * value
pair = int(np.argmax(expected - prior_js))
counterexample = {
    "agents_zero_based": [int(i[pair]), int(j[pair])],
    "root_names": ["H1", "H4", "H5"],
    "root_priors": archive["roots"][[i[pair], j[pair]]].tolist(),
    "prior_js": float(prior_js[pair]),
    "posterior_js_H2_false": float(branch_values[0][pair]),
    "posterior_js_H2_true": float(branch_values[1][pair]),
    "predictive_H2_true": predictive[[i[pair], j[pair]], 1].tolist(),
    "expected_equal_weight_js_under_pair_mixture": float(expected[pair]),
    "delta": float(expected[pair] - prior_js[pair]),
}
for d, prediction in enumerate(predictive[[i[pair], j[pair]]]):
    counterexample[f"expected_equal_weight_js_under_agent_{d + 1}"] = float(
        prediction[0] * branch_values[0][pair] + prediction[1] * branch_values[1][pair]
    )
kl_rows = []
for first, second in [(i[pair], j[pair]), (j[pair], i[pair])]:
    before = float(rel_entr(flat[first], flat[second]).sum())
    after = [float(rel_entr(post[first], post[second]).sum()) for post in posteriors]
    expected_kl = float(predictive[first] @ np.asarray(after))
    predictive_kl = float(rel_entr(predictive[first], predictive[second]).sum())
    np.testing.assert_allclose(expected_kl, before - predictive_kl, atol=2e-14)
    kl_rows.append(
        {
            "first_agent": int(first),
            "second_agent": int(second),
            "prior_kl": before,
            "posterior_kl_H2_false": after[0],
            "posterior_kl_H2_true": after[1],
            "expected_kl_under_first_agent": expected_kl,
            "predictive_kl": predictive_kl,
        }
    )
pd.DataFrame(kl_rows).to_csv(DATA / "counterexample-directed-kl.csv", index=False)

# Shared downstream CPTs retain exactly the root-distribution JS, before and after evidence.
root_error = max(
    float(np.max(np.abs(js_pairs(joints) - js_pairs(joints.sum(axis=(4, 5))))))
    for joints in archive["joints"]
)
assert root_error < 1e-12
symmetry_error = 0.0
frozen = []
for path in DATA.glob("reliability-complete-mixed-opposite-*.npz"):
    run = np.load(path)
    h, counts, actual = run["h"], run["payoffs"], run["actual_reliability"]
    successes = counts[:, actual, 0].cumsum(axis=0)
    trials = counts[:, actual, 1].cumsum(axis=0)
    private_posterior = expit(
        logit(h[0, actual])[None, :] + (2 * successes - trials) * np.log(0.55 / 0.45)
    )
    symmetry_error = max(
        symmetry_error, float(np.max(np.abs(h[:, actual] - private_posterior)))
    )
    inactive = h[0] <= 0.5
    np.testing.assert_allclose(
        h[:, inactive], np.tile(h[0, inactive], (len(h), 1)), atol=2e-13
    )
    frozen.append(int(inactive.sum()))
assert symmetry_error < 2e-12
checks = {
    "equal_weight_js_counterexample": counterexample,
    "max_probability_roundoff_correction": max_roundoff,
    "max_root_vs_full_joint_js_error": root_error,
    "max_opposite_model_private_only_posterior_error": symmetry_error,
    "initially_inactive_agents_frozen": sum(frozen),
    "opposite_model_total_agents": 144,
}
(DATA / "additional-checks.json").write_text(json.dumps(checks, indent=2))

rows = []
for name, values in endpoints.groupby("scenario", sort=False):
    row = {"scenario": name, "runs": len(values)}
    for metric in [
        "mean_h",
        "sd_h",
        "h_js",
        "h_brier",
        "mean_s",
        "trust_reliable",
        "trust_unreliable",
        "trust_brier",
        "joint_kl",
        "marginal_product_kl",
    ]:
        row[metric] = values[metric].mean()
        row[metric + "_sd_across_runs"] = values[metric].std(ddof=1)
    for count in ["all_correct", "no_testers", "some_wrong"]:
        row[count + "_runs"] = int(values[count].sum())
    crossings = []
    for seed in values.seed:
        run = times[(times.scenario == name) & (times.seed == seed)]
        hit = run[run.mean_h > 0.95]
        crossings.append(int(hit["round"].min()) if len(hit) else np.nan)
    row["ever_mean_h_above_95_runs"] = int(np.isfinite(crossings).sum())
    row["median_first_crossing_among_hits"] = (
        float(np.nanmedian(crossings)) if np.isfinite(crossings).any() else np.nan
    )
    rows.append(row)
pd.DataFrame(rows).to_csv(DATA / "summary.csv", index=False)
print(json.dumps(checks, indent=2))
