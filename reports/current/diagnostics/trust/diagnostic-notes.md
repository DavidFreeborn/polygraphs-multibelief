# Trust diagnostic investigation

Scope: 100 paired seeds (1729–1828), cycle topology, disinformation,
60 agents, 500 rounds, initial source reliability .90. These checks concern
the same disinformation setting used in the current report.

1. Replay each original baseline/trust evidence history under both inference
rules. Baseline log odds equal initial log odds plus private log likelihood
ratio and the sum of incoming source log likelihood ratios. Trust log odds
replace each source log ratio L with
log((.9 exp(L)+.1)/(.9+.1 exp(L))).
This contribution lies between -log(9) and log(9). The private contribution
is unchanged and unbounded. Report counts include only fresh source trials.

2. Force every agent to experiment with B on every round, with all other
parameters and random streams unchanged. forced_worker.cpp differs from the
main worker only by this policy intervention and the diagnostic condition filter.

```sh
clang++ -O3 -std=c++17 forced_worker.cpp -lz -o forced_worker
./forced_worker forced-data 500 100 forced 1729
```

The two forced models receive byte-identical evidence. All 200 final
posteriors were independently replayed using the main Python posterior
function. The recorded maximum log-posterior error is in forced-diagnostics.json.
The reported interval is a normal approximation using paired seed differences.

These diagnostics evaluate posterior belief in the true H=1 world, not
calibration across both possible worlds or overall decision-theoretic utility.
They support an interaction between source ambiguity and endogenous
experimentation; they do not provide an additive causal decomposition.

## Same-evidence replay

```json
{
  "baseline_history": {
    "base_on_same_data": 0.8771871412082819,
    "trust_on_same_data": 0.8766190206486625,
    "oracle_on_same_data": 0.9448369809746765,
    "B_trials_per_agent": 373.1455,
    "fraction_never_experimented": 0.0575,
    "fraction_final_H_below_half": 0.1395,
    "source_trust_by_final_H_group": {
      "wrong": {
        "good": 0.7035113268156434,
        "bad": 0.8862153874487669
      },
      "right": {
        "good": 0.9745696538196087,
        "bad": 0.19503158778981142
      }
    },
    "max_replay_error": 1.970645868709653e-14
  },
  "trust_history": {
    "base_on_same_data": 0.807025380015359,
    "trust_on_same_data": 0.7412620449189959,
    "oracle_on_same_data": 0.8652306072290108,
    "B_trials_per_agent": 301.96383333333335,
    "fraction_never_experimented": 0.18,
    "fraction_final_H_below_half": 0.3566666666666667,
    "source_trust_by_final_H_group": {
      "wrong": {
        "good": 0.7609480103862921,
        "bad": 0.8861509389947462
      },
      "right": {
        "good": 0.9630434332769003,
        "bad": 0.2802234525644285
      }
    },
    "max_replay_error": 1.13464793116691e-13
  }
}
```

## Forced experimentation

```json
{
  "runs": 200,
  "seeds": 100,
  "topology": "cycle",
  "environment": "disinformation",
  "intervention": "All 60 agents perform B for 500 rounds, regardless of posterior. Inference rules, priors and random streams unchanged.",
  "mean_final_baseline": 0.9711939881162971,
  "mean_final_trust": 0.9922229787469061,
  "mean_B_trials_per_agent": 500,
  "paired_mean_trust_minus_baseline": 0.02102899063060911,
  "paired_difference_SE": 0.0016936564588852284,
  "normal_approx_95pct_interval": [
    0.017709423971194065,
    0.024348557290024157
  ],
  "independent_max_log_posterior_error": 7.673861546209082e-13,
  "same_evidence_between_models": true
}
```
