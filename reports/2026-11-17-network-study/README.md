# Multibelief networks: controlled 60-agent study

Open `results.html` for the self-contained report, `experiment-settings.html`
for the complete protocol, or `figure-atlas.pdf` for printable figures.

The study uses 72 conditions, 20 repetitions, 500 rounds, 60 agents and 10 trials
per active agent. All requested belief and divergence measures are reported for
the whole population and separately for ordinary agents. The trust experiments
compare initial source reliability 0.5 with the actual reliable proportion.

From an installed checkout:

```sh
python -m pytest tests
python reports/2026-11-17-network-study/run_experiments.py --rounds 500 --repeats 20
python reports/2026-11-17-network-study/validate_results.py
python reports/2026-11-17-network-study/source_exposure.py
python reports/2026-11-17-network-study/write_settings.py
python reports/2026-11-17-network-study/make_report.py
```

Set `DGLBACKEND=pytorch`, `OMP_NUM_THREADS=1`, and `OPENBLAS_NUM_THREADS=1`.
Run configurations and versions are recorded in `data/settings.json` and
`data/execution.json`. `data/metrics.csv` and `data/nodes.csv` contain run-level
measurements; each condition directory contains raw success/trial histories,
priors, graph edges and posterior snapshots. `data/raw-sha256.json` identifies
the raw records, and `data/validation.json` records independent replay checks.

The HTML embeds its figures and opens offline. Links to supplementary files
require the surrounding directory. No simulations are launched by opening it.
