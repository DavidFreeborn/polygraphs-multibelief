# Multibelief study

[Results](https://davidfreeborn.github.io/polygraphs-multibelief/) · [Experiment setup](https://davidfreeborn.github.io/polygraphs-multibelief/experiment-settings.html) · [Measures](https://davidfreeborn.github.io/polygraphs-multibelief/measures.html)

The report compares Baseline, Converging, Polarizing, Factionalizing and
Trust-in-Neighbors across cycle, connected ER .1 and connected ER .2 networks,
in pristine, misinformation and disinformation environments. It shows
45 settings with 100 runs each. Every run has 60 agents and 500 rounds.

To read the report offline, download [report-and-settings.zip](report-and-settings.zip),
extract it and open `results.html`. The ZIP also contains the experiment setup
and measure definitions.

G means that B has a higher success probability than A. Each agent starts with
a randomly drawn probability for G, matched across the four belief models.
The settings explain how the underlying joint priors are constructed.

## Reproduce the simulations

From the repository root, with a C++17 compiler, zlib and Python 3:

```sh
cd reports/current
clang++ -O3 -std=c++17 worker.cpp -lz -o worker
./worker data 500 100 all 1729
python3 validate.py
python3 make_report.py
```

The worker generates all 4,500 runs. Source reliability starts at .90 in every
Trust-in-Neighbors environment, including pristine. In Factionalizing, H and S
are the upper nodes, X and Y the lower nodes, and G=1 means X=Y.

`study-design.json` specifies the seeds and recorded fields.
`summary.json` stores the plotted means and quartiles. The worker regenerates
the individual evidence and posterior records; those records are not stored
in Git. To rebuild the HTML from the saved aggregates without running new
simulations, use `python3 make_report.py` in this directory.

## Checks

`validation.json` records independent posterior replay, matched G trajectories,
the analytical Bala–Goyal comparison, source-reliability checks and checks of
sampled actions and statistics. The simulations used the standalone C++ runner;
native PolyGraphs/DGL and R execution was not performed for this study.

With the fork’s Python/DGL dependencies installed, `replay_polygraphs.py`
replays saved evidence through the native operations. Add `--all` to check
every run.

The [trust diagnostics](diagnostics/trust/diagnostic-notes.md) describe the
additional fixed-evidence and forced-experimentation checks.

## Publish the report

Run `python3 publish_pages.py --output /tmp/polygraphs-pages` to prepare the
HTML sheets for the `gh-pages` branch. The published copies request that search
engines omit them from their indexes. The report is still public and can be
shared or linked by anyone with its URL.
