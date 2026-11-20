# PolyGraphs
<p align="center">
  <img src="https://akoliousis.com/polygraphs/eu_email_core.webp" alt="Visualisation of EU Email Core network" height="250">
</p>

<p align="center"><a href="https://akoliousis.com/polygraphs/">Documentation</a> | <a href="https://polygraphs.sites.northeastern.edu/">Website</a> | <a href="https://pypi.org/project/polygraphs/">PyPi</a></p>

PolyGraphs simulates learning in social networks using PyTorch and DGL on CPUs
and GPUs. This fork gives each agent a Bayesian belief network and adds models
that learn a hypothesis together with the reliability of neighbouring agents.

## Results

Read the [results online](https://davidfreeborn.github.io/polygraphs-multibelief/),
with links to the experiment setup and definitions of the measures.
The [study directory](reports/current) contains the source, plotted data and
validation records. You can also [download the three report sheets](reports/current/report-and-settings.zip)
and read them offline.

The report shows 100 runs per setting across three network topologies and
three informational environments. These runs used the standalone C++ program.
Independent Python calculations checked its posteriors and statistics;
the study was not run through native PolyGraphs/DGL or R.

## Installation

Install this fork from source in a Python 3.10 environment. The PyPI release
does not include the multibelief operations.

```sh
git clone https://github.com/DavidFreeborn/polygraphs-multibelief.git
cd polygraphs-multibelief
python -m pip install -r requirements-multibelief.txt
```

The default inference backend uses the R packages bnlearn and gRain:

```r
install.packages(c("bnlearn", "gRain", "jsonlite"))
```

Put `Rscript` on your path, or set `POLYGRAPHS_RSCRIPT` to its full path.
To use the independent Python backend, set `multibelief.backend: numpy`.
The reference multibelief operations run on CPU.

Native PyTorch alternatives, `TorchMultiBeliefOp` and
`TorchBayesianReliabilityOp`, keep inference and sampling in float64 tensors.
They support all five current study models and retain the existing operators
and C++ runner. See the [tensor operator guide](docs/guide/ops/torch-multibelief.md),
the `torch-*.yaml` examples, and the
[executed validation and timings](reports/2026-11-19-torch-validation).
CPU execution is verified; CUDA comparison tests are provided but GPU execution
has not been verified.

## Running simulations

```sh
polygraphs -f examples/multibelief/meta-cycle.yaml
python -m pytest tests
```

The [multibelief guide](docs/guide/ops/multibelief.md) explains the models,
configuration and evidence semantics. The [examples](examples/multibelief)
cover multibelief learning, propaganda and decision controls, three
factionalization cases, learned source reliability and a custom belief network.
Network examples share fresh observations with neighbours; the separately
named shared-evidence examples give every agent the same observations.

For the rest of PolyGraphs, see the upstream
[getting started guide](https://akoliousis.com/polygraphs/guide/introduction/getting-started),
[installation instructions](https://akoliousis.com/polygraphs/guide/introduction/install-from-source),
[platform guide](https://akoliousis.com/polygraphs/guide/introduction/platform-guide)
and [analysis documentation](https://akoliousis.com/polygraphs/guide/simulations/processing-results).

## Papers About PolyGraphs
Ball, B., Koliousis, A., Mohanan, A. & Peacey, M. [Computational philosophy: reflections on the PolyGraphs project](https://doi.org/10.1057/s41599-024-02619-z). Humanit Soc Sci Commun 11, 186 (2024).

Ball, B., Koliousis, A., Mohanan, A. & Peacey, M. [Misinformation and higher-order evidence](https://doi.org/10.1057/s41599-024-03806-8). Humanit Soc Sci Commun 11, 1294 (2024).

## Contributing
Please file an [issue](https://github.com/DavidFreeborn/polygraphs-multibelief/issues) for this fork. The original project is maintained at [alexandroskoliousis/polygraphs](https://github.com/alexandroskoliousis/polygraphs).

