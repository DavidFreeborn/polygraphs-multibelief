# PolyGraphs
<p align="center">
  <img src="https://akoliousis.com/polygraphs/eu_email_core.webp" alt="Visualisation of EU Email Core network" height="250">
</p>

<p align="center"><a href="https://akoliousis.com/polygraphs/">Documentation</a> | <a href="https://polygraphs.sites.northeastern.edu/">Website</a> | <a href="https://pypi.org/project/polygraphs/">PyPi</a></p>

PolyGraphs is a scaleable framework for performing simulations on networks built using PyTorch and DGL that can run on CPUs and GPUs.

This fork adds Bayesian belief networks to individual agents. `MultiBeliefOp`
propagates evidence through connected beliefs; `BayesianReliabilityOp` jointly
learns a hypothesis and the reliability of each neighbour. Existing PolyGraphs
operations remain available.

## Multibelief Networks

Install this fork from source in a Python 3.10 environment:

```bash
git clone https://github.com/DavidFreeborn/polygraphs-multibelief.git
cd polygraphs-multibelief
python -m pip install -r requirements-multibelief.txt
```

The default backend uses the **R** packages bnlearn and gRain for exact inference:

```r
install.packages(c("bnlearn", "gRain", "jsonlite"))
```

Put `Rscript` on your path, or set `POLYGRAPHS_RSCRIPT` to its full path.
An independent exact Python backend is available with `multibelief.backend: numpy`.
These new operations currently run on CPU.

```bash
polygraphs -f examples/multibelief/meta-cycle.yaml
python -m pytest tests
```

The [multibelief guide](docs/guide/ops/multibelief.md) explains the models,
configuration, evidence semantics and validation. The [examples](examples/multibelief)
include the two multibelief models, propaganda and decision controls, the three
factionalization cases, learned source reliability, and a custom belief network.

The factionalization examples share fresh observations with social neighbours;
separately named shared-evidence examples provide cross-checks. The
[small simulation study](reports/2026-11-17) includes figures, saved trajectories,
KL/JS diagnostics and an audit of the divergence claims.

## Getting Started
The upstream PolyGraphs release requires an appropriately configured version of PyTorch and DGL; see the [getting started guide](https://akoliousis.com/polygraphs/guide/introduction/getting-started). The PyPI release does not contain this fork's multibelief operations:

```bash
pip install polygraphs
```

You can run simulations using a configuration file with the `polygraphs` command:

``` bash
polygraphs -f test.yaml
```

### Installing from Source
Advanced users can [install from source](https://akoliousis.com/polygraphs/guide/introduction/install-from-source), see the documentation for more details on running PolyGraphs on the [platform guide](https://akoliousis.com/polygraphs/guide/introduction/platform-guide).

## Analysing Simulation Results
Results from simulations can be processed using the [analysis module](https://akoliousis.com/polygraphs/guide/simulations/processing-results). 

## Papers About PolyGraphs
Ball, B., Koliousis, A., Mohanan, A. & Peacey, M. [Computational philosophy: reflections on the PolyGraphs project](https://doi.org/10.1057/s41599-024-02619-z). Humanit Soc Sci Commun 11, 186 (2024).

Ball, B., Koliousis, A., Mohanan, A. & Peacey, M. [Misinformation and higher-order evidence](https://doi.org/10.1057/s41599-024-03806-8). Humanit Soc Sci Commun 11, 1294 (2024).

## Contributing
Please file an [issue](https://github.com/DavidFreeborn/polygraphs-multibelief/issues) for this fork. The original project is maintained at [alexandroskoliousis/polygraphs](https://github.com/alexandroskoliousis/polygraphs).
