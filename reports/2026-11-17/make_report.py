"""Render the saved study as figures, a PDF atlas and a standalone HTML report."""

import base64
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import font_manager
import numpy as np
import pandas as pd

from run_study import js_pairs, marginal_probabilities, metrics

HERE = Path(__file__).resolve().parent
DATA, FIGURES = HERE / "data", HERE / "figures"
FIGURES.mkdir(exist_ok=True)
BG, INK, GRID = "#faf8f3", "#233746", "#dedfd9"
TEAL, PURPLE, CORAL, GOLD = "#087f8c", "#7653a2", "#c15d44", "#a67d21"
COLORS = [TEAL, PURPLE, CORAL, GOLD]
for name in ["calibri.ttf", "georgia.ttf"]:
    path = Path("C:/Windows/Fonts") / name
    if path.exists():
        font_manager.fontManager.addfont(str(path))
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Calibri", "DejaVu Sans"],
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.edgecolor": GRID,
        "grid.color": GRID,
        "grid.linewidth": 0.65,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "axes.axisbelow": True,
        "lines.linewidth": 2.2,
        "legend.frameon": False,
        "legend.fontsize": 11,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)
social = pd.read_csv(DATA / "social-timeseries.csv")
endpoints = pd.read_csv(DATA / "social-endpoints.csv")
factions = pd.read_csv(DATA / "faction-metrics.csv")
networked = pd.read_csv(DATA / "networked-faction-timeseries.csv")
sensitivity = pd.read_csv(DATA / "prior-sensitivity.csv")
audit = json.loads((DATA / "faction-audit.json").read_text())
checks = json.loads((DATA / "additional-checks.json").read_text())
execution = json.loads((DATA / "execution.json").read_text())
atlas = PdfPages(HERE / "figure-atlas.pdf")
saved = []


def canvas(rows, columns, title, subtitle, height=8.2):
    fig, axes = plt.subplots(rows, columns, figsize=(13.4, height), squeeze=False)
    fig.subplots_adjust(
        left=0.07, right=0.97, top=0.75, bottom=0.16, hspace=0.55, wspace=0.30
    )
    fig.text(0.055, 0.963, title, fontfamily="Georgia", fontsize=25, va="top")
    fig.text(0.055, 0.903, subtitle, fontsize=12.5, va="top", color="#5b6a72")
    for ax in axes.flat:
        ax.grid(axis="y")
        ax.tick_params(pad=7)
        ax.title.set_y(1.02)
    return fig, axes


def finish(fig, filename, footer):
    fig.text(0.055, 0.035, footer, color="#647279", fontsize=10.5)
    fig.savefig(FIGURES / f"{filename}.png", dpi=185, facecolor=BG)
    fig.savefig(FIGURES / f"{filename}.svg", facecolor=BG)
    svg = FIGURES / f"{filename}.svg"
    svg.write_text(
        "\n".join(
            line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()
        )
        + "\n",
        encoding="utf-8",
    )
    atlas.savefig(fig, facecolor=BG)
    saved.append(filename)
    plt.close(fig)


def band(ax, table, metric, color, label, style="-", full_range=False):
    group = table.groupby("round")[metric]
    mean = group.mean()
    low = group.min() if full_range else group.quantile(0.1)
    high = group.max() if full_range else group.quantile(0.9)
    ax.fill_between(mean.index, low, high, color=color, alpha=0.10, linewidth=0)
    ax.plot(mean.index, mean, color=color, linestyle=style, label=label)
    ax.set_xlabel("Round")
    return mean


def networked_figure():
    fig, axes = canvas(
        2,
        3,
        "Local evidence first divides, then settles",
        "Five beliefs per agent · 60 agents · neighbour sharing · four seeds per topology",
    )
    items = [
        ("mean_H2", "Belief in the observed hypothesis", "Mean P(H2 = true)"),
        ("sd_H2", "Disagreement about H2", "SD across agents"),
        ("joint_js", "Whole-network disagreement", "Joint JS (nats)"),
        (
            "joint_kl",
            "KL sees the mismatched tails",
            "Joint KL, ordered-pair mean (nats)",
        ),
        (
            "marginal_product_js",
            "Disagreement across individual beliefs",
            "Marginal-product JS (nats)",
        ),
        (
            "mean_received_trials",
            "How much evidence reaches an agent?",
            "Mean cumulative received trials",
        ),
    ]
    for ax, (metric, title, ylabel) in zip(axes.flat, items):
        for topology, color, label in [
            ("cycle", TEAL, "Cycle · two neighbours"),
            ("complete", PURPLE, "Complete · 59 neighbours"),
        ]:
            band(
                ax,
                networked[networked.topology == topology],
                metric,
                color,
                label,
                full_range=True,
            )
        ax.set(title=title, ylabel=ylabel, xlim=(0, 40))
        ax.set_xticks([0, 10, 20, 30, 40])
    axes[0, 0].set_ylim(0.45, 1.04)
    axes[0, 1].set_ylim(bottom=0)
    axes[0, 2].axhline(0.2539185669426351, color="#879399", ls=":", lw=1.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.055, 0.855), ncol=2)
    finish(
        fig,
        "01-networked-factions",
        "Lines: run means; bands: full range of four runs. Same initial priors. Evidence volume differs between topologies.\nH2 becomes settled; substantial disagreement about the remaining beliefs survives.",
    )


def geometry_figure():
    fig, axes = canvas(
        1,
        3,
        "One signal, three geometries of belief",
        "Shared-evidence cross-checks · every agent receives the same 20 positive signals",
        height=6.8,
    )
    for case, ax, pair, title in zip(
        [1, 2, 3],
        axes.flat,
        [("H1", "H2"), ("H1", "H3"), ("H4", "H5")],
        [
            "Two beliefs · agreement",
            "Three beliefs · two directions",
            "Five beliefs · four directions",
        ],
    ):
        archive = np.load(DATA / f"faction-{case}.npz")
        names = list(archive["names"])
        history = np.array(
            [marginal_probabilities(joint) for joint in archive["joints"]]
        )
        x, y = names.index(pair[0]), names.index(pair[1])
        for agent in range(history.shape[1]):
            points = history[:, agent, [x, y]]
            code = int(points[-1, 0] > 0.5) * 2 + int(points[-1, 1] > 0.5)
            color = TEAL if case == 1 else COLORS[code]
            ax.plot(points[:, 0], points[:, 1], color=color, alpha=0.7, lw=1.1)
            ax.scatter(
                *points[0], s=21, edgecolor=color, facecolor=BG, lw=0.8, zorder=3
            )
            ax.scatter(*points[-1], s=27, color=color, edgecolor=BG, lw=0.5, zorder=4)
        ax.set(
            xlabel=f"P({pair[0]} = true)",
            ylabel=f"P({pair[1]} = true)",
            title=title,
            xlim=(-0.03, 1.03),
            ylim=(-0.03, 1.03),
            aspect="equal",
        )
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.grid(alpha=0.7)
    finish(
        fig,
        "02-shared-evidence-geometry",
        "Hollow points: initial beliefs. Filled points: final beliefs. Lines: exact trajectories; colours indicate final quadrants.\nThese are common-evidence controls, separate from the social-network experiments.",
    )


def networked_geometry_figure():
    fig, axes = canvas(
        1,
        2,
        "Local evidence changes the route to factions",
        "Five-node model · first pre-specified seed (1729) · every path follows one agent",
        height=7.6,
    )
    for ax, topology, title in zip(
        axes.flat,
        ["cycle", "complete"],
        [
            "Cycle · different local data histories",
            "Complete · the same batch reaches everyone",
        ],
    ):
        archive = np.load(DATA / f"networked-faction-{topology}-1729.npz")
        names = list(archive["names"])
        history = np.array(
            [marginal_probabilities(joint) for joint in archive["joints"]]
        )
        x, y = names.index("H4"), names.index("H5")
        for agent in range(60):
            points = history[:, agent, [x, y]]
            code = int(points[-1, 0] > 0.5) * 2 + int(points[-1, 1] > 0.5)
            color = COLORS[code]
            ax.plot(points[:, 0], points[:, 1], color=color, alpha=0.5, lw=1)
            ax.scatter(
                *points[0], s=23, edgecolor=color, facecolor=BG, lw=0.8, zorder=3
            )
            ax.scatter(*points[-1], s=30, color=color, edgecolor=BG, lw=0.5, zorder=4)
        ax.set(
            title=title,
            xlabel="P(H4 = true)",
            ylabel="P(H5 = true)",
            xlim=(-0.03, 1.03),
            ylim=(-0.03, 1.03),
            aspect="equal",
        )
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.grid(alpha=0.7)
    finish(
        fig,
        "09-networked-faction-geometry",
        "Hollow: initial belief. Filled: round 40. Both networks approach the same conditional-on-H2 endpoint in this batch.\nThe paths differ because evidence is local on the cycle; no averaging of neighbours' beliefs is used.",
    )


def audit_figure():
    fig, axes = canvas(
        2,
        2,
        "Why did five-node JS increase?",
        "A reproducible 0.366% rise · outcome-dependent and sensitive to the initial population",
        height=8.9,
    )
    ax = axes[0, 0]
    case = factions[factions.case == 3]
    ax.plot(case["round"], case.joint_js - case.joint_js.iloc[0], color=PURPLE)
    ax.axhline(0, color="#89959a", lw=1)
    ax.set(
        title="The fixed positive-evidence trajectory",
        xlabel="Shared positive signals",
        ylabel="Change in joint JS (nats)",
    )
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3))
    ax = axes[0, 1]
    bins = np.linspace(-0.016, 0.008, 25)
    for kind, color, label in [
        ("uniform", TEAL, "Independent uniform priors"),
        ("scrambled_halton", PURPLE, "Scrambled Halton priors"),
    ]:
        values = sensitivity[sensitivity.kind == kind].delta_js
        ax.hist(
            values,
            bins=bins,
            color=color,
            alpha=0.42,
            label=f"{label}: {(values > 0).sum()}/100 increase",
        )
    ax.axvline(0, color="#89959a", lw=1)
    ax.axvline(
        audit["final_js"] - audit["initial_js"],
        color=CORAL,
        ls="--",
        lw=1.7,
        label="Original Halton instance",
    )
    ax.set(
        title="Changing only the initial priors",
        xlabel="Final − initial joint JS (nats)",
        ylabel="Populations",
    )
    ax.legend(fontsize=9.3, loc="upper left")
    ax = axes[1, 0]
    outcomes = pd.read_csv(DATA / "outcomes-20_signals.csv")
    ax.plot(outcomes.outcome, outcomes.delta_js, color="#adb4b5", lw=1.3)
    ax.scatter(
        outcomes.outcome,
        outcomes.delta_js,
        s=22 + 1300 * outcomes.population_mean_predictive_probability,
        c=np.where(outcomes.delta_js > 0, CORAL, TEAL),
        edgecolor=BG,
        linewidth=0.7,
    )
    ax.axhline(0, color="#89959a", lw=1)
    ax.set(
        title="All possible counts, not just 20 positives",
        xlabel="Positive signals out of 20",
        ylabel="Change in joint JS (nats)",
    )
    ax.text(
        0.03,
        0.95,
        "Dot area reflects predictive probability",
        transform=ax.transAxes,
        va="top",
        fontsize=10,
    )
    ax = axes[1, 1]
    archive = np.load(DATA / "faction-3.npz")["joints"]
    before, after = js_pairs(archive[0]), js_pairs(archive[-1])
    ax.scatter(
        before,
        after,
        s=9,
        c=np.where(after > before, CORAL, TEAL),
        alpha=0.46,
        linewidth=0,
        rasterized=True,
    )
    ax.plot([0, 0.69], [0, 0.69], color="#89959a", lw=1, ls="--")
    ax.set(
        title="Individual pairs move in both directions",
        xlabel="Initial pairwise JS",
        ylabel="Final pairwise JS",
        xlim=(0, 0.69),
        ylim=(0, 0.69),
    )
    ax.text(
        0.03,
        0.94,
        "53.8% of agent pairs move apart",
        transform=ax.transAxes,
        va="top",
        fontsize=11,
    )
    finish(
        fig,
        "03-divergence-audit",
        "Five-node control: 60 agents; 1,770 unordered pairs. No random data in the fixed positive-signal path.\nThe outcome average decreases here; individual outcomes and some pairs' expected equal-weight JS can increase.",
    )


def faction_metrics_figure():
    fig, axes = canvas(
        2,
        3,
        "Agreement, dispersion and dependence are distinct",
        "Shared-evidence controls · compare changes within each model's state space",
    )
    items = [
        ("joint_js", "Joint JS", "Nats"),
        ("joint_kl", "Joint KL", "Ordered-pair mean (nats)"),
        ("mean_sd", "Spread of marginal beliefs", "Mean SD across hypotheses"),
        ("marginal_product_js", "Marginal-product JS", "Nats"),
        ("marginal_product_kl", "Marginal-product KL", "Ordered-pair mean (nats)"),
        (
            "mean_total_correlation",
            "Dependence inside each agent",
            "Mean total correlation (nats)",
        ),
    ]
    for ax, (metric, title, ylabel) in zip(axes.flat, items):
        for case, color, label in [
            (1, TEAL, "Two beliefs"),
            (2, PURPLE, "Three beliefs"),
            (3, CORAL, "Five beliefs"),
        ]:
            data = factions[factions.case == case]
            ax.plot(data["round"], data[metric], color=color, label=label)
        ax.set(
            title=title, xlabel="Shared positive signals", ylabel=ylabel, xlim=(0, 20)
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper left", bbox_to_anchor=(0.055, 0.855))
    finish(
        fig,
        "04-shared-evidence-measures",
        "Total correlation is KL(joint || product of marginals) inside an agent, not correlation across agents.\nKL is unbounded; JS is bounded by ln(2). Their numerical magnitudes are not interchangeable.",
    )


def learning_figure():
    fig, axes = canvas(
        2,
        2,
        "An early disagreement peak, then learning",
        "12 agents · 12 fixed seeds · 10 trials per active agent · true success probability 0.55",
        height=8.9,
    )
    variants = [
        ("baseline-cycle", TEAL, "One belief · cycle", "-"),
        ("meta-cycle", PURPLE, "H + S · cycle", "-"),
        ("baseline-complete", TEAL, "One belief · complete", "--"),
        ("meta-complete", PURPLE, "H + S · complete", "--"),
    ]
    for ax, metric, title, ylabel in zip(
        axes.flat,
        ["mean_h", "h_js", "sd_h", "mean_s"],
        [
            "Learning the correct hypothesis",
            "Disagreement first grows on cycles",
            "Dispersion within each population",
            "Learning evidence quality",
        ],
        [
            "Mean P(H = true)",
            "Mean pairwise JS about H (nats)",
            "SD of P(H = true)",
            "Mean P(S = high quality)",
        ],
    ):
        for scenario, color, label, style in variants:
            if metric == "mean_s" and scenario.startswith("baseline"):
                continue
            band(ax, social[social.scenario == scenario], metric, color, label, style)
        ax.set(title=title, ylabel=ylabel, xlim=(0, 80))
    axes[0, 0].set_ylim(0, 1.04)
    axes[1, 1].set_ylim(0, 1.04)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="upper left", bbox_to_anchor=(0.055, 0.865))
    finish(
        fig,
        "05-learning-and-disagreement",
        "Lines: means across 12 runs. Bands: 10th–90th percentiles across runs, not confidence intervals.\nWithin-population SD is plotted separately. The same seeds give matched initial H priors, not identical evidence histories.",
    )


def propaganda_figure():
    fig, axes = canvas(
        2,
        2,
        "Different targets for distrust, different trajectories",
        "Cycle networks · weak negative signals target H or S with probability 0.1 per agent per round",
        height=8.9,
    )
    variants = [
        ("meta-cycle", PURPLE, "No propaganda"),
        ("meta-decision-control-cycle", GOLD, "Hypothesis-threshold decision"),
        ("meta-propaganda-cycle", TEAL, "Signal against S"),
        ("hypothesis-propaganda-cycle", CORAL, "Signal against H"),
    ]
    for ax, metric, title, ylabel in zip(
        axes.flat,
        ["mean_h", "mean_s", "h_brier", "decision_disagreement"],
        [
            "Belief in the true hypothesis",
            "Belief in high-quality evidence",
            "Accuracy, not just agreement",
            "Dependencies can change the action",
        ],
        [
            "Mean P(H = true)",
            "Mean P(S = high quality)",
            "Mean Brier error about H · lower is better",
            "Fraction: payoff rule differs from H > 0.5",
        ],
    ):
        for scenario, color, label in variants:
            band(ax, social[social.scenario == scenario], metric, color, label)
        ax.set(title=title, ylabel=ylabel, xlim=(0, 80))
    axes[0, 0].set_ylim(0, 1.04)
    axes[0, 1].set_ylim(0, 1.04)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="upper left", bbox_to_anchor=(0.055, 0.865))
    finish(
        fig,
        "06-propaganda-and-decisions",
        "12 fixed seeds; bands are 10th–90th percentile run spread. Negative signal likelihood: [1, 0.5].\nThese exploratory comparisons include adaptive sampling and different realised histories; they do not isolate a universal causal effect.",
    )


def reliability_figure():
    fig, axes = canvas(
        2,
        2,
        "Bayesian trust can learn—or remain unidentified",
        "Complete networks · 12 agents · mixed cases contain three persistently unreliable reporters",
        height=8.9,
    )
    variants = [
        ("reliability-complete-all-reliable", TEAL, "All reliable; noise alternative"),
        (
            "reliability-complete-mixed-uninformative",
            PURPLE,
            "Three uninformative sources",
        ),
        (
            "reliability-complete-mixed-opposite",
            CORAL,
            "Three opposite sources; symmetric prior",
        ),
    ]
    for ax, metric, title, ylabel in [
        (axes[0, 0], "mean_h", "Hypothesis learning", "Mean P(H = true)"),
        (
            axes[1, 0],
            "h_brier",
            "Hypothesis accuracy",
            "Mean Brier error · lower is better",
        ),
    ]:
        for scenario, color, label in variants:
            band(ax, social[social.scenario == scenario], metric, color, label)
        ax.set(title=title, ylabel=ylabel, xlim=(0, 80))
    for ax, suffix, title in [
        (axes[0, 1], "uninformative", "Learning which sources are informative"),
        (axes[1, 1], "opposite", "Opposite reporting has a symmetry"),
    ]:
        data = social[social.scenario == f"reliability-complete-mixed-{suffix}"]
        band(ax, data, "trust_reliable", TEAL, "Actually reliable sources")
        band(ax, data, "trust_unreliable", CORAL, "Actually unreliable sources")
        ax.axhline(0.5, color="#879399", lw=1, ls=":")
        ax.set(
            title=title,
            ylabel="Mean P(source is reliable)",
            ylim=(0, 1.04),
            xlim=(0, 80),
        )
        ax.legend(
            fontsize=10,
            loc="lower left" if suffix == "uninformative" else "upper right",
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=2,
        loc="upper left",
        bbox_to_anchor=(0.055, 0.865),
        fontsize=10.5,
    )
    finish(
        fig,
        "07-bayesian-reliability",
        "12 seeds; bands: 10th–90th percentile run spread. The report model changes between noise and opposite conditions.\nWith a 50:50 reliable/opposite prior, testimony alone is exactly uninformative about H; private evidence still informs H.",
    )


def trust_structure_figure():
    fig, axes = canvas(
        1,
        2,
        "What the symmetric trust model cannot learn",
        "First pre-specified seed (1729) · opposite-reporting model · no selection for visual appeal",
        height=7.3,
    )
    fig.subplots_adjust(right=0.90, bottom=0.19, top=0.75, wspace=0.40)
    run = np.load(DATA / "reliability-complete-mixed-opposite-1729.npz")
    h, actual = run["h"], run["actual_reliability"]
    order = np.argsort(h[-1])
    cmap = LinearSegmentedColormap.from_list(
        "belief", ["#eee9df", "#a5c8c4", "#126c78"]
    )
    ax = axes[0, 0]
    plot = ax.imshow(
        h[:, order].T,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=0,
        vmax=1,
        extent=(-0.5, 80.5, -0.5, 11.5),
    )
    ax.set(
        title="Hypothesis belief through time",
        xlabel="Round",
        ylabel="Agent, sorted by final belief",
    )
    ax.set_yticks(range(12), order)
    ax.grid(False)
    fig.colorbar(plot, ax=ax, fraction=0.04, pad=0.025, label="P(H = true)")
    ax = axes[0, 1]
    trust = np.full((12, 12), np.nan)
    trust[run["destinations"], run["sources"]] = run["trust"][-1]
    columns = np.r_[np.flatnonzero(actual), np.flatnonzero(~actual)]
    plot = ax.imshow(
        trust[np.ix_(order, columns)],
        aspect="equal",
        origin="lower",
        cmap=cmap,
        vmin=0,
        vmax=1,
    )
    ax.set(
        title="Who trusts whom at the end?",
        xlabel="Source agent († = actually unreliable)",
        ylabel="Recipient agent",
    )
    ax.set_xticks(
        range(12), [f"{i}{'†' if not actual[i] else ''}" for i in columns], fontsize=10
    )
    ax.set_yticks(range(12), order)
    ax.axvline(8.5, color=CORAL, lw=1.3)
    ax.grid(False)
    fig.colorbar(plot, ax=ax, fraction=0.04, pad=0.025, label="P(source is reliable)")
    finish(
        fig,
        "08-trust-structure",
        "Across all 12 runs, all 77 agents initially at H ≤ 0.5 remain at their starting H belief in this symmetric model.\nSaved histories verify the private-evidence-only posterior to 4.4 × 10⁻¹⁵ for the 108 actually reliable agents.",
    )


def picture(name, caption):
    encoded = base64.b64encode((FIGURES / f"{name}.png").read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{encoded}" alt="{caption}"><figcaption>{caption} <a href="figures/{name}.svg">Vector SVG</a></figcaption></figure>'


def report():
    summary = pd.read_csv(DATA / "summary.csv")
    labels = {
        "baseline-cycle": "One belief · cycle",
        "baseline-complete": "One belief · complete",
        "meta-cycle": "H + S · cycle",
        "meta-complete": "H + S · complete",
        "meta-decision-control-cycle": "H + S · threshold decision",
        "meta-propaganda-cycle": "Negative signal against S",
        "hypothesis-propaganda-cycle": "Negative signal against H",
        "reliability-complete-all-reliable": "Trust · all reliable",
        "reliability-complete-mixed-uninformative": "Trust · three uninformative",
        "reliability-complete-mixed-opposite": "Trust · three opposite",
    }
    table = "<table><thead><tr><th>Scenario</th><th>Mean H</th><th>SD within runs</th><th>JS about H</th><th>Brier error</th><th>All agents H &gt; .95</th></tr></thead><tbody>"
    for _, row in summary.iterrows():
        table += f"<tr><td>{labels[row.scenario]}</td><td>{row.mean_h:.3f}</td><td>{row.sd_h:.3f}</td><td>{row.h_js:.4f}</td><td>{row.h_brier:.4f}</td><td>{int(row.all_correct_runs)}/12 runs</td></tr>"
    table += "</tbody></table>"
    counter = checks["equal_weight_js_counterexample"]
    directed = pd.read_csv(DATA / "counterexample-directed-kl.csv")
    kl_table = "<table><tr><th>Direction</th><th>Prior KL</th><th>Expected posterior KL</th><th>Expectation under</th></tr>"
    for _, row in directed.iterrows():
        kl_table += f"<tr><td>{int(row.first_agent)} → {int(row.second_agent)}</td><td>{row.prior_kl:.6f}</td><td>{row.expected_kl_under_first_agent:.6f}</td><td>Agent {int(row.first_agent)}</td></tr>"
    kl_table += "</table>"
    content = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Beliefs within networks — simulation study</title><style>
:root{{--paper:{BG};--ink:{INK};--muted:#607078;--teal:{TEAL};--rule:#d8ddd8}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:18px/1.65 Calibri,'Segoe UI',sans-serif}}main{{max-width:1220px;margin:auto;padding:70px 40px 90px}}header{{max-width:960px;border-top:5px solid var(--teal);padding-top:25px}}.eyebrow{{font-size:13px;letter-spacing:.16em;text-transform:uppercase;color:var(--teal);font-weight:700}}h1,h2,h3{{font-family:Georgia,serif;font-weight:400;line-height:1.2}}h1{{font-size:60px;letter-spacing:-.035em;margin:16px 0 24px}}h2{{font-size:36px;margin:65px 0 20px}}h3{{font-size:25px;margin:32px 0 12px}}p{{max-width:920px;margin:16px 0}}.lede{{font-size:23px;color:#41555f}}a{{color:var(--teal);text-underline-offset:3px}}nav{{display:flex;flex-wrap:wrap;gap:8px 24px;border-block:1px solid var(--rule);padding:18px 0;margin:35px 0;font-size:15px}}.facts{{display:grid;grid-template-columns:repeat(3,1fr);gap:26px;margin:35px 0}}.fact{{border-left:2px solid var(--teal);padding-left:18px}}.number{{display:block;font:34px Georgia,serif}}.small{{font-size:14px;color:var(--muted)}}figure{{margin:30px -15px 42px}}figure img{{width:100%;height:auto;display:block}}figcaption{{font-size:15px;line-height:1.5;color:var(--muted);padding:8px 15px;max-width:1060px}}figcaption a{{margin-left:12px}}.note{{border-left:3px solid var(--teal);padding:12px 22px;margin:26px 0;background:#f0f2ec;max-width:950px}}.equation{{font:18px/1.8 'Cambria Math',Georgia,serif;background:#f0f2ec;padding:20px;overflow:auto;margin:20px 0}}table{{border-collapse:collapse;width:100%;font-size:15px;margin:25px 0}}th{{text-align:left;font-weight:600;border-bottom:2px solid var(--rule);padding:10px 12px}}td{{padding:9px 12px;border-bottom:1px solid var(--rule);font-variant-numeric:tabular-nums}}td:not(:first-child),th:not(:first-child){{text-align:right}}.table-wrap{{overflow-x:auto}}code{{font-size:.87em;background:#eeeee8;padding:2px 5px}}details{{border-block:1px solid var(--rule);padding:18px 0;margin-top:24px}}summary{{cursor:pointer;font-weight:600}}footer{{margin-top:60px;border-top:1px solid var(--rule);padding-top:20px;color:var(--muted);font-size:14px}}@media(max-width:700px){{main{{padding:35px 20px}}h1{{font-size:42px}}h2{{font-size:29px}}.facts{{grid-template-columns:1fr}}figure{{margin-inline:-10px}}body{{font-size:17px}}}}
</style></head><body><main>
<header><div class="eyebrow">PolyGraphs · research notebook · 17 November 2026</div><h1>Beliefs within networks</h1><p class="lede">Local evidence, persistent factions and the limits of Bayesian trust.</p><p>A small, reproducible study of agents with internal Bayesian networks. The main experiments share fresh observations with social neighbours. Common-evidence experiments are retained as separate mathematical cross-checks.</p></header>
<nav><a href="#networked">Networked factions</a><a href="#audit">The divergence puzzle</a><a href="#kl">KL versus JS</a><a href="#learning">Learning & propaganda</a><a href="#trust">Bayesian trust</a><a href="#methods">Methods & downloads</a></nav>
<div class="facts"><div class="fact"><span class="number">128</span><span class="small">short social-network runs</span></div><div class="fact"><span class="number">200</span><span class="small">five-node prior-sensitivity checks</span></div><div class="fact"><span class="number">under 2 min</span><span class="small">combined simulation and audit compute</span></div></div>
<section id="networked"><h2>Local evidence creates a transient disagreement peak</h2><p>In the five-node networked model, each active agent conducts 10 Bernoulli trials, then shares the complete fresh counts with its neighbours. The cycle gives each agent two neighbours; the complete graph gives it 59. Both start with the same 60-agent Halton population used in the shared-evidence control.</p><p>On the cycle, mean joint JS rises from <strong>0.252992 to 0.323707 in one round</strong>, then falls towards 0.253919. Complete-network agents receive the same large batch and approach the endpoint in one round. The comparison therefore includes a large difference in evidence volume, shown explicitly below.</p>
{picture('01-networked-factions', 'Networked five-node model. Four seeds per topology; bands show the full range, not a confidence interval.')}
{picture('09-networked-faction-geometry', 'The same five-node belief structure inside two social networks. Shared initial priors; distinct paths to the same limiting faction geometry in these runs.')}
<div class="note"><strong>Agreeing about H2 is not agreeing about everything.</strong> All eight runs settle H2, but differences about the upstream hypotheses remain. The eventual whole-network divergence is close to the shared-evidence conditioning limit, not zero. These are finite-horizon observations, not a proof that any general social-network model must converge.</div></section>
<section id="audit"><h2>The slight increase is real—and interpretable</h2><p>The original shared-evidence five-node instance has joint JS <strong>0.252992 → 0.253919</strong>, a 0.366% increase. Its ordered-pair mean joint KL also increases: <strong>1.396911 → 1.402020</strong>. This is not an artefact of using JS, rounding error or different observations reaching different agents: every agent in this control receives the same evidence.</p><p>The prescribed all-positive sequence approaches conditioning on H2 = true. For these priors, that outcome raises joint JS slightly; H2 = false instead lowers it to 0.245135. The original control contains no random data draw. Its increase is a property of the selected population and selected evidence outcome.</p>
{picture('03-divergence-audit', 'Sensitivity to initial priors and evidence outcomes. The fixed positive sequence is not an average over possible observations.')}
<p>Across 100 independently sampled uniform-prior populations, 22 increased; across 100 scrambled-Halton populations, 3 increased. Both sets had a negative average change. Changing the unscrambled Halton population size also changes the sign: 15 and 120 agents decrease, while 30 and 60 increase. These counts describe this sensitivity check, not a universal probability of divergence.</p><p>The posterior was checked against R bnlearn/gRain, against a direct normalized likelihood calculation, and the JS calculation against SciPy. Both sides of the exact KL and weighted-JS chain identities agree to below 6 × 10⁻¹⁵. Shared downstream CPTs also make the full-joint JS equal to the JS of the root distributions here; deleting an irrelevant array axis is not the explanation.</p>
{picture('02-shared-evidence-geometry', 'Common-evidence controls reproduce the distinct geometries of the three belief structures. They are not neighbour-sharing simulations.')}
{picture('04-shared-evidence-measures', 'Full-joint disagreement, marginal disagreement, dispersion and internal dependence are different diagnostics.')}</section>
<section id="kl"><h2>What the expectation result actually guarantees</h2><p>Directed KL obeys the chain-rule identity below. The expectation uses the first agent’s predictions for D. A realised observation can increase divergence; the expectation cannot. The inequality is strict only if the two predictive distributions of D differ.</p>
<div class="equation">E<sub>D ∼ P</sub>[ KL(P(X | D) || Q(X | D)) ]<br>= KL(P(X,D) || Q(X,D)) − KL(P(D) || Q(D)).</div>
<p>For fresh data generated by a common likelihood, the initial joint KL over (X,D) equals the initial KL over X. For an existing observed node, X denotes the remaining variables. Neither statement assumes that the realised posterior divergence must fall.</p>
<p>JS has a related identity, but its mixture weights update. Let m(d) = [p(d)+q(d)]/2 and w(d) = p(d)/[p(d)+q(d)]. Then:</p>
<div class="equation">JS<sub>½</sub>(P(X,D), Q(X,D))<br>= JS<sub>½</sub>(P(D), Q(D)) + Σ<sub>d</sub> m(d) JS<sub>w(d)</sub>(P(X | d), Q(X | d)).</div>
<p>Ordinary JS sets both weights back to ½ after every observation. That is a different quantity. Consequently, the KL expectation result does not immediately establish contraction of equal-weight posterior JS under the shared-CPT model.</p>
<details open><summary>A concrete equal-weight JS counterexample inside the five-node population</summary><p>Agents 34 and 47 (zero-based) have root priors (H1,H4,H5) = (0.765625, 0.901235, 0.088) and (0.046875, 0.197531, 0.768), with the same downstream CPTs. They assign H2 = true probabilities 0.387590 and 0.594032.</p><div class="table-wrap"><table><tr><th>Quantity</th><th>Equal-weight JS</th></tr><tr><td>Before observing H2</td><td>0.526462</td></tr><tr><td>After H2 = false</td><td>0.538771</td></tr><tr><td>After H2 = true</td><td>0.519710</td></tr><tr><td>Expected posterior JS under agent 34</td><td>0.531383</td></tr><tr><td>Expected posterior JS under agent 47</td><td>0.527448</td></tr></table></div><p>Both agents expect an increase in ordinary JS. The directed-KL and properly weighted-JS identities still hold. Across this population, 383 of 1,770 pairs have an increased expected equal-weight JS under their pair-mixture predictive distribution, although the population average decreases.</p></details>
<h3>The same pair obeys the directed-KL theorem</h3><div class="table-wrap">{kl_table}</div><p>The directions use different predictive measures. A population’s symmetrized KL summary should not silently be substituted into an identity for a single directed KL under its first agent’s predictions.</p>
<p><strong>Implication for the paper:</strong> retain the distinction between realised outcomes and expectations; state the directed-KL expectation theorem precisely; and replace the immediate extension to ordinary JS with the weighted identity or a suitably restricted claim. This affects the information-theoretic guarantee. It does not remove the demonstrated possibility of Bayesian faction formation.</p><p>Here “shared conditional relationships” means shared parent-to-child CPTs, as in the implemented examples. It does not mean agreement on every reverse conditional probability: those generally differ when priors differ. A revised theorem should make this distinction explicit.</p></section>
<section id="learning"><h2>The social-network intuition is visible in the data</h2><p>For the one-belief cycle, mean JS about H rises from 0.0945 to 0.1512 at round 9 before falling nearly to zero. The H + S cycle peaks at 0.1412 at round 8. In each cycle condition, 11 of 12 runs rise above their initial disagreement within the first ten rounds. The complete-network averages instead peak at the start, although some individual runs still increase.</p>
{picture('05-learning-and-disagreement', 'The early rise is observed in these cycle networks, not assumed by the updating rule. The bands describe variation between runs.')}
<p>Signals against S lowered final mean belief in high-quality evidence to 0.265, while final mean H remained 0.999. Signals against H left mean H at 0.726, with substantial within-run dispersion. These contrasts are illustrative; 12 runs with adaptive experimentation do not establish a universal effect of either propaganda mechanism.</p>
{picture('06-propaganda-and-decisions', 'The H + S model separates doubts about a conclusion from doubts about the strength of its evidence. The action-rule discrepancy uses the full joint posterior.')}
<h3>Endpoints across all twelve seeds</h3><p class="small">Means over runs at round 80. Within-run SD is averaged across runs, not pooled across all agents. “All agents H &gt; .95” counts entire runs, not agents. JS and KL use natural logarithms.</p><div class="table-wrap">{table}</div></section>
<section id="trust"><h2>A Bayesian model can be honest about what it cannot identify</h2><p>With uninformative reporting as the alternative, final average trust was <strong>0.839 for actually reliable sources and 0.175 for uninformative sources</strong>. Eleven of twelve mixed-source runs ended with every agent above H = 0.95; one run stopped experimenting. Learning trust was useful, but it did not guarantee success.</p>
{picture('07-bayesian-reliability', 'Different report models produce different identification problems. The symmetric opposite-reporting model is not simply a stronger noise condition.')}
<p>The opposite-reporting case has an exact symmetry. With prior P(R = reliable) = ½, conditional on the reported trial counts, every source’s entire report history has the same marginal likelihood under H and not-H: flipping H and R together preserves the report likelihood. Marginalizing R therefore gives Bayes factor 1 for H. More testimony alone cannot break that symmetry.</p>
<div class="equation">½ L(reports | H, reliable) + ½ L(reports | H, opposite)<br>= ½ L(reports | not-H, reliable) + ½ L(reports | not-H, opposite).</div>
<p>Private evidence can change H, and beliefs about H can change trust. But testimony does not add information about H in this exact symmetric setup. All 77 agents initially at H ≤ 0.5 across the twelve runs never experiment and retain their initial H belief. For the 108 actually reliable agents, the saved report counts also give their private counts: a direct private-only posterior matches every saved H trajectory within 4.4 × 10⁻¹⁵.</p>
{picture('08-trust-structure', 'The first pre-specified seed illustrates the mechanism. The analytic symmetry and trajectory checks, rather than this single image, establish the result.')}
<p>A non-symmetric reliability prior, independently calibrated source, or forced exploratory trials could break aspects of this trap. Those interventions were not simulated in this approved batch.</p></section>
<section id="methods"><h2>Methods, scope and reproducibility</h2><p>The initial batch contained ten scenarios × twelve seeds (1729–1740) × twelve agents × eighty rounds, with ten trials per active agent and ε = 0.05. All used fixed horizons. The additional five-node comparison used sixty agents, two topologies, four seeds and forty rounds, with observation probabilities 0.35/0.65. Both batches used exact Python inference; the shared-evidence posterior audit additionally used R bnlearn 5.2.1 and gRain 1.4.6.</p><p>Simulations use the actual PolyGraphs operations on DGL graphs. Every active source generates fresh counts once, sends them unchanged to its outgoing neighbours, and uses its own data once. Received reports are not re-forwarded. Communication is local, not filtered by agreement or report content. All agents decide from pre-round beliefs and update after the round’s evidence is generated.</p><p>KL summaries average directed KL over all ordered distinct agent pairs, equivalently half the Jeffreys divergence averaged over unordered pairs. Full log posteriors were retained in the networked five-node runs, so underflow of tiny probabilities does not create spurious infinite KL. For JS about H in the initial batch, probability roundoff of at most 7.2 × 10⁻¹⁵ was clipped to [0,1]; no substantive probability mass was smoothed. The raw saved arrays remain available.</p><p>The model-selection and population comparisons are exploratory and small. There are no significance claims, no parameter search for attractive outcomes and no hidden discarded seeds. Different network degrees change the amount of evidence received, so topology and evidence volume are not isolated. Stable polarization, calibration and long-run convergence would require additional designs.</p>
<p><a href="figure-atlas.pdf">Download the nine-figure PDF atlas</a> · <a href="data/summary.csv">Endpoint summary CSV</a> · <a href="data/social-timeseries.csv">Social time-series CSV</a> · <a href="data/networked-faction-timeseries.csv">Networked faction time series</a> · <a href="data/faction-metrics.csv">Shared-control metrics</a></p>
<p><a href="run_study.py">Initial study script</a> · <a href="run_networked_factions.py">Networked comparison script</a> · <a href="analyse_checks.py">Independent checks</a> · <a href="make_report.py">Figure and report source</a>. Re-running simulation scripts requires the configured PolyGraphs environment and the intended compute authorization.</p>
<p>Sources: <a href="https://doi.org/10.1007/s11229-024-04491-5">Freeborn (2024), Rational factionalization</a>; the supplied multibeliefs draft; <a href="https://arxiv.org/abs/1712.04561">O’Connor & Weatherall, Scientific Polarization</a>; <a href="https://github.com/DavidFreeborn/polygraphs-multibelief">PolyGraphs multibelief fork</a>. The source-reliability model is a new Bayesian extension, conditional on its report model and experiment schedule.</p></section>
<footer>Local exploratory study · all computed values and figures derive from the saved arrays in this directory · no external resources are required to view this HTML report.</footer>
</main></body></html>"""
    (HERE / "results.html").write_text(content, encoding="utf-8")


if __name__ == "__main__":
    networked_figure()
    networked_geometry_figure()
    geometry_figure()
    audit_figure()
    faction_metrics_figure()
    learning_figure()
    propaganda_figure()
    reliability_figure()
    trust_structure_figure()
    atlas.close()
    report()
    print(f"Rendered {len(saved)} figures, PDF atlas and standalone results.html")
