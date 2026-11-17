"""Render a self-contained scientific report from saved experiment tables."""

import base64
import html
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from run_experiments import MODELS, NETWORKS, TREATMENTS

HERE = Path(__file__).resolve().parent
DATA, FIG = HERE / "data", HERE / "figures"
FIG.mkdir(exist_ok=True)
LABEL = {
    "baseline": "Two-node baseline",
    "converging": "Three-node converging candidate",
    "polarising": "Three-node polarising candidate",
    "factionalising": "Five-hypothesis factionalisation",
    "cycle": "Cycle",
    "complete": "Complete",
    "er01": "Erdos-Renyi p = 0.1",
    "er05": "Erdos-Renyi p = 0.5",
    "honest": "No unreliable agents",
    "misinformation": "6 misinformants",
    "disinformation": "6 disinformants",
}
COLORS = {"honest": "#21618c", "misinformation": "#a56600", "disinformation": "#a43545"}
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "lines.linewidth": 1.8,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)
M = pd.read_csv(DATA / "metrics.csv")
B = pd.read_csv(DATA / "nodes.csv")
SETTINGS = json.loads((DATA / "execution.json").read_text())
END = M[M["round"] == SETTINGS["rounds"]].copy()
atlas = PdfPages(HERE / "figure-atlas.pdf")
parts = []
figure_number = 0


def band(ax, frame, key, label, color, style="-", xmax=None):
    if frame.empty or frame[key].isna().all():
        return
    summary = frame.groupby("round")[key].agg(
        mean="mean", lo=lambda x: x.quantile(0.25), hi=lambda x: x.quantile(0.75)
    )
    ax.plot(summary.index, summary["mean"], label=label, color=color, linestyle=style)
    ax.fill_between(
        summary.index,
        summary["lo"],
        summary["hi"],
        color=color,
        alpha=0.12,
        linewidth=0,
    )
    ax.set_xlabel("Round")
    ax.set_xlim(0, SETTINGS["rounds"] if xmax is None else xmax)


def save(fig, filename, caption):
    global figure_number
    figure_number += 1
    fig.savefig(
        FIG / f"{filename}.png", dpi=130, bbox_inches="tight", facecolor="white"
    )
    fig.savefig(FIG / f"{filename}.svg", bbox_inches="tight", facecolor="white")
    svg = FIG / f"{filename}.svg"
    svg.write_text(
        "\n".join(
            line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()
        )
        + "\n",
        encoding="utf-8",
    )
    footer = fig.text(
        0.02,
        -0.025,
        textwrap.fill(f"Figure {figure_number}. {caption}", 155),
        ha="left",
        va="top",
        fontsize=8,
    )
    atlas.savefig(fig, bbox_inches="tight")
    footer.remove()
    plt.close(fig)
    encoded = base64.b64encode((FIG / f"{filename}.png").read_bytes()).decode()
    return f'<figure id="fig-{figure_number}"><img loading="lazy" src="data:image/png;base64,{encoded}" alt="{html.escape(caption,quote=True)}"><figcaption><strong>Figure {figure_number}.</strong> {caption}</figcaption></figure>'


def node_figure(model, network, ordinary=False):
    frame = B[(B.model == model) & (B.network == network)]
    names = frame.node.drop_duplicates().tolist()
    cols = min(3, len(names))
    blocks = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(
        blocks * 2,
        cols,
        figsize=(12, blocks * 5.1),
        squeeze=False,
        layout="constrained",
    )
    pop = "Ordinary agents only" if ordinary else "All 60 agents"
    fig.suptitle(
        f"{LABEL[model]} | {LABEL[network]}\n{pop}: beliefs and between-agent dispersion",
        fontsize=14,
    )
    prefix = "ordinary_" if ordinary else ""
    for index, name in enumerate(names):
        row, col = divmod(index, cols)
        for offset, key in [(0, "mean"), (blocks, "sd")]:
            ax = axes[row + offset, col]
            for treatment in TREATMENTS:
                band(
                    ax,
                    frame[(frame.node == name) & (frame.treatment == treatment)],
                    prefix + key,
                    LABEL[treatment],
                    COLORS[treatment],
                )
            ax.set_title(
                (
                    "Next-trial success probability"
                    if name == "D_next"
                    else f"P({name} = 1)"
                )
                + (" | mean" if key == "mean" else " | agent SD")
            )
            ax.set_ylabel("Mean probability" if key == "mean" else "Population SD")
            if key == "mean":
                ax.set_ylim((0.44, 0.56) if name == "D_next" else (-0.02, 1.02))
            else:
                ax.set_ylim(bottom=0)
    for k in range(len(names), blocks * cols):
        r, c = divmod(k, cols)
        axes[r, c].set_visible(False)
        axes[r + blocks, c].set_visible(False)
    axes[0, 0].legend(fontsize=8)
    return save(
        fig,
        f"{model}-{network}-nodes" + ("-ordinary" if ordinary else ""),
        f"{LABEL[model]}, {LABEL[network]}; {pop.lower()}. Each line averages 20 run-level means or population SDs; shading is the 25th–75th percentile across runs. D_next is a prediction for a fresh trial, not a repeatedly conditioned data variable. All treatments have 10 trials per active agent and epsilon = 0.05.",
    )


def measures_figure(model, network, ordinary=False):
    frame = M[(M.model == model) & (M.network == network)]
    prefix = "ordinary_" if ordinary else ""
    keys = [
        (prefix + "joint_kl", "Joint KL: ordered-pair mean", "Nats"),
        (prefix + "product_kl", "Marginal-product KL: ordered-pair mean", "Nats"),
        (prefix + "joint_js", "Joint JS: pair mean", "Nats"),
        (prefix + "product_js", "Marginal-product JS: pair mean", "Nats"),
        (
            prefix + "dependence",
            "Dependence inside an agent",
            "Mean total correlation (nats)",
        ),
        (
            "active_fraction",
            "Agents choosing B next round",
            "Fraction of all 60 agents",
        ),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
    pop = "ordinary agents" if ordinary else "all agents"
    fig.suptitle(
        f"{LABEL[model]} | {LABEL[network]}\nBelief-distribution measures: {pop}",
        fontsize=14,
    )
    for ax, (key, title, unit) in zip(axes.flat, keys):
        for treatment in TREATMENTS:
            band(
                ax,
                frame[frame.treatment == treatment],
                key,
                LABEL[treatment],
                COLORS[treatment],
            )
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.set_ylim(bottom=0)
        if key.endswith("js"):
            ax.set_ylim(0, np.log(2))
    axes[0, 0].legend(fontsize=8)
    return save(
        fig,
        f"{model}-{network}-measures" + ("-ordinary" if ordinary else ""),
        f"{LABEL[model]}, {LABEL[network]}. Divergences compare {pop}' persistent-hypothesis distributions; D is excluded. KL uses ordered distinct pairs; JS uses equally weighted unordered pairs. Marginal products discard within-agent dependence. Lower-right always describes all 60 agents' next-round choices. Lines are run means; bands are interquartile ranges, not confidence intervals. Axes for KL are independently scaled; JS axes share [0, ln 2].",
    )


def trust_figure(network, treatment):
    frame = M[
        (M.model == "trust") & (M.network == network) & (M.treatment == treatment)
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
    fig.suptitle(
        f"Bayesian source reliability | {LABEL[network]}\n{LABEL[treatment]}",
        fontsize=14,
    )
    colors = ["#21618c", "#a43545"]
    for prior, color in zip(sorted(frame.trust_prior.unique()), colors):
        f = frame[frame.trust_prior == prior]
        label = f"Initial trust {prior:g}"
        band(axes[0, 0], f, "mean_good", label, color)
        band(axes[0, 1], f, "trust_reliable", label + ", reliable", color)
        band(axes[0, 1], f, "trust_unreliable", label + ", unreliable", color, "--")
        band(axes[0, 2], f, "initial_a_mean", label + ", initially A", color, "--")
        band(axes[0, 2], f, "initial_b_mean", label + ", initially B", color)
        band(axes[1, 0], f, "active_fraction", label, color)
        band(axes[1, 1], f, "trust_brier", label, color)
        band(axes[1, 2], f, "reporting_edge_fraction", label, color)
    titles = [
        "Mean P(B is good)",
        "Trust by actual source type",
        "Beliefs by initial action",
        "Fraction choosing B next round",
        "Source-type Brier score",
        "Edges with any source report",
    ]
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title)
        ax.set_ylabel(
            "Mean squared error" if "Brier" in title else "Probability / fraction"
        )
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=7, loc="best")
    return save(
        fig,
        f"trust-{network}-{treatment}",
        f"Bayesian trust, {LABEL[network]}, {LABEL[treatment].lower()}. Twenty runs per prior; means and interquartile bands. Trust averages directed recipient–source edges, grouped by the source's actual type. Solid/dashed source curves denote reliable/unreliable sources; cohort curves distinguish agents initially choosing B/A. The final panel counts edges whose source has reported at least once, explaining why some reliability beliefs remain untested. Prior 1 in the honest treatment is a certainty benchmark, not learned reliability.",
    )


def endpoint_table():
    frame = END[END.model != "trust"]
    summary = frame.groupby(["model", "network", "treatment"]).agg(
        mean_good=("mean_good", "mean"),
        ordinary_good=("ordinary_good", "mean"),
        all_correct=("all_correct", "sum"),
        active=("active_fraction", "mean"),
        joint_js=("joint_js", "mean"),
        product_js=("product_js", "mean"),
        joint_kl=("joint_kl", "mean"),
        product_kl=("product_kl", "mean"),
    )
    summary.to_csv(DATA / "endpoint-summary.csv")
    return summary.reset_index().to_html(
        index=False, float_format=lambda x: f"{x:.4f}", border=0
    )


def overview():
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    colors = ["#21618c", "#a43545", "#44846c", "#a56600"]
    for ax, model in zip(axes.flat, MODELS):
        for network, color in zip(NETWORKS, colors):
            frame = M[
                (M.model == model) & (M.network == network) & (M.treatment == "honest")
            ]
            band(ax, frame, "mean_good", LABEL[network], color, xmax=100)
        ax.set_title(LABEL[model])
        ax.set_ylabel("Mean P(B's success probability > 0.5)")
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=8)
    return save(
        fig,
        "overview-early-learning",
        "Early learning without unreliable agents, rounds 0–100. The quantity is posterior probability that B is strictly better than A: H for baseline, (H,S)=(1,1) for the converging candidate, H=S for the polarising candidate, and H2 for factionalisation. It is not necessarily the marginal of the node labelled H. Bands show run-to-run interquartile spread.",
    )


def geometry():
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.5), layout="constrained")
    fig.suptitle(
        "Belief geometry without unreliable sources | prespecified seed 1729",
        fontsize=14,
    )
    for col, network in enumerate(NETWORKS):
        for row, model in enumerate(["polarising", "factionalising"]):
            z = np.load(DATA / f"{model}-{network}-honest" / "1729.npz")
            names = z["names"].tolist()
            x, y = ("H", "S") if model == "polarising" else ("H4", "H5")
            b = z["beliefs"]
            ax = axes[row, col]
            for agent in range(60):
                ax.plot(
                    b[:, agent, names.index(x)],
                    b[:, agent, names.index(y)],
                    color="#21618c",
                    alpha=0.16,
                    lw=0.6,
                )
            ax.scatter(
                b[0, :, names.index(x)],
                b[0, :, names.index(y)],
                s=12,
                facecolors="none",
                edgecolors="#555",
                lw=0.6,
            )
            ax.scatter(
                b[-1, :, names.index(x)],
                b[-1, :, names.index(y)],
                s=16,
                color="#a43545",
            )
            ax.set(
                xlim=(0, 1),
                ylim=(0, 1),
                xlabel=f"P({x}=1)",
                ylabel=f"P({y}=1)",
                title=LABEL[network],
            )
    return save(
        fig,
        "belief-geometry",
        "Within-run belief trajectories: polarising H–S above, factionalising H4–H5 below. Hollow markers are initial beliefs; red markers are round 500; faint lines connect sampled rounds. The seed was fixed in advance, not selected for appearance. These plots show latent disagreement even when agents agree on the value of B; they do not prove categorical faction membership or asymptotic convergence.",
    )


def exposure_figure():
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    for ax, model in zip(axes.flat, MODELS):
        for network, color in zip(
            NETWORKS, ["#21618c", "#a43545", "#44846c", "#a56600"]
        ):
            f = M[
                (M.model == model) & (M.network == network) & (M.treatment == "honest")
            ]
            avg = f.groupby("round")[["received_mean", "mean_good"]].mean()
            ax.plot(avg.received_mean, avg.mean_good, label=LABEL[network], color=color)
        ax.set(
            xlabel="Mean cumulative B trials received per agent",
            ylabel="Mean good-B belief",
            title=LABEL[model],
            xlim=(0, 5000),
            ylim=(0, 1.02),
        )
        ax.legend(fontsize=8)
    return save(
        fig,
        "evidence-exposure",
        "Learning plotted against mean cumulative evidence received, without unreliable sources. The first 5,000 observations per agent are shown where reached. Points connect run-averaged values at common sampled rounds; this is an exposure diagnostic, not a matched-evidence causal comparison. Sparse networks may not reach the right edge, and averaging hides heterogeneous evidence histories.",
    )


def trust_heatmap(network):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    fig.suptitle(
        f"Disinformation and trust | {LABEL[network]} | seed 1729", fontsize=14
    )
    for col, prior in enumerate([0.5, 0.9]):
        z = np.load(DATA / f"trust-{network}-disinformation-{prior}" / "1729.npz")
        order = np.argsort(np.exp(z["initial_log_joint"][:, 1]))
        im = axes[0, col].pcolormesh(
            z["rounds"],
            np.arange(60),
            z["beliefs"][:, order, 0].T,
            vmin=0,
            vmax=1,
            cmap="viridis",
            shading="nearest",
            rasterized=True,
        )
        axes[0, col].set(
            title=f"Initial trust {prior}: P(H=1)",
            xlabel="Round",
            ylabel="Agent, ordered by initial P(H)",
        )
        matrix = np.full((60, 60), np.nan)
        matrix[z["trust_recipients"], z["trust_sources"]] = z["final_trust"]
        im2 = axes[1, col].imshow(
            matrix[order], vmin=0, vmax=1, cmap="viridis", aspect="auto", origin="lower"
        )
        axes[1, col].set(
            title="Final trust in each source",
            xlabel="Source ID",
            ylabel="Recipient, same ordering",
        )
        axes[1, col].set_xticks(
            z["unreliable_ids"],
            labels=[str(x) + "*" for x in z["unreliable_ids"]],
            fontsize=8,
            rotation=45,
        )
    fig.colorbar(im, ax=axes[0, :], label="P(H=1)", shrink=0.8)
    fig.colorbar(im2, ax=axes[1, :], label="P(source reliable)", shrink=0.8)
    return save(
        fig,
        f"trust-matrix-{network}",
        f"{LABEL[network]}, disinformation, prespecified seed 1729. Top: agents ordered by initial H belief; bottom: final trust using the same recipient order. Columns compare priors .50 and .90. Starred source IDs are the six actual disinformants. White cells are absent social edges or self-pairs, not zero trust. Rows are individual agents in one illustrative run; population-level findings use all 20 runs.",
    )


def main():
    minutes = SETTINGS["elapsed_seconds"] / 60
    parts.append(
        f'<h1>Learning, disagreement and source reliability in multibelief networks</h1><p class="subtitle">Controlled PolyGraphs experiments · 17 November 2026</p><p>1,440 simulations; 60 agents; 500 rounds; 10 trials per active agent; epsilon = 0.05. Simulation execution: {minutes:.1f} minutes. Exact inference, fixed seeds, complete results.</p>'
    )
    parts.append(
        '<nav><a href="#design">Design and models</a> · <a href="#findings">Findings</a> · <a href="#plots">Complete plots</a> · <a href="#trust">Bayesian trust</a> · <a href="#endpoints">Endpoint tables</a> · <a href="#limits">Interpretation and limits</a></nav>'
    )
    parts.append(
        '<h2 id="design">1. Design and model definitions</h2><p>Every agent observes its own experiments and fresh experiments from immediate neighbours. Received evidence is not forwarded again. Choices are synchronous and maximise immediate expected payoff. A pays with known probability 0.5; genuine B trials succeed with probability 0.55. Agents choosing A collect no evidence about B. All runs continue to round 500; flat trajectories can reflect stopped experimentation rather than successful learning.</p>'
    )
    parts.append(
        "<pre>Baseline:        H → D\nThree-node:     H → D ← S\nFactionalising: H4 → H3 ← H5; H1 → H2 ← H3; H2 → D\nTrust:          H → report_j ← R_j, separately for each neighbour j</pre>"
    )
    parts.append(
        "<p>D always denotes a fresh observation. Baseline has one persistent hypothesis; each three-node model has two; factionalisation has five plus D. We report the predictive probability of the next D, not the already observed binary outcome. Initial root probabilities are independent uniform draws across agents, paired across conditions.</p><table><tr><th>Model</th><th>Conditional probabilities</th><th>What evidence can identify</th></tr><tr><td>Baseline</td><td>P(D=1|H=0,1) = .45, .55</td><td>Whether B is good</td></tr><tr><td>Converging candidate</td><td>Parent order (H,S): 00=.45, 01=.50, 10=.50, 11=.55</td><td>Persistent .55 data favour 11</td></tr><tr><td>Polarising candidate</td><td>00=.55, 01=.45, 10=.45, 11=.55</td><td>Agreement H=S, not whether both are true or both false</td></tr><tr><td>Factionalisation</td><td>P(H3=1|H4,H5) and P(H2=1|H1,H3): 00=.9, 01=.1, 10=.1, 11=.9. P(D=1|H2)=.45,.55</td><td>H2; other hypotheses remain only indirectly constrained</td></tr></table>"
    )
    parts.append(
        "<p>Social networks are a cycle (degree 2), complete (degree 59), and independent Erdos–Renyi graphs with p=.1 or .5 (expected degrees 5.9 and 29.5). ER graphs are not conditioned on connectivity; every saved graph can be inspected. Network density changes the evidence volume as well as connectivity. These are not equal-evidence comparisons.</p><p>The four belief models have three treatments: all reliable, exactly six misinformants sampling B outcomes at .50, or exactly six disinformants sampling at .45. Unreliable identities are random but their count is fixed; identities, graphs, root draws and underlying trial uniforms are paired across comparisons. The basic models follow native unreliable-observation semantics, including agents using their own altered observations. The trust model instead retains genuine private trials and replaces unreliable outgoing reports. Ordinary-agent supplementary plots help distinguish population effects; the contrast between model families is not a pure intervention on trust alone.</p>"
    )
    parts.append('<h2 id="findings">2. Findings and how to read them</h2>')
    # Generated numerical findings: no narrative assumes the candidate model succeeds.
    for model in MODELS:
        frame = END[(END.model == model) & (END.treatment == "honest")]
        group = frame.groupby("network").agg(
            good=("mean_good", "mean"), js=("joint_js", "mean"), kl=("joint_kl", "mean")
        )
        entries = "; ".join(
            f'{LABEL[n]}: good-B belief {group.loc[n,"good"]:.3f}, joint JS {group.loc[n,"js"]:.3f}'
            for n in NETWORKS
        )
        parts.append(
            f"<p><strong>{LABEL[model]}:</strong> round-500 averages — {entries}.</p>"
        )
    parts.append(
        "<p>The central distinction is between learning which treatment pays and agreeing about every latent hypothesis. In the polarising model, 00 and 11 have identical likelihoods for every possible B history: their posterior odds retain their initial ratio. In the five-hypothesis model, evidence directly distinguishes only H2. Persistent disagreement on background hypotheses is therefore compatible with near-unanimous support for B. The converging model has an identifiable .55 state, but agents may stop experimenting before identifying it.</p>"
    )
    parts.append(overview())
    parts.append(exposure_figure())
    parts.append(geometry())
    parts.append(
        "<p><strong>A mean near .5 need not mean individual indecision.</strong> Background-node means can remain near .5 while agents occupy quite different positions. Read those means alongside the between-agent SDs and belief-geometry plots. Conversely, a change in within-agent total correlation measures dependence among hypotheses inside a belief distribution, not social alignment between agents.</p>"
    )
    parts.append(
        "<p><strong>KL and JS tell different parts of the story.</strong> In the baseline disinformation cycle, mean pairwise JS fell from .102 to .049 nats, while mean ordered-pair KL rose from .492 to 11.38 nats. The high mean good-B belief therefore does not imply that all disagreement disappeared. JS is bounded and can fall as disagreement becomes concentrated in a minority; KL strongly penalises near-zero probabilities assigned to possibilities that other agents favour. These are realised network histories with unequal evidence, not a violation of a common-evidence expectation theorem.</p>"
    )
    parts.append(
        "<p><strong>Connectivity qualification:</strong> 4 of the 20 ER p=.1 graphs were disconnected (three had two components; one had three). All sampled p=.5 graphs were connected. The sparse ER results therefore include isolated-component effects; they should not be read as a pure comparison of communication density among connected networks.</p>"
    )
    for model in MODELS:
        frame = END[END.model == model]
        values = frame.groupby(["network", "treatment"])["ordinary_good"].mean()
        parts.append(
            f'<p><strong>Source-treatment comparison, {LABEL[model].lower()}:</strong> ordinary-agent good-B belief at round 500 on the cycle is {values.loc[("cycle","honest")]:.3f} with no unreliable sources, {values.loc[("cycle","misinformation")]:.3f} with misinformation and {values.loc[("cycle","disinformation")]:.3f} with disinformation. The corresponding complete-network values are {values.loc[("complete","honest")]:.3f}, {values.loc[("complete","misinformation")]:.3f} and {values.loc[("complete","disinformation")]:.3f}. These are descriptive averages over the same 20 seeds; full network-specific results follow.</p>'
        )
    parts.append(
        "<h3>Definitions used in every divergence panel</h3><p><strong>Joint KL</strong> is the mean KL(P_i || P_j) over ordered distinct agent pairs. <strong>Joint JS</strong> is mean equal-weight Jensen–Shannon divergence over unordered pairs, not its square-root distance. <strong>Marginal-product</strong> measures replace each agent’s joint distribution with the product of its own node marginals. All logarithms are natural: units are nats. Total correlation is KL(joint || own marginal product), averaged over agents. No epsilon smoothing is used. Divergences exclude fresh D. For the one-hypothesis baseline, joint and marginal-product measures must coincide and total correlation is zero.</p><p>Each curve averages a statistic calculated separately in each run. Shading shows its 25th–75th percentiles across 20 runs; it is not a confidence interval. Agent SD uses denominator N, and its shaded band is the variation of that SD across runs. JS axes have a common upper bound ln 2. KL axes are separately scaled to retain detail.</p>"
    )
    parts.append(
        '<h2 id="plots">3. Complete belief-model results</h2><p>Each model/network combination below contains every requested mean, between-agent SD, joint KL, marginal-product KL, joint JS and marginal-product JS. Main figures use all agents. Expand the supplement for the same measurements restricted to ordinary agents (54 in affected treatments, 60 in the control).</p>'
    )
    for model in MODELS:
        parts.append(f"<h3>{LABEL[model]}</h3>")
        for network in NETWORKS:
            parts.append(f"<h4>{LABEL[network]}</h4>")
            parts.append(node_figure(model, network))
            parts.append(measures_figure(model, network))
            parts.append(
                "<details><summary>Ordinary-agent supplement: all node and divergence plots</summary>"
            )
            parts.append(node_figure(model, network, True))
            parts.append(measures_figure(model, network, True))
            parts.append("</details>")
    parts.append(
        '<h2 id="trust">4. Learning which neighbours are informative</h2><p>Each agent has a persistent reliability variable for every neighbour. Exact inference retains P(H) times the product of P(R_j|H), rather than incorrectly treating posterior reliability and H as independent. Report batches have binomial likelihoods; the number of trials and experimental schedule are conditioned on, not modelled as additional evidence. Agents do not infer reliability directly from opinion distance.</p><p>We compare initial trust .50 with the actual reliable proportion: .90 when six sources are unreliable and 1.00 when all are reliable. These are independent priors on source types, not knowledge that exactly six agents are unreliable. In the all-reliable condition agents entertain an uninformative alternative; in disinformation they entertain reversed evidence. Prior 1 makes reliability certain and cannot learn otherwise.</p><p><strong>The symmetry trap.</strong> With reversed evidence and trust .50, a source’s successes equally support “good B/reliable source” and “bad B/reversing source”. Marginalising source type cancels that source’s evidence about H. Private experiments can still teach the agent about H and thereby about the source. An agent initially choosing A takes no private B samples and remains at its initial H belief in this exact symmetric model. The .90 prior breaks this balance. This is a property of this particular likelihood and prior, not a general failure of Bayesian social learning.</p>'
    )
    for network in NETWORKS:
        parts.append(f"<h3>{LABEL[network]}</h3>")
        for treatment in TREATMENTS:
            parts.append(trust_figure(network, treatment))
    parts.append(trust_heatmap("cycle"))
    parts.append(trust_heatmap("complete"))
    parts.append(
        "<p><strong>Trust is not automatically beneficial.</strong> In the disinformation cycle, ordinary agents’ mean good-B belief was about .967 in the baseline and .775 in the trust model with prior .90. Introducing uncertainty about source reliability can impede learning as well as protect against bad evidence. This contrast is descriptive: the two families also differ in what unreliable agents privately observe, so it does not isolate the causal effect of source inference.</p>"
    )
    parts.append(
        "<p><strong>Trust results in numbers:</strong> under reversed reports and prior .50, mean final H belief was 0.565 on every network type (to three decimals), as expected when testimony cancels and paired private experiments drive H. The initially A cohort remained at a mean of 0.245. With prior .90, mean final H was 0.776 on the cycle, 0.970 on ER p=.1, and approximately 1 on the complete and ER p=.5 networks. Thus breaking symmetry enabled social learning but did not guarantee successful learning on sparse networks. With uninformative reports, final mean H was approximately .983–.988 on the cycle and nearly 1 on the dense graphs; unreliable-source trust fell substantially. These are finite-run averages, not a claim that Bayesian trust universally defeats misinformation.</p>"
    )
    exposure = pd.read_csv(DATA / "source-exposure-summary.csv")
    parts.append(
        '<details><summary>Source exposure: which reliability beliefs had a chance to be tested?</summary><p>The following final-round table separates sources that never supplied a report from sources that supplied at least 100 trials. Fractions are within actual source type, averaged over runs; trust means average available within-run edge means. The 100-trial cutoff is a descriptive diagnostic, not a stopping rule or an unbiased comparison: experimentation is endogenous. Empty groups have no trust estimate.</p><div class="tablewrap">'
        + exposure.to_html(index=False, float_format=lambda x: f"{x:.4f}", border=0)
        + "</div></details>"
    )
    trust_summary = (
        END[END.model == "trust"]
        .groupby(["network", "treatment", "trust_prior"])
        .agg(
            mean_good=("mean_good", "mean"),
            ordinary_good=("ordinary_good", "mean"),
            trust_reliable=("trust_reliable", "mean"),
            trust_unreliable=("trust_unreliable", "mean"),
            initial_a=("initial_a_mean", "mean"),
            initial_b=("initial_b_mean", "mean"),
            active=("active_fraction", "mean"),
        )
    )
    trust_summary.to_csv(DATA / "trust-endpoints.csv")
    parts.append(
        '<h2 id="endpoints">5. Complete round-500 summaries</h2><p>all_correct counts runs in which every agent assigns more than .95 probability to B being strictly better, out of 20. This is not necessarily agreement on all latent beliefs. active is the fraction choosing B next round. All entries other than counts are averages across runs.</p><div class="tablewrap">'
        + endpoint_table()
        + '</div><h3>Trust conditions</h3><div class="tablewrap">'
        + trust_summary.reset_index().to_html(
            index=False, float_format=lambda x: f"{x:.4f}", border=0
        )
        + "</div>"
    )
    parts.append(
        '<h2 id="limits">6. Interpretation, validation and limitations</h2><p>These are finite-horizon, small-ensemble experiments. Twenty paired runs reveal trajectories and large differences; they do not establish precise rare-event probabilities, asymptotic convergence or universal rankings of network structures. In particular, residual disagreement at round 500 should not automatically be described as permanent polarisation. Identical-likelihood latent states are a separate, structural source of non-identifiability.</p><p>More neighbours usually mean more observations per round. Raw records and summary tables include cumulative observations received so that evidence exposure can be examined. Endogenous stopping also matters: agents who cease testing may depend on neighbours for recovery. Sparse graphs can preserve independent experimentation but can also isolate misleading evidence or inactive regions.</p><p>Joint KL need not decline along a realised history, even under common evidence. In these network experiments agents often see different histories, so a common-evidence expectation theorem is not a monotonicity guarantee for the plotted population statistics. Nor does it impose monotonicity on the equal-weight pairwise JS used here.</p><p>Validation includes matched-path comparisons against native Bala–Goyal, comparisons of batched inference with the reference operations, the existing R bnlearn/gRain checks, independent replay from stored counts, and independent divergence calculations. See the separate experiment-settings report and machine-readable validation.json for exact checks and tolerances. The fast runs use exact NumPy/SciPy inference, not a fresh R process each round.</p><p>Replication materials: <a href="experiment-settings.html">full experiment settings</a>, <a href="data/settings.json">machine-readable settings</a>, <a href="data/validation.json">validation results</a>, <a href="data/metrics.csv">run-level metrics</a>, <a href="data/nodes.csv">node means and SDs</a>, <a href="figure-atlas.pdf">all figures as a PDF atlas</a>. All plots are embedded in this HTML and remain visible offline; supplemental file links require the project directory.</p>'
    )
    parts.append(
        "<p>Across-model comparisons require care: the joint state spaces have different dimensions and the induced prior probability of good B is not matched across models. Larger joint divergence alone is not a controlled measure of the effect of adding beliefs. The strongest controlled contrasts here are within a model across paired networks and source treatments.</p>"
    )
    parts.append(
        '<h3>References</h3><ul><li>Freeborn, O’Connor and Weatherall. Multiple Beliefs in Epistemic Networks (supplied draft). The present three-node CPTs are explicit experimental variants, not a verbatim replication of its strong/weak meta-belief case.</li><li>Freeborn (2024). <a href="https://doi.org/10.1007/s11229-024-04491-5">Rational factionalization for agents with probabilistically related beliefs</a>. Internal five-hypothesis structure retained; networked data collection and epsilon .05 are this study’s design.</li><li>O’Connor and Weatherall (2018). <a href="https://arxiv.org/abs/1712.04561">Scientific Polarization</a>. Bayesian source inference here replaces their distance-based trust rule.</li><li>Ball et al. (2024). <a href="https://doi.org/10.1057/s41599-024-02619-z">Computational philosophy: reflections on the PolyGraphs project</a>.</li><li><a href="https://github.com/alexandroskoliousis/polygraphs">Upstream PolyGraphs</a>; <a href="https://github.com/DavidFreeborn/polygraphs-multibelief">multibelief fork</a>.</li></ul>'
    )
    css = "body{font:18px/1.6 Georgia,serif;color:#202830;background:white;max-width:1160px;margin:48px auto;padding:0 28px}h1{font-size:36px;line-height:1.2}h2{font-size:27px;border-top:1px solid #cbd0d4;padding-top:28px;margin-top:50px}h3{font-size:23px}h4{font-size:20px}a{color:#21618c}nav,.subtitle,figcaption,table,summary{font-family:Arial,sans-serif}nav{font-size:15px}figure{margin:32px 0 48px}img{width:100%;height:auto}figcaption{font-size:14px;line-height:1.5;color:#35424c}table{border-collapse:collapse;font-size:12px;width:100%}td,th{padding:7px 9px;border-bottom:1px solid #d5dade;text-align:left}th{background:#f0f2f3}pre{padding:20px;background:#f5f6f7;overflow:auto}.tablewrap{overflow:auto}summary{cursor:pointer;font-size:16px;padding:12px;background:#f0f2f3}details{margin:20px 0}@media print{nav,summary{display:none}details{display:block}figure{break-inside:avoid}body{max-width:none;font-size:11pt}}"
    (HERE / "results.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Multibelief network experiments</title><style>'
        + css
        + "</style><body>"
        + "".join(parts)
        + "</body></html>",
        encoding="utf-8",
    )
    atlas.close()
    print(f"Rendered {figure_number} figures", flush=True)


if __name__ == "__main__":
    main()
