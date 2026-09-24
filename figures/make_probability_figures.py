#!/usr/bin/env python3
"""Publication figures for the AIIM manuscript.

The palette is intentionally restricted to clinical green, blue, and purple;
red is reserved for deterioration. Probability trajectories are read from the
author-supplied evidence tables, and Stage-2 estimates are read from the
reconstructed ten-split analysis. No information-theoretic quantities are
plotted in this manuscript.
"""
import os
import textwrap
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.lines as mlines
from matplotlib.patches import FancyBboxPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
PROB = pd.read_csv(os.path.join(HERE, "probability_trajectories.csv"))
CONTRASTS = pd.read_csv(os.path.join(HERE, "probability_contrasts.csv"))

# ---- clinical palette ----
GREEN = "#0B6B53"
GREEN_DARK = "#084A3A"
GREEN_LIGHT = "#BFE3D5"
BLUE = "#3568A8"
PURPLE = "#7251A3"
RED = "#B63C3C"
GREY = "#6F7D77"
ALPHA_COLORS = {1.0: GREEN, 0.3: BLUE, 0.1: PURPLE}
BASELINE = "#333333"
GRID = "#B8B8B8"
TECH_MARKER = {"distill_sft": "o", "step_by_step": "s", "sft_puro": "D"}
# neutral palette for schematics (skill: keep schematics neutral, one accent)
INKC = "#2B2B2B"
NEU = "#ECEAE4"      # light neutral fill
NEU2 = "#D8D4CA"     # secondary neutral
ACCENT = GREEN
ACCENT_L = GREEN_LIGHT

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.edgecolor": "#333333", "axes.labelcolor": "#111111",
    "text.color": "#111111", "xtick.color": "#333333", "ytick.color": "#333333",
    "axes.linewidth": 0.8, "figure.dpi": 150,
})


def save(fig, name):
    fig.savefig(os.path.join(HERE, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(HERE, name + ".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("wrote", name)


def acc_row(cond, split, regime, subset):
    r = ACCDF[(ACCDF.cond == cond) & (ACCDF.split == split) &
            (ACCDF.regime == regime) & (ACCDF.subset == subset)]
    return r.iloc[0]


# ---------------------------------------------------------------------------
# FIG 2 (main). Visual explanation of the rationale-prefix probability trace.
# The rationale is an excerpt from a Portuguese output in the study archive;
# the plotted curve is the aggregate teacher-correct trajectory, not an
# item-specific trace for the illustrated question.
# ---------------------------------------------------------------------------
def fig_probability_explainer():
    fig = plt.figure(figsize=(9.6, 3.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.24)
    axl = fig.add_subplot(gs[0, 0])
    axr = fig.add_subplot(gs[0, 1])
    axl.set_xlim(0, 1)
    axl.set_ylim(0, 1)
    axl.axis("off")

    axl.text(
        0.03, 0.94, "Portuguese rationale prefix",
        ha="left", va="top", fontsize=11, fontweight="bold"
    )

    fragments = [
        "A paciente é jovem, sem histórico psiquiátrico prévio,",
        "com início recente e alterações de comportamento,",
        "pensamentos persecutórios, agressividade e insônia.",
        "Cefaleia constante e febre sugerem uma condição orgânica,",
        "como uma infecção do sistema nervoso central,",
        "capaz de causar sintomas psicóticos e neurológicos agudos.",
        "A ausência de história psiquiátrica e o início súbito reforçam a suspeita.",
    ]
    reveal_fracs = [0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.0]
    reveal_colors = [
        "#E8F4EF", "#D8EDE4", "#C7E5D9", "#ACD9C8",
        "#8FCBB5", "#68B79C", "#3D9678",
    ]
    y0, line_h = 0.82, 0.099
    for j, (frag, frac, fc) in enumerate(
        zip(fragments, reveal_fracs, reveal_colors)
    ):
        y = y0 - j * line_h
        axl.add_patch(Rectangle(
            (0.06, y - 0.037), 0.80, 0.072,
            facecolor=fc, edgecolor="none"
        ))
        axl.text(
            0.075, y, frag, ha="left", va="center",
            fontsize=5.9 if j == len(fragments) - 1 else 7.05,
            color="white" if j == len(fragments) - 1 else INKC
        )
        axl.text(
            0.94, y, f"{int(frac * 100)}%",
            ha="right", va="center", fontsize=7.2,
            color=GREEN_DARK, fontweight="bold"
        )
    axl.text(
        0.06, 0.145,
        "Explicit final-answer span removed before scoring",
        ha="left", va="bottom", fontsize=6.8, color=RED, style="italic"
    )

    for condition, color, marker, linestyle, label in [
        ("coherent", GREEN, "o", "-", "Coherent rationale"),
        ("shuffled", BLUE, "s", "--", "Shuffled rationale"),
    ]:
        d = PROB[
            (PROB.teacher_status == "correct") &
            (PROB.condition == condition)
        ].sort_values("fraction")
        axr.plot(
            d.fraction, d["mean"], color=color, marker=marker,
            linestyle=linestyle, linewidth=2.0, markersize=5.2, label=label
        )
        if condition == "coherent":
            point_colors = ["#B7C2BD"] + reveal_colors
            axr.scatter(
                d.fraction, d["mean"], c=point_colors, s=58,
                edgecolors="white", linewidths=0.8, zorder=5
            )

    axr.set_title(
        "Conditional probability of correct answer",
        loc="left", fontsize=11, fontweight="bold"
    )
    axr.set_xlabel(r"Rationale prefix $r_{1:f}$")
    axr.set_ylabel(
        r"$\left\langle P_\theta(y\mid q,r_{1:f})\right\rangle$"
    )
    axr.set_xlim(-0.03, 1.03)
    axr.set_ylim(0.64, 0.92)
    axr.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    axr.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    axr.grid(axis="y", alpha=0.22)
    axr.spines[["top", "right"]].set_visible(False)
    axr.legend(frameon=False, loc="upper left", fontsize=8.2)

    fig.subplots_adjust(left=0.04, right=0.99, top=0.94, bottom=0.18)
    save(fig, "fig_probability_explainer_v2")


# ---------------------------------------------------------------------------
# FIG 3 (main). Three-panel rationale-conditioned choice probabilities.
# ---------------------------------------------------------------------------
def fig_probability_trajectories():
    fig, axes = plt.subplots(
        1, 3, figsize=(10.6, 3.35),
        gridspec_kw={"width_ratios": [1.08, 1.0, 1.0]}
    )

    # Panel A: concrete rationale prefixes used by the diagnostic.
    ax = axes[0]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(
        "A  Teacher-correct rationale prefix", loc="left", fontweight="bold"
    )
    fragments = [
        "A paciente é jovem, sem histórico psiquiátrico prévio,",
        "com início recente e alterações de comportamento,",
        "pensamentos persecutórios, agressividade e insônia.",
        "Cefaleia e febre sugerem uma condição orgânica,",
        "como uma infecção do sistema nervoso central,",
        "capaz de causar sintomas psicóticos e neurológicos.",
        "O início súbito reforça a suspeita.",
    ]
    reveal_fracs = [0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.0]
    reveal_colors = [
        "#E8F4EF", "#D8EDE4", "#C7E5D9", "#ACD9C8",
        "#8FCBB5", "#68B79C", "#3D9678",
    ]
    y0, line_h = 0.82, 0.102
    for j, (frag, frac, fc) in enumerate(
        zip(fragments, reveal_fracs, reveal_colors)
    ):
        y = y0 - j * line_h
        ax.add_patch(Rectangle(
            (0.02, y - 0.037), 0.82, 0.072,
            facecolor=fc, edgecolor="none"
        ))
        ax.text(
            0.04, y, frag, ha="left", va="center", fontsize=5.75,
            color="white" if j == len(fragments) - 1 else INKC
        )
        ax.text(
            0.97, y, f"{int(frac * 100)}%",
            ha="right", va="center", fontsize=6.7,
            color=GREEN_DARK, fontweight="bold"
        )
    ax.text(
        0.02, 0.075, "Final-answer span removed",
        ha="left", va="bottom", fontsize=6.2, color=RED, style="italic"
    )

    styles = {
        "coherent": dict(color=GREEN, marker="o", ls="-", label="Coherent"),
        "shuffled": dict(color=BLUE, marker="s", ls="--", label="Shuffled"),
    }

    for ax, status, panel, title in [
        (axes[1], "correct", "B", "Teacher correct"),
        (axes[2], "wrong", "C", "Teacher wrong"),
    ]:
        for condition in ["coherent", "shuffled"]:
            d = PROB[(PROB.teacher_status == status) &
                     (PROB.condition == condition)].sort_values("fraction")
            st = styles[condition].copy()
            if status == "wrong" and condition == "coherent":
                st["color"] = RED
            ax.fill_between(
                d.fraction.to_numpy(), d.ci_lo.to_numpy(), d.ci_hi.to_numpy(),
                color=st["color"], alpha=0.14, linewidth=0
            )
            ax.plot(
                d.fraction, d["mean"], color=st["color"], marker=st["marker"],
                ls=st["ls"], lw=1.8, ms=4.2, label=st["label"]
            )
        ax.set_title(f"{panel}  {title}", loc="left", fontweight="bold")
        ax.set_xlabel(r"Rationale prefix $r_{1:f}$")
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
        ax.grid(axis="y", alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)

    axes[1].set_ylabel(
        r"$\left\langle P_\theta(y\mid q,r_{1:f})\right\rangle$"
    )
    axes[1].set_ylim(0.62, 0.92)
    axes[2].set_ylim(0.10, 0.33)
    axes[1].legend(frameon=False, fontsize=7.5, loc="upper left")

    fig.subplots_adjust(
        left=0.025, right=0.99, top=0.88, bottom=0.21, wspace=0.34
    )
    save(fig, "fig_probability_trajectories")


# ---------------------------------------------------------------------------
# FIG 1 (main). Stage-2 transfer subset B under reasoning: forest by technique x alpha.
# ---------------------------------------------------------------------------
def fig_stage2_forest():
    order = [  # (cond, technique, alpha, label)
        ("distill_sft", "distill_sft", 1.0, "distill-SFT $\\alpha$=1.0"),
        ("distill_sft_a0.1", "distill_sft", 0.1, "distill-SFT $\\alpha$=0.1"),
        ("distill_sft_a0.3", "distill_sft", 0.3, "distill-SFT $\\alpha$=0.3"),
        ("step_by_step", "step_by_step", 1.0, "Step-by-step $\\alpha$=1.0"),
        ("step_by_step_a0.1", "step_by_step", 0.1, "Step-by-step $\\alpha$=0.1"),
        ("step_by_step_a0.3", "step_by_step", 0.3, "Step-by-step $\\alpha$=0.3"),
    ]
    fig, ax = plt.subplots(figsize=(6.8, 3.25))
    ys = list(range(len(order)))[::-1]
    for y, (cond, tech, a, lab) in zip(ys, order):
        r = acc_row(cond, "abc", "reasoning", "B")
        c = ALPHA_COLORS[a]
        # 95% interval (thin) — 80% not stored, so we show mean + 95% CI + point
        ax.hlines(y, r.ci_lo, r.ci_hi, color=c, lw=1.6, alpha=0.9)
        ax.vlines([r.ci_lo, r.ci_hi], y - 0.12, y + 0.12, color=c, lw=1.2, alpha=0.9)
        ax.scatter(r.acc, y, marker=TECH_MARKER[tech], color=c, s=80,
                   edgecolors="white", linewidths=0.9, zorder=5)
        ax.text(r.ci_hi + 0.004, y, f"{r.acc*100:.1f}", va="center",
                fontsize=8.5, color=c, fontweight="bold")
    # reference: pure-SFT direct answering on B (no-transfer baseline)
    sft = acc_row("sft_puro", "abc", "answer_only", "B").acc
    ax.axvline(sft, color=BASELINE, lw=1.1, ls="--", alpha=0.7)
    ax.text(sft + 0.002, max(ys) - 0.45, "Pure SFT direct", color=BASELINE,
            fontsize=7.5, ha="left", va="top")
    ax.set_yticks(ys); ax.set_yticklabels([o[3] for o in order], fontsize=9)
    ax.set_xlabel("Accuracy on transfer subset B under reasoning (95% CI)")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.set_xlim(0.28, 0.48)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    save(fig, "fig_stage2_forest")


# ---------------------------------------------------------------------------
# FIG 2 (main). Loss-weight dose-response on hard questions (alpha on x).
# ---------------------------------------------------------------------------
def fig_alpha_dose():
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    alphas = [1.0, 0.3, 0.1]
    x = [0, 1, 2]

    def series(tech, regime, subset="H"):
        out = []
        for a in alphas:
            cond = tech if a == 1.0 else f"{tech}_a{a:g}"
            out.append(acc_row(cond, "hard", regime, subset).acc)
        return out

    # concatenated: circles; step: squares. regime: solid=direct, dashed=reasoning.
    tcol = {"distill_sft": GREEN, "step_by_step": BLUE}
    for tech, name in [("distill_sft", "distill-SFT"), ("step_by_step", "Step-by-step")]:
        ax.plot(x, series(tech, "answer_only"), "-", marker=TECH_MARKER[tech],
                color=tcol[tech], label=f"{name}, direct")
        ax.plot(x, series(tech, "reasoning"), "--", marker=TECH_MARKER[tech],
                color=tcol[tech], mfc="white", label=f"{name}, reasoning")
    sft = acc_row("sft_puro", "hard", "answer_only", "H").acc
    ax.axhline(sft, color=GRID, ls=":", lw=1.1)
    ax.text(2.02, sft, "pure SFT", fontsize=8, va="center", color="gray")
    ax.set_xticks(x); ax.set_xticklabels([f"{a:g}" for a in alphas])
    ax.set_xlabel("Rationale loss weight $\\alpha$")
    ax.set_ylabel("Accuracy on hard items")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=7.5, frameon=True, loc="center right")
    fig.tight_layout()
    save(fig, "fig_alpha_dose")


# ---------------------------------------------------------------------------
# FIG 3 (main). ENEM: paired reasoning degradation vs base (effect plot, zero ref).
# Numbers from results/results_forgetting_enem.txt (Axis 2, paired vs base, alpha=1.0).
# ---------------------------------------------------------------------------
def fig_forgetting():
    rows = [  # label, paired delta vs base, colour
        ("Pure SFT", -0.120, RED),
        ("Step-by-step", -0.044, BLUE),
        ("distill-SFT", -0.042, GREEN),
    ]
    fig, ax = plt.subplots(figsize=(5.8, 2.45))
    ys = list(range(len(rows)))[::-1]
    for y, (lab, d, c) in zip(ys, rows):
        col = c
        ax.hlines(y, 0, d, color=col, lw=6, alpha=0.85)
        ax.scatter(d, y, color=col, s=80, edgecolors="white", linewidths=0.9, zorder=5)
        ax.text(d, y + 0.22, f"{d*100:.1f} pp", va="bottom", ha="center",
                fontsize=9, color=col, fontweight="bold")
    ax.axvline(0, color="black", lw=1.2, ls="--", alpha=0.65)
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows])
    ax.set_ylim(-0.5, len(rows) - 0.3)
    ax.set_xlabel("Paired change in ENEM reasoning accuracy vs. baseline model")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.set_xlim(-0.14, 0.02)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    save(fig, "fig_forgetting")


# ---------------------------------------------------------------------------
# FIG 4 (main). Accuracy vs cost on hard items (kept, restyled to alpha palette).
# ---------------------------------------------------------------------------
def fig_cost():
    # accuracy + latency clusters read from the results/finais.txt efficiency block.
    pts = [  # cond, alpha, tech, regime, acc, secs
        ("distill_sft", 1.0, "distill_sft", "answer_only", 0.155, 0.0172),
        ("distill_sft_a0.3", 0.3, "distill_sft", "answer_only", 0.1807, 0.0169),
        ("distill_sft_a0.1", 0.1, "distill_sft", "answer_only", 0.1832, 0.0170),
        ("step_by_step", 1.0, "step_by_step", "answer_only", 0.1941, 0.0172),
        ("step_by_step_a0.3", 0.3, "step_by_step", "answer_only", 0.2084, 0.0169),
        ("step_by_step_a0.1", 0.1, "step_by_step", "answer_only", 0.2173, 0.0168),
        ("sft_puro", 1.0, "sft_puro", "answer_only", 0.1911, 0.0173),
        ("distill_sft", 1.0, "distill_sft", "reasoning", 0.2371, 1.3867),
        ("distill_sft_a0.3", 0.3, "distill_sft", "reasoning", 0.2238, 1.1852),
        ("distill_sft_a0.1", 0.1, "distill_sft", "reasoning", 0.2406, 1.1775),
        ("step_by_step", 1.0, "step_by_step", "reasoning", 0.2312, 1.2455),
        ("step_by_step_a0.3", 0.3, "step_by_step", "reasoning", 0.2262, 1.3252),
        ("step_by_step_a0.1", 0.1, "step_by_step", "reasoning", 0.2203, 1.3096),
    ]
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for cond, a, tech, regime, acc, secs in pts:
        c = ALPHA_COLORS[a] if tech != "sft_puro" else BASELINE
        ax.scatter(secs, acc, marker=TECH_MARKER[tech], color=c, s=70,
                   edgecolors="white", linewidths=0.8, zorder=3)
    ax.set_xscale("log"); ax.set_xlim(0.01, 3.0); ax.set_ylim(0.14, 0.26)
    ax.set_xlabel("Latency (s / question, log scale)")
    ax.set_ylabel("Accuracy on hard items")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.annotate("", xy=(1.20, 0.152), xytext=(0.018, 0.152),
                arrowprops=dict(arrowstyle="<->", color=GRID, lw=1.1))
    ax.text(0.14, 0.158, r"$\approx$77$\times$ latency", ha="center",
            fontsize=8.5, color="gray")
    hc = mlines.Line2D([], [], color="gray", marker="o", ls="None", label="distill-SFT")
    hs = mlines.Line2D([], [], color="gray", marker="s", ls="None", label="Step-by-step")
    hd = mlines.Line2D([], [], color=BASELINE, marker="D", ls="None", label="Pure SFT")
    ax.legend(handles=[hc, hs, hd], fontsize=7.5, frameon=True, loc="lower right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    save(fig, "fig_cost")


# ---------------------------------------------------------------------------
# Appendix accuracy grid: condition x (subset,regime) heatmap.
# ---------------------------------------------------------------------------
def fig_appendix_grid():
    conds = ["sft_puro", "distill_sft", "distill_sft_a0.3", "distill_sft_a0.1",
             "step_by_step", "step_by_step_a0.3", "step_by_step_a0.1"]
    clabel = {"sft_puro": "Pure SFT", "distill_sft": "distill-SFT $\\alpha$1.0",
              "distill_sft_a0.3": "distill-SFT $\\alpha$0.3", "distill_sft_a0.1": "distill-SFT $\\alpha$0.1",
              "step_by_step": "Step $\\alpha$1.0", "step_by_step_a0.3": "Step $\\alpha$0.3",
              "step_by_step_a0.1": "Step $\\alpha$0.1"}
    cols = [("abc", "answer_only", "A"), ("abc", "answer_only", "B"), ("abc", "answer_only", "C"),
            ("abc", "reasoning", "A"), ("abc", "reasoning", "B"), ("abc", "reasoning", "C"),
            ("hard", "answer_only", "H"), ("hard", "reasoning", "H")]
    collab = ["A dir", "B dir", "C dir", "A reas", "B reas", "C reas", "D dir", "D reas"]
    M = np.full((len(conds), len(cols)), np.nan)
    for i, cond in enumerate(conds):
        for j, (sp, rg, sub) in enumerate(cols):
            r = ACCDF[(ACCDF.cond == cond) & (ACCDF.split == sp) &
                    (ACCDF.regime == rg) & (ACCDF.subset == sub)]
            if len(r):
                M[i, j] = r.iloc[0].acc
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    im = ax.imshow(M, aspect="auto", cmap="YlGnBu", vmin=0.1, vmax=0.95)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(collab, rotation=45, ha="right")
    ax.set_yticks(range(len(conds))); ax.set_yticklabels([clabel[c] for c in conds])
    for i in range(len(conds)):
        for j in range(len(cols)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if M[i, j] > 0.55 else "black")
    plt.colorbar(im, ax=ax, label="Accuracy")
    fig.tight_layout()
    save(fig, "fig_appendix_grid")


# ---------------------------------------------------------------------------
# FIG 0 (main). Stage-1 content x format on subset B (the organising 2x2).
# Means from results/finais.txt (shuffle vs shuffle_notag) and HANDOFF Stage-1
# table; Stage-1 has no stored CIs, so this is a value matrix, not a forest.
# ---------------------------------------------------------------------------
def fig_stage1():
    mat = np.array([[0.390, 0.391],   # coherent: tagged, tag-free
                    [0.237, 0.285]])  # shuffled: tagged, tag-free
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    im = ax.imshow(mat, cmap="YlGnBu", vmin=0.20, vmax=0.42, aspect="auto")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Tagged", "Tag-free"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Coherent", "Shuffled"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{mat[i, j]*100:.1f}", ha="center", va="center",
                    fontsize=15, fontweight="bold",
                    color="white" if mat[i, j] > 0.33 else "#111111")
    ax.set_title("Subset-B accuracy (%): content $\\times$ format", fontsize=11)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    save(fig, "fig_stage1")


# ---------------------------------------------------------------------------
# Study-design schematic: pipeline + A/B/C/D subsets.
# ---------------------------------------------------------------------------
def fig_design():
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")

    def box(x, y, w, h, text, fc, tc=INKC, bold=False, fs=8.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.04,rounding_size=0.10",
                     linewidth=1.0, edgecolor=INKC, facecolor=fc))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=tc, fontweight="bold" if bold else "normal")

    def arrow(x1, y1, x2, y2, c=INKC):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                     mutation_scale=11, linewidth=1.1, color=c))

    # pipeline row
    box(0.2, 3.5, 2.5, 1.15, "Teacher\nQwen3-32B", NEU, bold=True)
    box(3.5, 3.5, 2.7, 1.15, "Portuguese\nrationale + answer", NEU)
    box(7.0, 3.5, 2.5, 1.15, "Student\nQwen3-4B + LoRA", ACCENT, tc="white", bold=True)
    arrow(2.7, 4.07, 3.5, 4.07); arrow(6.2, 4.07, 7.0, 4.07)
    ax.text(4.85, 3.36, "teacher-correct rationales only", ha="center", va="top",
            fontsize=6.8, style="italic", color="#7A756B")

    # subset row
    ax.text(0.2, 2.55, "Test items partitioned by teacher / baseline-student correctness",
            ha="left", va="center", fontsize=8, fontweight="bold", color=INKC)
    subs = [("A", "both correct", NEU2, INKC),
            ("B", "teacher right,\nstudent wrong", ACCENT, "white"),
            ("C", "student right,\nteacher wrong", NEU2, INKC),
            ("D", "both wrong\n(hard)", NEU2, INKC)]
    for i, (k, d, c, tc) in enumerate(subs):
        box(0.2 + i * 3.0, 0.35, 2.7, 1.35, f"{k}\n{d}", c, tc=tc, bold=True, fs=8.5)
    ax.annotate("transfer thermometer", xy=(3.2 + 1.35, 1.72), xytext=(3.2 + 1.35, 2.15),
                ha="center", fontsize=6.8, color=ACCENT, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=1.1))
    save(fig, "fig1_design")


if __name__ == "__main__":
    fig_probability_explainer()
    fig_probability_trajectories()
    print("done")
