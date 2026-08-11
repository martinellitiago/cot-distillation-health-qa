#!/usr/bin/env python3
"""AIM figures, rebuilt to follow skill:visual-paper-clinical-ai.

Design language (see .claude/skills/visual-paper-clinical-ai/reference.md):
- colour by alpha (fixed map), baseline dark grey, technique by marker;
- main text = forest / effect plots with hierarchical intervals + pp annotations,
  zero reference on EFFECT plots;
- appendix = full grid heatmap (granularity);
- captions written from the printed values; sober interpretation.

Source of truth: results/acuracia.txt (Stage 2, means + 95% bootstrap CI).
Stage-1 means and ENEM come from the result files (results/finais.txt,
results/results_forgetting_enem.txt) and are annotated inline; Stage-1 lacks
stored CIs, so it is shown as points with the paired-test annotation we do have.
Every number is from a file, none from memory.
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.lines as mlines

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))
ACCDF = pd.read_csv(os.path.join(RESULTS, "accuracy.csv"))

# ---- palette (skill) ----
ALPHA_COLORS = {1.0: "#E53E3E", 0.3: "#3FB37F", 0.1: "#FF8500"}
BASELINE = "#333333"
GRID = "#B8B8B8"
TECH_MARKER = {"distill_sft": "o", "step_by_step": "s", "pure_sft": "D"}
# neutral palette for schematics (skill: keep schematics neutral, one accent)
INKC = "#2B2B2B"
NEU = "#ECEAE4"      # light neutral fill
NEU2 = "#D8D4CA"     # secondary neutral
ACCENT = "#0F766E"   # single accent (teal), consistent with data figures
ACCENT_L = "#CDE3E0"  # light accent

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
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
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
    sft = acc_row("pure_sft", "abc", "answer_only", "B").acc
    ax.axvline(sft, color=BASELINE, lw=1.1, ls="--", alpha=0.7)
    ax.text(sft, len(order) - 0.4, " pure SFT (direct)", color=BASELINE,
            fontsize=8, ha="left", va="top")
    ax.set_yticks(ys); ax.set_yticklabels([o[3] for o in order], fontsize=9)
    ax.set_xlabel("Accuracy on transfer subset B under reasoning (95% CI)")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.set_xlim(0.28, 0.48)
    ax.grid(axis="x", alpha=0.25)
    hc = mlines.Line2D([], [], color="gray", marker="o", ls="None", label="distill-SFT")
    hs = mlines.Line2D([], [], color="gray", marker="s", ls="None", label="Step-by-step")
    a10 = mlines.Line2D([], [], color=ALPHA_COLORS[1.0], lw=4, label="$\\alpha$=1.0")
    a03 = mlines.Line2D([], [], color=ALPHA_COLORS[0.3], lw=4, label="$\\alpha$=0.3")
    a01 = mlines.Line2D([], [], color=ALPHA_COLORS[0.1], lw=4, label="$\\alpha$=0.1")
    ax.legend(handles=[hc, hs, a10, a03, a01], loc="center left",
              frameon=True, fontsize=8, ncol=1)
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
    tcol = {"distill_sft": "#0F766E", "step_by_step": BASELINE}
    for tech, name in [("distill_sft", "distill-SFT"), ("step_by_step", "Step-by-step")]:
        ax.plot(x, series(tech, "answer_only"), "-", marker=TECH_MARKER[tech],
                color=tcol[tech], label=f"{name}, direct")
        ax.plot(x, series(tech, "reasoning"), "--", marker=TECH_MARKER[tech],
                color=tcol[tech], mfc="white", label=f"{name}, reasoning")
    sft = acc_row("pure_sft", "hard", "answer_only", "H").acc
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
        ("Pure SFT", -0.120, BASELINE),
        ("Step-by-step", -0.044, "#0F766E"),
        ("distill-SFT", -0.042, "#0F766E"),
    ]
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    ys = list(range(len(rows)))[::-1]
    for y, (lab, d, c) in zip(ys, rows):
        col = BASELINE if lab == "Pure SFT" else "#0F766E"
        ax.hlines(y, 0, d, color=col, lw=6, alpha=0.85)
        ax.scatter(d, y, color=col, s=80, edgecolors="white", linewidths=0.9, zorder=5)
        ax.text(d, y + 0.22, f"{d*100:.1f} pp", va="bottom", ha="center",
                fontsize=9, color=col, fontweight="bold")
    ax.axvline(0, color="black", lw=1.2, ls="--", alpha=0.65)
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows])
    ax.set_ylim(-0.5, len(rows) - 0.3)
    ax.set_xlabel("Paired change in ENEM reasoning accuracy vs. untrained base")
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
        ("pure_sft", 1.0, "pure_sft", "answer_only", 0.1911, 0.0173),
        ("distill_sft", 1.0, "distill_sft", "reasoning", 0.2371, 1.3867),
        ("distill_sft_a0.3", 0.3, "distill_sft", "reasoning", 0.2238, 1.1852),
        ("distill_sft_a0.1", 0.1, "distill_sft", "reasoning", 0.2406, 1.1775),
        ("step_by_step", 1.0, "step_by_step", "reasoning", 0.2312, 1.2455),
        ("step_by_step_a0.3", 0.3, "step_by_step", "reasoning", 0.2262, 1.3252),
        ("step_by_step_a0.1", 0.1, "step_by_step", "reasoning", 0.2203, 1.3096),
    ]
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for cond, a, tech, regime, acc, secs in pts:
        c = ALPHA_COLORS[a] if tech != "pure_sft" else BASELINE
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
# FIG A1 (appendix). Full accuracy grid: condition x (subset,regime) heatmap.
# ---------------------------------------------------------------------------
def fig_appendix_grid():
    conds = ["pure_sft", "distill_sft", "distill_sft_a0.3", "distill_sft_a0.1",
             "step_by_step", "step_by_step_a0.3", "step_by_step_a0.1"]
    clabel = {"pure_sft": "Pure SFT", "distill_sft": "Concat $\\alpha$1.0",
              "distill_sft_a0.3": "Concat $\\alpha$0.3", "distill_sft_a0.1": "Concat $\\alpha$0.1",
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
# FIG design (main, Fig 1). Neutral schematic: pipeline + A/B/C/H subsets.
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
    ax.text(0.2, 2.55, "Test items partitioned by teacher / untrained-student correctness",
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
    fig_design()
    fig_stage1()
    fig_stage2_forest()
    fig_alpha_dose()
    fig_forgetting()
    fig_cost()
    fig_appendix_grid()
    print("done")
