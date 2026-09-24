#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compact conceptual figures for the AIIM manuscript.

The diagrams use light and dark clinical green, with neutral ink only for text
and structure. The proposed distill-SFT arm is explicitly highlighted.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
INK = "#26332E"
GREEN = "#0B6B53"
GREEN_DARK = "#084A3A"
GREEN_LIGHT = "#BFE3D5"
GREEN_PALE = "#E7F4EF"
GREY = "#6F7D77"
BLUE = "#35618D"
RED = "#B84A4A"

plt.rcParams.update({"font.family": "serif", "text.color": INK})


def save(fig, name):
    fig.savefig(os.path.join(HERE, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(HERE, name + ".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("wrote", name)


def box(ax, x, y, w, h, text, fc, tc=INK, bold=True, fs=8.5,
        edge=INK, lw=1.0):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        fc=fc, ec=edge, lw=lw, zorder=3
    ))
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center", fontsize=fs, color=tc,
        fontweight="bold" if bold else "normal", zorder=4
    )


def arrow(ax, x1, y1, x2, y2, color=GREEN, lw=1.4):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>",
        mutation_scale=13, lw=lw, color=color, zorder=2
    ))


def fig1_design():
    fig, ax = plt.subplots(figsize=(7.4, 2.75))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.7)
    ax.axis("off")

    box(ax, 0.825, 3.25, 2.35, 0.92, "Teacher\nQwen3-32B", GREEN_LIGHT)
    box(ax, 4.625, 3.25, 2.75, 0.92,
        "Portuguese rationale\n+ answer", GREEN_PALE)
    box(ax, 8.675, 3.25, 2.65, 0.92,
        "Student\nQwen3-4B + LoRA", GREEN, tc="white")
    arrow(ax, 3.175, 3.71, 4.625, 3.71)
    arrow(ax, 7.375, 3.71, 8.675, 3.71)
    ax.text(
        6.00, 3.08, "answer-verified rationales only",
        ha="center", va="top", fontsize=6.8, style="italic", color=GREY
    )

    ax.text(
        6.00, 2.47,
        "Test items partitioned by teacher and baseline-student correctness",
        ha="center", va="center", fontsize=8, fontweight="bold"
    )
    subsets = [
        ("A", "both correct", GREEN_PALE, INK),
        ("B", "teacher correct\nstudent wrong", GREEN, "white"),
        ("C", "student correct\nteacher wrong", GREEN_PALE, INK),
        ("D", "both wrong\nheld-out difficult", GREEN_PALE, INK),
    ]
    for i, (key, description, fill, text_color) in enumerate(subsets):
        box(
            ax, 0.25 + i * 2.95, 0.48, 2.60, 1.25,
            f"{key}\n{description}", fill, tc=text_color, fs=8.2
        )
    ax.annotate(
        "transfer thermometer", xy=(4.50, 1.74), xytext=(4.50, 2.16),
        ha="center", fontsize=6.8, color=GREEN, fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.1)
    )
    save(fig, "fig1_design")


def fig2_techniques():
    """Compact two-panel overview; Panel B preserves the original drawing."""
    fig = plt.figure(figsize=(10.2, 2.62))
    axa = fig.add_axes([0.015, 0.14, 0.46, 0.64])
    axb = fig.add_axes([0.515, 0.14, 0.47, 0.64])
    for ax in (axa, axb):
        ax.set_xlim(0, 12)
        ax.set_ylim(1.12, 5.18)
        ax.axis("off")

    fig.text(0.018, 0.86, "A", fontsize=11.2, fontweight="bold",
             color=GREEN_DARK, va="center")
    fig.text(0.047, 0.86, "Rationale influence diagnosis",
             fontsize=9.2, fontweight="bold", va="center")
    fig.text(0.512, 0.86, "B", fontsize=11.2, fontweight="bold",
             color=GREEN_DARK, va="center")
    fig.text(0.541, 0.86, "Answer-verified distillation",
             fontsize=9.2, fontweight="bold", va="center")

    # Panel A: compact diagnostic workflow.
    box(
        axa, 3.10, 4.48, 5.80, 0.46,
        "Inject teacher rationale\n" + r"into QA input $q$",
        GREEN_PALE, fs=6.1
    )
    arrow(axa, 6.00, 4.44, 6.00, 3.97, color=GREEN_DARK, lw=1.1)
    box(
        axa, 3.40, 3.45, 5.20, 0.46,
        "Reveal prefixes\n" + r"$r_{1:f},\ f=0,.15,\ldots,1$",
        GREEN_LIGHT, fs=6.3
    )
    arrow(axa, 6.00, 3.41, 6.00, 2.94, color=GREEN_DARK, lw=1.1)
    box(
        axa, 3.40, 2.42, 5.20, 0.46,
        "Student Qwen3-4B\n" + r"reads $q+r_{1:f}$",
        GREEN, tc="white", fs=6.3
    )
    arrow(axa, 6.00, 2.38, 6.00, 1.95, color=GREEN_DARK, lw=1.1)

    box(axa, 0.35, 1.35, 3.30, 0.54,
        "Choice logits\n" + r"$z_A,\ldots,z_E$", GREEN_PALE, fs=6.8)
    box(axa, 4.35, 1.35, 3.30, 0.54,
        "Restricted\nsoftmax over A--E", GREEN_LIGHT, fs=6.7)
    box(axa, 8.35, 1.35, 3.30, 0.54,
        "Extract\n" + r"$P(\mathrm{gold}\mid q,r_{1:f})$",
        GREEN, tc="white", fs=6.6)
    arrow(axa, 3.65, 1.62, 4.35, 1.62, color=GREEN_DARK, lw=1.0)
    arrow(axa, 7.65, 1.62, 8.35, 1.62, color=GREEN_DARK, lw=1.0)

    # Panel B: the original training-target illustration, uniformly reduced.
    columns = [2.0, 6.0, 10.0]
    box(
        axb, 0.45, 4.35, 11.10, 0.58,
        "Masked prompt: question + alternatives (A--E)",
        GREEN_PALE, fs=7.3
    )
    for center in columns:
        arrow(axb, center, 4.32, center, 3.82, color=INK, lw=1.1)

    titles = ["Pure SFT", "distill-SFT", "Step-by-step"]
    for center, title in zip(columns, titles):
        axb.text(
            center, 3.72, title, ha="center", va="top", fontsize=7.6,
            fontweight="bold",
            color=GREEN if title == "distill-SFT" else INK
        )

    box(
        axb, 0.85, 2.18, 2.30, 0.72,
        "Answer: X\n(weight 1)", GREEN, tc="white", fs=6.5
    )
    axb.text(
        2.0, 1.93, "one answer-only target",
        ha="center", va="top", fontsize=5.8, style="italic", color=GREY
    )

    axb.add_patch(FancyBboxPatch(
        (4.54, 1.58), 2.92, 1.54,
        boxstyle="round,pad=0.04,rounding_size=0.09",
        fc="none", ec=GREEN, lw=2.0, zorder=2
    ))
    box(
        axb, 4.70, 2.40, 2.60, 0.62,
        "Answer: X\n(weight 1)", GREEN, tc="white", fs=5.9
    )
    box(
        axb, 4.70, 1.70, 2.60, 0.58,
        "rationale\n(weight alpha)", GREEN_LIGHT, fs=5.6
    )
    axb.text(
        6.0, 1.45, "one joint target",
        ha="center", va="top", fontsize=5.8, style="italic",
        color=GREEN, fontweight="bold"
    )

    box(
        axb, 8.70, 2.40, 2.60, 0.62,
        "Answer: X\n(weight 1)", GREEN, tc="white", fs=5.9
    )
    box(
        axb, 8.70, 1.70, 2.60, 0.58,
        "rationale\n(weight alpha)", GREEN_LIGHT, fs=5.6
    )
    axb.text(
        10.0, 1.50, "two separate tasks per eligible item",
        ha="center", va="top", fontsize=5.8, style="italic", color=GREY
    )

    save(fig, "fig2_techniques")


def fig_method_workflow():
    """Three-panel method schematic linking the probability trace to training."""
    fig, ax = plt.subplots(figsize=(7.4, 5.35))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8.35)
    ax.axis("off")

    # Panel A: rationale-conditioned probability tracing.
    ax.text(0.18, 8.04, "A", fontsize=10.5, fontweight="bold", color=GREEN_DARK)
    ax.text(
        0.55, 8.04, "Pre-training probability tracing",
        fontsize=9.2, fontweight="bold", va="center"
    )
    box(
        ax, 0.25, 6.35, 2.00, 1.05,
        "Question + rationale\nfinal answer span\nremoved",
        GREEN_PALE, fs=6.6
    )
    box(
        ax, 2.70, 6.35, 2.15, 1.05,
        "Reveal prefixes\n0, .15, .30, ... , 1",
        GREEN_LIGHT, fs=7.4
    )
    box(
        ax, 5.30, 6.35, 1.55, 1.05,
        "Baseline\nQwen3-4B",
        GREEN, tc="white", fs=7.4
    )
    box(
        ax, 7.30, 6.35, 1.55, 1.05,
        "Choice logits\nzA ... zE",
        GREEN_PALE, fs=7.2
    )
    box(
        ax, 9.30, 6.35, 2.35, 1.05,
        "Restricted softmax\nextract P(gold)",
        GREEN_LIGHT, fs=7.4
    )
    for x1, x2 in [(2.25, 2.70), (4.85, 5.30), (6.85, 7.30), (8.85, 9.30)]:
        arrow(ax, x1, 6.88, x2, 6.88, color=GREEN_DARK, lw=1.2)

    ax.text(
        1.25, 6.14, "coherent or fixed cross-question shuffle",
        ha="center", va="top", fontsize=5.9, style="italic", color=GREY
    )
    arrow(ax, 10.47, 6.32, 10.47, 5.70, color=GREEN_DARK, lw=1.2)
    box(
        ax, 2.55, 4.67, 6.95, 0.92,
        "Repeat across fractions and conditions\n"
        "stratify teacher correct/wrong; unique-stem cluster bootstrap",
        GREEN_PALE, fs=6.3, bold=False
    )
    box(
        ax, 10.05, 4.72, 1.55, 0.82,
        "Figure 2\ntrajectories",
        GREEN, tc="white", fs=6.5
    )
    arrow(ax, 9.50, 5.13, 10.05, 5.13, color=GREEN_DARK, lw=1.1)
    ax.plot([0.15, 11.85], [4.28, 4.28], color=GREEN_LIGHT, lw=1.1)

    # Panel B: training target constructions retained from the original figure.
    ax.text(0.18, 3.95, "B", fontsize=10.5, fontweight="bold", color=GREEN_DARK)
    ax.text(
        0.55, 3.95, "Answer-verified training targets",
        fontsize=9.2, fontweight="bold", va="center"
    )
    ax.text(0.32, 3.32, "Pure SFT", ha="left", va="center",
            fontsize=7.0, fontweight="bold")
    ax.text(0.32, 2.31, "Proposed\ndistill-SFT", ha="left", va="center",
            fontsize=6.5, fontweight="bold", color=GREEN)
    ax.text(0.32, 1.39, "Step-by-step", ha="left", va="center",
            fontsize=6.8, fontweight="bold")

    # Pure SFT row.
    box(ax, 2.70, 3.04, 1.65, 0.56, "Answer X  (1)", GREEN, tc="white", fs=6.6)

    # Proposed joint target row.
    ax.add_patch(FancyBboxPatch(
        (2.58, 1.91), 4.45, 0.82,
        boxstyle="round,pad=0.035,rounding_size=0.07",
        fc="none", ec=GREEN, lw=1.8, zorder=2
    ))
    box(ax, 2.73, 2.04, 2.60, 0.54, "rationale  (alpha)", GREEN_LIGHT, fs=6.5)
    box(ax, 5.45, 2.04, 1.43, 0.54, "Answer X  (1)", GREEN, tc="white", fs=6.0)

    # Two-task baseline row.
    box(ax, 2.70, 1.11, 1.78, 0.56, "label task:\nAnswer X", GREEN, tc="white", fs=5.9)
    ax.text(4.65, 1.39, "+", ha="center", va="center", fontsize=8, color=GREY)
    box(ax, 4.84, 1.11, 2.05, 0.56, "rationale task\n(alpha)", GREEN_LIGHT, fs=5.9)
    ax.text(
        3.78, 0.56,
        "Prompt tokens are masked.\n"
        "Only answer-verified rationales carry rationale loss.",
        ha="center", va="center", fontsize=5.9, color=GREY
    )

    # Panel C: why the middle grid point is alpha=.3 rather than .5.
    ax.plot([7.35, 7.35], [0.45, 4.05], color=GREEN_LIGHT, lw=1.1)
    ax.text(7.60, 3.95, "C", fontsize=10.5, fontweight="bold", color=GREEN_DARK)
    ax.text(
        7.97, 3.95, "Loss-mass spacing",
        fontsize=9.2, fontweight="bold", va="center"
    )
    ax.text(
        7.60, 3.53,
        r"answer share = $n_A / (\alpha n_R + n_A)$",
        fontsize=6.8, color=GREY
    )

    rows = [
        (1.0, 0.02, True),
        (0.5, 0.04, False),
        (0.3, 0.06, True),
        (0.1, 0.17, True),
    ]
    bar_x, bar_w = 8.95, 2.18
    for idx, (alpha, answer_share, selected) in enumerate(rows):
        y = 2.92 - idx * 0.65
        rationale_share = 1.0 - answer_share
        edge = GREEN_DARK if alpha == 0.3 else (INK if selected else GREY)
        line_style = "-" if selected else "--"
        ax.text(
            7.62, y + 0.17, rf"$\alpha={alpha:g}$",
            fontsize=6.8, va="center",
            color=INK if selected else GREY,
            fontweight="bold" if alpha == 0.3 else "normal"
        )
        ax.add_patch(Rectangle(
            (bar_x, y), bar_w * rationale_share, 0.34,
            facecolor=GREEN_LIGHT if selected else "#E5E8E6",
            edgecolor="none", zorder=2
        ))
        ax.add_patch(Rectangle(
            (bar_x + bar_w * rationale_share, y),
            bar_w * answer_share, 0.34,
            facecolor=GREEN if selected else GREY,
            edgecolor="none", zorder=2
        ))
        ax.add_patch(Rectangle(
            (bar_x, y), bar_w, 0.34,
            facecolor="none", edgecolor=edge, linewidth=1.2,
            linestyle=line_style, zorder=3
        ))
        ax.text(
            11.28, y + 0.17, f"{round(answer_share * 100):d}% answer",
            fontsize=6.6, va="center", color=INK if selected else GREY,
            fontweight="bold" if alpha == 0.3 else "normal"
        )
        if not selected:
            ax.text(
                8.08, y - 0.12, "not evaluated",
                fontsize=5.9, color=GREY, style="italic"
            )

    ax.text(
        9.72, 0.52,
        r"$\alpha=0.3$: about $3\times$ the answer share of $\alpha=1$"
        "\n" r"$\alpha=0.5$: only about $2\times$",
        ha="center", va="center", fontsize=5.8, color=GREEN_DARK,
        fontweight="bold"
    )

    save(fig, "fig_method_workflow")


if __name__ == "__main__":
    fig1_design()
    fig2_techniques()
    fig_method_workflow()
    print("done")

