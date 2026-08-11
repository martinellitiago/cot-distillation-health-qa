#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent paired tests for the paper's key Stage-2 claims.

This script exists separately from stage2_analyze_full.py because it computes
the seed-level accuracies directly from the inference files, with no shared code
path -- so a bug in the main analysis pipeline cannot silently propagate into the
headline numbers. It is the source of the abstract's

    distill-SFT beats step-by-step on subset B under reasoning
    (0.427 vs 0.405; paired p = 0.025)

which is a SUBSET-restricted test. Earlier versions of the main analysis computed
the paired t only on the whole split, where this effect is invisible; that gap is
now closed there too, but this file remains the independent check.

For each claim it reports the paired t across the ten seeds, the Wilcoxon
signed-rank as sensitivity, and how many seeds moved in the claimed direction --
because a p-value with three seeds carrying it is a different fact from a
p-value with nine.

Usage
  python key_tests_independent.py --root outputs_stage2
"""
import argparse
from pathlib import Path

import pandas as pd
from scipy import stats

SEEDS = [8, 12, 17, 23, 25, 31, 37, 44, 52, 61]

# condition label -> (run-directory technique, alpha suffix)
# `sft_puro` is the archived Portuguese identifier; `pure_sft` the current one.
CONDITION_DIRS = {
    "distill_sft":       ("distill_sft", ""),
    "distill_sft_a0.1":  ("distill_sft", "_a01"),
    "distill_sft_a0.3":  ("distill_sft", "_a03"),
    "pure_sft":          ("pure_sft", ""),
    "step_by_step":      ("step_by_step", ""),
    "step_by_step_a0.1": ("step_by_step", "_a01"),
    "step_by_step_a0.3": ("step_by_step", "_a03"),
}
LEGACY_DIR_NAMES = {"pure_sft": "sft_puro"}
LEGACY_COLUMNS = {"acertou": "correct"}

# (claim, condition_1, condition_2, split, regime, subset)
# The reported difference is always condition_2 minus condition_1.
COMPARISONS = [
    ("alpha repair, hard direct",
     "distill_sft", "distill_sft_a0.1", "hard", "answer_only", "ALL"),
    ("alpha 0.3 repair, hard direct",
     "distill_sft", "distill_sft_a0.3", "hard", "answer_only", "ALL"),
    ("closed gap vs pure SFT, hard direct",
     "distill_sft_a0.1", "pure_sft", "hard", "answer_only", "ALL"),
    ("closed gap vs step-by-step, hard direct",
     "distill_sft_a0.1", "step_by_step", "hard", "answer_only", "ALL"),
    ("reasoning preserved, hard",
     "distill_sft", "distill_sft_a0.1", "hard", "reasoning", "ALL"),
    ("reasoning preserved, ABC",
     "distill_sft", "distill_sft_a0.1", "abc", "reasoning", "ALL"),
    ("step-by-step regularisation vs pure SFT",
     "pure_sft", "step_by_step_a0.1", "hard", "answer_only", "ALL"),
    ("step-by-step reasoning cost at alpha 0.3",
     "step_by_step", "step_by_step_a0.3", "abc", "reasoning", "ALL"),
    ("distill-SFT vs step-by-step on subset B under reasoning",
     "step_by_step", "distill_sft", "abc", "reasoning", "B"),
]


def run_dir(root: Path, condition: str, seed: int, epochs: int):
    technique, suffix = CONDITION_DIRS[condition]
    candidates = [technique] + ([LEGACY_DIR_NAMES[technique]]
                                if technique in LEGACY_DIR_NAMES else [])
    for name in candidates:
        d = root / f"{name}_seed{seed}_ep{epochs}{suffix}"
        if d.exists():
            return d
    raise FileNotFoundError(
        f"no run directory for {condition} seed {seed} under {root} "
        f"(tried {candidates})")


def seed_accuracies(root, condition, split, regime, subset="ALL", epochs=2):
    values = []
    for seed in SEEDS:
        f = run_dir(root, condition, seed, epochs) / f"infer_{split}_{regime}.csv"
        frame = pd.read_csv(f).rename(columns=LEGACY_COLUMNS)
        if subset != "ALL":
            frame = frame[frame["subset_code"].astype(str).str.upper() == subset]
        if frame.empty:
            raise ValueError(f"{f} has no rows for subset {subset}")
        values.append(float(frame["correct"].mean()))
    return values


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="outputs_stage2")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--out", default="key_significance_tests.csv")
    args = ap.parse_args()
    root = Path(args.root)

    rows = []
    for claim, first, second, split, regime, subset in COMPARISONS:
        a = seed_accuracies(root, first, split, regime, subset, args.epochs)
        b = seed_accuracies(root, second, split, regime, subset, args.epochs)
        t = stats.ttest_rel(b, a)
        w = stats.wilcoxon(b, a, alternative="two-sided")
        rows.append({
            "claim": claim,
            "cond_1": first, "cond_2": second,
            "split": split, "regime": regime, "subset": subset,
            "acc_1": round(sum(a) / len(a), 4),
            "acc_2": round(sum(b) / len(b), 4),
            "delta_2_minus_1": round(sum(y - x for x, y in zip(a, b)) / len(a), 4),
            "paired_t_p": round(float(t.pvalue), 4),
            "wilcoxon_p": round(float(w.pvalue), 4),
            "seeds_favouring_2": sum(y > x for x, y in zip(a, b)),
            "ties": sum(y == x for x, y in zip(a, b)),
            "n_seeds": len(a),
        })

    result = pd.DataFrame(rows)
    result.to_csv(args.out, index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 50):
        print(result.to_string(index=False))
    print(f"\nsaved={args.out}")
    print("\nNote: non-significant rows mean the difference was NOT DETECTED. They do "
          "not establish equivalence -- no equivalence margin was prespecified.")


if __name__ == "__main__":
    main()
