#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rebuild the A/B/C/D labels in the existing split files after re-parsing the
teacher's answer -- WITHOUT retraining, re-evaluating the student, or re-seeding.

Why this is cheap: the stratum is a GLOBAL per-question property
(teacher-correct x student-correct). The ten seeds only control the 85/15
partition, so a label correction is a single relabel that applies identically to
every seed. And the student's correctness is NOT affected by a teacher-parser
change; it is recoverable from the existing subset code, since A and C are
exactly the strata where the student was right.

Per seed and per split file:
  1. student_correct <- (old subset_code in {A, C})
  2. teacher_correct <- (re-parsed teacher letter == gold)
  3. new subset      <- f(teacher_correct, student_correct)
  4. write the corrected file and report what moved

WATCH THE B<->D FLIPS. Those two strata live in DIFFERENT pools: B belongs to the
ABC pool and D to the hard pool. An item that flips between them has to physically
move between split files, which this script flags but does not do -- move it
yourself or regenerate the pools from the corrected labels.

NOTE: the paper's results use the ORIGINAL teacher labels. This script exists so
the relabelling is reproducible, not because the released numbers depend on it;
see `teacher_parser_fixed.py` for why neither parser is ground truth.

Usage
  python regenerate_splits.py --splits-dir data/splits_stage2 \
      --teacher-file results/teacher_generations_fixed.pkl \
      --out-dir data/splits_stage2_relabelled
"""
import argparse
import re
from pathlib import Path

import pandas as pd

SEEDS = [8, 12, 17, 23, 25, 31, 37, 44, 52, 61]


def normalise_stem(s):
    """Join key: collapse whitespace, lowercase. Must match build_dataset.py."""
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def subset_of(teacher_ok, student_ok):
    if teacher_ok and student_ok:
        return "A"
    if teacher_ok and not student_ok:
        return "B"
    if not teacher_ok and student_ok:
        return "C"
    return "D"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits-dir", default="data/splits_stage2")
    ap.add_argument("--teacher-file", default="results/teacher_generations_fixed.pkl")
    ap.add_argument("--out-dir", default="data/splits_stage2_relabelled")
    ap.add_argument("--subset-col", default="subset_code")
    ap.add_argument("--gold-col", default="resposta")
    ap.add_argument("--answer-col", default="letra_fix",
                    help="teacher answer column to relabel from")
    args = ap.parse_args()

    teacher = pd.read_pickle(args.teacher_file)
    teacher["_key"] = teacher["enunciado"].map(normalise_stem)
    answer = teacher[args.answer_col].astype(str).str.strip().str.upper()
    gold = teacher[args.gold_col].astype(str).str.strip().str.upper()
    teacher_correct_by_stem = dict(zip(teacher["_key"], answer == gold))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    total_changed = total_missing = total_cross_pool = 0

    for seed in SEEDS:
        for name in (f"df_train_seed{seed}.pkl",
                     f"df_test_abc_seed{seed}.pkl",
                     f"df_test_hard_seed{seed}.pkl"):
            path = Path(args.splits_dir) / name
            if not path.exists():
                print(f"[skip] {path} does not exist")
                continue
            df = pd.read_pickle(path).copy()
            old = df[args.subset_col].astype(str).str.upper()
            student_ok = old.isin(["A", "C"])          # unaffected by the parser
            keys = df["enunciado"].map(normalise_stem)
            teacher_ok = keys.map(teacher_correct_by_stem)

            missing = teacher_ok.isna()
            total_missing += int(missing.sum())
            # no match in the teacher archive -> keep the old verdict
            teacher_ok = teacher_ok.fillna(old.isin(["A", "B"]))

            new = pd.Series([subset_of(t, s) for t, s in zip(teacher_ok, student_ok)],
                            index=df.index)
            changed = new != old
            total_changed += int(changed.sum())

            cross_pool = (((old == "B") & (new == "D")) | ((old == "D") & (new == "B")))
            total_cross_pool += int(cross_pool.sum())

            df[args.subset_col] = new
            df.to_pickle(out_dir / name)
            if changed.sum() or missing.sum():
                print(f"{name}: changed {changed.sum()}, unmatched {missing.sum()}, "
                      f"B<->D {cross_pool.sum()}")

    print(f"\nTOTAL: relabelled={total_changed}  unmatched={total_missing}  "
          f"B<->D (cross-pool)={total_cross_pool}")
    print(f"written to {out_dir}/ (same seeds, same items)")
    if total_cross_pool:
        print("WARNING: B<->D flips cross the ABC/hard pool boundary. Move those items "
              "between df_test_abc/df_train and df_test_hard, or regenerate the pools "
              "from the corrected labels -- relabelling alone leaves them in the wrong pool.")


if __name__ == "__main__":
    main()
