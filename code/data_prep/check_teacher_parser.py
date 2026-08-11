#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Divergence report between the two teacher-answer parsers.

Prints how often they disagree, how the teacher's accuracy moves, and -- the part
that matters -- a sample of the actual disagreements with the tail of each
generation, so a human can read them and judge which letter the teacher really
settled on. Regular expressions cannot settle that; people can.

Writes every disagreement to a CSV for manual audit.

Measured over the 4,260 archived generations:
    disagreement          77 items (1.81%)
    explicit-answer stratum   87.2% of items, parsers agree on 99.70%
    ambiguous stratum         12.8% of items, parsers agree on 87.87%
    teacher accuracy      0.749 (original) -> 0.743 (audit)

The audit parser is not uniformly better -- it recovers 18 items the original got
wrong and loses 44 it got right. That is why the released dataset ships both
verdicts and treats neither as ground truth. Logit-scoring the teacher over A-E
would sidestep the problem entirely and is the right long-term fix.

Usage
  python check_teacher_parser.py --teacher-file results/teacher_generations.pkl
"""
import argparse
import re
import sys

import pandas as pd

# strong "conclusion" phrases; the LAST occurrence wins
STRONG = re.compile(
    r'(?:resposta|alternativa|op[çc][ãa]o|letra)\s*(?:correta|final|é|e)?'
    r'[\s:é\-\.]*\*{0,2}\s*\(?\s*([A-E])\b', re.IGNORECASE)
BOLD = re.compile(r'\*\*\s*\(?\s*([A-E])\s*\)?\s*\*{0,2}')
PAREN = re.compile(r'(?<![A-Za-z0-9])([A-E])\s*\)')


def parse_final(text):
    """The answer that appears LAST, which for this teacher is its conclusion."""
    if not isinstance(text, str) or not text.strip():
        return None
    candidates = [(m.start(), m.group(1).upper()) for m in STRONG.finditer(text)]
    bold = list(BOLD.finditer(text))
    if bold:
        candidates.append((bold[-1].start(), bold[-1].group(1).upper()))
    if candidates:
        return max(candidates, key=lambda x: x[0])[1]
    paren = list(PAREN.finditer(text))
    return paren[-1].group(1).upper() if paren else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher-file", required=True)
    ap.add_argument("--out", default="teacher_parser_divergences.csv")
    ap.add_argument("--show", type=int, default=20)
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    df = pd.read_pickle(args.teacher_file)
    df["audit"] = [parse_final(o) or parse_final(p)
                   for o, p in zip(df["output"], df["pensamento"])]
    df["original"] = df["letra"].astype(str).str.strip().str.upper()
    df["gold"] = df["resposta"].astype(str).str.strip().str.upper()
    df["audit"] = df["audit"].fillna("?")

    n = len(df)
    disagree = df["audit"] != df["original"]
    df["correct_original"] = df["original"] == df["gold"]
    df["correct_audit"] = df["audit"] == df["gold"]
    flipped = df["correct_original"] != df["correct_audit"]

    print(f"total={n}  audit parser found nothing on {(df['audit']=='?').sum()} items")
    print(f"parsers disagree      : {disagree.sum()} ({100*disagree.mean():.2f}%)")
    print(f"teacher verdict flips : {flipped.sum()} ({100*flipped.mean():.2f}%)")
    print(f"teacher accuracy      : original={df['correct_original'].mean():.4f}  "
          f"audit={df['correct_audit'].mean():.4f}")
    print(f"  original right / audit wrong: "
          f"{int((df['correct_original'] & ~df['correct_audit']).sum())}")
    print(f"  audit right / original wrong: "
          f"{int((df['correct_audit'] & ~df['correct_original']).sum())}")
    print()
    print("sample disagreements -- read the tail and judge which letter is the "
          "teacher's FINAL answer:")
    for _, r in df[disagree].head(args.show).iterrows():
        tail = str(r["output"])[-100:].replace("\n", " ")
        flag = "FLIP" if r["correct_original"] != r["correct_audit"] else "    "
        print(f"  original={r['original']} audit={r['audit']} gold={r['gold']} "
              f"[{flag}] | ...{tail}")

    df[disagree][["original", "audit", "gold", "output"]].to_csv(args.out, index=False)
    print(f"\nwrote {args.out} (every disagreement, for manual audit)")


if __name__ == "__main__":
    main()
