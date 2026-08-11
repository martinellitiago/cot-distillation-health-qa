#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit parser for the teacher's stated answer.

The teacher (Qwen3-32B) answers in free-form Portuguese, so its chosen letter has
to be extracted with regular expressions. The original extractor used the FIRST
match in its bold-letter fallbacks, which returns an early letter when the model
marks each option in bold while deliberating -- a bias toward A, the first listed
alternative.

This parser extracts the FINAL answer instead, with an explicit priority order:

  P1  an explicit phrase: 'Resposta [correta/final][:] X', or
      'alternativa/opção/letra correta [é/:] [a/o] X'          (last occurrence)
  P2  the last '**X)**' -- the canonical final-answer format. Requiring the
      closing paren avoids matching 'vitamina **D**' or 'grupo **A**'.
  P3  a bold '**X**' but only within the last 60 characters, i.e. a genuine
      conclusion rather than an aside
  P4  fallback: the last 'X)' anywhere

READ THIS BEFORE USING IT AS A LABEL.

Against the answer key, this parser is not uniformly better than the original --
it is differently wrong. Over the 4,260 teacher generations the two disagree on
77 items (1.81%):

    original right / audit wrong : 44
    audit right / original wrong : 18
    both wrong                   : 15

so teacher accuracy moves from 0.749 (original) to 0.743 (audit), i.e. DOWN.
Extracting a final answer from free text is brittle in both directions, and
neither parser is ground truth.

Consequently the released dataset keeps the ORIGINAL parser's verdict as
`teacher_correct` -- it is what the paper's A/B/C/D partition was built on -- and
ships this parser's verdict alongside as `teacher_correct_audit`. Substituting
one for the other would silently redefine the strata.

The robust argument does not depend on resolving this: the strata are defined per
item, and every technique is compared on the SAME partition, so a mislabelled
item is mislabelled identically in every arm and cannot create a difference
between them. Answer-extraction brittleness is a known problem in this
literature; logit-scoring the teacher over A-E would remove it entirely and is
the right long-term fix.

Usage
  python fix_teacher_answer_parser.py --teacher-file results/teacher_generations.pkl
"""
import argparse
import re

_EXPLICIT_PHRASE = [
    re.compile(r'resposta\s*(?:correta|final)?\s*(?:é|e|:|-|\s)*(?:a|o)?\s*'
               r'\*{0,2}\s*\(?\s*([A-E])\b', re.IGNORECASE),
    re.compile(r'(?:alternativa|op[çc][ãa]o|letra)\s*correta\s*(?:é|e|:|-|\s)*(?:a|o)?\s*'
               r'\*{0,2}\s*\(?\s*([A-E])\b', re.IGNORECASE),
]
_BOLD_WITH_PAREN = re.compile(r'\*\*\s*\(?\s*([A-E])\s*\)')   # **X)
_BOLD_LETTER = re.compile(r'\*\*\s*([A-E])\s*\*\*')           # **X**
_LETTER_PAREN = re.compile(r'(?<![A-Za-z0-9])([A-E])\s*\)')   # X)

_CONCLUSION_WINDOW = 60      # chars from the end within which a bare bold letter counts


def extract_final_answer(text):
    """Return the letter the teacher settled on, or None."""
    if not isinstance(text, str) or not text.strip():
        return None

    candidates = []                                   # (priority, position, letter)
    for rx in _EXPLICIT_PHRASE:
        for m in rx.finditer(text):
            candidates.append((3, m.start(), m.group(1).upper()))
    for m in _BOLD_WITH_PAREN.finditer(text):
        candidates.append((2, m.start(), m.group(1).upper()))
    for m in _BOLD_LETTER.finditer(text):
        if m.start() >= len(text) - _CONCLUSION_WINDOW:
            candidates.append((1, m.start(), m.group(1).upper()))

    if candidates:
        # highest priority first, then the LATEST occurrence at that priority
        return max(candidates, key=lambda c: (c[0], c[1]))[2]

    matches = list(_LETTER_PAREN.finditer(text))
    return matches[-1].group(1).upper() if matches else None


def parse_row(output, reasoning):
    """The answer usually sits in `output`; fall back to the reasoning field."""
    return extract_final_answer(output) or extract_final_answer(reasoning)


def main():
    import sys
    import pandas as pd

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher-file", required=True,
                    help="pickle with columns output, pensamento, letra, resposta")
    ap.add_argument("--out", default=None, help="where to write the annotated pickle")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    df = pd.read_pickle(args.teacher_file)
    df["letra_fix"] = [parse_row(o, p) for o, p in zip(df["output"], df["pensamento"])]
    original = df["letra"].astype(str).str.strip().str.upper()
    gold = df["resposta"].astype(str).str.strip().str.upper()
    audit = df["letra_fix"].fillna("?").astype(str).str.strip().str.upper()

    disagree = original != audit
    acc_original = (original == gold).mean()
    acc_audit = (audit == gold).mean()

    print(f"total={len(df)}  audit parser found nothing on {(audit == '?').sum()} items")
    print(f"parsers disagree on {disagree.sum()} items ({100*disagree.mean():.2f}%)")
    print(f"teacher accuracy: original={acc_original:.4f}  audit={acc_audit:.4f} "
          f"({100*(acc_audit-acc_original):+.2f} pp)")
    print(f"  original right / audit wrong: {int(((original==gold) & (audit!=gold)).sum())}")
    print(f"  audit right / original wrong: {int(((audit==gold) & (original!=gold)).sum())}")
    print(f"  both wrong                  : "
          f"{int(((audit!=gold) & (original!=gold) & disagree).sum())}")

    # regression probes: the exact failures this parser was written to fix
    for probe, want in [("alternativa correta é a E", "E"),
                        ("Resposta: C) IV A", "C"),
                        ("**C) A ineficiência", "C")]:
        got = extract_final_answer(probe)
        status = "ok" if got == want else "FAIL"
        print(f"  probe {probe!r:40s} -> {got} (want {want}) [{status}]")

    if args.out:
        df.to_pickle(args.out)
        print(f"wrote {args.out} (with the letra_fix column)")


if __name__ == "__main__":
    main()
