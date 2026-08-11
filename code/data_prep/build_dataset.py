#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the released dataset files from the internal source archives.

This is the provenance record for data/: it documents exactly how the two
published files were produced, so that the release is auditable rather than
merely asserted.

Inputs (internal, not redistributed)
  multiple_choice.json          the full item bank, all exam years
  qwen32_qa_thinkings_fixed.pkl teacher generations with the corrected answer
                                parser (see fix_teacher_answer_parser.py)

Outputs (published)
  data/questions_2024_2025.jsonl   one item per line: id, source, exam_year,
                                   stem, options, gold_answer, has_image
  data/teacher_rationales.jsonl    one item per line: id, rationale,
                                   teacher_answer_raw, teacher_answer_fixed,
                                   teacher_correct, n_tokens

Scope: the study population is the 2024 and 2025 examinations. Earlier years
(2011-2023) exist in the source bank but were never used in any experiment, and
are not part of this release.

The two files are joined on the item `id`. Teacher rationales are matched to
items by a whitespace-normalised stem, the same key used by
regenerate_splits.py, because the teacher archive does not carry the item id.

Usage
  python build_release_dataset.py \
      --item-bank    /path/to/multiple_choice.json \
      --teacher-file /path/to/qwen32_qa_thinkings_fixed.pkl \
      --out-dir      ../../data
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd

STUDY_YEARS = [2024, 2025]


def normalise_stem(s):
    """Join key: collapse whitespace and lowercase. Must stay identical to the key
    used in regenerate_splits.py or the two files will disagree."""
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--item-bank", required=True)
    ap.add_argument("--teacher-file", required=True)
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- items -----------------------------------------------------------
    bank = pd.DataFrame(json.load(open(args.item_bank, encoding="utf-8")))
    print(f"item bank: {len(bank)} items across {bank['prova'].nunique()} exam years")
    items = bank[bank["prova"].isin(STUDY_YEARS)].copy()
    print(f"study population ({'/'.join(map(str, STUDY_YEARS))}): {len(items)} items")
    print(items["origem"].value_counts().to_string())

    items["id"] = items["id"].astype(str)
    questions = [{
        "id": r["id"],
        "source": r["origem"],
        "exam_year": int(r["prova"]),
        "stem": r["enunciado"],
        "options": (r["alternativas"] if isinstance(r["alternativas"], dict)
                    else json.loads(str(r["alternativas"]).replace("'", '"'))),
        "gold_answer": str(r["resposta"]).strip().upper(),
        "has_image": bool(r["contains_img"]),
    } for _, r in items.iterrows()]

    qpath = out_dir / "questions_2024_2025.jsonl"
    with open(qpath, "w", encoding="utf-8") as f:
        for rec in questions:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\nwrote {qpath} ({len(questions)} items, "
          f"{sum(r['has_image'] for r in questions)} of them reference an image)")

    # ---- teacher rationales ----------------------------------------------
    teacher = pd.read_pickle(args.teacher_file)
    print(f"\nteacher archive: {len(teacher)} generations")

    stem_to_id = {normalise_stem(r["stem"]): r["id"] for r in questions}
    teacher["_key"] = teacher["enunciado"].map(normalise_stem)
    teacher["id"] = teacher["_key"].map(stem_to_id)

    unmatched = int(teacher["id"].isna().sum())
    if unmatched:
        print(f"[warning] {unmatched} teacher rows did not match an item stem "
              f"and are excluded")
    teacher = teacher[teacher["id"].notna()].copy()

    # `teacher_correct` uses the ORIGINAL parser, because that is what defined the
    # A/B/C/D partition every experiment in the paper was run on. The audit
    # parser's answer is published alongside it as `teacher_answer_audit` so the
    # disagreement is inspectable, but it is NOT the study label -- swapping it in
    # would silently redefine the subsets and detach the data from the results.
    gold = {r["id"]: r["gold_answer"] for r in questions}
    rationales = []
    for _, r in teacher.iterrows():
        original = str(r["letra"]).strip().upper()
        audit = str(r["letra_fix"]).strip().upper()
        rationales.append({
            "id": r["id"],
            "rationale": r["pensamento"],
            "teacher_answer": original,
            "teacher_answer_audit": audit,
            "parsers_disagree": bool(original != audit),
            "teacher_correct": bool(original == gold[r["id"]]),
            "teacher_correct_audit": bool(audit == gold[r["id"]]),
            "n_tokens": int(r["n_tokens"]) if pd.notna(r.get("n_tokens")) else None,
        })

    rpath = out_dir / "teacher_rationales.jsonl"
    with open(rpath, "w", encoding="utf-8") as f:
        for rec in rationales:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n = len(rationales)
    n_correct = sum(r["teacher_correct"] for r in rationales)
    n_correct_audit = sum(r["teacher_correct_audit"] for r in rationales)
    n_disagree = sum(r["parsers_disagree"] for r in rationales)
    print(f"wrote {rpath} ({n} rationales)")
    print(f"  teacher accuracy, ORIGINAL parser (the study label): {n_correct/n:.4f}")
    print(f"  teacher accuracy, AUDIT parser                     : {n_correct_audit/n:.4f}")
    print(f"  items where the two parsers disagree: {n_disagree} ({100*n_disagree/n:.2f}%)")
    print(f"    original right / audit wrong: "
          f"{sum(r['teacher_correct'] and not r['teacher_correct_audit'] for r in rationales)}")
    print(f"    audit right / original wrong: "
          f"{sum(r['teacher_correct_audit'] and not r['teacher_correct'] for r in rationales)}")


if __name__ == "__main__":
    main()
