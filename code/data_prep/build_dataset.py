#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the released dataset files from the internal source archives.

This is the provenance record for data/: it documents exactly how the two
published files were produced, so that the release is auditable rather than
merely asserted.

Inputs
  one of:
    multiple_choice.json             internal full item bank, all exam years; or
    data/questions_2024_2025.jsonl   the already released question table
  qwen32_qa_thinkings.pkl            internal teacher generations. If `letra_fix`
                                     is absent, the audit parser is run here.

Outputs (published)
  data/questions_2024_2025.jsonl   one item per line: id, source, exam_year,
                                   stem, options, gold_answer, has_image
  data/teacher_rationales.jsonl    one item per line: id, rationale,
                                   teacher_answer, teacher_answer_audit,
                                   teacher_correct, teacher_correct_audit,
                                   n_tokens

Scope: the study population is the 2024 and 2025 examinations. Earlier years
(2011-2023) exist in the source bank but were never used in any experiment, and
are not part of this release.

The two files are joined on the item `id`. Because the teacher archive does not
carry that id, teacher rows are matched one-to-one on normalised stem, complete
option set, and gold label. A stem by itself is deliberately not used as an id.

Usage
  python code/data_prep/build_dataset.py --item-bank /path/to/multiple_choice.json --teacher-file /path/to/qwen32_qa_thinkings.pkl --out-dir data

To repair an existing checkout without the internal item bank:
  python code/data_prep/build_dataset.py --questions-jsonl data/questions_2024_2025.jsonl --teacher-file /path/to/qwen32_qa_thinkings.pkl --out-dir data
"""
import argparse
import ast
import hashlib
import json
import re
from collections import defaultdict, deque
from pathlib import Path

import pandas as pd

STUDY_YEARS = [2024, 2025]


def normalise_stem(s):
    """Join key: collapse whitespace and lowercase. Must stay identical to the key
    used in regenerate_splits.py or the two files will disagree."""
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def parse_options(value):
    """Return an A--E dictionary without guessing by quote replacement."""
    if isinstance(value, dict):
        return {str(k).strip().upper(): str(v) for k, v in value.items()}
    parsed = ast.literal_eval(str(value))
    if not isinstance(parsed, dict):
        raise TypeError(f"options must be a mapping, got {type(parsed).__name__}")
    return {str(k).strip().upper(): str(v) for k, v in parsed.items()}


def normalise_options(value):
    options = parse_options(value)
    return tuple((k, normalise_stem(v)) for k, v in sorted(options.items()))


def cluster_id(stem):
    """Stable identifier for resampling repeated normalised question stems."""
    digest = hashlib.sha256(normalise_stem(stem).encode("utf-8")).hexdigest()[:16]
    return f"stem-{digest}"


def item_key(stem, options, gold):
    return (
        normalise_stem(stem),
        normalise_options(options),
        str(gold).strip().upper(),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--item-bank")
    source.add_argument("--questions-jsonl")
    ap.add_argument("--teacher-file", required=True)
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- items -----------------------------------------------------------
    if args.item_bank:
        with open(args.item_bank, encoding="utf-8") as handle:
            bank = pd.DataFrame(json.load(handle))
        print(f"item bank: {len(bank)} items across {bank['prova'].nunique()} exam years")
        items = bank[bank["prova"].isin(STUDY_YEARS)].copy()
        print(f"study population ({'/'.join(map(str, STUDY_YEARS))}): {len(items)} items")
        print(items["origem"].value_counts().to_string())

        items["id"] = items["id"].astype(str)
        questions = [{
            "id": r["id"],
            "question_cluster_id": cluster_id(r["enunciado"]),
            "source": r["origem"],
            "exam_year": int(r["prova"]),
            "stem": r["enunciado"],
            "options": parse_options(r["alternativas"]),
            "gold_answer": str(r["resposta"]).strip().upper(),
            "has_image": bool(r["contains_img"]),
        } for _, r in items.iterrows()]
    else:
        with open(args.questions_jsonl, encoding="utf-8") as handle:
            questions = [json.loads(line) for line in handle if line.strip()]
        for q in questions:
            q["id"] = str(q["id"])
            q["options"] = parse_options(q["options"])
            q["gold_answer"] = str(q["gold_answer"]).strip().upper()
            q["question_cluster_id"] = cluster_id(q["stem"])
        print(f"released question table: {len(questions)} items")

    if len(questions) != len({q["id"] for q in questions}):
        raise RuntimeError("Question ids are not unique")

    qpath = out_dir / "questions_2024_2025.jsonl"
    with open(qpath, "w", encoding="utf-8") as f:
        for rec in questions:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\nwrote {qpath} ({len(questions)} items, "
          f"{sum(r['has_image'] for r in questions)} of them reference an image)")

    # ---- teacher rationales ----------------------------------------------
    teacher = pd.read_pickle(args.teacher_file)
    print(f"\nteacher archive: {len(teacher)} generations")
    required = {"enunciado", "alternativas", "resposta", "pensamento", "letra"}
    missing = required - set(teacher.columns)
    if missing:
        raise KeyError(f"teacher archive is missing required columns: {sorted(missing)}")
    if "letra_fix" not in teacher.columns:
        from teacher_parser_fixed import parse_row
        if "output" not in teacher.columns:
            raise KeyError("audit parsing requires the teacher archive column 'output'")
        teacher["letra_fix"] = [
            parse_row(output, reasoning)
            for output, reasoning in zip(teacher["output"], teacher["pensamento"])
        ]
        print("computed letra_fix with teacher_parser_fixed.py")

    # A stem alone is not an item identifier: 4,260 evaluations contain 3,792
    # unique normalised stems.  The previous exporter used a dict from stem to
    # id, so every repeated stem inherited the final id in its group.  No row
    # was lost in the experiment, but the public join became many-to-one.  Match
    # on stem + complete option set + gold label and consume candidates once so
    # every teacher row receives exactly one stable item id.
    candidates = defaultdict(deque)
    question_by_id = {}
    for q in questions:
        candidates[item_key(q["stem"], q["options"], q["gold_answer"])].append(q["id"])
        question_by_id[q["id"]] = q

    matched_ids = []
    unmatched_rows = []
    for row_index, r in teacher.iterrows():
        key = item_key(r["enunciado"], r["alternativas"], r["resposta"])
        if not candidates[key]:
            unmatched_rows.append(int(row_index))
            matched_ids.append(None)
        else:
            matched_ids.append(candidates[key].popleft())

    leftovers = [item_id for queue in candidates.values() for item_id in queue]
    if unmatched_rows or leftovers:
        raise RuntimeError(
            "Teacher-to-question matching is not a bijection. "
            f"unmatched teacher rows={len(unmatched_rows)}, "
            f"unmatched question ids={len(leftovers)}. "
            "Do not fall back to a stem-only dict; inspect the source archives."
        )
    if len(set(matched_ids)) != len(matched_ids):
        raise RuntimeError("Teacher-to-question matching reused an item id")

    teacher["id"] = matched_ids
    print(f"teacher/question bijection: {len(matched_ids)} rows, "
          f"{len(set(matched_ids))} unique item ids")

    # `teacher_correct` uses the ORIGINAL parser, because that is what defined the
    # A/B/C/D partition every experiment in the paper was run on. The audit
    # parser's answer is published alongside it as `teacher_answer_audit` so the
    # disagreement is inspectable, but it is NOT the study label -- swapping it in
    # would silently redefine the subsets and detach the data from the results.
    rationales = []
    for _, r in teacher.iterrows():
        original = str(r["letra"]).strip().upper()
        audit = (str(r["letra_fix"]).strip().upper()
                 if pd.notna(r["letra_fix"]) else "?")
        question = question_by_id[r["id"]]
        gold = question["gold_answer"]
        source_gold = str(r["resposta"]).strip().upper()
        if source_gold != gold:
            raise RuntimeError(
                f"gold mismatch after item join for id={r['id']}: "
                f"teacher archive={source_gold}, item bank={gold}"
            )
        rationales.append({
            "id": r["id"],
            "question_cluster_id": question["question_cluster_id"],
            "rationale": r["pensamento"],
            "teacher_answer": original,
            "teacher_answer_audit": audit,
            "parsers_disagree": bool(original != audit),
            "teacher_correct": bool(original == gold),
            "teacher_correct_audit": bool(audit == gold),
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

    audit = {
        "n_questions": len(questions),
        "n_teacher_rationales": len(rationales),
        "n_unique_item_ids_questions": len({r["id"] for r in questions}),
        "n_unique_item_ids_rationales": len({r["id"] for r in rationales}),
        "n_unique_question_clusters": len({r["question_cluster_id"] for r in questions}),
        "n_repeated_stem_evaluations": (
            len(questions) - len({r["question_cluster_id"] for r in questions})
        ),
        "item_join_is_bijective": (
            {r["id"] for r in questions} == {r["id"] for r in rationales}
        ),
    }
    audit_path = out_dir / "mapping_audit.json"
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"wrote {audit_path}: {audit}")


if __name__ == "__main__":
    main()
