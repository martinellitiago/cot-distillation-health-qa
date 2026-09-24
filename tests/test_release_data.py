"""Release gate for the public question/rationale tables.

These checks prevent a repeated question stem from silently collapsing multiple
evaluations onto one item id.  The paper analyses 4,260 evaluation rows grouped
into 3,792 normalised-stem clusters; those are different units and both must be
represented explicitly in the release.
"""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(name):
    path = ROOT / "data" / name
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_question_rationale_join_is_bijective():
    questions = load_jsonl("questions_2024_2025.jsonl")
    rationales = load_jsonl("teacher_rationales.jsonl")

    assert len(questions) == len(rationales) == 4260
    question_ids = [row["id"] for row in questions]
    rationale_ids = [row["id"] for row in rationales]
    assert len(set(question_ids)) == 4260
    assert len(set(rationale_ids)) == 4260
    assert set(question_ids) == set(rationale_ids)


def test_cluster_ids_encode_repeated_stems():
    questions = load_jsonl("questions_2024_2025.jsonl")
    rationales = load_jsonl("teacher_rationales.jsonl")
    q_cluster = {row["id"]: row["question_cluster_id"] for row in questions}
    r_cluster = {row["id"]: row["question_cluster_id"] for row in rationales}

    assert len(set(q_cluster.values())) == 3792
    assert q_cluster == r_cluster


def test_options_and_teacher_labels_are_internally_consistent():
    questions = load_jsonl("questions_2024_2025.jsonl")
    rationales = load_jsonl("teacher_rationales.jsonl")
    question_by_id = {row["id"]: row for row in questions}

    assert {len(row["options"]) for row in questions} == {4, 5}
    assert sum(len(row["options"]) == 4 for row in questions) == 409
    for row in rationales:
        gold = question_by_id[row["id"]]["gold_answer"]
        assert row["teacher_correct"] == (row["teacher_answer"] == gold)
        assert row["teacher_correct_audit"] == (row["teacher_answer_audit"] == gold)


def test_mapping_audit_records_the_release_gate():
    audit = json.loads((ROOT / "data" / "mapping_audit.json").read_text(encoding="utf-8"))
    assert audit == {
        "n_questions": 4260,
        "n_teacher_rationales": 4260,
        "n_unique_item_ids_questions": 4260,
        "n_unique_item_ids_rationales": 4260,
        "n_unique_question_clusters": 3792,
        "n_repeated_stem_evaluations": 468,
        "item_join_is_bijective": True,
    }


def main():
    checks = [
        test_question_rationale_join_is_bijective,
        test_cluster_ids_encode_repeated_stems,
        test_options_and_teacher_labels_are_internally_consistent,
        test_mapping_audit_records_the_release_gate,
    ]
    for check in checks:
        check()
        print(f"PASS {check.__name__}")
    print("RELEASE DATA GATE: PASS")


if __name__ == "__main__":
    main()
