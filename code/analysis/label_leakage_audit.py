#!/usr/bin/env python3
"""CPU-only lexical-leakage sensitivity analysis for the rationale diagnostic.

The input item table must be in the same row order as the stored probability
arrays. The script deliberately uses transparent lexical diagnostics rather
than embeddings or a second language model:

1. exact normalised containment of the gold-option phrase;
2. stem-adjusted content-token coverage of each option by the rationale;
3. gold coverage minus the best distractor coverage;
4. coherent-minus-shuffled effects after excluding overt/high-overlap cases.

This does not detect all semantic paraphrases. It tests the narrower reviewer
concern that the probability result is driven by copied option wording.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


LETTERS = tuple("ABCDE")
STOPWORDS_PT = {
    "a", "ao", "aos", "aquela", "aquelas", "aquele", "aqueles", "aquilo",
    "as", "ate", "com", "como", "da", "das", "de", "dela", "dele", "do",
    "dos", "e", "ela", "elas", "ele", "eles", "em", "entre", "era", "essa",
    "essas", "esse", "esses", "esta", "estas", "este", "estes", "foi", "ha",
    "isso", "isto", "ja", "lhe", "lhes", "mais", "mas", "na", "nas", "no",
    "nos", "o", "os", "ou", "para", "pela", "pelas", "pelo", "pelos", "por",
    "qual", "que", "se", "sem", "ser", "seu", "seus", "sua", "suas", "tem",
    "um", "uma", "umas", "uns",
}


def normalise_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"</?think>", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def strip_explicit_answer_span(value: Any) -> str:
    """Remove common standalone final-answer lines before measuring overlap.

    The probability runner already removes its explicit final-answer span. This
    conservative text-side analogue prevents the audit from counting
    ``Resposta: A`` as lexical overlap. It intentionally does not remove option
    wording embedded in explanatory sentences, which is the signal under audit.
    """
    text = "" if value is None else str(value)
    text = re.sub(r"(?i)</?think>", " ", text)
    answer_line = re.compile(
        r"(?im)^\s*(?:"
        r"resposta(?:\s+(?:correta|final))?|"
        r"alternativa\s+correta|gabarito|"
        r"answer|final\s+answer"
        r")\s*[:\-]?\s*\**\s*[A-E]\s*\**"
        r"(?:\s*[\)\].:\-]\s*.*)?$"
    )
    return answer_line.sub(" ", text).strip()


def content_tokens(value: Any) -> set[str]:
    return {
        tok for tok in normalise_text(value).split()
        if len(tok) > 1 and tok not in STOPWORDS_PT
    }


def parse_options(value: Any) -> dict[str, str]:
    parsed = value
    if isinstance(parsed, str):
        for loader in (ast.literal_eval, json.loads):
            try:
                parsed = loader(parsed)
                break
            except Exception:
                pass
    if isinstance(parsed, dict):
        out = {}
        for key, text in parsed.items():
            letter = str(key).strip().upper()[:1]
            if letter in LETTERS:
                out[letter] = str(text)
        return out
    if isinstance(parsed, (list, tuple)):
        return {letter: str(text) for letter, text in zip(LETTERS, parsed)}
    if isinstance(value, str):
        matches = re.findall(
            r"(?:^|\n)\s*([A-E])\s*[\)\].:\-]\s*(.+?)(?=\n\s*[A-E]\s*[\)\].:\-]|\Z)",
            value,
            flags=re.I | re.S,
        )
        return {letter.upper(): text.strip() for letter, text in matches}
    return {}


def normalise_gold(value: Any) -> str:
    text = normalise_text(value).upper()
    if text and text[0] in LETTERS:
        return text[0]
    if text.isdigit() and 0 <= int(text) < 5:
        return LETTERS[int(text)]
    return ""


def pick_column(frame: pd.DataFrame, explicit: str | None, candidates: Iterable[str], label: str) -> str:
    if explicit:
        if explicit not in frame.columns:
            raise KeyError(f"--{label}-col={explicit!r} not found. Columns: {list(frame.columns)}")
        return explicit
    lower = {str(col).lower(): str(col) for col in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    raise KeyError(f"Could not infer {label} column. Columns: {list(frame.columns)}")


def load_items(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        obj = pd.read_pickle(path)
    elif suffix == ".parquet":
        obj = pd.read_parquet(path)
    elif suffix == ".csv":
        obj = pd.read_csv(path)
    elif suffix == ".jsonl":
        obj = pd.read_json(path, lines=True)
    elif suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            obj = json.load(handle)
    else:
        raise ValueError(f"Unsupported item-table format: {suffix}")
    if isinstance(obj, pd.DataFrame):
        return obj.reset_index(drop=True)
    if isinstance(obj, list):
        return pd.DataFrame(obj)
    if isinstance(obj, dict):
        for key in ("data", "items", "questions", "records"):
            if key in obj and isinstance(obj[key], list):
                return pd.DataFrame(obj[key])
        return pd.DataFrame(obj)
    raise TypeError(f"Unsupported object in {path}: {type(obj)}")


def lexical_table(
    frame: pd.DataFrame,
    question_col: str,
    options_col: str,
    gold_col: str,
    rationale_col: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row_number, row in frame.iterrows():
        question = row[question_col]
        rationale = strip_explicit_answer_span(row[rationale_col])
        options = parse_options(row[options_col])
        gold = normalise_gold(row[gold_col])
        stem_tokens = content_tokens(question)
        rationale_norm = normalise_text(rationale)
        rationale_tokens = content_tokens(rationale)

        coverage: dict[str, float] = {}
        exact: dict[str, bool] = {}
        substantive_exact: dict[str, bool] = {}
        for letter in LETTERS:
            option = options.get(letter, "")
            option_norm = normalise_text(option)
            option_tokens_all = content_tokens(option)
            option_tokens = option_tokens_all - stem_tokens
            if not option_tokens:
                option_tokens = option_tokens_all
            coverage[letter] = (
                len(option_tokens & rationale_tokens) / len(option_tokens)
                if option_tokens else np.nan
            )
            exact[letter] = bool(option_norm and option_norm in rationale_norm)
            substantive_exact[letter] = bool(
                exact[letter]
                and (len(option_tokens_all) >= 2 or len(option_norm) >= 12)
            )

        distractor_coverages = [
            coverage[letter] for letter in LETTERS
            if letter != gold and np.isfinite(coverage[letter])
        ]
        gold_coverage = coverage.get(gold, np.nan)
        best_distractor = max(distractor_coverages) if distractor_coverages else np.nan
        margin = (
            gold_coverage - best_distractor
            if np.isfinite(gold_coverage) and np.isfinite(best_distractor)
            else np.nan
        )
        rows.append(
            {
                "row": row_number,
                "cluster_id": normalise_text(question),
                "gold": gold,
                "rationale_words": len(normalise_text(rationale).split()),
                "exact_gold_any": exact.get(gold, False),
                "exact_gold_substantive": substantive_exact.get(gold, False),
                "exact_any_option_substantive": any(substantive_exact.values()),
                "gold_coverage": gold_coverage,
                "best_distractor_coverage": best_distractor,
                "gold_overlap_margin": margin,
            }
        )
    return pd.DataFrame(rows)


def cluster_bootstrap_effect(
    delta: np.ndarray,
    teacher_correct: np.ndarray,
    cluster_ids: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    replicates: int,
) -> dict[str, float | int]:
    valid = (
        mask.astype(bool)
        & np.isfinite(delta)
        & pd.notna(cluster_ids)
    )
    indices = np.flatnonzero(valid)
    if not len(indices):
        raise ValueError("No valid rows remain for this sensitivity condition.")
    tc = teacher_correct.astype(bool)
    if not tc[indices].any() or tc[indices].all():
        raise ValueError("Both teacher-correct and teacher-wrong rows are required.")

    clusters = np.unique(cluster_ids[indices])
    members = {cluster: indices[cluster_ids[indices] == cluster] for cluster in clusters}

    def statistic(sample_indices: np.ndarray) -> tuple[float, float, float]:
        correct = float(np.mean(delta[sample_indices][tc[sample_indices]]))
        wrong = float(np.mean(delta[sample_indices][~tc[sample_indices]]))
        return correct, wrong, correct - wrong

    observed = statistic(indices)
    boot = np.empty((replicates, 3), dtype=float)
    for b in range(replicates):
        sampled_clusters = rng.choice(clusters, size=len(clusters), replace=True)
        sampled_indices = np.concatenate([members[cluster] for cluster in sampled_clusters])
        boot[b] = statistic(sampled_indices)
    lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
    return {
        "n_items": int(len(indices)),
        "n_clusters": int(len(clusters)),
        "C_correct": observed[0],
        "C_correct_lo": lo[0],
        "C_correct_hi": hi[0],
        "C_wrong": observed[1],
        "C_wrong_lo": lo[1],
        "C_wrong_hi": hi[1],
        "interaction": observed[2],
        "interaction_lo": lo[2],
        "interaction_hi": hi[2],
    }


def load_vector(path: Path | None, expected_length: int, label: str) -> np.ndarray | None:
    if path is None:
        return None
    array = np.asarray(np.load(path, allow_pickle=True)).reshape(-1)
    if len(array) != expected_length:
        raise ValueError(f"{label} has {len(array)} rows; item table has {expected_length}.")
    return array


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=Path, required=True, help="Aligned item table (.pkl/.parquet/.csv/.json/.jsonl).")
    parser.add_argument("--question-col")
    parser.add_argument("--options-col")
    parser.add_argument("--gold-col")
    parser.add_argument("--rationale-col")
    parser.add_argument("--teacher-correct", type=Path, help="Boolean .npy array, e.g. tc_student_nothink.npy.")
    parser.add_argument("--coherent", type=Path, help="Coherent probability trajectory .npy array.")
    parser.add_argument("--shuffled", type=Path, help="Shuffled probability trajectory .npy array.")
    parser.add_argument("--cluster-ids", type=Path, help="Optional aligned cluster-ID .npy array.")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("label_leakage_audit"))
    args = parser.parse_args()

    frame = load_items(args.items)
    question_col = pick_column(
        frame, args.question_col,
        ("enunciado", "question", "questao", "pergunta", "stem", "prompt"),
        "question",
    )
    options_col = pick_column(
        frame, args.options_col,
        ("alternativas", "alternatives", "options", "choices"),
        "options",
    )
    gold_col = pick_column(
        frame, args.gold_col,
        ("resposta", "gold", "gabarito", "label", "answer"),
        "gold",
    )
    rationale_col = pick_column(
        frame, args.rationale_col,
        ("target_thinking", "thinking", "pensamento", "rationale", "reasoning", "teacher_rationale"),
        "rationale",
    )
    audit = lexical_table(frame, question_col, options_col, gold_col, rationale_col)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.out_dir / "label_leakage_items.csv", index=False)

    finite_margin = audit["gold_overlap_margin"].to_numpy(dtype=float)
    threshold = float(np.nanquantile(finite_margin, 0.90))
    exact = audit["exact_gold_substantive"].to_numpy(dtype=bool)
    high_margin = finite_margin >= threshold
    lexical_summary: dict[str, Any] = {
        "n_items": int(len(audit)),
        "n_unique_text_keys_after_audit_normalisation": int(audit["cluster_id"].nunique()),
        "exact_gold_any_fraction": float(audit["exact_gold_any"].mean()),
        "exact_gold_substantive_fraction": float(audit["exact_gold_substantive"].mean()),
        "exact_any_option_substantive_fraction": float(audit["exact_any_option_substantive"].mean()),
        "gold_coverage_median": float(np.nanmedian(audit["gold_coverage"])),
        "best_distractor_coverage_median": float(np.nanmedian(audit["best_distractor_coverage"])),
        "gold_overlap_margin_median": float(np.nanmedian(finite_margin)),
        "gold_overlap_margin_q90": threshold,
    }

    effect_summary: list[dict[str, Any]] = []
    supplied = (args.teacher_correct, args.coherent, args.shuffled)
    if any(path is not None for path in supplied) and not all(path is not None for path in supplied):
        parser.error("--teacher-correct, --coherent, and --shuffled must be supplied together.")
    if all(path is not None for path in supplied):
        n = len(audit)
        teacher_correct = load_vector(args.teacher_correct, n, "teacher-correct")
        coherent = np.asarray(np.load(args.coherent), dtype=float)
        shuffled = np.asarray(np.load(args.shuffled), dtype=float)
        if coherent.ndim != 2 or shuffled.ndim != 2 or coherent.shape != shuffled.shape:
            raise ValueError("Coherent and shuffled arrays must be aligned 2D arrays of identical shape.")
        if coherent.shape[0] != n:
            raise ValueError(f"Trajectory arrays have {coherent.shape[0]} rows; item table has {n}.")
        cluster_ids = load_vector(args.cluster_ids, n, "cluster-ids")
        if cluster_ids is None:
            cluster_ids = audit["cluster_id"].to_numpy(dtype=object)
        lexical_summary["n_bootstrap_clusters"] = int(len(np.unique(cluster_ids)))
        delta = coherent[:, -1] - shuffled[:, -1]
        conditions = {
            "all_items": np.ones(n, dtype=bool),
            "exclude_exact_gold": ~exact,
            "exclude_top_decile_overlap_margin": ~high_margin,
            "exclude_exact_or_top_decile": ~(exact | high_margin),
        }
        rng = np.random.default_rng(args.seed)
        for name, mask in conditions.items():
            result = cluster_bootstrap_effect(
                delta, teacher_correct.astype(bool), cluster_ids, mask, rng, args.bootstrap
            )
            effect_summary.append({"condition": name, **result})
        pd.DataFrame(effect_summary).to_csv(
            args.out_dir / "label_leakage_effect_sensitivity.csv", index=False
        )

    payload = {
        "columns": {
            "question": question_col,
            "options": options_col,
            "gold": gold_col,
            "rationale": rationale_col,
        },
        "lexical_summary": lexical_summary,
        "effect_sensitivity": effect_summary,
        "interpretation": (
            "This audit detects overt lexical overlap, not all semantic paraphrases. "
            "Exclusion sensitivities reuse stored inference and are not masked-rationale re-inference."
        ),
    }
    with (args.out_dir / "label_leakage_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

