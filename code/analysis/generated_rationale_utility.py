#!/usr/bin/env python3
"""Functional utility audit for generated rationales.

Primary comparison:
  distill-SFT vs step-by-step, alpha=1.0, ten paired data splits.

The frozen base Qwen3-4B reader scores the restricted A--E distribution under:
  (1) question only;
  (2) question + the generated rationale;
  (3) question + a cross-question-shuffled generated rationale.

This measures answer-directed external-reader utility. It is not a factuality,
clinical-quality, calibration, or mechanistic-faithfulness assessment.
"""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from tqdm import tqdm
from unsloth import FastLanguageModel


LETTERS = np.array(list("ABCDE"))
DEFAULT_SEEDS = [8, 12, 17, 23, 25, 31, 37, 44, 52, 61]
DEFAULT_TECHNIQUES = ["distill_sft", "step_by_step"]


def normalize_question(text: object) -> str:
    value = unicodedata.normalize("NFKD", str(text))
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = re.sub(r"\s+", " ", value.lower()).strip()
    return value


def question_id(text: object) -> str:
    return hashlib.sha1(normalize_question(text).encode("utf-8")).hexdigest()[:16]


def format_options(value: object) -> str:
    if isinstance(value, str):
        value = ast.literal_eval(value)
    if not isinstance(value, dict):
        raise TypeError(f"alternativas must be dict-like, received {type(value)}")
    return "\n".join(
        f"{str(key).strip().upper()}) {text}"
        for key, text in sorted(value.items())
    )


def strip_generated_answer(text: object) -> str:
    """Keep the generated rationale and remove explicit answer material."""
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""
    value = str(text).strip()
    if not value:
        return ""

    # In reasoning inference, the generated rationale normally precedes </think>.
    if "</think>" in value:
        value = value.split("</think>", 1)[0]

    value = re.sub(r"(?is)^.*?<think>\s*", "", value)
    value = re.split(
        r"(?is)\b(?:resposta|resposta\s+final|alternativa\s+correta|answer)"
        r"\s*[:\-]",
        value,
        maxsplit=1,
    )[0]
    value = re.sub(r"(?is)</?think>", "", value)
    value = re.sub(
        r"(?is)^\s*vamos\s+analisar\s+a\s+quest[aã]o\s*:\s*",
        "",
        value,
    )
    return value.strip()


def derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    if n < 2:
        raise ValueError("Need at least two items for cross-question shuffling")
    order = rng.permutation(n)
    mapping = np.empty(n, dtype=int)
    mapping[order] = np.roll(order, 1)
    if np.any(mapping == np.arange(n)):
        raise AssertionError("Derangement construction produced a fixed point")
    return mapping


def render_prompt(tokenizer, row: pd.Series, rationale: str) -> str:
    content = (
        f"Questão:\n{row['enunciado']}\n\n"
        f"Alternativas:\n{format_options(row['alternativas'])}\n\n"
    )
    if rationale:
        content += f"Raciocínio de apoio:\n{rationale}\n\n"
    content += (
        "Com base na questão e, quando presente, no raciocínio de apoio, "
        "responda apenas com a letra da alternativa correta."
    )
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    # Force the next scored token to be the answer letter.
    return prompt.rstrip() + "\nResposta:"


def letter_token_ids(tokenizer) -> list[int]:
    ids = []
    for letter in LETTERS:
        encoded = tokenizer.encode(" " + letter, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(
                f"Expected one token for continuation ' {letter}', got {encoded}. "
                "Use the same tokenizer/model as the executed Qwen3 protocol."
            )
        ids.append(int(encoded[0]))
    return ids


@torch.no_grad()
def score_prompts(
    model,
    tokenizer,
    prompts: list[str],
    batch_size: int,
    max_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return restricted A--E probabilities and input-token counts."""
    FastLanguageModel.for_inference(model)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "right"
    candidate_ids = letter_token_ids(tokenizer)

    all_probs: list[np.ndarray] = []
    all_lengths: list[np.ndarray] = []
    for start in tqdm(
        range(0, len(prompts), batch_size),
        desc="reader scoring",
        unit="batch",
        leave=True,
        dynamic_ncols=True,
        mininterval=0.5,
    ):
        batch = prompts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        ).to(model.device)
        outputs = model(**encoded)
        next_logits = outputs.logits[:, -1, :].float()[:, candidate_ids]
        probabilities = torch.softmax(next_logits, dim=-1)
        all_probs.append(probabilities.cpu().numpy())
        all_lengths.append(encoded["attention_mask"].sum(dim=1).cpu().numpy())

    return np.concatenate(all_probs), np.concatenate(all_lengths)


def probability_for_letter(matrix: np.ndarray, labels: pd.Series) -> np.ndarray:
    index = {letter: idx for idx, letter in enumerate(LETTERS)}
    positions = np.array([index.get(str(value).strip().upper(), -1) for value in labels])
    result = np.full(len(labels), np.nan, dtype=float)
    valid = positions >= 0
    result[valid] = matrix[np.arange(len(labels))[valid], positions[valid]]
    return result


def load_aligned_unit(
    splits_dir: Path,
    outputs_root: Path,
    technique: str,
    seed: int,
    split: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if split == "B":
        data_path = splits_dir / f"df_test_abc_seed{seed}.pkl"
        output_name = "infer_abc_reasoning.csv"
    elif split == "D":
        data_path = splits_dir / f"df_test_hard_seed{seed}.pkl"
        output_name = "infer_hard_reasoning.csv"
    else:
        raise ValueError(split)

    output_path = (
        outputs_root / f"{technique}_seed{seed}_ep2" / output_name
    )
    data = pd.read_pickle(data_path).reset_index(drop=False).rename(
        columns={"index": "original_row"}
    )
    generated = pd.read_csv(output_path)

    if "gen" not in generated.columns:
        raise KeyError(
            f"{output_path} has no 'gen' column. Use the original inference CSV, "
            "not a reconstructed compact CSV."
        )
    if len(data) != len(generated):
        raise AssertionError(
            f"Length mismatch for {technique}, seed={seed}, split={split}: "
            f"data={len(data)}, generated={len(generated)}"
        )

    gold_data = data["resposta"].astype(str).str.strip().str.upper()
    gold_output = generated["gold"].astype(str).str.strip().str.upper()
    if not gold_data.equals(gold_output):
        raise AssertionError(
            f"Gold/order mismatch for {technique}, seed={seed}, split={split}"
        )
    if "subset_code" in generated and "subset_code" in data:
        if not (
            generated["subset_code"].astype(str).str.upper().reset_index(drop=True)
            == data["subset_code"].astype(str).str.upper().reset_index(drop=True)
        ).all():
            raise AssertionError(
                f"Subset/order mismatch for {technique}, seed={seed}, split={split}"
            )

    if split == "B":
        keep = data["subset_code"].astype(str).str.upper().eq("B")
        data = data.loc[keep].reset_index(drop=True)
        generated = generated.loc[keep.to_numpy()].reset_index(drop=True)

    if split == "D" and len(data) != 202:
        raise AssertionError(
            f"Expected the Stage-2 held-out D set to contain 202 items, got {len(data)}"
        )
    return data, generated


def score_unit(
    model,
    tokenizer,
    splits_dir: Path,
    outputs_root: Path,
    technique: str,
    seed: int,
    split: str,
    batch_size: int,
    max_length: int,
    shuffle_seed: int,
) -> pd.DataFrame:
    data, generated = load_aligned_unit(
        splits_dir, outputs_root, technique, seed, split
    )
    rationales = generated["gen"].map(strip_generated_answer).tolist()
    rng = np.random.default_rng(
        shuffle_seed + 1009 * seed + (0 if technique == "distill_sft" else 1)
    )
    permutation = derangement(len(data), rng)
    shuffled = [rationales[idx] for idx in permutation]

    prompts_base = [
        render_prompt(tokenizer, row, "") for _, row in data.iterrows()
    ]
    prompts_coherent = [
        render_prompt(tokenizer, row, rationale)
        for (_, row), rationale in zip(data.iterrows(), rationales)
    ]
    prompts_shuffled = [
        render_prompt(tokenizer, row, rationale)
        for (_, row), rationale in zip(data.iterrows(), shuffled)
    ]
    all_prompts = prompts_base + prompts_coherent + prompts_shuffled
    probabilities, lengths = score_prompts(
        model, tokenizer, all_prompts, batch_size, max_length
    )
    n = len(data)
    p_base, p_coherent, p_shuffled = np.split(probabilities, [n, 2 * n])
    l_base, l_coherent, l_shuffled = np.split(lengths, [n, 2 * n])

    gold = generated["gold"].astype(str).str.strip().str.upper()
    pred = generated["pred"].astype(str).str.strip().str.upper()
    result = pd.DataFrame(
        {
            "technique": technique,
            "seed": seed,
            "split": split,
            "original_row": data["original_row"].to_numpy(),
            "question_id": data["enunciado"].map(question_id),
            "gold": gold,
            "pred": pred,
            "generator_correct": generated["acertou"].astype(int),
            "rationale_chars": [len(value) for value in rationales],
            "rationale_empty": [int(not value) for value in rationales],
            "input_tokens_base": l_base,
            "input_tokens_coherent": l_coherent,
            "input_tokens_shuffled": l_shuffled,
            "p_gold_base": probability_for_letter(p_base, gold),
            "p_gold_coherent": probability_for_letter(p_coherent, gold),
            "p_gold_shuffled": probability_for_letter(p_shuffled, gold),
            "p_pred_base": probability_for_letter(p_base, pred),
            "p_pred_coherent": probability_for_letter(p_coherent, pred),
            "p_pred_shuffled": probability_for_letter(p_shuffled, pred),
        }
    )
    for idx, letter in enumerate(LETTERS):
        result[f"p_{letter}_base"] = p_base[:, idx]
        result[f"p_{letter}_coherent"] = p_coherent[:, idx]
        result[f"p_{letter}_shuffled"] = p_shuffled[:, idx]

    result["d_gold"] = result["p_gold_coherent"] - result["p_gold_base"]
    result["C_gold"] = result["p_gold_coherent"] - result["p_gold_shuffled"]
    result["d_pred"] = result["p_pred_coherent"] - result["p_pred_base"]
    result["C_pred"] = result["p_pred_coherent"] - result["p_pred_shuffled"]
    return result


def safe_wilcoxon(left: pd.Series, right: pd.Series) -> float:
    if len(left) < 2 or len(right) < 2:
        return float("nan")
    delta = left.to_numpy() - right.to_numpy()
    if np.allclose(delta, 0):
        return 1.0
    return float(stats.wilcoxon(left, right).pvalue)


def safe_paired_t(left: pd.Series, right: pd.Series) -> float:
    if len(left) < 2 or len(right) < 2:
        return float("nan")
    return float(stats.ttest_rel(left, right).pvalue)


def hierarchical_bootstrap_difference(
    frame: pd.DataFrame,
    col_distill: str,
    col_step: str,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    seeds = frame["seed"].unique()
    values = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        seed_effects = []
        for seed in sampled_seeds:
            group = frame.loc[frame["seed"].eq(seed)]
            sampled_rows = rng.integers(0, len(group), size=len(group))
            sampled = group.iloc[sampled_rows]
            seed_effects.append(
                np.nanmean(
                    sampled[col_distill].to_numpy()
                    - sampled[col_step].to_numpy()
                )
            )
        values[b] = np.nanmean(seed_effects)
    return tuple(np.nanpercentile(values, [2.5, 97.5]))


def summarise(scores: pd.DataFrame, out_dir: Path, n_bootstrap: int) -> None:
    metric_columns = ["d_gold", "C_gold", "d_pred", "C_pred"]
    summary_rows = []
    for (technique, split), group0 in scores.groupby(["technique", "split"]):
        masks = {
            "all": np.ones(len(group0), dtype=bool),
            "generator_correct": group0["generator_correct"].eq(1).to_numpy(),
            "generator_wrong": group0["generator_correct"].eq(0).to_numpy(),
        }
        for group_name, mask in masks.items():
            group = group0.loc[mask]
            for metric in metric_columns:
                per_seed = group.groupby("seed")[metric].mean()
                summary_rows.append(
                    {
                        "technique": technique,
                        "split": split,
                        "group": group_name,
                        "metric": metric,
                        "n_evaluations": len(group),
                        "n_seeds": per_seed.notna().sum(),
                        "mean_across_seed_means": per_seed.mean(),
                        "sd_across_seed_means": per_seed.std(ddof=1),
                    }
                )
    pd.DataFrame(summary_rows).to_csv(
        out_dir / "summary_by_technique.csv", index=False
    )

    key = ["seed", "split", "original_row", "question_id"]
    keep = key + ["generator_correct"] + metric_columns
    distill = scores.loc[scores["technique"].eq("distill_sft"), keep].rename(
        columns={
            "generator_correct": "correct_distill",
            **{metric: f"{metric}_distill" for metric in metric_columns},
        }
    )
    step = scores.loc[scores["technique"].eq("step_by_step"), keep].rename(
        columns={
            "generator_correct": "correct_step",
            **{metric: f"{metric}_step" for metric in metric_columns},
        }
    )
    paired = distill.merge(step, on=key, how="inner", validate="one_to_one")
    if len(paired) != len(distill) or len(paired) != len(step):
        raise AssertionError("The technique comparison is not fully paired")

    rng = np.random.default_rng(20260727)
    comparison_rows = []
    for split, split_frame in paired.groupby("split"):
        masks = {
            "all": np.ones(len(split_frame), dtype=bool),
            "both_correct": (
                split_frame["correct_distill"].eq(1)
                & split_frame["correct_step"].eq(1)
            ).to_numpy(),
            "both_wrong": (
                split_frame["correct_distill"].eq(0)
                & split_frame["correct_step"].eq(0)
            ).to_numpy(),
        }
        for group_name, mask in masks.items():
            group = split_frame.loc[mask]
            if group.empty:
                continue
            for metric in metric_columns:
                d_col, s_col = f"{metric}_distill", f"{metric}_step"
                per_seed = group.groupby("seed")[[d_col, s_col]].mean().dropna()
                delta = per_seed[d_col] - per_seed[s_col]
                ci_lo, ci_hi = hierarchical_bootstrap_difference(
                    group, d_col, s_col, n_bootstrap, rng
                )
                comparison_rows.append(
                    {
                        "split": split,
                        "group": group_name,
                        "metric": metric,
                        "n_paired_evaluations": len(group),
                        "n_seeds": len(per_seed),
                        "distill_mean": per_seed[d_col].mean(),
                        "step_mean": per_seed[s_col].mean(),
                        "delta_distill_minus_step": delta.mean(),
                        "delta_ci_lo": ci_lo,
                        "delta_ci_hi": ci_hi,
                        "paired_t_p": safe_paired_t(
                            per_seed[d_col], per_seed[s_col]
                        ),
                        "wilcoxon_p": safe_wilcoxon(
                            per_seed[d_col], per_seed[s_col]
                        ),
                        "splits_favour_distill": int((delta > 0).sum()),
                        "splits_favour_step": int((delta < 0).sum()),
                        "split_ties": int((delta == 0).sum()),
                    }
                )

    comparisons = pd.DataFrame(comparison_rows)
    comparisons.to_csv(out_dir / "paired_comparisons.csv", index=False)

    print("\n=== HEADLINE: coherent-minus-shuffled gold probability ===")
    headline = comparisons.loc[
        comparisons["metric"].eq("C_gold")
        & comparisons["group"].isin(["all", "both_correct", "both_wrong"])
    ].copy()
    print(
        headline[
            [
                "split",
                "group",
                "n_paired_evaluations",
                "distill_mean",
                "step_mean",
                "delta_distill_minus_step",
                "delta_ci_lo",
                "delta_ci_hi",
                "paired_t_p",
                "wilcoxon_p",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", default="outputs_etapa2")
    parser.add_argument("--splits-dir", default="data/splits_etapa2")
    parser.add_argument("--out-dir", default="generated_rationale_utility")
    parser.add_argument("--model-name", default="unsloth/Qwen3-4B")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--techniques", nargs="+", default=DEFAULT_TECHNIQUES
    )
    parser.add_argument("--splits", nargs="+", choices=["B", "D"], default=["B", "D"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--shuffle-seed", type=int, default=20260727)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs_root = Path(args.outputs_root)
    splits_dir = Path(args.splits_dir)
    out_dir = Path(args.out_dir)
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    expected_cache = [
        cache_dir / f"scores_{technique}_seed{seed}_{split}.parquet"
        for technique in args.techniques
        for seed in args.seeds
        for split in args.splits
    ]
    missing = [path for path in expected_cache if args.force or not path.exists()]

    model = tokenizer = None
    if missing:
        print(
            f"Loading frozen reader {args.model_name}; "
            f"{len(missing)}/{len(expected_cache)} units require scoring."
        )
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_name,
            max_seq_length=args.max_length,
            dtype=None,
            load_in_4bit=False,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    try:
        unit_counter = 0
        total_units = len(expected_cache)
        for technique in args.techniques:
            for seed in args.seeds:
                for split in args.splits:
                    unit_counter += 1
                    cache_path = (
                        cache_dir
                        / f"scores_{technique}_seed{seed}_{split}.parquet"
                    )
                    if cache_path.exists() and not args.force:
                        print(
                            f"[{unit_counter}/{total_units}] "
                            f"[cache] {cache_path.name}"
                        )
                        continue
                    print(
                        f"\n[{unit_counter}/{total_units}] "
                        f"Scoring technique={technique}, seed={seed}, split={split}"
                    )
                    result = score_unit(
                        model=model,
                        tokenizer=tokenizer,
                        splits_dir=splits_dir,
                        outputs_root=outputs_root,
                        technique=technique,
                        seed=seed,
                        split=split,
                        batch_size=args.batch_size,
                        max_length=args.max_length,
                        shuffle_seed=args.shuffle_seed,
                    )
                    result.to_parquet(cache_path, index=False)
                    print(
                        f"saved {cache_path} | n={len(result)} | "
                        f"empty rationale={result['rationale_empty'].mean():.2%} | "
                        f"max input tokens={result['input_tokens_coherent'].max()}"
                    )
    finally:
        if model is not None:
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()

    scores = pd.concat(
        [pd.read_parquet(path) for path in expected_cache],
        ignore_index=True,
    )
    scores.to_parquet(out_dir / "per_item_scores.parquet", index=False)
    summarise(scores, out_dir, args.bootstrap)

    metadata = {
        "reader": args.model_name,
        "techniques": args.techniques,
        "seeds": args.seeds,
        "splits": args.splits,
        "estimands": {
            "d_gold": "P(gold|question, coherent rationale) - P(gold|question)",
            "C_gold": "P(gold|question, coherent rationale) - P(gold|question, shuffled rationale)",
            "d_pred": "P(generator answer|question, coherent rationale) - P(generator answer|question)",
            "C_pred": "P(generator answer|question, coherent rationale) - P(generator answer|question, shuffled rationale)",
        },
        "scope": (
            "External-reader functional utility only; not a factuality, "
            "clinical-quality, calibration, or mechanistic-faithfulness test."
        ),
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    print(f"\nDONE. Results saved to {out_dir.resolve()}")


if __name__ == "__main__":
    main()

